#include <bits/stdc++.h>
#include <omp.h>
#include <mpi.h>

int NUM_THREADS = 4;
using namespace std;

struct Edge {
    int source, destination;
    double weight;
    Edge(int src, int dest, double w) : source(src), destination(dest), weight(w) {}
};

struct PairHash {
    size_t operator()(const pair<int, int>& p) const {
        return hash<int>{}(p.first) ^ (hash<int>{}(p.second) << 1);
    }
};

using Graph = unordered_map<int, unordered_map<int, double>>;
using MultiObjectiveGraph = vector<Graph>;
using EdgeKey = pair<int, int>;

double& access_with_default(Graph& g, int key1, int key2, double default_value = 4.0) {
    return g[key1][key2];
}

Graph buildOriginalGraph(const vector<vector<Edge>>& graphDT, const vector<vector<Edge>>& changedEdgesDT) {
    Graph graph;
    for (const auto& edges : graphDT) {
        for (const auto& edge : edges) {
            graph[edge.source + 1][edge.destination + 1] = edge.weight;
        }
    }
    for (const auto& edges : changedEdgesDT) {
        for (const auto& edge : edges) {
            if (edge.weight > 0) {
                graph[edge.source + 1][edge.destination + 1] = edge.weight;
            } else {
                graph[edge.source + 1].erase(edge.destination + 1);
                if (graph[edge.source + 1].empty()) graph.erase(edge.source + 1);
            }
        }
    }
    return graph;
}

void serializeGraph(const Graph& graph, vector<int>& nodes, vector<int>& neighbors, vector<double>& weights, int& maxNode) {
    maxNode = 0;
    nodes.clear();
    neighbors.clear();
    weights.clear();
    for (const auto& [node, neighborMap] : graph) {
        maxNode = max(maxNode, node);
        for (const auto& [neighbor, weight] : neighborMap) {
            maxNode = max(maxNode, neighbor);
            nodes.push_back(node);
            neighbors.push_back(neighbor);
            weights.push_back(weight);
        }
    }
}

Graph deserializeGraph(const vector<int>& nodes, const vector<int>& neighbors, const vector<double>& weights) {
    Graph graph;
    for (size_t i = 0; i < nodes.size(); ++i) {
        graph[nodes[i]][neighbors[i]] = weights[i];
    }
    return graph;
}

bool bellmanFordMOSP(const Graph& ensembleGraph, const MultiObjectiveGraph& originalGraphs, int source, int numObjectives, unordered_map<int, vector<double>>& distances, vector<int>& newParent, vector<vector<int>>& ssspTree, vector<vector<int>>& objParents, unordered_map<int, double>& ensembleDistances, int rank, int size) {
    int maxNode = 0;
    for (const auto& [node, neighbors] : ensembleGraph) {
        maxNode = max(maxNode, node);
        for (const auto& [neighbor, _] : neighbors) {
            maxNode = max(maxNode, neighbor);
        }
    }

    newParent.assign(maxNode + 1, -1);
    objParents.assign(numObjectives, vector<int>(maxNode + 1, -1));
    distances.clear();
    ensembleDistances.clear();
    ssspTree.assign(maxNode, {});

    // Initialize distances
    for (const auto& [node, _] : ensembleGraph) {
        distances[node] = vector<double>(numObjectives, numeric_limits<double>::infinity());
        ensembleDistances[node] = numeric_limits<double>::infinity();
        for (const auto& [neighbor, _] : ensembleGraph.at(node)) {
            distances[neighbor] = vector<double>(numObjectives, numeric_limits<double>::infinity());
            ensembleDistances[neighbor] = numeric_limits<double>::infinity();
        }
    }
    distances[source] = vector<double>(numObjectives, 0.0);
    newParent[source] = -1;
    for (int i = 0; i < numObjectives; ++i) {
        objParents[i][source] = -1;
    }
    ensembleDistances[source] = 0.0;

    // Divide nodes among MPI processes
    vector<int> localNodes;
    for (const auto& [node, _] : ensembleGraph) {
        if (node % size == rank) {
            localNodes.push_back(node);
        }
    }

    // Precompute edges for efficiency
    vector<tuple<int, int, double>> edges;
    for (const auto& [u, neighbors] : ensembleGraph) {
        for (const auto& [v, weight] : neighbors) {
            edges.emplace_back(u, v, weight);
        }
    }

    // Run Bellman-Ford for ensemble graph
    vector<double> localMinDistances(maxNode + 1, numeric_limits<double>::infinity());
    vector<int> localParents(maxNode + 1, -1);
    for (int node = 1; node <= maxNode; ++node) {
        localMinDistances[node] = ensembleDistances[node];
        localParents[node] = newParent[node];
    }

    for (size_t i = 0; i < maxNode; ++i) {
        bool localUpdated = false;

        #pragma omp parallel for num_threads(NUM_THREADS) reduction(||:localUpdated)
        for (size_t j = 0; j < edges.size(); ++j) {
            auto [u, v, weight] = edges[j];
            if (localNodes.empty() || find(localNodes.begin(), localNodes.end(), u) != localNodes.end()) {
                if (!isinf(ensembleDistances[u]) && ensembleDistances[u] + weight < localMinDistances[v]) {
                    #pragma omp atomic write
                    localMinDistances[v] = ensembleDistances[u] + weight;
                    #pragma omp atomic write
                    localParents[v] = u;
                    localUpdated = true;
                }
            }
        }

        // Synchronize only when necessary
        int localUpdateFlag = localUpdated ? 1 : 0;
        int globalUpdateFlag = 0;
        MPI_Allreduce(&localUpdateFlag, &globalUpdateFlag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (!globalUpdateFlag) break;

        // Synchronize distances and parents
        MPI_Allreduce(MPI_IN_PLACE, localMinDistances.data(), maxNode + 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, localParents.data(), maxNode + 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        for (int node = 1; node <= maxNode; ++node) {
            ensembleDistances[node] = localMinDistances[node];
            if (localParents[node] != -1) {
                newParent[node] = localParents[node];
            }
        }
    }

    // Check for negative-weight cycles
    bool hasNegativeCycle = false;
    #pragma omp parallel for num_threads(NUM_THREADS) reduction(||:hasNegativeCycle)
    for (size_t j = 0; j < edges.size(); ++j) {
        auto [u, v, weight] = edges[j];
        if (!isinf(ensembleDistances[u]) && ensembleDistances[u] + weight < ensembleDistances[v]) {
            hasNegativeCycle = true;
        }
    }
    int globalNegativeCycle = hasNegativeCycle ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &globalNegativeCycle, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (globalNegativeCycle) return false;

    // Build ssspTree
    for (int v = 1; v <= maxNode; ++v) {
        if (newParent[v] != -1 && newParent[v] != 0) {
            ssspTree[newParent[v] - 1].push_back(v);
        }
    }

    // Compute distances for each objective
    for (size_t i = 0; i < maxNode; ++i) {
        bool localUpdated = false;

        #pragma omp parallel for num_threads(NUM_THREADS) reduction(||:localUpdated)
        for (size_t j = 0; j < edges.size(); ++j) {
            auto [u, v, _] = edges[j];
            if (localNodes.empty() || find(localNodes.begin(), localNodes.end(), u) != localNodes.end()) {
                for (int k = 0; k < numObjectives; ++k) {
                    double weightObj = originalGraphs[k].count(u) && originalGraphs[k].at(u).count(v) ? originalGraphs[k].at(u).at(v) : numeric_limits<double>::infinity();
                    if (!isinf(weightObj) && !isinf(distances[u][k])) {
                        double newDist = distances[u][k] + weightObj;
                        bool updated = false;
                        #pragma omp critical
                        {
                            if (newDist < distances[v][k]) {
                                distances[v][k] = newDist;
                                objParents[k][v] = u;
                                updated = true;
                            }
                        }
                        if (updated) localUpdated = true;
                    }
                }
            }
        }

        // Synchronize objective distances and parents
        for (int k = 0; k < numObjectives; ++k) {
            vector<double> allObjDistances(maxNode + 1, numeric_limits<double>::infinity());
            vector<int> allObjParents(maxNode + 1, -1);
            for (int node = 1; node <= maxNode; ++node) {
                allObjDistances[node] = distances[node][k];
                allObjParents[node] = objParents[k][node];
            }
            MPI_Allreduce(MPI_IN_PLACE, allObjDistances.data(), maxNode + 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, allObjParents.data(), maxNode + 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            for (int node = 1; node <= maxNode; ++node) {
                distances[node][k] = allObjDistances[node];
                if (allObjParents[node] != -1) {
                    objParents[k][node] = allObjParents[node];
                }
            }
        }

        // Check if further updates are needed
        int localUpdateFlag = localUpdated ? 1 : 0;
        int globalUpdateFlag = 0;
        MPI_Allreduce(&localUpdateFlag, &globalUpdateFlag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (!globalUpdateFlag) break;
    }

    return true;
}

bool operator==(const Edge& lhs, const Edge& rhs) {
    return lhs.source == rhs.source && lhs.destination == rhs.destination && lhs.weight == rhs.weight;
}

pair<int, MultiObjectiveGraph> readMultiObjectiveGraph(ifstream& inputFile, bool isGraph, vector<vector<vector<Edge>>>& predecessor) {
    string line;
    MultiObjectiveGraph graphs;
    int numObjectives = 0, maxNodes = 0;

    while (getline(inputFile, line)) {
        if (line.find("obj") == 0) {
            numObjectives++;
            getline(inputFile, line);
            istringstream headerStream(line);
            int numRows, numCols, numNonZero;
            if (!(headerStream >> numRows >> numCols >> numNonZero)) {
                cerr << "Error: Invalid header format for objective " << numObjectives << endl;
                exit(1);
            }
            maxNodes = max(maxNodes, max(numRows, numCols));
            vector<vector<Edge>> DTMatrix(numRows);
            if (isGraph) {
                predecessor.emplace_back(vector<vector<Edge>>(numCols));
            }
            vector<string> lines(numNonZero);
            for (int i = 0; i < numNonZero; ++i) {
                if (!getline(inputFile, lines[i])) {
                    cerr << "Error: Incomplete data for objective " << numObjectives << endl;
                    exit(1);
                }
            }
            #pragma omp parallel for num_threads(NUM_THREADS)
            for (int i = 0; i < numNonZero; ++i) {
                istringstream lineStream(lines[i]);
                int row, col;
                double value;
                if (!(lineStream >> row >> col >> value)) {
                    cerr << "Error: Invalid line format at line " << i + 1 << " for objective " << numObjectives << endl;
                    exit(1);
                }
                if (row < 1 || row > numRows || col < 1 || col > numCols) {
                    cerr << "Error: Invalid vertex indices at line " << i + 1 << " for objective " << numObjectives << endl;
                    exit(1);
                }
                #pragma omp critical
                {
                    DTMatrix[row - 1].emplace_back(row - 1, col - 1, value);
                    if (isGraph && numObjectives == 1) {
                        predecessor.back()[col - 1].emplace_back(row, col, value);
                    }
                }
            }
            graphs.push_back(buildOriginalGraph(DTMatrix, {}));
        }
    }
    return {numObjectives, graphs};
}

Graph createEnsembleGraph(const MultiObjectiveGraph& graphs, int numObjectives) {
    Graph ensembleGraph;
    unordered_map<EdgeKey, int, PairHash> edgeCount;

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (size_t i = 0; i < graphs.size(); ++i) {
        unordered_map<EdgeKey, int, PairHash> localEdgeCount;
        for (const auto& [u, neighbors] : graphs[i]) {
            for (const auto& [v, _] : neighbors) {
                localEdgeCount[{u, v}]++;
            }
        }
        #pragma omp critical
        for (const auto& [edge, count] : localEdgeCount) {
            edgeCount[edge] += count;
        }
    }

    for (const auto& [edge, count] : edgeCount) {
        ensembleGraph[edge.first][edge.second] = numObjectives - count + 1;
    }
    return ensembleGraph;
}

void printEnsembleGraph(const Graph& ensembleGraph, int rank) {
    if (rank == 0) {
        cout << "Ensemble Graph:\n";
        for (const auto& [u, neighbors] : ensembleGraph) {
            for (const auto& [v, weight] : neighbors) {
                cout << "Edge (" << u << ", " << v << ") with weight " << weight << "\n";
            }
        }
    }
}

void printShortestPathTree(const vector<vector<int>>& ssspTree, int rank) {
    if (rank == 0) {
        cout << "Final MOSP Tree:\n";
        for (size_t i = 0; i < ssspTree.size(); ++i) {
            cout << "Node " << (i + 1) << ": ";
            for (int child : ssspTree[i]) {
                cout << child << " ";
            }
            cout << "\n";
        }
    }
}

void printEnsembleShortestPaths(int source, const vector<int>& parents, const unordered_map<int, double>& distances, int maxNode, int rank) {
    if (rank == 0) {
        cout << "\nShortest Paths in Ensemble Graph from Source Node " << source << ":\n";
        for (int node = 1; node <= maxNode; ++node) {
            if (node == source || distances.find(node) == distances.end() || isinf(distances.at(node))) continue;
            vector<int> path;
            int current = node;
            while (current != -1) {
                path.push_back(current);
                current = parents[current];
            }
            if (path.back() != source) continue;
            cout << "To node " << node << ": ";
            for (int i = path.size() - 1; i >= 0; --i) {
                cout << path[i];
                if (i > 0) cout << " -> ";
            }
            cout << " (Distance: " << (isinf(distances.at(node)) ? "inf" : to_string(distances.at(node))) << ")\n";
        }
    }
}

void printShortestPaths(int source, const vector<vector<int>>& objParents, const unordered_map<int, vector<double>>& distances, int maxNode, int numObjectives, int rank) {
    if (rank == 0) {
        cout << "\nShortest Paths from Source Node " << source << " in Ensemble Graph:\n";
        for (int node = 1; node <= maxNode; ++node) {
            if (node == source || distances.find(node) == distances.end()) continue;
            bool hasValidPath = false;
            vector<vector<int>> paths(numObjectives);
            for (int k = 0; k < numObjectives; ++k) {
                if (!isinf(distances.at(node)[k])) {
                    int current = node;
                    while (current != -1) {
                        paths[k].push_back(current);
                        current = objParents[k][current];
                    }
                    if (!paths[k].empty() && paths[k].back() == source) {
                        hasValidPath = true;
                    } else {
                        paths[k].clear();
                    }
                }
            }
            if (!hasValidPath) continue;
            cout << "To node " << node << ":\n";
            for (int k = 0; k < numObjectives; ++k) {
                cout << "  Objective " << (k + 1) << ": ";
                if (!paths[k].empty()) {
                    for (int i = paths[k].size() - 1; i >= 0; --i) {
                        cout << paths[k][i];
                        if (i > 0) cout << " -> ";
                    }
                    cout << " (Distance: " << (isinf(distances.at(node)[k]) ? "inf" : to_string(distances.at(node)[k])) << ")\n";
                } else {
                    cout << "unreachable (Distance: inf)\n";
                }
            }
        }
    }
}

void generateDotFile(const Graph& ensembleGraph, const vector<int>& newParent, const unordered_map<int, vector<double>>& distances, int source, int maxNode, int numObjectives, int rank) {
    if (rank == 0) {
        ofstream dotFile("ensemble_graph.dot");
        if (!dotFile.is_open()) {
            cerr << "Unable to open file for DOT output: ensemble_graph.dot" << endl;
            return;
        }

        dotFile << "digraph EnsembleGraph {\n";
        dotFile << "    rankdir=LR;\n";
        dotFile << "    node [shape=circle, style=filled, fillcolor=lightgrey];\n";

        for (int node = 1; node <= maxNode; ++node) {
            string label = to_string(node);
            if (distances.find(node) != distances.end()) {
                label += "\\n(";
                for (int k = 0; k < numObjectives; ++k) {
                    label += isinf(distances.at(node)[k]) ? "inf" : to_string(distances.at(node)[k]);
                    if (k < numObjectives - 1) label += ",";
                }
                label += ")";
            }
            dotFile << "    " << node << " [label=\"" << label << "\"";
            if (node == source) dotFile << ", fillcolor=yellow";
            dotFile << "];\n";
        }

        set<pair<int, int>> treeEdges;
        for (int v = 1; v <= maxNode; ++v) {
            if (newParent[v] != -1 && newParent[v] != 0) {
                treeEdges.emplace(newParent[v], v);
            }
        }

        for (const auto& [u, neighbors] : ensembleGraph) {
            for (const auto& [v, weight] : neighbors) {
                dotFile << "    " << u << " -> " << v << " [label=\"" << weight << "\"";
                if (treeEdges.count({u, v})) {
                    dotFile << ", color=blue, penwidth=2";
                }
                dotFile << "];\n";
            }
        }

        dotFile << "}\n";
        dotFile.close();
        cout << "Generated DOT file: ensemble_graph.dot\n";
        cout << "Run 'dot -Tpng ensemble_graph.dot -o ensemble_graph.png' to generate a PNG image.\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np <num_processes> ./program -g <graph_file> -c <changed_edges_file> -s <source_node>\n";
        }
        MPI_Finalize();
        return 1;
    }

    string graphFile, changedEdgesFile;
    int sourceNode;
    for (int i = 1; i < argc; i += 2) {
        string option(argv[i]);
        string argument(argv[i + 1]);
        if (option == "-g") graphFile = argument;
        else if (option == "-c") changedEdgesFile = argument;
        else if (option == "-s") sourceNode = stoi(argument);
        else if (option == "-t") NUM_THREADS = stoi(argument);
        else {
            if (rank == 0) cerr << "Invalid option: " << option << "\n";
            MPI_Finalize();
            return 1;
        }
    }

    int numObjectives = 0, maxNodes = 0;
    MultiObjectiveGraph multiGraphs;
    vector<vector<vector<Edge>>> predecessor;

    if (rank == 0) {
        ifstream graphInputFile(graphFile);
        if (!graphInputFile.is_open()) {
            cerr << "Unable to open graph file: " << graphFile << endl;
            MPI_Finalize();
            return 1;
        }
        auto [no, mg] = readMultiObjectiveGraph(graphInputFile, true, predecessor);
        numObjectives = no;
        multiGraphs = mg;
        graphInputFile.close();

        for (const auto& graph : multiGraphs) {
            for (const auto& [node, neighbors] : graph) {
                maxNodes = max(maxNodes, node);
                for (const auto& [neighbor, _] : neighbors) {
                    maxNodes = max(maxNodes, neighbor);
                }
            }
        }
    }

    MPI_Bcast(&numObjectives, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < numObjectives; ++i) {
        vector<int> nodes, neighbors;
        vector<double> weights;
        int localMaxNode = 0;
        if (rank == 0) {
            serializeGraph(multiGraphs[i], nodes, neighbors, weights, localMaxNode);
        }
        int edgeCount = nodes.size();
        MPI_Bcast(&edgeCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            nodes.resize(edgeCount);
            neighbors.resize(edgeCount);
            weights.resize(edgeCount);
        }
        MPI_Bcast(nodes.data(), edgeCount, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(neighbors.data(), edgeCount, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(weights.data(), edgeCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        Graph graph = deserializeGraph(nodes, neighbors, weights);
        if (rank != 0) multiGraphs.push_back(graph);
    }

    MultiObjectiveGraph multiChangedGraphs;
    if (rank == 0) {
        ifstream changedEdgesInputFile(changedEdgesFile);
        if (!changedEdgesInputFile.is_open()) {
            cerr << "Unable to open changed edges file: " << changedEdgesFile << endl;
            MPI_Finalize();
            return 1;
        }
        auto [noc, mcg] = readMultiObjectiveGraph(changedEdgesInputFile, false, predecessor);
        multiChangedGraphs = mcg;
        changedEdgesInputFile.close();
        if (numObjectives != noc) {
            cerr << "Error: Number of objectives in graph and changes files do not match\n";
            MPI_Finalize();
            return 1;
        }
    }

    for (int i = 0; i < numObjectives; ++i) {
        vector<int> nodes, neighbors;
        vector<double> weights;
        int localMaxNode = 0;
        if (rank == 0) {
            serializeGraph(multiChangedGraphs[i], nodes, neighbors, weights, localMaxNode);
        }
        int edgeCount = nodes.size();
        MPI_Bcast(&edgeCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            nodes.resize(edgeCount);
            neighbors.resize(edgeCount);
            weights.resize(edgeCount);
        }
        MPI_Bcast(nodes.data(), edgeCount, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(neighbors.data(), edgeCount, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(weights.data(), edgeCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        Graph graph = deserializeGraph(nodes, neighbors, weights);
        if (rank != 0) multiChangedGraphs.push_back(graph);
    }

    MultiObjectiveGraph updatedGraphs = multiGraphs;
    for (size_t i = 0; i < numObjectives; ++i) {
        for (const auto& [u, neighbors] : multiChangedGraphs[i]) {
            for (const auto& [v, weight] : neighbors) {
                if (weight > 0) {
                    updatedGraphs[i][u][v] = weight;
                } else {
                    updatedGraphs[i][u].erase(v);
                    if (updatedGraphs[i][u].empty()) updatedGraphs[i].erase(u);
                }
            }
        }
    }

    Graph ensembleGraph = createEnsembleGraph(updatedGraphs, numObjectives);
    printEnsembleGraph(ensembleGraph, rank);

    unordered_map<int, vector<double>> distances;
    vector<int> newParent;
    vector<vector<int>> objParents, ssspTree;
    unordered_map<int, double> ensembleDistances;
    bool success = bellmanFordMOSP(ensembleGraph, updatedGraphs, sourceNode, numObjectives, distances, newParent, ssspTree, objParents, ensembleDistances, rank, size);

    if (rank == 0) {
        if (success) {
            cout << "Shortest distances from node " << sourceNode << " (Actual Weights):\n";
            for (const auto& [node, dist] : distances) {
                if (node > 0) {
                    cout << "node " << node << ": ";
                    for (int k = 0; k < numObjectives; ++k) {
                        cout << "Obj" << (k + 1) << ": " << (isinf(dist[k]) ? "inf" : to_string(dist[k]));
                        if (k < numObjectives - 1) cout << ", ";
                    }
                    for (int k = 0; k < numObjectives; ++k) {
                        cout << ", parent Obj" << (k + 1) << ": " << objParents[k][node];
                    }
                    cout << "\n";
                }
            }
            printShortestPaths(sourceNode, objParents, distances, maxNodes, numObjectives, rank);
            printEnsembleShortestPaths(sourceNode, newParent, ensembleDistances, maxNodes, rank);
            generateDotFile(ensembleGraph, newParent, distances, sourceNode, maxNodes, numObjectives, rank);
        } else {
            cout << "The ensemble graph contains a negative-weight cycle.\n";
        }
    }

    printShortestPathTree(ssspTree, rank);
    MPI_Finalize();
    return 0;
}