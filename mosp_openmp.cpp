#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>

int NUM_THREADS = 4;

struct Edge {
    int source, destination;
    double weight;
    Edge(int src, int dest, double w) : source(src), destination(dest), weight(w) {}
};

using Graph = std::vector<std::vector<std::pair<int, double>>>;
using MultiObjectiveGraph = std::vector<Graph>;
using EdgeList = std::vector<Edge>;

Graph buildOriginalGraph(const std::vector<EdgeList>& graphDT, const std::vector<EdgeList>& changedEdgesDT, int maxNodes) {
    Graph graph(maxNodes + 1);
    for (const auto& edges : graphDT) {
        for (const auto& edge : edges) {
            graph[edge.source + 1].emplace_back(edge.destination + 1, edge.weight);
        }
    }
    for (const auto& edges : changedEdgesDT) {
        for (const auto& edge : edges) {
            if (edge.weight > 0) {
                auto& adj = graph[edge.source + 1];
                auto it = std::find_if(adj.begin(), adj.end(), 
                    [&](const auto& p) { return p.first == edge.destination + 1; });
                if (it != adj.end()) {
                    it->second = edge.weight;
                } else {
                    adj.emplace_back(edge.destination + 1, edge.weight);
                }
            }
        }
    }
    return graph;
}

bool bellmanFordMOSP(const Graph& ensembleGraph, const MultiObjectiveGraph& originalGraphs, int source, int numObjectives, 
                    std::vector<std::vector<double>>& distances, std::vector<int>& newParent, 
                    std::vector<std::vector<int>>& ssspTree, std::vector<std::vector<int>>& objParents, 
                    std::vector<double>& ensembleDistances, int maxNodes) {
    distances.assign(maxNodes + 1, std::vector<double>(numObjectives, std::numeric_limits<double>::infinity()));
    ensembleDistances.assign(maxNodes + 1, std::numeric_limits<double>::infinity());
    newParent.assign(maxNodes + 1, -1);
    objParents.assign(numObjectives, std::vector<int>(maxNodes + 1, -1));
    ssspTree.assign(maxNodes + 1, {});

    distances[source] = std::vector<double>(numObjectives, 0.0);
    ensembleDistances[source] = 0.0;
    newParent[source] = -1;
    for (int i = 0; i < numObjectives; ++i) {
        objParents[i][source] = -1;
    }

    std::vector<Edge> edges;
    for (int u = 1; u <= maxNodes; ++u) {
        for (const auto& [v, weight] : ensembleGraph[u]) {
            edges.emplace_back(u, v, weight);
        }
    }

    bool continueLoop = true;
    for (size_t i = 0; i < maxNodes && continueLoop; ++i) {
        bool updated = false;
        #pragma omp parallel num_threads(NUM_THREADS) reduction(||:updated)
        {
            #pragma omp for schedule(dynamic, 100)
            for (size_t j = 0; j < edges.size(); ++j) {
                int u = edges[j].source, v = edges[j].destination;
                double weight = edges[j].weight;

                // Update ensemble distances
                double newEnsembleDist = ensembleDistances[u] + weight;
                if (newEnsembleDist < ensembleDistances[v]) {
                    #pragma omp critical(ensemble_update)
                    {
                        if (newEnsembleDist < ensembleDistances[v]) {
                            ensembleDistances[v] = newEnsembleDist;
                            newParent[v] = u;
                            updated = true;
                        }
                    }
                }

                // Update objective distances
                for (int k = 0; k < numObjectives; ++k) {
                    double objWeight = std::numeric_limits<double>::infinity();
                    for (const auto& [dest, w] : originalGraphs[k][u]) {
                        if (dest == v) {
                            objWeight = w;
                            break;
                        }
                    }
                    if (!std::isinf(objWeight) && !std::isinf(distances[u][k])) {
                        double newDist = distances[u][k] + objWeight;
                        if (newDist < distances[v][k]) {
                            #pragma omp critical(objective_update)
                            {
                                if (newDist < distances[v][k]) {
                                    distances[v][k] = newDist;
                                    objParents[k][v] = u;
                                    updated = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        continueLoop = updated;
    }

    // Check for negative-weight cycles
    for (const auto& edge : edges) {
        int u = edge.source, v = edge.destination;
        double weight = edge.weight;
        if (!std::isinf(ensembleDistances[u]) && ensembleDistances[u] + weight < ensembleDistances[v]) {
            return false;
        }
    }

    // Build SSSP tree
    for (int v = 1; v <= maxNodes; ++v) {
        if (newParent[v] != -1 && newParent[v] != 0) {
            ssspTree[newParent[v]].push_back(v);
        }
    }

    return true;
}

std::pair<int, MultiObjectiveGraph> readMultiObjectiveGraph(std::ifstream& inputFile, bool isGraph, int& maxNodes) {
    std::string line;
    MultiObjectiveGraph graphs;
    int numObjectives = 0;
    maxNodes = 0;

    std::vector<EdgeList> allEdges;
    while (std::getline(inputFile, line)) {
        if (line.find("obj") == 0) {
            numObjectives++;
            std::getline(inputFile, line);
            std::istringstream headerStream(line);
            int numRows, numCols, numNonZero;
            if (!(headerStream >> numRows >> numCols >> numNonZero)) {
                std::cerr << "Error: Invalid header format for objective " << numObjectives << std::endl;
                exit(1);
            }
            maxNodes = std::max(maxNodes, std::max(numRows, numCols));
            EdgeList edges;
            for (int i = 0; i < numNonZero; ++i) {
                if (!std::getline(inputFile, line)) {
                    std::cerr << "Error: Incomplete data for objective " << numObjectives << std::endl;
                    exit(1);
                }
                std::istringstream lineStream(line);
                int row, col;
                double value;
                if (!(lineStream >> row >> col >> value)) {
                    std::cerr << "Error: Invalid line format for objective " << numObjectives << std::endl;
                    exit(1);
                }
                if (row < 1 || row > numRows || col < 1 || col > numCols) {
                    std::cerr << "Error: Invalid vertex indices for objective " << numObjectives << std::endl;
                    exit(1);
                }
                edges.emplace_back(row - 1, col - 1, value);
            }
            allEdges.push_back(edges);
            graphs.push_back(buildOriginalGraph({edges}, {}, maxNodes));
        }
    }
    return {numObjectives, graphs};
}

Graph createEnsembleGraph(const MultiObjectiveGraph& graphs, int numObjectives, int maxNodes) {
    Graph ensembleGraph(maxNodes + 1);
    std::vector<std::vector<std::vector<int>>> threadEdgeCount(NUM_THREADS, 
        std::vector<std::vector<int>>(maxNodes + 1, std::vector<int>(maxNodes + 1, 0)));

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < graphs.size(); ++i) {
            for (int u = 1; u <= maxNodes; ++u) {
                for (const auto& [v, _] : graphs[i][u]) {
                    threadEdgeCount[tid][u][v]++;
                }
            }
        }
    }

    // Merge thread-local counts
    std::vector<std::vector<int>> edgeCount(maxNodes + 1, std::vector<int>(maxNodes + 1, 0));
    for (int tid = 0; tid < NUM_THREADS; ++tid) {
        for (int u = 1; u <= maxNodes; ++u) {
            for (int v = 1; v <= maxNodes; ++v) {
                edgeCount[u][v] += threadEdgeCount[tid][u][v];
            }
        }
    }

    for (int u = 1; u <= maxNodes; ++u) {
        for (int v = 1; v <= maxNodes; ++v) {
            if (edgeCount[u][v] > 0) {
                ensembleGraph[u].emplace_back(v, numObjectives - edgeCount[u][v] + 1);
            }
        }
    }

    return ensembleGraph;
}

void printEnsembleGraph(const Graph& graph, int maxNodes) {
    std::cout << "Ensemble Graph:\n";
    for (int u = 1; u <= maxNodes; ++u) {
        for (const auto& [v, weight] : graph[u]) {
            std::cout << "Edge (" << u << ", " << v << ") with weight " << weight << "\n";
        }
    }
}

void printShortestPathTree(const std::vector<std::vector<int>>& ssspTree, int maxNodes) {
    std::cout << "Final MOSP Tree:\n";
    for (int i = 1; i <= maxNodes; ++i) {
        if (!ssspTree[i].empty()) {
            std::cout << "Node " << i << ": ";
            for (int child : ssspTree[i]) {
                std::cout << child << " ";
            }
            std::cout << "\n";
        }
    }
}

void printEnsembleShortestPaths(int source, const std::vector<int>& parents, const std::vector<double>& distances, int maxNodes) {
    std::cout << "\nShortest Paths in Ensemble Graph from Source Node " << source << ":\n";
    for (int node = 1; node <= maxNodes; ++node) {
        if (node == source || std::isinf(distances[node])) continue;
        std::vector<int> path;
        int current = node;
        while (current != -1) {
            path.push_back(current);
            current = parents[current];
        }
        if (path.back() != source) continue;
        std::cout << "To node " << node << ": ";
        for (int i = path.size() - 1; i >= 0; --i) {
            std::cout << path[i];
            if (i > 0) std::cout << " -> ";
        }
        std::cout << " (Distance: " << distances[node] << ")\n";
    }
}

void printShortestPaths(int source, const std::vector<std::vector<int>>& objParents, 
                        const std::vector<std::vector<double>>& distances, int maxNodes, int numObjectives) {
    std::cout << "\nShortest Paths from Source Node " << source << ":\n";
    for (int node = 1; node <= maxNodes; ++node) {
        if (node == source) continue;
        bool hasValidPath = false;
        std::vector<std::vector<int>> paths(numObjectives);
        for (int k = 0; k < numObjectives; ++k) {
            if (!std::isinf(distances[node][k])) {
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
        std::cout << "To node " << node << ":\n";
        for (int k = 0; k < numObjectives; ++k) {
            std::cout << "  Objective " << (k + 1) << ": ";
            if (!paths[k].empty()) {
                for (int i = paths[k].size() - 1; i >= 0; --i) {
                    std::cout << paths[k][i];
                    if (i > 0) std::cout << " -> ";
                }
                std::cout << " (Distance: " << distances[node][k] << ")\n";
            } else {
                std::cout << "unreachable (Distance: inf)\n";
            }
        }
    }
}

void generateDotFile(const Graph& ensembleGraph, const std::vector<int>& newParent, 
                     const std::vector<std::vector<double>>& distances, int source, int maxNodes, int numObjectives) {
    std::ofstream dotFile("ensemble_graph.dot");
    if (!dotFile.is_open()) {
        std::cerr << "Unable to open file for DOT output: ensemble_graph.dot" << std::endl;
        return;
    }

    dotFile << "digraph EnsembleGraph {\n";
    dotFile << "    rankdir=LR;\n";
    dotFile << "    node [shape=circle, style=filled, fillcolor=lightgrey];\n";

    for (int node = 1; node <= maxNodes; ++node) {
        std::string label = std::to_string(node);
        label += "\\n(";
        for (int k = 0; k < numObjectives; ++k) {
            label += std::isinf(distances[node][k]) ? "inf" : std::to_string(distances[node][k]);
            if (k < numObjectives - 1) label += ",";
        }
        label += ")";
        dotFile << "    " << node << " [label=\"" << label << "\"";
        if (node == source) dotFile << ", fillcolor=yellow";
        dotFile << "];\n";
    }

    std::set<std::pair<int, int>> treeEdges;
    for (int v = 1; v <= maxNodes; ++v) {
        if (newParent[v] != -1 && newParent[v] != 0) {
            treeEdges.emplace(newParent[v], v);
        }
    }

    for (int u = 1; u <= maxNodes; ++u) {
        for (const auto& [v, weight] : ensembleGraph[u]) {
            dotFile << "    " << u << " -> " << v << " [label=\"" << weight << "\"";
            if (treeEdges.count(std::make_pair(u, v))) {
                dotFile << ", color=blue, penwidth=2";
            }
            dotFile << "];\n";
        }
    }

    dotFile << "}\n";
    dotFile.close();
    std::cout << "Generated DOT file: ensemble_graph.dot\n";
    std::cout << "Run 'dot -Tpng ensemble_graph.dot -o ensemble_graph.png' to generate a PNG image.\n";
}

int main(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();
    if (argc < 6) {
        std::cerr << "Usage: ./program -g <graph_file> -c <changed_edges_file> -s <source_node>\n";
        return 1;
    }
    std::string graphFile, changedEdgesFile;
    int sourceNode;

    for (int i = 1; i < argc; i += 2) {
        std::string option(argv[i]);
        if (option == "-g") graphFile = argv[i + 1];
        else if (option == "-c") changedEdgesFile = argv[i + 1];
        else if (option == "-s") sourceNode = std::stoi(argv[i + 1]);
        else if (option == "-t") NUM_THREADS = std::stoi(argv[i + 1]);
        else {
            std::cerr << "Invalid option: " << option << "\n";
            return 1;
        }
    }

    std::ifstream graphInputFile(graphFile);
    if (!graphInputFile.is_open()) {
        std::cerr << "Unable to open graph file: " << graphFile << std::endl;
        return 1;
    }
    int maxNodes;
    auto [numObjectives, multiGraphs] = readMultiObjectiveGraph(graphInputFile, true, maxNodes);
    graphInputFile.close();

    std::ifstream changedEdgesInputFile(changedEdgesFile);
    if (!changedEdgesInputFile.is_open()) {
        std::cerr << "Unable to open changed edges file: " << changedEdgesFile << std::endl;
        return 1;
    }
    int maxNodesChanges;
    auto [numObjectivesChanges, multiChangedGraphs] = readMultiObjectiveGraph(changedEdgesInputFile, false, maxNodesChanges);
    changedEdgesInputFile.close();

    if (numObjectives != numObjectivesChanges) {
        std::cerr << "Error: Number of objectives in graph and changes files do not match\n";
        return 1;
    }
    maxNodes = std::max(maxNodes, maxNodesChanges);

    MultiObjectiveGraph updatedGraphs(numObjectives, Graph(maxNodes + 1));
    for (size_t i = 0; i < numObjectives; ++i) {
        EdgeList graphEdges;
        for (int u = 1; u <= maxNodes; ++u) {
            for (const auto& [v, weight] : multiGraphs[i][u]) {
                graphEdges.emplace_back(u - 1, v - 1, weight);
            }
        }
        EdgeList changedEdges;
        for (int u = 1; u <= maxNodes; ++u) {
            for (const auto& [v, weight] : multiChangedGraphs[i][u]) {
                changedEdges.emplace_back(u - 1, v - 1, weight);
            }
        }
        updatedGraphs[i] = buildOriginalGraph({graphEdges}, {changedEdges}, maxNodes);
    }

    Graph ensembleGraph = createEnsembleGraph(updatedGraphs, numObjectives, maxNodes);
    printEnsembleGraph(ensembleGraph, maxNodes);

    std::vector<std::vector<double>> distances;
    std::vector<int> newParent;
    std::vector<std::vector<int>> objParents, ssspTree;
    std::vector<double> ensembleDistances;
    bool success = bellmanFordMOSP(ensembleGraph, updatedGraphs, sourceNode, numObjectives, distances, 
                                   newParent, ssspTree, objParents, ensembleDistances, maxNodes);

    if (success) {
        std::cout << "Shortest distances from node " << sourceNode << " (Actual Weights):\n";
        for (int node = 1; node <= maxNodes; ++node) {
            if (distances[node] != std::vector<double>(numObjectives, std::numeric_limits<double>::infinity())) {
                std::cout << "node " << node << ": ";
                for (int k = 0; k < numObjectives; ++k) {
                    std::cout << "Obj" << (k + 1) << ": " << (std::isinf(distances[node][k]) ? "inf" : std::to_string(distances[node][k]));
                    if (k < numObjectives - 1) std::cout << ", ";
                }
                for (int k = 0; k < numObjectives; ++k) {
                    std::cout << ", parent Obj" << (k + 1) << ": " << objParents[k][node];
                }
                std::cout << "\n";
            }
        }
        printShortestPaths(sourceNode, objParents, distances, maxNodes, numObjectives);
        printEnsembleShortestPaths(sourceNode, newParent, ensembleDistances, maxNodes);
        generateDotFile(ensembleGraph, newParent, distances, sourceNode, maxNodes, numObjectives);
    } else {
        std::cout << "The ensemble graph contains a negative-weight cycle.\n";
    }

    printShortestPathTree(ssspTree, maxNodes);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\nExecution Time: " << duration.count() << " milliseconds\n";
    return 0;
}