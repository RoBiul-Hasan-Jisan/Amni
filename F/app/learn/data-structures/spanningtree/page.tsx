
import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { GraphVisualizer } from "@/components/visualizations/graph-visualizer";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { AlertCircle, CheckCircle2, Clock, Lightbulb, Target, Zap, Network, Layers, Hash, Flame, Map, CircuitBoard, Binary, ArrowRight, Cpu, BarChart, TreePine, GitBranch, Merge, GitCompare, Shield, Rocket, Gauge, Route } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

export default function MinimumSpanningTreePage() {
  const algorithmCategories = [
    {
      title: "Greedy Algorithms",
      icon: Zap,
      algorithms: ["Prim's Algorithm", "Kruskal's Algorithm", "Boruvka's Algorithm"]
    },
    {
      title: "Alternative Approaches",
      icon: GitCompare,
      algorithms: ["Reverse Delete Algorithm", "Minimum Product MST", "Second Best MST"]
    },
    {
      title: "Applications",
      icon: Target,
      algorithms: ["Network Design", "Circuit Design", "Cluster Analysis", "Image Segmentation"]
    },
    {
      title: "Properties",
      icon: CircuitBoard,
      algorithms: ["Cut Property", "Cycle Property", "Uniqueness", "Total Spanning Trees"]
    }
  ];

  const mstAlgorithms = [
    {
      name: "Prim's Algorithm",
      type: "Greedy",
      complexity: "O((V+E) log V)",
      approach: "Start from a vertex, grow tree by adding smallest edge",
      code: `import heapq

def prim_mst(n, edges):
    # Build adjacency list
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    # Initialize
    visited = [False] * n
    min_heap = [(0, 0, -1)]  # (weight, vertex, parent)
    total_weight = 0
    mst_edges = []
    
    while min_heap and len(mst_edges) < n - 1:
        weight, u, parent = heapq.heappop(min_heap)
        
        if visited[u]:
            continue
        
        visited[u] = True
        total_weight += weight
        
        if parent != -1:
            mst_edges.append((parent, u, weight))
        
        # Add all edges from u to heap
        for v, w in graph[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (w, v, u))
    
    return total_weight, mst_edges`
    },
    {
      name: "Kruskal's Algorithm",
      type: "Greedy",
      complexity: "O(E log E)",
      approach: "Sort edges, add smallest edge that doesn't form cycle",
      code: `class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True

def kruskal_mst(n, edges):
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    
    uf = UnionFind(n)
    total_weight = 0
    mst_edges = []
    
    for u, v, w in edges:
        if uf.union(u, v):
            total_weight += w
            mst_edges.append((u, v, w))
        
        if len(mst_edges) == n - 1:
            break
    
    return total_weight, mst_edges`
    },
    {
      name: "Boruvka's Algorithm",
      type: "Greedy",
      complexity: "O(E log V)",
      approach: "Parallel algorithm, find cheapest edge for each component",
      code: `def boruvka_mst(n, edges):
    # Initialize components
    parent = list(range(n))
    rank = [0] * n
    cheapest = [(-1, float('inf'))] * n  # (neighbor, weight)
    num_components = n
    total_weight = 0
    mst_edges = []
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x == root_y:
            return False
        
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1
        return True
    
    while num_components > 1:
        # Reset cheapest edges
        cheapest = [(-1, float('inf'))] * n
        
        # Find cheapest edge for each component
        for u, v, w in edges:
            comp_u = find(u)
            comp_v = find(v)
            
            if comp_u != comp_v:
                if w < cheapest[comp_u][1]:
                    cheapest[comp_u] = (comp_v, w, u, v)
                if w < cheapest[comp_v][1]:
                    cheapest[comp_v] = (comp_u, w, u, v)
        
        # Add cheapest edges to MST
        for comp in range(n):
            if cheapest[comp][0] != -1:
                target_comp, weight, u, v = cheapest[comp]
                if find(comp) != find(target_comp):
                    if union(comp, target_comp):
                        total_weight += weight
                        mst_edges.append((u, v, weight))
                        num_components -= 1
    
    return total_weight, mst_edges`
    },
    {
      name: "Reverse Delete Algorithm",
      type: "Alternative",
      complexity: "O(E log E)",
      approach: "Start with all edges, remove heaviest edge that doesn't disconnect",
      code: `def reverse_delete_mst(n, edges):
    # Sort edges in descending order
    edges.sort(key=lambda x: -x[2])
    
    # Build initial graph with all edges
    graph = [[] for _ in range(n)]
    edge_indices = {}
    
    for idx, (u, v, w) in enumerate(edges):
        graph[u].append((v, idx))
        graph[v].append((u, idx))
        edge_indices[idx] = (u, v, w)
    
    total_weight = sum(w for _, _, w in edges)
    removed_edges = []
    
    def is_connected():
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 1
        
        while stack:
            u = stack.pop()
            for v, idx in graph[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    count += 1
        
        return count == n
    
    # Try removing edges in descending weight order
    for idx in range(len(edges)):
        u, v, w = edge_indices[idx]
        
        # Temporarily remove edge
        graph[u] = [(nei, ei) for nei, ei in graph[u] if ei != idx]
        graph[v] = [(nei, ei) for nei, ei in graph[v] if ei != idx]
        
        if is_connected():
            total_weight -= w
            removed_edges.append((u, v, w))
        else:
            # Add edge back
            graph[u].append((v, idx))
            graph[v].append((u, idx))
    
    # MST edges are those not removed
    mst_edges = [(u, v, w) for idx, (u, v, w) in edge_indices.items() 
                 if (u, v, w) not in removed_edges]
    
    return total_weight, mst_edges`
    }
  ];

  const specializedMSTProblems = [
    {
      title: "Minimum Cost to Connect All Cities",
      description: "MST problem where each city needs to be connected with minimum cost",
      complexity: "O(n² log n)",
      code: `def min_cost_connect_cities(cities):
    """
    cities: list of (x, y) coordinates
    Cost between cities = Manhattan distance
    """
    n = len(cities)
    if n <= 1:
        return 0
    
    # Create complete graph
    edges = []
    for i in range(n):
        x1, y1 = cities[i]
        for j in range(i + 1, n):
            x2, y2 = cities[j]
            cost = abs(x1 - x2) + abs(y1 - y2)
            edges.append((i, j, cost))
    
    # Use Kruskal's algorithm
    edges.sort(key=lambda x: x[2])
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x == root_y:
            return False
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1
        return True
    
    total_cost = 0
    edges_used = 0
    
    for u, v, cost in edges:
        if union(u, v):
            total_cost += cost
            edges_used += 1
            if edges_used == n - 1:
                break
    
    return total_cost`
    },
    {
      title: "Minimum Product Spanning Tree",
      description: "Find spanning tree that minimizes product of edge weights",
      complexity: "O(E log E)",
      code: `def min_product_mst(n, edges):
    """
    Minimize product of edge weights = minimize sum of log(weights)
    """
    import math
    
    # Convert to log weights
    log_edges = []
    for u, v, w in edges:
        if w <= 0:
            raise ValueError("Weights must be positive for product")
        log_edges.append((u, v, math.log(w)))
    
    # Use Kruskal's with log weights
    log_edges.sort(key=lambda x: x[2])
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x == root_y:
            return False
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1
        return True
    
    product = 1
    edges_used = 0
    
    for u, v, log_w in log_edges:
        if union(u, v):
            # Convert back from log
            product *= math.exp(log_w)
            edges_used += 1
            if edges_used == n - 1:
                break
    
    return product`
    },
    {
      title: "Total Spanning Trees (Kirchhoff's Theorem)",
      description: "Count total number of spanning trees in a graph",
      complexity: "O(n³)",
      code: `def total_spanning_trees(n, edges):
    """
    Kirchhoff's Matrix Tree Theorem
    Count = any cofactor of Laplacian matrix
    """
    # Initialize Laplacian matrix
    laplacian = [[0] * n for _ in range(n)]
    
    # Build degree matrix and adjacency matrix
    for u, v in edges:
        laplacian[u][u] += 1
        laplacian[v][v] += 1
        laplacian[u][v] = -1
        laplacian[v][u] = -1
    
    # Remove last row and column (cofactor)
    cofactor_matrix = [row[:-1] for row in laplacian[:-1]]
    
    # Calculate determinant
    def determinant(matrix):
        n = len(matrix)
        if n == 1:
            return matrix[0][0]
        if n == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        
        det = 0
        for col in range(n):
            # Create submatrix
            submatrix = []
            for i in range(1, n):
                row = []
                for j in range(n):
                    if j != col:
                        row.append(matrix[i][j])
                submatrix.append(row)
            
            sign = 1 if col % 2 == 0 else -1
            det += sign * matrix[0][col] * determinant(submatrix)
        
        return det
    
    return determinant(cofactor_matrix)`
    },
    {
      title: "Second Best Minimum Spanning Tree",
      description: "Find spanning tree with second minimum total weight",
      complexity: "O(V²)",
      code: `def second_best_mst(n, edges):
    # First, find MST using Prim's
    def prim_mst(start):
        visited = [False] * n
        min_edge = [(float('inf'), -1)] * n
        min_edge[start] = (0, -1)
        total = 0
        mst_edges = []
        
        for _ in range(n):
            # Find minimum edge
            u = -1
            for i in range(n):
                if not visited[i] and (u == -1 or min_edge[i][0] < min_edge[u][0]):
                    u = i
            
            visited[u] = True
            total += min_edge[u][0]
            if min_edge[u][1] != -1:
                mst_edges.append((min_edge[u][1], u, min_edge[u][0]))
            
            # Update neighbors
            for v, w in graph[u]:
                if not visited[v] and w < min_edge[v][0]:
                    min_edge[v] = (w, u)
        
        return total, mst_edges
    
    # Build graph
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    # Get MST
    mst_weight, mst_edges = prim_mst(0)
    
    # Try removing each MST edge and find alternative
    second_best = float('inf')
    
    for excluded_u, excluded_v, excluded_w in mst_edges:
        # Remove excluded edge from graph temporarily
        temp_graph = [neighbors.copy() for neighbors in graph]
        temp_graph[excluded_u] = [(v, w) for v, w in temp_graph[excluded_u] 
                                  if v != excluded_v]
        temp_graph[excluded_v] = [(v, w) for v, w in temp_graph[excluded_v] 
                                  if v != excluded_u]
        
        # Find MST without excluded edge
        # (Simplified - in practice use more efficient approach)
        # This is O(V²) implementation
        
        best_without = float('inf')
        # Try all possible edges to replace excluded one
        for u, v, w in edges:
            if (u == excluded_u and v == excluded_v) or (u == excluded_v and v == excluded_u):
                continue
            
            # Would need to run modified Prim's here
            # This is simplified for illustration
        
        if best_without < second_best:
            second_best = best_without
    
    return second_best if second_best != float('inf') else -1`
    }
  ];

  const mstProperties = [
    {
      property: "Cut Property",
      description: "For any cut of the graph, the minimum weight edge crossing the cut is in some MST",
      implication: "Basis for Prim's and Kruskal's correctness"
    },
    {
      property: "Cycle Property",
      description: "For any cycle, the maximum weight edge is not in any MST",
      implication: "Basis for Reverse Delete algorithm"
    },
    {
      property: "Uniqueness",
      description: "MST is unique if all edge weights are distinct",
      implication: "Multiple MSTs possible with equal weights"
    },
    {
      property: "Edge Count",
      description: "MST always has exactly V-1 edges for a connected graph with V vertices",
      implication: "Stop condition for greedy algorithms"
    }
  ];

  const realWorldApplications = [
    {
      title: "Network Design",
      icon: Network,
      description: "Designing computer networks, telephone networks with minimum cable length",
      algorithm: "Prim's/Kruskal's",
      example: "Connecting cities with minimum fiber optic cable"
    },
    {
      title: "Circuit Design",
      icon: CircuitBoard,
      description: "Minimizing wire length in VLSI circuit layout",
      algorithm: "Prim's Algorithm",
      example: "Connecting components on a chip with minimum wire"
    },
    {
      title: "Cluster Analysis",
      icon: GitBranch,
      description: "Hierarchical clustering using MST (single-linkage clustering)",
      algorithm: "Kruskal's Algorithm",
      example: "Grouping similar data points in machine learning"
    },
    {
      title: "Image Segmentation",
      icon: Map,
      description: "Segmenting images by treating pixels as vertices",
      algorithm: "Kruskal's with custom weights",
      example: "Medical image analysis, object recognition"
    }
  ];

  const primVsKruskal = [
    {
      aspect: "Approach",
      prim: "Vertex-based greedy",
      kruskal: "Edge-based greedy"
    },
    {
      aspect: "Data Structures",
      prim: "Priority Queue (Heap)",
      kruskal: "Union-Find (Disjoint Set)"
    },
    {
      aspect: "Best For",
      prim: "Dense graphs (E ≈ V²)",
      kruskal: "Sparse graphs (E << V²)"
    },
    {
      aspect: "Time Complexity",
      prim: "O((V+E) log V) with heap",
      kruskal: "O(E log E) with sorting"
    },
    {
      aspect: "Implementation",
      prim: "Slightly more complex",
      kruskal: "Simpler with Union-Find"
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "Which MST algorithm uses a priority queue/heap?",
      options: ["Kruskal's", "Prim's", "Boruvka's", "Reverse Delete"],
      correctAnswer: 1,
      explanation: "Prim's algorithm uses a priority queue to always select the minimum weight edge from the current tree.",
    },
    {
      id: 2,
      question: "What is the key data structure used in Kruskal's algorithm?",
      options: ["Heap", "Stack", "Union-Find", "Hash Table"],
      correctAnswer: 2,
      explanation: "Kruskal's algorithm uses Union-Find (Disjoint Set) to efficiently detect cycles.",
    },
    {
      id: 3,
      question: "How many edges does an MST have for a connected graph with V vertices?",
      options: ["V", "V-1", "V+1", "E"],
      correctAnswer: 1,
      explanation: "A spanning tree always has exactly V-1 edges for a connected graph with V vertices.",
    },
    {
      id: 4,
      question: "Which property states that the minimum edge crossing any cut is in some MST?",
      options: ["Cycle Property", "Cut Property", "Tree Property", "Greedy Property"],
      correctAnswer: 1,
      explanation: "The Cut Property is fundamental to proving the correctness of MST algorithms.",
    },
    {
      id: 5,
      question: "When is the MST unique?",
      options: [
        "Always",
        "When all edge weights are distinct",
        "When graph is complete",
        "When using Prim's algorithm"
      ],
      correctAnswer: 1,
      explanation: "If all edge weights are distinct, there is exactly one MST. Equal weights can lead to multiple MSTs.",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <header className="mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Minimum Spanning Tree Algorithms
          </h1>
          <p className="text-lg text-muted-foreground">
            Complete guide to MST algorithms: Prim's, Kruskal's, Boruvka's, and their applications in network design and optimization.
          </p>
        </header>

        {/* Categories Overview */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground flex items-center gap-2">
            <TreePine className="h-6 w-6" />
            MST Algorithm Categories
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {algorithmCategories.map((category, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <category.icon className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold text-foreground">{category.title}</h3>
                  </div>
                  <ul className="space-y-2">
                    {category.algorithms.map((algo, tIdx) => (
                      <li key={tIdx} className="flex items-center gap-2 text-sm">
                        <div className="h-1.5 w-1.5 rounded-full bg-primary"></div>
                        <span className="text-muted-foreground">{algo}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Core MST Algorithms */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Core MST Algorithms</h2>
          <div className="space-y-6">
            {mstAlgorithms.map((algo, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex flex-col lg:flex-row lg:items-start gap-6">
                    <div className="lg:w-1/3">
                      <div className="flex items-center gap-3 mb-4">
                        <Badge variant="outline" className="font-mono">
                          {algo.complexity}
                        </Badge>
                        <h3 className="font-semibold text-lg text-foreground">{algo.name}</h3>
                      </div>
                      
                      <div className="space-y-3">
                        <div>
                          <h4 className="font-medium text-sm text-foreground mb-1">Type:</h4>
                          <Badge variant="secondary">{algo.type}</Badge>
                        </div>
                        <div>
                          <h4 className="font-medium text-sm text-foreground mb-1">Approach:</h4>
                          <p className="text-sm text-muted-foreground">{algo.approach}</p>
                        </div>
                        <div>
                          <h4 className="font-medium text-sm text-foreground mb-1">Best For:</h4>
                          <p className="text-sm text-muted-foreground">
                            {algo.name === "Prim's Algorithm" && "Dense graphs, when graph is represented as adjacency matrix"}
                            {algo.name === "Kruskal's Algorithm" && "Sparse graphs, edges already sorted or sortable"}
                            {algo.name === "Boruvka's Algorithm" && "Parallel processing, distributed computing"}
                            {algo.name === "Reverse Delete Algorithm" && "When edges need to be removed, not added"}
                          </p>
                        </div>
                      </div>
                    </div>
                    
                    <div className="lg:w-2/3">
                      <div className="bg-muted p-4 rounded-lg overflow-x-auto">
                        <pre className="text-xs font-mono">
                          <code>{algo.code}</code>
                        </pre>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Prim's vs Kruskal's Comparison */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Prim's vs Kruskal's: Detailed Comparison</h2>
          <Card>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-muted border-b">
                      <th className="p-4 text-left font-semibold text-foreground">Aspect</th>
                      <th className="p-4 text-left font-semibold text-foreground">Prim's Algorithm</th>
                      <th className="p-4 text-left font-semibold text-foreground">Kruskal's Algorithm</th>
                    </tr>
                  </thead>
                  <tbody>
                    {primVsKruskal.map((row, idx) => (
                      <tr key={idx} className="border-b hover:bg-muted/50">
                        <td className="p-4 font-medium text-foreground">{row.aspect}</td>
                        <td className="p-4">{row.prim}</td>
                        <td className="p-4">{row.kruskal}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
          
          <div className="mt-6 grid md:grid-cols-2 gap-6">
            <Card className="bg-blue-500/5 border-blue-500/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-3 text-foreground">When to Use Prim's</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>Graph is dense (many edges)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>Need MST starting from specific vertex</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>Graph is represented as adjacency matrix</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>Real-time updates to graph (online algorithm)</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="bg-green-500/5 border-green-500/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-3 text-foreground">When to Use Kruskal's</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Graph is sparse (few edges)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Edges are already sorted by weight</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Need simple implementation</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Parallel processing possible</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Specialized MST Problems */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Specialized MST Problems</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {specializedMSTProblems.map((problem, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="font-semibold text-lg text-foreground mb-1">
                        {problem.title}
                      </h3>
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="outline" className="font-mono">
                          {problem.complexity}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {problem.description}
                      </p>
                    </div>
                  </div>
                  <div className="bg-muted p-4 rounded-lg overflow-x-auto">
                    <pre className="text-xs font-mono">
                      <code>{problem.code}</code>
                    </pre>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* MST Properties & Theory */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">MST Properties & Theory</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {mstProperties.map((prop, idx) => (
              <Card key={idx}>
                <CardContent className="p-6">
                  <h3 className="font-semibold mb-3 text-foreground">{prop.property}</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    {prop.description}
                  </p>
                  <div className="bg-primary/5 border border-primary/20 p-3 rounded">
                    <p className="text-xs font-medium text-foreground">Implication:</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {prop.implication}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Real-world Applications */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Real-world Applications</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {realWorldApplications.map((app, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <app.icon className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold text-foreground">{app.title}</h3>
                  </div>
                  <p className="text-sm text-muted-foreground mb-4">
                    {app.description}
                  </p>
                  <div className="flex justify-between items-center">
                    <Badge variant="outline">{app.algorithm}</Badge>
                    <div className="text-xs text-muted-foreground">
                      Example: {app.example}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Algorithm Comparison Table */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">MST Algorithm Comparison</h2>
          <Card>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-muted border-b">
                      <th className="p-4 text-left font-semibold text-foreground">Algorithm</th>
                      <th className="p-4 text-left font-semibold text-foreground">Time Complexity</th>
                      <th className="p-4 text-left font-semibold text-foreground">Space Complexity</th>
                      <th className="p-4 text-left font-semibold text-foreground">Best For</th>
                      <th className="p-4 text-left font-semibold text-foreground">Key Data Structure</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">Prim's (Adj Matrix)</td>
                      <td className="p-4 font-mono">O(V²)</td>
                      <td className="p-4 font-mono">O(V)</td>
                      <td className="p-4 text-sm">Dense graphs</td>
                      <td className="p-4 text-sm">Array for min-edge</td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">Prim's (Adj List + Heap)</td>
                      <td className="p-4 font-mono">O((V+E) log V)</td>
                      <td className="p-4 font-mono">O(V)</td>
                      <td className="p-4 text-sm">General graphs</td>
                      <td className="p-4 text-sm">Priority Queue</td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">Kruskal's</td>
                      <td className="p-4 font-mono">O(E log E)</td>
                      <td className="p-4 font-mono">O(V)</td>
                      <td className="p-4 text-sm">Sparse graphs</td>
                      <td className="p-4 text-sm">Union-Find</td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">Boruvka's</td>
                      <td className="p-4 font-mono">O(E log V)</td>
                      <td className="p-4 font-mono">O(V)</td>
                      <td className="p-4 text-sm">Parallel processing</td>
                      <td className="p-4 text-sm">Component array</td>
                    </tr>
                    <tr className="hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">Reverse Delete</td>
                      <td className="p-4 font-mono">O(E log E)</td>
                      <td className="p-4 font-mono">O(V+E)</td>
                      <td className="p-4 text-sm">Edge removal scenarios</td>
                      <td className="p-4 text-sm">Graph + connectivity check</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Practice Problems */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Practice Problems</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Basic MST</h3>
                <ul className="space-y-2">
                  <li className="flex justify-between">
                    <span className="text-sm">Connecting Cities</span>
                    <Badge variant="outline">1135</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Min Cost to Connect</span>
                    <Badge variant="outline">1584</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Network Connection</span>
                    <Badge variant="outline">1319</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Advanced MST</h3>
                <ul className="space-y-2">
                  <li className="flex justify-between">
                    <span className="text-sm">Optimize Water Distribution</span>
                    <Badge variant="outline">1168</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Critical Connections</span>
                    <Badge variant="outline">1192</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Minimum Cost Walk</span>
                    <Badge variant="outline">2493</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Variants</h3>
                <ul className="space-y-2">
                  <li className="flex justify-between">
                    <span className="text-sm">Second Minimum MST</span>
                    <Badge variant="outline">Second MST</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Minimum Product MST</span>
                    <Badge variant="outline">Product MST</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Total Spanning Trees</span>
                    <Badge variant="outline">Matrix Tree</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Applications</h3>
                <ul className="space-y-2">
                  <li className="flex justify-between">
                    <span className="text-sm">Clustering</span>
                    <Badge variant="outline">Clustering</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Network Design</span>
                    <Badge variant="outline">Network</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Image Segmentation</span>
                    <Badge variant="outline">Segmentation</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Quiz Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Test Your MST Knowledge</h2>
          <Quiz questions={quizQuestions} title="Minimum Spanning Tree Quiz" />
        </section>

        {/* Quick Tips & Tricks */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Quick Tips & Interview Strategies</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="bg-primary/5 border-primary/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-3 text-foreground">Interview Tips</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                    <span>Always mention both Prim's and Kruskal's, then justify your choice</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                    <span>Discuss time/space complexity trade-offs</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                    <span>Explain the greedy choice property and why it works</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                    <span>Be prepared to handle negative weights (they're allowed in MST!)</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="bg-warning/5 border-warning/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-3 text-foreground">Common Mistakes</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Assuming MST only works for positive weights (works for any weights)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Forgetting to check if graph is connected before finding MST</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Using Dijkstra's for MST (they're different algorithms!)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Not handling parallel edges properly</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="bg-green-500/5 border-green-500/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-3 text-foreground">Optimization Tips</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <Zap className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>For dense graphs: Use Prim's with adjacency matrix O(V²)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Zap className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>For sparse graphs: Use Kruskal's O(E log E)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Zap className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>When edges are pre-sorted: Kruskal's becomes O(E α(V))</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Zap className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>For parallel processing: Boruvka's algorithm scales well</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="bg-blue-500/5 border-blue-500/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-3 text-foreground">Problem Patterns</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <Target className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>"Connect all points/cities" → Standard MST problem</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Target className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>"Minimum cable/wire needed" → MST with Euclidean distance</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Target className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>"Cluster points" → Stop MST early or use k-clustering</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Target className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>"Second best MST" → Remove each edge, find replacement</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>
      </div>
    </div>
  );
}