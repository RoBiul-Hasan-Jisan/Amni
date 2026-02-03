
import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { GraphVisualizer } from "@/components/visualizations/graph-visualizer";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { AlertCircle, CheckCircle2, Clock, Lightbulb, Target, Zap, Network, Layers, Hash, Flame, Map, CircuitBoard, Binary, ArrowRight, Cpu, BarChart, CpuIcon, GitCompare, Shield, Rocket, Gauge, Route } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

export default function ShortestPathPage() {
  const algorithmCategories = [
    {
      title: "Single Source",
      icon: Target,
      algorithms: ["Dijkstra", "Bellman-Ford", "DAG Shortest Path", "D'Esopo-Pape"]
    },
    {
      title: "All Pairs",
      icon: Network,
      algorithms: ["Floyd-Warshall", "Johnson's Algorithm"]
    },
    {
      title: "Special Cases",
      icon: Zap,
      algorithms: ["Unweighted (BFS)", "Binary Graph", "Dial's Algorithm", "0-1 BFS"]
    },
    {
      title: "Advanced",
      icon: Rocket,
      algorithms: ["Multistage Graph", "Minimum Mean Cycle", "Minimum Weight Cycle"]
    }
  ];

  const algorithmComparison = [
    {
      name: "Dijkstra's Algorithm",
      complexity: "O((V+E) log V)",
      weights: "Non-negative only",
      graph: "Directed/Undirected",
      bestFor: "Single source, non-negative weights",
      code: `import heapq

def dijkstra(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]  # (distance, vertex)
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        # Skip outdated entries
        if current_dist > dist[u]:
            continue
        
        for v, weight in graph[u]:
            new_dist = current_dist + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    
    return dist`
    },
    {
      name: "Bellman-Ford Algorithm",
      complexity: "O(VE)",
      weights: "Any (detects negative cycles)",
      graph: "Directed",
      bestFor: "Negative weights, small graphs",
      code: `def bellman_ford(n, edges, start):
    dist = [float('inf')] * n
    dist[start] = 0
    
    # Relax edges V-1 times
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break
    
    # Check for negative cycles
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # Negative cycle detected
    
    return dist`
    },
    {
      name: "Floyd-Warshall Algorithm",
      complexity: "O(V³)",
      weights: "Any (no negative cycles)",
      graph: "Directed/Undirected",
      bestFor: "All pairs shortest path, small V",
      code: `def floyd_warshall(n, edges):
    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Set diagonal to 0
    for i in range(n):
        dist[i][i] = 0
    
    # Add direct edges
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # Check for negative cycles
    for i in range(n):
        if dist[i][i] < 0:
            return None  # Negative cycle detected
    
    return dist`
    },
    {
      name: "Johnson's Algorithm",
      complexity: "O(VE + V² log V)",
      weights: "Any (no negative cycles)",
      graph: "Directed/Undirected",
      bestFor: "Sparse graphs, all pairs",
      code: `def johnsons_algorithm(n, edges):
    # Step 1: Add super source
    super_source = n
    modified_edges = edges.copy()
    for i in range(n):
        modified_edges.append((super_source, i, 0))
    
    # Step 2: Run Bellman-Ford from super source
    h = bellman_ford(n + 1, modified_edges, super_source)
    if h is None:
        return None  # Negative cycle detected
    
    # Step 3: Re-weight edges
    reweighted_edges = []
    for u, v, w in edges:
        reweighted_edges.append((u, v, w + h[u] - h[v]))
    
    # Step 4: Run Dijkstra from each vertex
    result = []
    for source in range(n):
        dist = dijkstra_with_edges(n, reweighted_edges, source)
        # Re-adjust distances
        for i in range(n):
            if dist[i] != float('inf'):
                dist[i] = dist[i] - h[source] + h[i]
        result.append(dist)
    
    return result`
    }
  ];

  const specializedAlgorithms = [
    {
      title: "Shortest Path in DAG",
      description: "Topological sort + DP for O(V+E) time",
      complexity: "O(V+E)",
      code: `def dag_shortest_path(n, edges, start):
    # Build adjacency list
    graph = [[] for _ in range(n)]
    indegree = [0] * n
    
    for u, v, w in edges:
        graph[u].append((v, w))
        indegree[v] += 1
    
    # Topological sort
    order = []
    queue = deque([i for i in range(n) if indegree[i] == 0])
    
    while queue:
        u = queue.popleft()
        order.append(u)
        for v, _ in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    
    # DP for shortest path
    dist = [float('inf')] * n
    dist[start] = 0
    
    for u in order:
        if dist[u] != float('inf'):
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
    
    return dist`
    },
    {
      title: "Dial's Algorithm",
      description: "Bucket-based Dijkstra for small integer weights",
      complexity: "O(W*V + E)",
      code: `def dials_algorithm(n, edges, start, max_weight):
    # Create adjacency list
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    
    # Initialize distances
    dist = [float('inf')] * n
    dist[start] = 0
    
    # Create buckets (0 to max_weight * V)
    num_buckets = max_weight * n + 1
    buckets = [set() for _ in range(num_buckets)]
    buckets[0].add(start)
    
    idx = 0
    while True:
        # Find next non-empty bucket
        while idx < num_buckets and not buckets[idx]:
            idx += 1
        
        if idx == num_buckets:
            break
        
        # Process bucket
        current = buckets[idx].pop()
        
        for v, w in graph[current]:
            new_dist = dist[current] + w
            if new_dist < dist[v]:
                # Remove from old bucket if present
                old_bucket = int(dist[v])
                if old_bucket < num_buckets and v in buckets[old_bucket]:
                    buckets[old_bucket].remove(v)
                
                # Update distance and add to new bucket
                dist[v] = new_dist
                new_bucket = int(new_dist)
                buckets[new_bucket].add(v)
    
    return dist`
    },
    {
      title: "0-1 BFS",
      description: "Deque-based BFS for 0/1 weighted graphs",
      complexity: "O(V+E)",
      code: `from collections import deque

def zero_one_bfs(n, edges, start):
    # Build adjacency list
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    
    # Initialize
    dist = [float('inf')] * n
    dist[start] = 0
    dq = deque([start])
    
    while dq:
        u = dq.popleft()
        
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    dq.appendleft(v)  # Push to front for 0-weight
                else:
                    dq.append(v)      # Push to back for 1-weight
    
    return dist`
    },
    {
      title: "Multistage Graph (DP)",
      description: "Dynamic programming for staged graphs",
      complexity: "O(V+E)",
      code: `def multistage_graph_shortest_path(n, stages, edges):
    """
    stages: list of vertices in each stage
    edges: (u, v, w) where u is in earlier stage than v
    """
    # Initialize DP table
    dp = [float('inf')] * n
    next_vertex = [-1] * n
    
    # Last stage vertices have cost 0
    for v in stages[-1]:
        dp[v] = 0
    
    # Process stages backwards
    for stage_idx in range(len(stages)-2, -1, -1):
        for u in stages[stage_idx]:
            for v, w in graph[u]:  # v in next stage
                if dp[v] + w < dp[u]:
                    dp[u] = dp[v] + w
                    next_vertex[u] = v
    
    # Reconstruct path
    path = []
    current = stages[0][0]  # Start from first vertex of first stage
    while current != -1:
        path.append(current)
        current = next_vertex[current]
    
    return dp[stages[0][0]], path`
    }
  ];

  const advancedAlgorithms = [
    {
      title: "Minimum Mean Weight Cycle",
      description: "Karp's algorithm for finding cycle with minimum average weight",
      complexity: "O(VE)",
      code: `def minimum_mean_cycle(n, edges):
    # Karp's algorithm
    # F[k][v] = minimum weight of path with exactly k edges from source to v
    F = [[float('inf')] * n for _ in range(n+1)]
    F[0][0] = 0  # Source is vertex 0
    
    # Compute F
    for k in range(1, n+1):
        for u, v, w in edges:
            if F[k-1][u] != float('inf'):
                F[k][v] = min(F[k][v], F[k-1][u] + w)
    
    # Compute minimum mean weight
    min_mean = float('inf')
    for v in range(n):
        if F[n][v] != float('inf'):
            max_ratio = -float('inf')
            for k in range(n):
                if F[k][v] != float('inf'):
                    ratio = (F[n][v] - F[k][v]) / (n - k)
                    max_ratio = max(max_ratio, ratio)
            min_mean = min(min_mean, max_ratio)
    
    return min_mean`
    },
    {
      title: "Minimum Weight Cycle",
      description: "Find cycle with minimum total weight",
      complexity: "O(V³)",
      code: `def minimum_weight_cycle(n, edges):
    # Floyd-Warshall for all pairs shortest path
    dist = [[float('inf')] * n for _ in range(n)]
    
    # Initialize
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)
    
    # Run Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # Find minimum cycle
    min_cycle = float('inf')
    for u, v, w in edges:
        if dist[v][u] != float('inf'):
            # Cycle: u -> v -> ... -> u
            cycle_weight = w + dist[v][u]
            min_cycle = min(min_cycle, cycle_weight)
    
    return min_cycle if min_cycle != float('inf') else -1`
    },
    {
      title: "D'Esopo-Pape Algorithm",
      description: "Queue-based shortest path, often faster than Dijkstra in practice",
      complexity: "O(2^V) worst, O(E) average",
      code: `from collections import deque

def desopo_pape(n, edges, start):
    # Build adjacency list
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    
    # Initialize
    dist = [float('inf')] * n
    dist[start] = 0
    in_queue = [False] * n
    m = [2] * n  # 0: in queue, 1: never in queue, 2: currently in queue
    
    dq = deque([start])
    m[start] = 0
    
    while dq:
        u = dq.popleft()
        in_queue[u] = False
        
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                
                if m[v] == 2:
                    # Was in queue, push to front
                    dq.appendleft(v)
                elif m[v] == 1:
                    # Never in queue, push to back
                    dq.append(v)
                
                m[v] = 0
                in_queue[v] = True
    
    return dist`
    },
    {
      title: "Shortest Path in Binary Graph",
      description: "BFS for unweighted graph, 0-1 BFS for 0/1 weighted",
      complexity: "O(V+E)",
      code: `from collections import deque

def shortest_path_binary_grid(grid):
    # Grid BFS for unweighted binary grid
    n = len(grid)
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    
    # 8 directions including diagonals
    directions = [
        (1,0), (-1,0), (0,1), (0,-1),
        (1,1), (1,-1), (-1,1), (-1,-1)
    ]
    
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    visited = [[False] * n for _ in range(n)]
    visited[0][0] = True
    
    while queue:
        r, c, dist = queue.popleft()
        
        if r == n-1 and c == n-1:
            return dist
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < n and 0 <= nc < n and 
                grid[nr][nc] == 0 and not visited[nr][nc]):
                visited[nr][nc] = True
                queue.append((nr, nc, dist + 1))
    
    return -1`
    }
  ];

  const algorithmProperties = [
    {
      property: "Optimality",
      description: "All algorithms find optimal shortest paths when conditions are met",
      algorithms: ["Dijkstra", "Bellman-Ford", "Floyd-Warshall", "Johnson's"]
    },
    {
      property: "Negative Weights",
      description: "Only Bellman-Ford and Floyd-Warshall handle negative weights",
      algorithms: ["Bellman-Ford", "Floyd-Warshall"]
    },
    {
      property: "Negative Cycles",
      description: "Detection possible in Bellman-Ford and Floyd-Warshall",
      algorithms: ["Bellman-Ford", "Floyd-Warshall"]
    },
    {
      property: "Greedy vs DP",
      description: "Dijkstra is greedy, Floyd-Warshall is dynamic programming",
      algorithms: ["Dijkstra (Greedy)", "Floyd-Warshall (DP)"]
    }
  ];

  const realWorldApplications = [
    {
      title: "GPS Navigation",
      icon: Map,
      description: "Dijkstra's algorithm powers most GPS routing systems",
      algorithm: "Dijkstra/A*",
      complexity: "O(E log V)"
    },
    {
      title: "Network Routing",
      icon: Network,
      description: "Bellman-Ford used in distance-vector routing protocols",
      algorithm: "Bellman-Ford",
      complexity: "O(VE)"
    },
    {
      title: "Flight Scheduling",
      icon: Rocket,
      description: "Johnson's algorithm for all-pairs shortest flights",
      algorithm: "Johnson's",
      complexity: "O(VE + V² log V)"
    },
    {
      title: "Circuit Design",
      icon: CircuitBoard,
      description: "Floyd-Warshall for timing analysis in circuits",
      algorithm: "Floyd-Warshall",
      complexity: "O(V³)"
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "Which algorithm is most efficient for single-source shortest path with non-negative weights?",
      options: ["Bellman-Ford", "Dijkstra", "Floyd-Warshall", "Johnson's"],
      correctAnswer: 1,
      explanation: "Dijkstra's algorithm with binary heap gives O((V+E) log V) time for non-negative weights.",
    },
    {
      id: 2,
      question: "What is the main advantage of Bellman-Ford over Dijkstra?",
      options: [
        "Faster for all graphs",
        "Handles negative weights",
        "Requires less memory",
        "Easier to implement"
      ],
      correctAnswer: 1,
      explanation: "Bellman-Ford can handle negative edge weights, while Dijkstra cannot.",
    },
    {
      id: 3,
      question: "Which algorithm is best for finding all-pairs shortest paths in a sparse graph?",
      options: ["Floyd-Warshall", "Run Dijkstra from each vertex", "Johnson's", "Bellman-Ford from each vertex"],
      correctAnswer: 2,
      explanation: "Johnson's algorithm re-weights edges to use Dijkstra for all vertices, efficient for sparse graphs.",
    },
    {
      id: 4,
      question: "What is the time complexity of Floyd-Warshall algorithm?",
      options: ["O(V²)", "O(V³)", "O(VE)", "O(V² log V)"],
      correctAnswer: 1,
      explanation: "Floyd-Warshall uses three nested loops over V vertices, giving O(V³) time complexity.",
    },
    {
      id: 5,
      question: "When should you use 0-1 BFS instead of Dijkstra?",
      options: [
        "For graphs with negative cycles",
        "For graphs with only 0 and 1 edge weights",
        "For graphs with more than 1000 vertices",
        "For directed acyclic graphs"
      ],
      correctAnswer: 1,
      explanation: "0-1 BFS using deque is optimal O(V+E) for graphs with only 0 and 1 edge weights.",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <header className="mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Shortest Path Algorithms Masterclass
          </h1>
          <p className="text-lg text-muted-foreground">
            Comprehensive guide to all shortest path algorithms with implementations, comparisons, and applications.
          </p>
        </header>

        {/* Categories Overview */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground flex items-center gap-2">
            <Route className="h-6 w-6" />
            Algorithm Categories
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

        {/* Core Algorithms */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Core Shortest Path Algorithms</h2>
          <div className="space-y-6">
            {algorithmComparison.map((algo, idx) => (
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
                          <h4 className="font-medium text-sm text-foreground mb-1">Weights:</h4>
                          <p className="text-sm text-muted-foreground">{algo.weights}</p>
                        </div>
                        <div>
                          <h4 className="font-medium text-sm text-foreground mb-1">Graph Type:</h4>
                          <p className="text-sm text-muted-foreground">{algo.graph}</p>
                        </div>
                        <div>
                          <h4 className="font-medium text-sm text-foreground mb-1">Best For:</h4>
                          <p className="text-sm text-muted-foreground">{algo.bestFor}</p>
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

        {/* Algorithm Comparison Table */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Algorithm Comparison Table</h2>
          <Card>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-muted border-b">
                      <th className="p-4 text-left font-semibold text-foreground">Algorithm</th>
                      <th className="p-4 text-left font-semibold text-foreground">Type</th>
                      <th className="p-4 text-left font-semibold text-foreground">Time Complexity</th>
                      <th className="p-4 text-left font-semibold text-foreground">Space Complexity</th>
                      <th className="p-4 text-left font-semibold text-foreground">Handles Negative Weights</th>
                      <th className="p-4 text-left font-semibold text-foreground">Detects Negative Cycles</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">Dijkstra (Heap)</td>
                      <td className="p-4">Single Source</td>
                      <td className="p-4 font-mono">O((V+E) log V)</td>
                      <td className="p-4 font-mono">O(V)</td>
                      <td className="p-4">
                        <Badge variant="destructive">No</Badge>
                      </td>
                      <td className="p-4">
                        <Badge variant="destructive">No</Badge>
                      </td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">Bellman-Ford</td>
                      <td className="p-4">Single Source</td>
                      <td className="p-4 font-mono">O(VE)</td>
                      <td className="p-4 font-mono">O(V)</td>
                      <td className="p-4">
                        <Badge variant="default">Yes</Badge>
                      </td>
                      <td className="p-4">
                        <Badge variant="default">Yes</Badge>
                      </td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">Floyd-Warshall</td>
                      <td className="p-4">All Pairs</td>
                      <td className="p-4 font-mono">O(V³)</td>
                      <td className="p-4 font-mono">O(V²)</td>
                      <td className="p-4">
                        <Badge variant="default">Yes</Badge>
                      </td>
                      <td className="p-4">
                        <Badge variant="default">Yes</Badge>
                      </td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">Johnson's</td>
                      <td className="p-4">All Pairs</td>
                      <td className="p-4 font-mono">O(VE + V² log V)</td>
                      <td className="p-4 font-mono">O(V²)</td>
                      <td className="p-4">
                        <Badge variant="default">Yes</Badge>
                      </td>
                      <td className="p-4">
                        <Badge variant="default">Yes</Badge>
                      </td>
                    </tr>
                    <tr className="hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">0-1 BFS</td>
                      <td className="p-4">Single Source</td>
                      <td className="p-4 font-mono">O(V+E)</td>
                      <td className="p-4 font-mono">O(V)</td>
                      <td className="p-4">
                        <Badge variant="destructive">0/1 only</Badge>
                      </td>
                      <td className="p-4">
                        <Badge variant="destructive">No</Badge>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Specialized Algorithms */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Specialized Algorithms</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {specializedAlgorithms.map((algo, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="font-semibold text-lg text-foreground mb-1">
                        {algo.title}
                      </h3>
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="outline" className="font-mono">
                          {algo.complexity}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {algo.description}
                      </p>
                    </div>
                  </div>
                  <div className="bg-muted p-4 rounded-lg overflow-x-auto">
                    <pre className="text-xs font-mono">
                      <code>{algo.code}</code>
                    </pre>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Advanced Algorithms */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Advanced Algorithms</h2>
          <Accordion type="multiple" className="space-y-4">
            {advancedAlgorithms.map((algo, idx) => (
              <AccordionItem key={idx} value={`algo-${idx}`} className="border rounded-lg">
                <AccordionTrigger className="px-6 hover:no-underline">
                  <div className="flex items-center justify-between w-full pr-4">
                    <div className="flex items-center gap-3">
                      <Badge variant="secondary" className="font-mono">
                        {algo.complexity}
                      </Badge>
                      <span className="font-semibold text-foreground">{algo.title}</span>
                    </div>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="px-6 pb-4">
                  <p className="text-sm text-muted-foreground mb-4">
                    {algo.description}
                  </p>
                  <CodeBlock
                    language="python"
                    code={algo.code}
                    className="mt-2"
                  />
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
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
                    <span className="text-xs font-mono text-muted-foreground">
                      {app.complexity}
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Algorithm Properties */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Algorithm Properties</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {algorithmProperties.map((prop, idx) => (
              <Card key={idx}>
                <CardContent className="p-6">
                  <h3 className="font-semibold mb-3 text-foreground">{prop.property}</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    {prop.description}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {prop.algorithms.map((algo, aIdx) => (
                      <Badge key={aIdx} variant="secondary">
                        {algo}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* When to Use Which Algorithm */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">When to Use Which Algorithm?</h2>
          <Card>
            <CardContent className="p-6">
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                    <Target className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2 text-foreground">Single Source, Non-negative Weights</h4>
                    <p className="text-sm text-muted-foreground">
                      Use <Badge variant="outline">Dijkstra's Algorithm</Badge> with binary heap. For small integer weights, consider <Badge variant="outline">Dial's Algorithm</Badge>.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-4">
                  <div className="h-8 w-8 rounded-full bg-warning/10 flex items-center justify-center shrink-0">
                    <AlertCircle className="h-4 w-4 text-warning" />
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2 text-foreground">Single Source, Negative Weights Possible</h4>
                    <p className="text-sm text-muted-foreground">
                      Use <Badge variant="outline">Bellman-Ford Algorithm</Badge>. It can detect negative cycles and works with any weights.
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-4">
                  <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                    <Network className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2 text-foreground">All Pairs Shortest Path</h4>
                    <p className="text-sm text-muted-foreground">
                      For dense graphs: <Badge variant="outline">Floyd-Warshall</Badge> (O(V³)). 
                      For sparse graphs: <Badge variant="outline">Johnson's Algorithm</Badge> (O(VE + V² log V)).
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start gap-4">
                  <div className="h-8 w-8 rounded-full bg-green-500/10 flex items-center justify-center shrink-0">
                    <Zap className="h-4 w-4 text-green-500" />
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2 text-foreground">Special Cases Optimization</h4>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      <li>• 0/1 weights: Use <Badge variant="outline">0-1 BFS</Badge> with deque</li>
                      <li>• Unweighted: Use <Badge variant="outline">BFS</Badge></li>
                      <li>• DAG: Use <Badge variant="outline">Topological Sort + DP</Badge></li>
                      <li>• Multistage Graph: Use <Badge variant="outline">Dynamic Programming</Badge></li>
                    </ul>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Quiz Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Test Your Knowledge</h2>
          <Quiz questions={quizQuestions} title="Shortest Path Algorithms Quiz" />
        </section>

        {/* Practice Problems */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Practice Problems</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Dijkstra Problems</h3>
                <ul className="space-y-2">
                  <li className="flex justify-between">
                    <span className="text-sm">Network Delay Time</span>
                    <Badge variant="outline">743</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Cheapest Flights</span>
                    <Badge variant="outline">787</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Path With Minimum Effort</span>
                    <Badge variant="outline">1631</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Bellman-Ford Problems</h3>
                <ul className="space-y-2">
                  <li className="flex justify-between">
                    <span className="text-sm">Currency Exchange</span>
                    <Badge variant="outline">Currency</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Arbitrage</span>
                    <Badge variant="outline">Arbitrage</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Negative Cycle Detection</span>
                    <Badge variant="outline">Bellman-Ford</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">All Pairs Problems</h3>
                <ul className="space-y-2">
                  <li className="flex justify-between">
                    <span className="text-sm">Find the City</span>
                    <Badge variant="outline">1334</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Minimum Cost to Connect</span>
                    <Badge variant="outline">1584</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Network Rank</span>
                    <Badge variant="outline">1615</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Special Cases</h3>
                <ul className="space-y-2">
                  <li className="flex justify-between">
                    <span className="text-sm">Shortest Path Binary Matrix</span>
                    <Badge variant="outline">1091</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Minimum Cost to Make Valid</span>
                    <Badge variant="outline">1368</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Swim in Rising Water</span>
                    <Badge variant="outline">778</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Quick Reference */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Quick Reference Guide</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <Card className="bg-primary/5 border-primary/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Time Complexity Cheatsheet</h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex justify-between">
                    <span>Dijkstra (Heap):</span>
                    <span className="font-mono">O(E log V)</span>
                  </li>
                  <li className="flex justify-between">
                    <span>Bellman-Ford:</span>
                    <span className="font-mono">O(VE)</span>
                  </li>
                  <li className="flex justify-between">
                    <span>Floyd-Warshall:</span>
                    <span className="font-mono">O(V³)</span>
                  </li>
                  <li className="flex justify-between">
                    <span>0-1 BFS:</span>
                    <span className="font-mono">O(V+E)</span>
                  </li>
                  <li className="flex justify-between">
                    <span>BFS (Unweighted):</span>
                    <span className="font-mono">O(V+E)</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="bg-warning/5 border-warning/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Common Pitfalls</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Using Dijkstra with negative weights</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Forgetting to check for negative cycles</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Using Floyd-Warshall for large V</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Not handling disconnected graphs</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="bg-green-500/5 border-green-500/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Optimization Tips</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Use A* search with heuristic for grids</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>For 0/1 weights, prefer 0-1 BFS over Dijkstra</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>For DAGs, topological sort is optimal</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Use Dial's algorithm for small integer weights</span>
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