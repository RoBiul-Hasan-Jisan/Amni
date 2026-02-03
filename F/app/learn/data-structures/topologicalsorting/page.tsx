// app/learn/data-structures/topological-sorting/page.tsx
import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { GraphVisualizer } from "@/components/visualizations/graph-visualizer";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { AlertCircle, CheckCircle2, Clock, Lightbulb, Target, Zap, Network, Layers, Hash, Flame, Map, CircuitBoard, Binary, ArrowRight, Cpu, BarChart, ArrowUpDown, GitBranch, SortAsc, SortDesc, Route, ListOrdered, Calendar, Ticket, Package } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

export default function TopologicalSortingPage() {
  const algorithmCategories = [
    {
      title: "Sorting Algorithms",
      icon: SortAsc,
      algorithms: ["Kahn's Algorithm", "DFS-based Sort", "All Topological Sorts"]
    },
    {
      title: "DAG Applications",
      icon: GitBranch,
      algorithms: ["Longest Path in DAG", "Maximum Edges in DAG", "Course Schedule", "Task Scheduling"]
    },
    {
      title: "Advanced Concepts",
      icon: ArrowUpDown,
      algorithms: ["Departure Time Sort", "Lexicographical Sort", "Detect Cycle via Topological Sort"]
    },
    {
      title: "Real-world Problems",
      icon: Ticket,
      algorithms: ["Flight Itinerary", "Build Order", "Package Dependencies", "Event Scheduling"]
    }
  ];

  const topologicalAlgorithms = [
    {
      name: "Kahn's Algorithm (BFS-based)",
      type: "Iterative",
      complexity: "O(V+E)",
      approach: "Remove vertices with 0 in-degree, update neighbors",
      code: `from collections import deque

def topological_sort_kahn(n, edges):
    """
    Kahn's Algorithm for Topological Sort
    Returns: topological order or empty list if cycle exists
    """
    # Build adjacency list and in-degree array
    graph = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    # Initialize queue with vertices having 0 in-degree
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topological_order = []
    
    while queue:
        u = queue.popleft()
        topological_order.append(u)
        
        # Reduce in-degree of neighbors
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Check if topological sort exists (no cycle)
    if len(topological_order) == n:
        return topological_order
    else:
        return []  # Cycle detected`
    },
    {
      name: "DFS-based Topological Sort",
      type: "Recursive",
      complexity: "O(V+E)",
      approach: "Post-order DFS, vertices are added to order after processing all descendants",
      code: `def topological_sort_dfs(n, edges):
    """
    DFS-based Topological Sort using departure time
    Returns: topological order or empty list if cycle exists
    """
    # Build adjacency list
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
    
    visited = [0] * n  # 0=unvisited, 1=visiting, 2=visited
    stack = []
    has_cycle = False
    
    def dfs(u):
        nonlocal has_cycle
        if has_cycle:
            return
        
        visited[u] = 1  # Mark as visiting
        
        for v in graph[u]:
            if visited[v] == 0:
                dfs(v)
            elif visited[v] == 1:
                # Back edge found → cycle
                has_cycle = True
                return
        
        visited[u] = 2  # Mark as visited
        stack.append(u)  # Add to stack after all descendants
    
    for i in range(n):
        if visited[i] == 0:
            dfs(i)
            if has_cycle:
                return []
    
    # Reverse stack to get topological order
    return stack[::-1]`
    },
    {
      name: "All Topological Sorts",
      type: "Backtracking",
      complexity: "O(V! * E) worst, O(V+E) per sort",
      approach: "Generate all permutations with constraint satisfaction",
      code: `def all_topological_sorts(n, edges):
    """
    Generate all possible topological orders of a DAG
    Returns: list of all topological orders
    """
    # Build adjacency list and in-degree array
    graph = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    visited = [False] * n
    current_path = []
    all_orders = []
    
    def backtrack():
        # Try all vertices that can be next
        for v in range(n):
            # Vertex can be next if: not visited and in-degree is 0
            if not visited[v] and in_degree[v] == 0:
                # Choose v
                visited[v] = True
                current_path.append(v)
                
                # Reduce in-degree of neighbors
                for neighbor in graph[v]:
                    in_degree[neighbor] -= 1
                
                # Recursively generate remaining vertices
                backtrack()
                
                # Backtrack
                for neighbor in graph[v]:
                    in_degree[neighbor] += 1
                current_path.pop()
                visited[v] = False
        
        # If all vertices are in current_path, save this order
        if len(current_path) == n:
            all_orders.append(current_path.copy())
    
    backtrack()
    return all_orders`
    },
    {
      name: "Topological Sort with Departure Time",
      type: "DFS with timing",
      complexity: "O(V+E)",
      approach: "Track departure time in DFS, sort vertices by decreasing departure time",
      code: `def topological_sort_departure_time(n, edges):
    """
    Topological sort using departure time tracking
    Returns: vertices sorted by decreasing departure time
    """
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
    
    visited = [False] * n
    departure = [-1] * n
    time = 0
    has_cycle = False
    
    def dfs(u):
        nonlocal time, has_cycle
        visited[u] = True
        
        for v in graph[u]:
            if not visited[v]:
                dfs(v)
            elif departure[v] == -1:
                # Back edge detected (cycle)
                has_cycle = True
        
        # Record departure time
        departure[u] = time
        time += 1
    
    for i in range(n):
        if not visited[i]:
            dfs(i)
    
    if has_cycle:
        return []
    
    # Sort vertices by decreasing departure time
    vertices = list(range(n))
    vertices.sort(key=lambda x: departure[x], reverse=True)
    return vertices`
    }
  ];

  const dagApplications = [
    {
      title: "Longest Path in DAG",
      description: "Find longest path from source to all vertices using topological order",
      complexity: "O(V+E)",
      code: `def longest_path_in_dag(n, edges, source):
    """
    Find longest path distances from source in weighted DAG
    edges: (u, v, weight)
    Returns: longest distance array, -inf for unreachable
    """
    # Build adjacency list
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    
    # Get topological order
    def topological_sort():
        in_degree = [0] * n
        for u, v, _ in edges:
            in_degree[v] += 1
        
        queue = deque([i for i in range(n) if in_degree[i] == 0])
        order = []
        
        while queue:
            u = queue.popleft()
            order.append(u)
            for v, _ in graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        return order if len(order) == n else []
    
    topo_order = topological_sort()
    if not topo_order:
        return []  # Cycle detected
    
    # Initialize distances
    dist = [float('-inf')] * n
    dist[source] = 0
    
    # Process vertices in topological order
    for u in topo_order:
        if dist[u] != float('-inf'):
            for v, w in graph[u]:
                if dist[u] + w > dist[v]:
                    dist[v] = dist[u] + w
    
    return dist`
    },
    {
      title: "Maximum Edges in DAG",
      description: "Maximum edges that can be added to DAG without creating cycles",
      complexity: "O(V²)",
      code: `def max_edges_in_dag(n, edges):
    """
    Calculate maximum number of edges that can be added
    to a DAG without creating cycles
    Returns: maximum additional edges possible
    """
    # Get topological order
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
    
    # Kahn's algorithm for topological sort
    in_degree = [0] * n
    for u in range(n):
        for v in graph[u]:
            in_degree[v] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # For a DAG with n vertices, maximum possible edges = n*(n-1)/2
    max_possible = n * (n - 1) // 2
    
    # Count existing edges (including those that could be added)
    existing_edges = len(edges)
    
    # Maximum additional edges = max_possible - existing_edges
    return max_possible - existing_edges
    
# Alternative: More detailed version that considers reachability
def max_edges_with_reachability(n, edges):
    """
    More precise calculation considering reachability
    """
    # Build adjacency matrix for reachability
    reachable = [[False] * n for _ in range(n)]
    
    # Floyd-Warshall for transitive closure
    for u, v in edges:
        reachable[u][v] = True
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if reachable[i][k] and reachable[k][j]:
                    reachable[i][j] = True
    
    # Count missing edges that don't create cycles
    # An edge (u,v) can be added if not already reachable from v to u
    missing_edges = 0
    for u in range(n):
        for v in range(n):
            if u != v and not reachable[u][v] and not reachable[v][u]:
                # No path in either direction, can add edge
                missing_edges += 1
    
    return missing_edges // 2  # Divide by 2 because we counted each pair twice`
    },
    {
      title: "Find Itinerary from Tickets",
      description: "Reconstruct itinerary from list of tickets (Eulerian path in directed graph)",
      complexity: "O(E log E)",
      code: `from collections import defaultdict, deque

def find_itinerary(tickets):
    """
    Reconstruct itinerary from list of [from, to] tickets
    Returns: complete itinerary starting from "JFK"
    """
    # Build adjacency list with sorted destinations
    graph = defaultdict(list)
    for src, dst in tickets:
        graph[src].append(dst)
    
    # Sort destinations in reverse order for lexical DFS
    for src in graph:
        graph[src].sort(reverse=True)
    
    itinerary = []
    
    def dfs(airport):
        # Visit all destinations from current airport
        while graph[airport]:
            next_airport = graph[airport].pop()
            dfs(next_airport)
        itinerary.append(airport)
    
    dfs("JFK")
    return itinerary[::-1]  # Reverse to get correct order

# Alternative: Using Hierholzer's algorithm for Eulerian path
def find_itinerary_hierholzer(tickets):
    """
    Hierholzer's algorithm for finding Eulerian path
    """
    # Build graph and count in/out degrees
    graph = defaultdict(deque)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    
    for src, dst in tickets:
        graph[src].append(dst)
        out_degree[src] += 1
        in_degree[dst] += 1
    
    # Find starting airport (JFK or airport with out_degree > in_degree)
    start = "JFK"
    for airport in graph:
        if out_degree[airport] > in_degree[airport]:
            start = airport
            break
    
    # Hierholzer's algorithm
    path = []
    stack = [start]
    
    while stack:
        while graph[stack[-1]]:
            next_airport = graph[stack[-1]].popleft()
            stack.append(next_airport)
        path.append(stack.pop())
    
    return path[::-1]`
    },
    {
      title: "Course Schedule II",
      description: "Find order of courses given prerequisites (standard topological sort)",
      complexity: "O(V+E)",
      code: `def find_order(num_courses, prerequisites):
    """
    Course Schedule II - Find valid course order
    Returns: course order or empty list if impossible
    """
    # Build graph and in-degree array
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Kahn's algorithm
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    order = []
    
    while queue:
        course = queue.popleft()
        order.append(course)
        
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    if len(order) == num_courses:
        return order
    else:
        return []  # Cycle detected, impossible schedule`
    }
  ];

  const topologicalProperties = [
    {
      property: "DAG Definition",
      description: "Directed Acyclic Graph - no directed cycles",
      implication: "Topological sort only exists for DAGs"
    },
    {
      property: "Multiple Orders",
      description: "A DAG can have multiple valid topological orders",
      implication: "Use constraints (lexicographic) for specific order"
    },
    {
      property: "Uniqueness",
      description: "Topological order is unique iff Hamiltonian path exists",
      implication: "Single valid order means linear structure"
    },
    {
      property: "Cycle Detection",
      description: "Failed topological sort indicates cycle",
      implication: "Kahn's algorithm can detect cycles"
    }
  ];

  const realWorldApplications = [
    {
      title: "Build Systems",
      icon: Cpu,
      description: "Determining build order for source files with dependencies",
      algorithm: "Kahn's Algorithm",
      example: "Makefile dependency resolution, Maven/Gradle builds"
    },
    {
      title: "Course Scheduling",
      icon: Calendar,
      description: "Scheduling courses based on prerequisites",
      algorithm: "Topological Sort",
      example: "University course registration systems"
    },
    {
      title: "Task Scheduling",
      icon: Clock,
      description: "Scheduling tasks with dependencies in project management",
      algorithm: "Longest Path in DAG",
      example: "Critical path method in project management"
    },
    {
      title: "Package Management",
      icon: Package,
      description: "Resolving package dependencies for installation",
      algorithm: "DFS-based Sort",
      example: "apt-get, npm, pip dependency resolution"
    }
  ];

  const algorithmComparison = [
    {
      aspect: "Approach",
      kahn: "BFS-based, iterative",
      dfs: "DFS-based, recursive/iterative"
    },
    {
      aspect: "Cycle Detection",
      kahn: "Count vertices processed",
      dfs: "Detect back edges during DFS"
    },
    {
      aspect: "Memory",
      kahn: "O(V) queue + O(V) in-degree",
      dfs: "O(V) recursion stack + O(V) visited"
    },
    {
      aspect: "Best For",
      kahn: "When in-degrees are needed anyway",
      dfs: "When graph is already being traversed"
    },
    {
      aspect: "Parallelization",
      kahn: "Easier to parallelize",
      dfs: "Harder due to recursion"
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is the time complexity of Kahn's algorithm for topological sort?",
      options: ["O(V²)", "O(V log V)", "O(V+E)", "O(E log E)"],
      correctAnswer: 2,
      explanation: "Kahn's algorithm visits each vertex and edge once, giving O(V+E) time complexity.",
    },
    {
      id: 2,
      question: "Which algorithm uses DFS and a stack for topological sort?",
      options: ["Kahn's", "Tarjan's", "DFS-based", "BFS-based"],
      correctAnswer: 2,
      explanation: "DFS-based topological sort uses post-order traversal and a stack to record vertices.",
    },
    {
      id: 3,
      question: "What does it mean if topological sort returns fewer than V vertices?",
      options: [
        "Graph is complete",
        "Graph is disconnected",
        "Graph has a cycle",
        "Algorithm has a bug"
      ],
      correctAnswer: 2,
      explanation: "If topological sort processes fewer than V vertices, it indicates a cycle in the graph.",
    },
    {
      id: 4,
      question: "In Kahn's algorithm, which vertices are added to the queue initially?",
      options: [
        "All vertices",
        "Vertices with 0 out-degree",
        "Vertices with 0 in-degree",
        "Random vertices"
      ],
      correctAnswer: 2,
      explanation: "Kahn's algorithm starts with vertices having 0 in-degree (no incoming edges).",
    },
    {
      id: 5,
      question: "What is the maximum number of edges in a DAG with n vertices?",
      options: ["n", "n-1", "n(n-1)/2", "2^n"],
      correctAnswer: 2,
      explanation: "A complete DAG (tournament) can have at most n(n-1)/2 edges without creating cycles.",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <header className="mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Topological Sorting & DAG Algorithms
          </h1>
          <p className="text-lg text-muted-foreground">
            Complete guide to topological sorting, DAG algorithms, and their applications in scheduling, dependency resolution, and path finding.
          </p>
        </header>

        {/* Categories Overview */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground flex items-center gap-2">
            <SortAsc className="h-6 w-6" />
            Topological Sorting Categories
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
          <h2 className="text-2xl font-bold mb-6 text-foreground">Topological Sorting Algorithms</h2>
          <div className="space-y-6">
            {topologicalAlgorithms.map((algo, idx) => (
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
                            {algo.name === "Kahn's Algorithm (BFS-based)" && "Cycle detection, when in-degrees are needed"}
                            {algo.name === "DFS-based Topological Sort" && "When already doing DFS traversal, recursive solutions"}
                            {algo.name === "All Topological Sorts" && "Enumerating all possible valid orders"}
                            {algo.name === "Topological Sort with Departure Time" && "When need departure times for other algorithms"}
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

        {/* Kahn's vs DFS Comparison */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Kahn's vs DFS-based Topological Sort</h2>
          <Card>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-muted border-b">
                      <th className="p-4 text-left font-semibold text-foreground">Aspect</th>
                      <th className="p-4 text-left font-semibold text-foreground">Kahn's Algorithm</th>
                      <th className="p-4 text-left font-semibold text-foreground">DFS-based Algorithm</th>
                    </tr>
                  </thead>
                  <tbody>
                    {algorithmComparison.map((row, idx) => (
                      <tr key={idx} className="border-b hover:bg-muted/50">
                        <td className="p-4 font-medium text-foreground">{row.aspect}</td>
                        <td className="p-4">{row.kahn}</td>
                        <td className="p-4">{row.dfs}</td>
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
                <h3 className="font-semibold mb-3 text-foreground">When to Use Kahn's Algorithm</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>Need to detect cycles explicitly</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>When in-degrees are needed for other calculations</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>Prefer iterative over recursive solutions</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>Want BFS-like level-by-level processing</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="bg-green-500/5 border-green-500/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-3 text-foreground">When to Use DFS-based Algorithm</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Already performing DFS traversal</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Need departure times for other algorithms</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Graph is large and recursion depth is manageable</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Implementing lexicographically smallest order</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* DAG Applications */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">DAG Applications & Problems</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {dagApplications.map((app, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="font-semibold text-lg text-foreground mb-1">
                        {app.title}
                      </h3>
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="outline" className="font-mono">
                          {app.complexity}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {app.description}
                      </p>
                    </div>
                  </div>
                  <div className="bg-muted p-4 rounded-lg overflow-x-auto">
                    <pre className="text-xs font-mono">
                      <code>{app.code}</code>
                    </pre>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Topological Properties */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Topological Sorting Properties</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {topologicalProperties.map((prop, idx) => (
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
          <h2 className="text-2xl font-bold mb-6 text-foreground">Algorithm Complexity Comparison</h2>
          <Card>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-muted border-b">
                      <th className="p-4 text-left font-semibold text-foreground">Algorithm</th>
                      <th className="p-4 text-left font-semibold text-foreground">Time Complexity</th>
                      <th className="p-4 text-left font-semibold text-foreground">Space Complexity</th>
                      <th className="p-4 text-left font-semibold text-foreground">Cycle Detection</th>
                      <th className="p-4 text-left font-semibold text-foreground">Parallelizable</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">Kahn's Algorithm</td>
                      <td className="p-4 font-mono">O(V+E)</td>
                      <td className="p-4 font-mono">O(V)</td>
                      <td className="p-4">
                        <Badge variant="default">Yes</Badge>
                      </td>
                      <td className="p-4">
                        <Badge variant="default">Yes</Badge>
                      </td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">DFS-based</td>
                      <td className="p-4 font-mono">O(V+E)</td>
                      <td className="p-4 font-mono">O(V)</td>
                      <td className="p-4">
                        <Badge variant="default">Yes</Badge>
                      </td>
                      <td className="p-4">
                        <Badge variant="destructive">No</Badge>
                      </td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">All Topological Sorts</td>
                      <td className="p-4 font-mono">O(V! * E)</td>
                      <td className="p-4 font-mono">O(V)</td>
                      <td className="p-4">
                        <Badge variant="destructive">No</Badge>
                      </td>
                      <td className="p-4">
                        <Badge variant="destructive">No</Badge>
                      </td>
                    </tr>
                    <tr className="hover:bg-muted/50">
                      <td className="p-4 font-medium text-foreground">Departure Time Sort</td>
                      <td className="p-4 font-mono">O(V+E)</td>
                      <td className="p-4 font-mono">O(V)</td>
                      <td className="p-4">
                        <Badge variant="default">Yes</Badge>
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

        {/* Practice Problems */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Practice Problems</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Basic Topological Sort</h3>
                <ul className="space-y-2">
                  <li className="flex justify-between">
                    <span className="text-sm">Course Schedule</span>
                    <Badge variant="outline">207</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Course Schedule II</span>
                    <Badge variant="outline">210</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Alien Dictionary</span>
                    <Badge variant="outline">269</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">DAG Applications</h3>
                <ul className="space-y-2">
                  <li className="flex justify-between">
                    <span className="text-sm">Longest Path in DAG</span>
                    <Badge variant="outline">DAG Path</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Parallel Courses</span>
                    <Badge variant="outline">1136</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Minimum Height Trees</span>
                    <Badge variant="outline">310</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Advanced Problems</h3>
                <ul className="space-y-2">
                  <li className="flex justify-between">
                    <span className="text-sm">Sequence Reconstruction</span>
                    <Badge variant="outline">444</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Find Eventual Safe States</span>
                    <Badge variant="outline">802</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Loud and Rich</span>
                    <Badge variant="outline">851</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Real-world Applications</h3>
                <ul className="space-y-2">
                  <li className="flex justify-between">
                    <span className="text-sm">Reconstruct Itinerary</span>
                    <Badge variant="outline">332</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Verifying Alien Dictionary</span>
                    <Badge variant="outline">953</Badge>
                  </li>
                  <li className="flex justify-between">
                    <span className="text-sm">Sort Items by Groups</span>
                    <Badge variant="outline">1203</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Quiz Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Test Your Knowledge</h2>
          <Quiz questions={quizQuestions} title="Topological Sorting Quiz" />
        </section>

        {/* Quick Tips & Interview Strategies */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Interview Strategies & Tips</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="bg-primary/5 border-primary/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-3 text-foreground">Interview Tips</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                    <span>Always check for cycles first - mention both detection methods</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                    <span>Discuss when to use Kahn's vs DFS-based approach</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                    <span>Mention real-world applications (build systems, scheduling)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                    <span>Be prepared for lexicographically smallest order questions</span>
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
                    <span>Forgetting to handle disconnected graphs</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Not checking for cycles when graph might not be DAG</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Incorrectly implementing DFS cycle detection (3-color method)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Not considering multiple valid topological orders</span>
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
                    <span>Use Kahn's for cycle detection and when need in-degrees</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Zap className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Use DFS-based for lexicographic order or when doing DFS anyway</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Zap className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>For longest path in DAG: Topological sort + DP is optimal</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Zap className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Use min-heap in Kahn's for lexicographically smallest order</span>
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
                    <span>"Course prerequisites" → Standard topological sort</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Target className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>"Find build order" → Topological sort with cycle check</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Target className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>"Longest path in project" → DAG longest path algorithm</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Target className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span>"Flight itinerary" → Eulerian path in directed graph</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Visualization Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Topological Sort Visualization</h2>
          <Card>
            <CardContent className="p-6">
              <div className="mb-6">
                <h3 className="font-semibold mb-3 text-foreground">Visualizing the Process</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Watch how topological sort processes vertices in order of dependencies.
                  Vertices with no incoming edges are processed first.
                </p>
                
                <div className="grid md:grid-cols-3 gap-4 mb-6">
                  <div className="p-4 bg-muted rounded-lg">
                    <h4 className="font-semibold mb-2 text-foreground">Kahn's Algorithm Steps</h4>
                    <ol className="text-sm space-y-1 list-decimal pl-4">
                      <li>Calculate in-degrees for all vertices</li>
                      <li>Add vertices with in-degree 0 to queue</li>
                      <li>Process queue, reduce neighbor in-degrees</li>
                      <li>Add new vertices with in-degree 0</li>
                      <li>Repeat until queue empty</li>
                    </ol>
                  </div>
                  
                  <div className="p-4 bg-muted rounded-lg">
                    <h4 className="font-semibold mb-2 text-foreground">DFS-based Steps</h4>
                    <ol className="text-sm space-y-1 list-decimal pl-4">
                      <li>Perform DFS from unvisited vertices</li>
                      <li>Mark vertices as visited</li>
                      <li>After processing all descendants, add to stack</li>
                      <li>Reverse stack for topological order</li>
                      <li>Detect cycles with 3-color method</li>
                    </ol>
                  </div>
                  
                  <div className="p-4 bg-muted rounded-lg">
                    <h4 className="font-semibold mb-2 text-foreground">Cycle Detection</h4>
                    <ul className="text-sm space-y-1">
                      <li>• Kahn's: Final order length &lt; V</li>
                      <li>• DFS: Back edge in DFS tree</li>
                      <li>• Both: O(V+E) time complexity</li>
                      <li>• Kahn's: Easier to understand</li>
                    </ul>
                  </div>
                </div>
              </div>
              
              {/* Graph Visualizer would go here */}
              <div className="text-center py-8 border-2 border-dashed border-border rounded-lg">
                <Network className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-muted-foreground">Graph visualization would appear here</p>
                <p className="text-sm text-muted-foreground mt-2">Interactive demonstration of topological sort algorithm</p>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Quick Reference */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Quick Reference</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <Card className="bg-purple-500/5 border-purple-500/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Complexity Cheatsheet</h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex justify-between">
                    <span>Kahn's Algorithm:</span>
                    <span className="font-mono">O(V+E)</span>
                  </li>
                  <li className="flex justify-between">
                    <span>DFS-based Sort:</span>
                    <span className="font-mono">O(V+E)</span>
                  </li>
                  <li className="flex justify-between">
                    <span>All Topological Sorts:</span>
                    <span className="font-mono">O(V! * E)</span>
                  </li>
                  <li className="flex justify-between">
                    <span>Longest Path in DAG:</span>
                    <span className="font-mono">O(V+E)</span>
                  </li>
                  <li className="flex justify-between">
                    <span>Cycle Detection:</span>
                    <span className="font-mono">O(V+E)</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="bg-orange-500/5 border-orange-500/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Key Properties</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <div className="h-2 w-2 rounded-full bg-orange-500 mt-1.5 shrink-0"></div>
                    <span>Only works for Directed Acyclic Graphs (DAGs)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="h-2 w-2 rounded-full bg-orange-500 mt-1.5 shrink-0"></div>
                    <span>Multiple valid orders possible</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="h-2 w-2 rounded-full bg-orange-500 mt-1.5 shrink-0"></div>
                    <span>Hamiltonian path = unique topological order</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <div className="h-2 w-2 rounded-full bg-orange-500 mt-1.5 shrink-0"></div>
                    <span>Partial order → total order (linear extension)</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="bg-teal-500/5 border-teal-500/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Implementation Notes</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-teal-500 mt-0.5 shrink-0" />
                    <span>Kahn's: Use deque for queue, track in-degrees</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-teal-500 mt-0.5 shrink-0" />
                    <span>DFS: Use 3 colors for cycle detection</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-teal-500 mt-0.5 shrink-0" />
                    <span>Lex order: Use min-heap instead of queue</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-teal-500 mt-0.5 shrink-0" />
                    <span>Longest path: Topo sort + dynamic programming</span>
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

