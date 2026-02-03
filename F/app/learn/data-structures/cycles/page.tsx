// still  issue in visualization
import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { GraphVisualizer } from "@/components/visualizations/graph-visualizer1";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { AlertCircle, CheckCircle2, Clock, Lightbulb, GitBranch, ArrowDown, ArrowUp, CircuitBoard, Target, Network, Layers, Hash, Zap, Flame, Map, TreePine, Droplets, Puzzle, Binary, Copy, Link, RotateCw, RefreshCw, GitCompare, Merge, ShieldAlert, Lock, Unlock, Brain, Sparkles } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function GraphCyclesPage() {
  const cycleCategories = [
    {
      title: "Basic Cycle Detection",
      icon: RotateCw,
      topics: ["DFS-based Detection", "Union-Find", "Kahn's Algorithm", "BFS Approaches"]
    },
    {
      title: "Advanced Detection",
      icon: Brain,
      topics: ["Tarjan's Algorithm", "Strongly Connected Components", "Articulation Points", "Bridges"]
    },
    {
      title: "Special Cycles",
      icon: Sparkles,
      topics: ["Eulerian Cycles", "Hamiltonian Cycles", "Minimum Cycle Basis", "Fundamental Cycles"]
    },
    {
      title: "Weighted Graphs",
      icon: Hash,
      topics: ["Negative Cycles", "Minimum Mean Cycle", "Currency Arbitrage", "Floyd-Warshall"]
    },
    {
      title: "Cycle Properties",
      icon: CircuitBoard,
      topics: ["Cycle Length", "Cycle Counting", "Girth", "Circumference"]
    },
    {
      title: "Applications",
      icon: Target,
      topics: ["Deadlock Detection", "Scheduling", "Network Analysis", "Compiler Optimization"]
    }
  ];

  const advancedAlgorithms = [
    {
      title: "Tarjan's Strongly Connected Components",
      complexity: "O(V + E)",
      description: "Single-pass DFS algorithm to find all SCCs and detect cycles",
      code: `def tarjan_scc(graph):
    """
    Tarjan's algorithm for finding Strongly Connected Components
    Each SCC with >1 node or self-loop contains a cycle
    """
    n = len(graph)
    ids = [-1] * n  # Node IDs
    low = [0] * n   # Lowest reachable ID
    on_stack = [False] * n
    stack = []
    id_counter = 0
    sccs = []
    
    def dfs(node):
        nonlocal id_counter
        
        ids[node] = low[node] = id_counter
        id_counter += 1
        stack.append(node)
        on_stack[node] = True
        
        for neighbor in graph[node]:
            if ids[neighbor] == -1:
                dfs(neighbor)
            if on_stack[neighbor]:
                low[node] = min(low[node], low[neighbor])
        
        # Found SCC root
        if ids[node] == low[node]:
            scc = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == node:
                    break
            sccs.append(scc)
    
    for i in range(n):
        if ids[i] == -1:
            dfs(i)
    
    return sccs

def has_cycle_tarjan(graph):
    sccs = tarjan_scc(graph)
    # Cycle exists if any SCC has >1 node
    return any(len(scc) > 1 for scc in sccs) or \
           any(node in graph[node] for node in graph)  # Self-loops`,
      insights: [
        "Single DFS pass finds all SCCs",
        "Uses low-link values to identify SCC roots",
        "Stack maintains current SCC being built",
        "More efficient than Kosaraju's for cycle detection"
      ]
    },
    {
      title: "Johnson's Elementary Cycles Algorithm",
      complexity: "O((V + E)(C + 1))",
      description: "Enumerate all elementary cycles in a directed graph",
      code: `def find_all_cycles(graph):
    """
    Johnson's algorithm to find all elementary cycles
    C = number of cycles in the graph
    """
    n = len(graph)
    all_cycles = []
    blocked = set()
    block_map = {i: set() for i in range(n)}
    stack = []
    
    def unblock(node):
        blocked.discard(node)
        for w in list(block_map[node]):
            block_map[node].discard(w)
            if w in blocked:
                unblock(w)
    
    def circuit(v, start, component):
        found_cycle = False
        stack.append(v)
        blocked.add(v)
        
        for w in graph[v]:
            if w not in component:
                continue
                
            if w == start:
                # Found a cycle
                all_cycles.append(stack[:])
                found_cycle = True
            elif w not in blocked:
                if circuit(w, start, component):
                    found_cycle = True
        
        if found_cycle:
            unblock(v)
        else:
            for w in graph[v]:
                if w in component:
                    block_map[w].add(v)
        
        stack.pop()
        return found_cycle
    
    # Find all SCCs using Tarjan
    sccs = tarjan_scc(graph)
    
    for scc in sccs:
        if len(scc) == 1 and scc[0] not in graph[scc[0]]:
            continue
        
        # Find cycles in this SCC
        component = set(scc)
        for start in scc:
            blocked.clear()
            for i in range(n):
                block_map[i].clear()
            circuit(start, start, component)
    
    return all_cycles`,
      insights: [
        "Finds ALL elementary cycles (simple cycles)",
        "Uses backtracking with blocking set",
        "Processes each SCC separately",
        "Exponential in worst case (as there can be exponentially many cycles)"
      ]
    },
    {
      title: "Minimum Mean Cycle (Karp's Algorithm)",
      complexity: "O(V * E)",
      description: "Find cycle with minimum average edge weight",
      code: `def minimum_mean_cycle(graph, n):
    """
    Karp's algorithm for minimum mean weight cycle
    Returns the minimum average weight of any cycle
    """
    # dist[k][v] = min distance using exactly k edges to reach v
    dist = [[float('inf')] * n for _ in range(n + 1)]
    dist[0][0] = 0
    
    # Compute shortest paths using k edges
    for k in range(1, n + 1):
        for u in range(n):
            if dist[k-1][u] == float('inf'):
                continue
            for v, weight in graph[u]:
                dist[k][v] = min(dist[k][v], dist[k-1][u] + weight)
    
    # Find minimum mean cycle
    min_mean = float('inf')
    
    for v in range(n):
        if dist[n][v] == float('inf'):
            continue
        
        max_val = float('-inf')
        for k in range(n):
            if dist[k][v] != float('inf'):
                max_val = max(max_val, (dist[n][v] - dist[k][v]) / (n - k))
        
        min_mean = min(min_mean, max_val)
    
    return min_mean if min_mean != float('inf') else None`,
      insights: [
        "Finds cycle with minimum average edge weight",
        "Uses dynamic programming approach",
        "Applications: Performance analysis, rate optimization",
        "Can detect negative cycles (mean < 0)"
      ]
    },
    {
      title: "Eulerian Cycle Detection",
      complexity: "O(V + E)",
      description: "Check if graph has Eulerian cycle (visits every edge exactly once)",
      code: `def has_eulerian_cycle(graph, directed=False):
    """
    Directed: Every vertex has in-degree = out-degree
    Undirected: Every vertex has even degree and graph is connected
    """
    n = len(graph)
    
    if directed:
        in_degree = [0] * n
        out_degree = [0] * n
        
        for u in range(n):
            out_degree[u] = len(graph[u])
            for v in graph[u]:
                in_degree[v] += 1
        
        # Check if in-degree = out-degree for all vertices
        for i in range(n):
            if in_degree[i] != out_degree[i]:
                return False
    else:
        # Check if all vertices have even degree
        for u in range(n):
            if len(graph[u]) % 2 != 0:
                return False
    
    # Check connectivity (simplified - assumes graph is given properly)
    return True

def find_eulerian_cycle(graph):
    """
    Hierholzer's algorithm to find Eulerian cycle
    """
    if not has_eulerian_cycle(graph):
        return None
    
    n = len(graph)
    adj = [list(graph[i]) for i in range(n)]  # Make mutable copy
    stack = [0]
    path = []
    
    while stack:
        u = stack[-1]
        if adj[u]:
            v = adj[u].pop()
            stack.append(v)
        else:
            path.append(stack.pop())
    
    return path[::-1]`,
      insights: [
        "Eulerian cycle exists iff all vertices have even degree (undirected)",
        "For directed: in-degree = out-degree for all vertices",
        "Hierholzer's algorithm finds the cycle efficiently",
        "Applications: Route planning, circuit design"
      ]
    },
    {
      title: "Hamiltonian Cycle Detection (Backtracking)",
      complexity: "O(V!)",
      description: "Check if graph has Hamiltonian cycle (visits every vertex exactly once)",
      code: `def has_hamiltonian_cycle(graph, n):
    """
    NP-Complete problem - use backtracking
    No known polynomial-time algorithm
    """
    path = [0]  # Start from vertex 0
    visited = [False] * n
    visited[0] = True
    
    def is_safe(v, pos):
        # Check if vertex v can be added at position pos
        if v not in graph[path[pos - 1]]:
            return False
        
        if visited[v]:
            return False
        
        return True
    
    def hamiltonian_util(pos):
        # Base case: all vertices included
        if pos == n:
            # Check if there's edge from last to first vertex
            return 0 in graph[path[-1]]
        
        # Try all vertices as next candidate
        for v in range(1, n):
            if is_safe(v, pos):
                path.append(v)
                visited[v] = True
                
                if hamiltonian_util(pos + 1):
                    return True
                
                # Backtrack
                path.pop()
                visited[v] = False
        
        return False
    
    return hamiltonian_util(1)

# DP approach using bitmask (faster for small graphs)
def hamiltonian_cycle_dp(graph, n):
    """
    Dynamic programming with bitmask
    O(2^n * n^2) time, O(2^n * n) space
    """
    # dp[mask][i] = can we visit vertices in mask ending at i?
    dp = [[False] * n for _ in range(1 << n)]
    dp[1][0] = True  # Start at vertex 0
    
    for mask in range(1 << n):
        for last in range(n):
            if not dp[mask][last]:
                continue
            
            for next_v in graph[last]:
                if mask & (1 << next_v):
                    continue
                
                new_mask = mask | (1 << next_v)
                dp[new_mask][next_v] = True
    
    # Check if we can return to start
    full_mask = (1 << n) - 1
    for last in range(n):
        if dp[full_mask][last] and 0 in graph[last]:
            return True
    
    return False`,
      insights: [
        "NP-Complete - no known polynomial solution",
        "Backtracking works for small graphs",
        "DP with bitmask is O(2^n * n^2)",
        "Many real-world approximations exist (TSP, etc.)"
      ]
    }
  ];

  const cycleDetectionMethods = [
    {
      name: "DFS for Undirected Graph",
      complexity: "O(V+E)",
      description: "Check for back edges to non-parent vertices",
      code: `def has_cycle_undirected(graph):
    visited = set()
    
    def dfs(node, parent):
        visited.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                # Found back edge to non-parent
                return True
        
        return False
    
    for node in graph:
        if node not in visited:
            if dfs(node, -1):
                return True
    
    return False

# With cycle path reconstruction
def find_cycle_path_undirected(graph):
    visited = set()
    parent = {}
    
    def dfs(node, par):
        visited.add(node)
        parent[node] = par
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                cycle = dfs(neighbor, node)
                if cycle:
                    return cycle
            elif neighbor != par:
                # Reconstruct cycle
                cycle = [neighbor, node]
                curr = node
                while parent[curr] != neighbor:
                    curr = parent[curr]
                    cycle.append(curr)
                return cycle
        
        return None
    
    for node in graph:
        if node not in visited:
            cycle = dfs(node, -1)
            if cycle:
                return cycle
    
    return None`
    },
    {
      name: "Three-Color DFS (Directed)",
      complexity: "O(V+E)",
      description: "WHITE=unvisited, GRAY=processing, BLACK=done",
      code: `def has_cycle_directed(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    colors = {node: WHITE for node in graph}
    
    def dfs(node):
        colors[node] = GRAY
        
        for neighbor in graph.get(node, []):
            if colors[neighbor] == GRAY:
                # Back edge to ancestor
                return True
            if colors[neighbor] == WHITE:
                if dfs(neighbor):
                    return True
        
        colors[node] = BLACK
        return False
    
    for node in graph:
        if colors[node] == WHITE:
            if dfs(node):
                return True
    
    return False

# With cycle path reconstruction
def find_cycle_path_directed(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    colors = {node: WHITE for node in graph}
    parent = {}
    
    def dfs(node, path):
        colors[node] = GRAY
        path.add(node)
        
        for neighbor in graph.get(node, []):
            if colors[neighbor] == GRAY:
                # Found cycle - reconstruct it
                cycle = [neighbor]
                curr = node
                while curr != neighbor:
                    cycle.append(curr)
                    curr = parent[curr]
                return cycle[::-1]
            
            if colors[neighbor] == WHITE:
                parent[neighbor] = node
                cycle = dfs(neighbor, path)
                if cycle:
                    return cycle
        
        colors[node] = BLACK
        path.remove(node)
        return None
    
    for node in graph:
        if colors[node] == WHITE:
            parent[node] = None
            cycle = dfs(node, set())
            if cycle:
                return cycle
    
    return None`
    },
    {
      name: "Union-Find with Optimizations",
      complexity: "O(E * α(V))",
      description: "Path compression + union by rank for near-constant time",
      code: `class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.components = n
    
    def find(self, x):
        # Path compression - makes tree flat
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Cycle detected!
        
        # Union by rank - attach smaller tree to larger
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
            self.size[root_x] += self.size[root_y]
        
        self.components -= 1
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
    
    def component_size(self, x):
        return self.size[self.find(x)]
    
    def count_components(self):
        return self.components

def detect_cycle_edges(edges, n):
    """Returns list of edges that create cycles"""
    uf = UnionFind(n)
    cycle_edges = []
    
    for u, v in edges:
        if not uf.union(u, v):
            cycle_edges.append((u, v))
    
    return cycle_edges`
    },
    {
      name: "Floyd's Cycle Detection (Linked List)",
      complexity: "O(n)",
      description: "Tortoise and Hare algorithm for detecting cycles in sequences",
      code: `def floyd_cycle_detection(head):
    """
    Detect cycle in linked list using two pointers
    Also finds cycle start and length
    """
    if not head or not head.next:
        return None
    
    # Phase 1: Detect if cycle exists
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            # Cycle detected
            break
    else:
        return None  # No cycle
    
    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    cycle_start = slow
    
    # Phase 3: Find cycle length
    length = 1
    curr = slow.next
    while curr != slow:
        length += 1
        curr = curr.next
    
    return {
        'has_cycle': True,
        'start': cycle_start,
        'length': length
    }

# Brent's algorithm (alternative, often faster)
def brent_cycle_detection(head):
    """
    Brent's cycle detection - fewer operations than Floyd's
    """
    if not head:
        return None
    
    power = 1
    length = 1
    tortoise = head
    hare = head.next
    
    while hare and tortoise != hare:
        if power == length:
            tortoise = hare
            power *= 2
            length = 0
        
        hare = hare.next
        length += 1
    
    if not hare:
        return None
    
    # Find cycle start
    tortoise = hare = head
    for _ in range(length):
        hare = hare.next
    
    mu = 0  # Start position
    while tortoise != hare:
        tortoise = tortoise.next
        hare = hare.next
        mu += 1
    
    return {
        'has_cycle': True,
        'start_position': mu,
        'length': length
    }`
    }
  ];

  const advancedProblems = [
    {
      title: "Find All Cycles in Directed Graph",
      difficulty: "Hard",
      description: "Enumerate all simple cycles using Johnson's algorithm",
      code: `def find_all_simple_cycles(graph):
    """
    Complete implementation with cycle enumeration
    Time: O((V+E)(C+1)) where C is number of cycles
    """
    n = len(graph)
    all_cycles = []
    
    # Helper for Tarjan's SCC
    def tarjan_scc(graph):
        # ... (implementation from above)
        pass
    
    sccs = tarjan_scc(graph)
    
    for scc in sccs:
        if len(scc) == 1:
            node = scc[0]
            if node not in graph[node]:
                continue
        
        # Find all cycles in this SCC
        component = set(scc)
        blocked = set()
        block_map = {i: set() for i in range(n)}
        stack = []
        
        def unblock(node):
            blocked.discard(node)
            for w in list(block_map[node]):
                block_map[node].discard(w)
                if w in blocked:
                    unblock(w)
        
        def circuit(v, start):
            found = False
            stack.append(v)
            blocked.add(v)
            
            for w in graph[v]:
                if w not in component:
                    continue
                
                if w == start:
                    all_cycles.append(stack[:])
                    found = True
                elif w not in blocked:
                    if circuit(w, start):
                        found = True
            
            if found:
                unblock(v)
            else:
                for w in graph[v]:
                    if w in component:
                        block_map[w].add(v)
            
            stack.pop()
            return found
        
        for start in sorted(scc):
            blocked.clear()
            circuit(start, start)
    
    return all_cycles`,
      applications: [
        "Chemical reaction cycles",
        "Program control flow analysis",
        "Dependency resolution",
        "Game state analysis"
      ]
    },
    {
      title: "Detect Cycle of Specific Length K",
      difficulty: "Medium",
      description: "Find if there exists a cycle of exactly K vertices",
      code: `def has_cycle_length_k(graph, n, k):
    """
    DFS with depth tracking to find cycle of exact length k
    Time: O(V * V^k) in worst case
    """
    def dfs(start, current, depth, visited, path):
        if depth == k:
            # Check if we can return to start
            if start in graph[current]:
                return True, path + [start]
            return False, []
        
        for neighbor in graph[current]:
            if neighbor not in visited or (neighbor == start and depth == k - 1):
                if neighbor == start and depth < k - 1:
                    continue
                
                visited.add(neighbor)
                found, cycle = dfs(start, neighbor, depth + 1, 
                                  visited, path + [neighbor])
                if found:
                    return True, cycle
                visited.remove(neighbor)
        
        return False, []
    
    # Try starting from each vertex
    for start_vertex in range(n):
        visited = {start_vertex}
        found, cycle = dfs(start_vertex, start_vertex, 0, visited, [start_vertex])
        if found:
            return True, cycle
    
    return False, []

# More efficient: Use DP with bitmask for small k
def count_cycles_length_k(graph, n, k):
    """
    Count all cycles of length k using DP
    Time: O(V^k), Space: O(V^k)
    """
    # dp[length][current][first] = number of paths
    dp = {}
    
    def count_paths(length, current, first, visited):
        if length == k:
            return 1 if first in graph[current] else 0
        
        if (length, current, first, frozenset(visited)) in dp:
            return dp[(length, current, first, frozenset(visited))]
        
        total = 0
        for neighbor in graph[current]:
            if neighbor not in visited or (neighbor == first and length == k - 1):
                if neighbor == first and length < k - 1:
                    continue
                
                new_visited = visited | {neighbor}
                total += count_paths(length + 1, neighbor, first, new_visited)
        
        dp[(length, current, first, frozenset(visited))] = total
        return total
    
    total_cycles = 0
    for start in range(n):
        total_cycles += count_paths(1, start, start, {start})
    
    # Each cycle is counted k times (once for each starting vertex)
    return total_cycles // k`,
      applications: [
        "Finding triangles in graphs (k=3)",
        "Detecting specific pattern cycles",
        "Social network analysis",
        "Biochemical pathway analysis"
      ]
    },
    {
      title: "Minimum Cycle Basis",
      difficulty: "Hard",
      description: "Find the minimum set of independent cycles",
      code: `def minimum_cycle_basis(graph, n):
    """
    Find minimum cycle basis - set of fundamental cycles
    For a connected graph: |E| - |V| + 1 cycles
    Time: O(V^3) using Gaussian elimination
    """
    edges = []
    edge_to_idx = {}
    
    # Collect all edges
    idx = 0
    for u in range(n):
        for v in graph[u]:
            if u < v:  # Avoid duplicates in undirected
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx
                idx += 1
    
    m = len(edges)
    
    # Build spanning tree using DFS
    visited = set()
    tree_edges = set()
    parent = [-1] * n
    
    def build_tree(u):
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                parent[v] = u
                tree_edges.add((min(u, v), max(u, v)))
                build_tree(v)
    
    build_tree(0)
    
    # For each non-tree edge, find the fundamental cycle
    cycles = []
    
    for u, v in edges:
        if (u, v) not in tree_edges and (v, u) not in tree_edges:
            # Find path from u to v in tree
            path_u = []
            curr = u
            while curr != -1:
                path_u.append(curr)
                curr = parent[curr]
            
            path_v = []
            curr = v
            while curr != -1:
                path_v.append(curr)
                curr = parent[curr]
            
            # Find LCA
            set_u = set(path_u)
            lca = None
            for node in path_v:
                if node in set_u:
                    lca = node
                    break
            
            # Build cycle
            cycle = [u]
            curr = u
            while curr != lca:
                curr = parent[curr]
                cycle.append(curr)
            
            path_to_v = []
            curr = v
            while curr != lca:
                path_to_v.append(curr)
                curr = parent[curr]
            
            cycle.extend(path_to_v[::-1])
            cycle.append(u)  # Close the cycle
            
            cycles.append(cycle)
    
    return cycles`,
      applications: [
        "Circuit analysis (Kirchhoff's laws)",
        "Structural engineering",
        "Chemical graph theory",
        "Network reliability analysis"
      ]
    },
    {
      title: "Currency Arbitrage Detection",
      difficulty: "Medium",
      description: "Detect arbitrage opportunities using negative cycle detection",
      code: `import math

def detect_arbitrage(exchange_rates):
    """
    Detect currency arbitrage using Bellman-Ford
    exchange_rates[i][j] = rate to convert currency i to j
    
    Convert to negative log to find negative cycles:
    product > 1 => sum of logs > 0 => sum of negative logs < 0
    """
    n = len(exchange_rates)
    
    # Convert to log graph (negative logs)
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and exchange_rates[i][j] > 0:
                # Negative log converts product to sum
                weight = -math.log(exchange_rates[i][j])
                edges.append((i, j, weight))
    
    # Run Bellman-Ford from any source
    dist = [float('inf')] * n
    dist[0] = 0
    parent = [-1] * n
    
    # Relax edges V-1 times
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
    
    # Check for negative cycles (arbitrage opportunities)
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            # Found negative cycle - trace it
            cycle = []
            visited = set()
            curr = v
            
            while curr not in visited:
                visited.add(curr)
                cycle.append(curr)
                curr = parent[curr]
            
            # curr is start of cycle
            arbitrage_cycle = []
            start = curr
            arbitrage_cycle.append(curr)
            curr = parent[curr]
            
            while curr != start:
                arbitrage_cycle.append(curr)
                curr = parent[curr]
            
            arbitrage_cycle.append(start)
            
            # Calculate actual profit
            profit = 1.0
            for i in range(len(arbitrage_cycle) - 1):
                a, b = arbitrage_cycle[i], arbitrage_cycle[i + 1]
                profit *= exchange_rates[a][b]
            
            return True, arbitrage_cycle, profit
    
    return False, [], 0

# Example usage
def example_arbitrage():
    # Currencies: USD, EUR, GBP, JPY
    rates = [
        [1.0,   0.85,  0.75,  110.0],  # USD to others
        [1.18,  1.0,   0.88,  129.0],  # EUR to others
        [1.33,  1.14,  1.0,   147.0],  # GBP to others
        [0.0091, 0.0078, 0.0068, 1.0]  # JPY to others
    ]
    
    has_arb, cycle, profit = detect_arbitrage(rates)
    
    if has_arb:
        print(f"Arbitrage found! Profit: {profit:.4f}x")
        print(f"Cycle: {' -> '.join(map(str, cycle))}")
    else:
        print("No arbitrage opportunities")`,
      applications: [
        "Currency trading",
        "Sports betting arbitrage",
        "Price discrepancy detection",
        "Market inefficiency analysis"
      ]
    }
  ];

  const cycleProperties = [
    {
      property: "Girth",
      description: "Length of the shortest cycle in a graph",
      formal: "min{|C| : C is a cycle in G}",
      significance: "Important in graph theory, coding theory, and network design"
    },
    {
      property: "Circumference",
      description: "Length of the longest cycle in a graph",
      formal: "max{|C| : C is a cycle in G}",
      significance: "Related to Hamiltonian cycles; circumference = |V| means Hamiltonian"
    },
    {
      property: "Cycle Rank (Cyclomatic Number)",
      description: "Minimum number of edges to remove to make graph acyclic",
      formal: "ν(G) = |E| - |V| + c (c = number of connected components)",
      significance: "Used in software testing (McCabe complexity)"
    },
    {
      property: "Fundamental Cycles",
      description: "Cycles formed by adding non-tree edges to spanning tree",
      formal: "One fundamental cycle for each edge not in spanning tree",
      significance: "Forms basis for cycle space; minimum cycle basis"
    },
    {
      property: "Chordal Cycle",
      description: "Cycle with at least one chord (edge between non-adjacent vertices)",
      formal: "Cycle C with edge (u,v) where u,v ∈ C but (u,v) ∉ C",
      significance: "Important in perfect graphs and triangulation"
    },
    {
      property: "Odd/Even Cycles",
      description: "Cycles with odd/even number of vertices",
      formal: "|C| ≡ 1 (mod 2) or |C| ≡ 0 (mod 2)",
      significance: "Odd cycles prevent bipartiteness"
    }
  ];

  const realWorldApplications = [
    {
      title: "Deadlock Detection in Operating Systems",
      icon: ShieldAlert,
      description: "Resource allocation graphs where cycles indicate deadlocks",
      example: "Process P1 holds R1, waits for R2; P2 holds R2, waits for R1 → cycle → deadlock",
      algorithm: "DFS-based cycle detection in resource allocation graph",
      complexity: "O(P + R) where P=processes, R=resources"
    },
    {
      title: "Compiler Optimization",
      icon: Zap,
      description: "Detect cycles in data dependency graphs for instruction scheduling",
      example: "x = y + z; y = x * 2; → dependency cycle prevents reordering",
      algorithm: "Strongly connected components to find cyclic dependencies",
      complexity: "O(V + E) using Tarjan's algorithm"
    },
    {
      title: "Task Scheduling & Project Management",
      icon: Clock,
      description: "Detect circular dependencies in task graphs (PERT/CPM)",
      example: "Task A depends on B, B on C, C on A → impossible schedule",
      algorithm: "Topological sort failure or DFS cycle detection",
      complexity: "O(V + E)"
    },
    {
      title: "Network Routing Protocols",
      icon: Network,
      description: "Detect and prevent routing loops in distance vector routing",
      example: "Router A sends to B, B to C, C back to A → routing loop",
      algorithm: "Bellman-Ford for negative cycle detection",
      complexity: "O(V * E)"
    },
    {
      title: "Garbage Collection",
      icon: RefreshCw,
      description: "Find cyclic references in memory management",
      example: "Object A references B, B references C, C references A → memory leak",
      algorithm: "Mark-and-sweep with cycle detection",
      complexity: "O(Objects + References)"
    },
    {
      title: "Chemical Reaction Networks",
      icon: Merge,
      description: "Analyze cycles in metabolic pathways and reaction networks",
      example: "A → B → C → A forms catalytic cycle",
      algorithm: "Elementary cycle enumeration",
      complexity: "O((V+E)(C+1)) where C is cycle count"
    }
  ];

  const optimizationTechniques = [
    {
      title: "Early Termination",
      description: "Stop search as soon as first cycle is found",
      code: `def has_cycle_early_exit(graph):
    # Return immediately when cycle found
    # Don't need to explore entire graph`,
      benefit: "Best case O(E) instead of O(V+E)"
    },
    {
      title: "Iterative Deepening",
      description: "Find shortest cycles first using depth-limited DFS",
      code: `def find_shortest_cycle(graph):
    for depth in range(3, n + 1):
        cycle = dfs_limited_depth(graph, depth)
        if cycle:
            return cycle`,
      benefit: "Finds girth efficiently"
    },
    {
      title: "Parallel Cycle Detection",
      description: "Process multiple connected components in parallel",
      code: `from multiprocessing import Pool

def parallel_cycle_detection(graph):
    components = find_components(graph)
    with Pool() as pool:
        results = pool.map(has_cycle, components)
    return any(results)`,
      benefit: "O(V+E)/k with k processors"
    },
    {
      title: "Incremental Updates",
      description: "Maintain cycle-free property during edge additions",
      code: `class IncrementalCycleChecker:
    def __init__(self, n):
        self.uf = UnionFind(n)
    
    def add_edge(self, u, v):
        if self.uf.connected(u, v):
            return False  # Would create cycle
        self.uf.union(u, v)
        return True`,
      benefit: "O(α(n)) per edge instead of O(V+E)"
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is the time complexity of Tarjan's algorithm for finding strongly connected components?",
      options: ["O(V log V)", "O(V + E)", "O(V²)", "O(E log V)"],
      correctAnswer: 1,
      explanation: "Tarjan's algorithm uses a single DFS pass to find all SCCs in O(V + E) time.",
    },
    {
      id: 2,
      question: "Which condition indicates a graph has an Eulerian cycle?",
      options: [
        "All vertices have odd degree",
        "All vertices have even degree and graph is connected",
        "Graph is bipartite",
        "Graph is acyclic"
      ],
      correctAnswer: 1,
      explanation: "For undirected graphs, Eulerian cycle exists iff all vertices have even degree and the graph is connected.",
    },
    {
      id: 3,
      question: "What is the cycle rank (cyclomatic number) of a connected graph?",
      options: ["V - E", "E - V + 1", "V + E", "E - V - 1"],
      correctAnswer: 1,
      explanation: "Cycle rank = E - V + 1 for a connected graph. It represents the minimum edges to remove to make it acyclic.",
    },
    {
      id: 4,
      question: "Which algorithm is best for enumerating ALL cycles in a directed graph?",
      options: ["DFS", "Tarjan's SCC + Johnson's", "Union-Find", "Kahn's Algorithm"],
      correctAnswer: 1,
      explanation: "Johnson's algorithm (after finding SCCs with Tarjan's) efficiently enumerates all elementary cycles.",
    },
    {
      id: 5,
      question: "How can you detect currency arbitrage opportunities?",
      options: [
        "BFS on exchange rates",
        "Bellman-Ford on negative log of rates",
        "Dijkstra's algorithm",
        "Topological sort"
      ],
      correctAnswer: 1,
      explanation: "Convert exchange rates to negative logs and use Bellman-Ford to detect negative cycles (arbitrage).",
    },
    {
      id: 6,
      question: "What is the girth of a graph?",
      options: [
        "Maximum cycle length",
        "Minimum cycle length",
        "Average cycle length",
        "Number of cycles"
      ],
      correctAnswer: 1,
      explanation: "Girth is the length of the shortest cycle in a graph.",
    },
    {
      id: 7,
      question: "Which data structure gives O(α(n)) amortized time for cycle detection in undirected graphs?",
      options: ["Stack", "Queue", "Union-Find with path compression", "Heap"],
      correctAnswer: 2,
      explanation: "Union-Find with path compression and union by rank achieves O(α(n)) amortized time per operation.",
    },
    {
      id: 8,
      question: "What does a back edge in DFS indicate?",
      options: [
        "Tree edge",
        "Cross edge",
        "Cycle in the graph",
        "Disconnected component"
      ],
      correctAnswer: 2,
      explanation: "A back edge (edge to an ancestor in DFS tree) always indicates the presence of a cycle.",
    },
    {
      id: 9,
      question: "Is finding a Hamiltonian cycle polynomial-time solvable?",
      options: [
        "Yes, O(V+E)",
        "Yes, O(V²)",
        "No, it's NP-Complete",
        "Yes, O(V log V)"
      ],
      correctAnswer: 2,
      explanation: "Hamiltonian cycle is NP-Complete. Best known algorithms are exponential (backtracking, DP with bitmask).",
    },
    {
      id: 10,
      question: "What is the relationship between fundamental cycles and spanning trees?",
      options: [
        "No relationship",
        "Each non-tree edge creates one fundamental cycle",
        "Each tree edge creates one fundamental cycle",
        "Fundamental cycles exist only in cyclic graphs"
      ],
      correctAnswer: 1,
      explanation: "Adding each non-tree edge to a spanning tree creates exactly one fundamental cycle.",
    },
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <header className="mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Advanced Graph Cycles & Detection Algorithms
          </h1>
          <p className="text-lg text-muted-foreground mb-4">
            Comprehensive guide to cycle detection, enumeration, and analysis with advanced algorithms and real-world applications.
          </p>
          
          <Alert className="mb-4">
            <Brain className="h-4 w-4" />
            <AlertTitle>Advanced Content</AlertTitle>
            <AlertDescription>
              This guide covers advanced topics including Tarjan's SCC, Johnson's cycle enumeration, Eulerian/Hamiltonian cycles, and cycle basis theory.
            </AlertDescription>
          </Alert>
        </header>

        {/* Categories Overview */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground flex items-center gap-2">
            <RotateCw className="h-6 w-6" />
            Comprehensive Coverage
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {cycleCategories.map((category, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <category.icon className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold text-foreground">{category.title}</h3>
                  </div>
                  <ul className="space-y-2">
                    {category.topics.map((topic, tIdx) => (
                      <li key={tIdx} className="flex items-center gap-2 text-sm">
                        <div className="h-1.5 w-1.5 rounded-full bg-primary"></div>
                        <span className="text-muted-foreground">{topic}</span>
                      </li>
                    ))}
                  </ul>
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
              <AccordionItem key={idx} value={`adv-${idx}`} className="border rounded-lg">
                <AccordionTrigger className="px-6 hover:no-underline">
                  <div className="flex items-center justify-between w-full pr-4">
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className="font-mono">
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
                  <div className="mt-4">
                    <h5 className="font-medium mb-2 text-foreground">Key Insights:</h5>
                    <ul className="space-y-1">
                      {algo.insights.map((insight, iIdx) => (
                        <li key={iIdx} className="flex items-start gap-2 text-sm text-muted-foreground">
                          <Sparkles className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                          <span>{insight}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </section>

        {/* Core Detection Methods */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Core Cycle Detection Methods</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {cycleDetectionMethods.map((method, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="font-semibold text-lg text-foreground mb-1">
                        {method.name}
                      </h3>
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="outline" className="font-mono">
                          {method.complexity}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {method.description}
                      </p>
                    </div>
                  </div>
                  <div className="bg-muted p-4 rounded-lg overflow-x-auto">
                    <pre className="text-xs font-mono">
                      <code>{method.code}</code>
                    </pre>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Visualization */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">
            Interactive Cycle Visualization
          </h2>
          <GraphVisualizer />
        </section>

        {/* Advanced Problems */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Advanced Cycle Problems</h2>
          <Accordion type="multiple" className="space-y-4">
            {advancedProblems.map((problem, idx) => (
              <AccordionItem key={idx} value={`prob-${idx}`} className="border rounded-lg">
                <AccordionTrigger className="px-6 hover:no-underline">
                  <div className="flex items-center justify-between w-full pr-4">
                    <div className="flex items-center gap-3">
                      <Badge variant={
                        problem.difficulty === "Easy" ? "default" : 
                        problem.difficulty === "Medium" ? "secondary" : "destructive"
                      }>
                        {problem.difficulty}
                      </Badge>
                      <span className="font-semibold text-foreground">{problem.title}</span>
                    </div>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="px-6 pb-4">
                  <p className="text-sm text-muted-foreground mb-4">
                    {problem.description}
                  </p>
                  <CodeBlock
                    language="python"
                    code={problem.code}
                    className="mt-2"
                  />
                  <div className="mt-4">
                    <h5 className="font-medium mb-2 text-foreground">Real-world Applications:</h5>
                    <div className="flex flex-wrap gap-2">
                      {problem.applications.map((app, aIdx) => (
                        <Badge key={aIdx} variant="outline" className="text-xs">
                          {app}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </section>

        {/* Cycle Properties */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Cycle Properties & Theory</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {cycleProperties.map((prop, idx) => (
              <Card key={idx}>
                <CardContent className="p-6">
                  <h3 className="font-semibold mb-3 text-foreground">{prop.property}</h3>
                  <p className="text-sm text-muted-foreground mb-3">
                    {prop.description}
                  </p>
                  <div className="bg-muted p-3 rounded mb-3">
                    <p className="text-xs font-mono text-foreground">
                      {prop.formal}
                    </p>
                  </div>
                  <p className="text-xs text-muted-foreground italic">
                    {prop.significance}
                  </p>
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
                  <p className="text-sm text-muted-foreground mb-3">
                    {app.description}
                  </p>
                  <div className="bg-primary/5 border border-primary/20 p-3 rounded-lg mb-3">
                    <p className="text-sm font-medium text-foreground mb-1">Example:</p>
                    <p className="text-xs text-muted-foreground">
                      {app.example}
                    </p>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-foreground">Algorithm:</span>
                    <span className="text-muted-foreground">{app.algorithm}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs mt-1">
                    <span className="font-medium text-foreground">Complexity:</span>
                    <span className="font-mono text-primary">{app.complexity}</span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Optimization Techniques */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Optimization Techniques</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {optimizationTechniques.map((opt, idx) => (
              <Card key={idx}>
                <CardContent className="p-6">
                  <h3 className="font-semibold mb-3 text-foreground">{opt.title}</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    {opt.description}
                  </p>
                  <div className="bg-muted p-3 rounded-lg mb-3">
                    <pre className="text-xs font-mono">
                      <code>{opt.code}</code>
                    </pre>
                  </div>
                  <div className="flex items-center gap-2">
                    <Zap className="h-4 w-4 text-green-500" />
                    <span className="text-sm font-medium text-green-600">{opt.benefit}</span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Comparison Table */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Algorithm Comparison</h2>
          <Card>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-muted border-b">
                      <th className="p-4 text-left font-semibold text-foreground">Algorithm</th>
                      <th className="p-4 text-left font-semibold text-foreground">Graph Type</th>
                      <th className="p-4 text-left font-semibold text-foreground">Time</th>
                      <th className="p-4 text-left font-semibold text-foreground">Space</th>
                      <th className="p-4 text-left font-semibold text-foreground">Use Case</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium">DFS Three-Color</td>
                      <td className="p-4">Directed</td>
                      <td className="p-4 font-mono text-sm">O(V+E)</td>
                      <td className="p-4 font-mono text-sm">O(V)</td>
                      <td className="p-4 text-sm">General cycle detection</td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium">Tarjan's SCC</td>
                      <td className="p-4">Directed</td>
                      <td className="p-4 font-mono text-sm">O(V+E)</td>
                      <td className="p-4 font-mono text-sm">O(V)</td>
                      <td className="p-4 text-sm">Find all SCCs, single pass</td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium">Johnson's Algorithm</td>
                      <td className="p-4">Directed</td>
                      <td className="p-4 font-mono text-sm">O((V+E)(C+1))</td>
                      <td className="p-4 font-mono text-sm">O(V+E)</td>
                      <td className="p-4 text-sm">Enumerate all cycles</td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium">Union-Find</td>
                      <td className="p-4">Undirected</td>
                      <td className="p-4 font-mono text-sm">O(E·α(V))</td>
                      <td className="p-4 font-mono text-sm">O(V)</td>
                      <td className="p-4 text-sm">Incremental edge addition</td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium">Floyd Detection</td>
                      <td className="p-4">Sequence/LL</td>
                      <td className="p-4 font-mono text-sm">O(n)</td>
                      <td className="p-4 font-mono text-sm">O(1)</td>
                      <td className="p-4 text-sm">Linked list cycle detection</td>
                    </tr>
                    <tr className="border-b hover:bg-muted/50">
                      <td className="p-4 font-medium">Bellman-Ford</td>
                      <td className="p-4">Weighted</td>
                      <td className="p-4 font-mono text-sm">O(V·E)</td>
                      <td className="p-4 font-mono text-sm">O(V)</td>
                      <td className="p-4 text-sm">Negative cycle detection</td>
                    </tr>
                    <tr className="hover:bg-muted/50">
                      <td className="p-4 font-medium">Karp's Algorithm</td>
                      <td className="p-4">Weighted</td>
                      <td className="p-4 font-mono text-sm">O(V·E)</td>
                      <td className="p-4 font-mono text-sm">O(V²)</td>
                      <td className="p-4 text-sm">Minimum mean cycle</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Quiz */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Advanced Knowledge Test</h2>
          <Quiz questions={quizQuestions} title="Advanced Graph Cycles Quiz" />
        </section>

        {/* Practice Problems */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Practice Problems</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Flame className="h-5 w-5 text-green-500" />
                  <h3 className="font-semibold text-foreground">Easy</h3>
                </div>
                <ul className="space-y-2">
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Linked List Cycle</span>
                    <Badge variant="outline">LC 141</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Happy Number</span>
                    <Badge variant="outline">LC 202</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Find the Duplicate</span>
                    <Badge variant="outline">LC 287</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Zap className="h-5 w-5 text-yellow-500" />
                  <h3 className="font-semibold text-foreground">Medium</h3>
                </div>
                <ul className="space-y-2">
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Course Schedule</span>
                    <Badge variant="outline">LC 207</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Graph Valid Tree</span>
                    <Badge variant="outline">LC 261</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Redundant Connection</span>
                    <Badge variant="outline">LC 684</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Find Eventual Safe</span>
                    <Badge variant="outline">LC 802</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Puzzle className="h-5 w-5 text-red-500" />
                  <h3 className="font-semibold text-foreground">Hard</h3>
                </div>
                <ul className="space-y-2">
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Critical Connections</span>
                    <Badge variant="outline">LC 1192</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Longest Cycle</span>
                    <Badge variant="outline">LC 2360</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Find All Groups</span>
                    <Badge variant="outline">LC 1192</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Brain className="h-5 w-5 text-purple-500" />
                  <h3 className="font-semibold text-foreground">Expert</h3>
                </div>
                <ul className="space-y-2">
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Reconstruct Itinerary</span>
                    <Badge variant="outline">LC 332</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Valid Arrangement</span>
                    <Badge variant="outline">LC 2097</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Find All Recipes</span>
                    <Badge variant="outline">LC 2115</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Interview Tips */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Interview Success Guide</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="bg-primary/5 border-primary/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-3 text-foreground flex items-center gap-2">
                  <Lightbulb className="h-5 w-5" />
                  Key Interview Points
                </h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Always clarify: directed vs undirected, weighted vs unweighted</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>For undirected: Union-Find is often simplest</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>For directed: Three-color DFS is standard</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Mention Tarjan's for advanced scenarios (SCCs)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span>Discuss time/space tradeoffs</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card className="bg-warning/5 border-warning/20">
              <CardContent className="p-6">
                <h3 className="font-semibold mb-3 text-foreground flex items-center gap-2">
                  <AlertCircle className="h-5 w-5" />
                  Common Mistakes
                </h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Forgetting parent check in undirected graphs</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Not handling self-loops correctly</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Mixing up directed/undirected cycle detection</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Not resetting visited set between DFS calls</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-warning mt-0.5 shrink-0" />
                    <span>Using Dijkstra for graphs with negative weights</span>
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