
import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { GraphVisualizer } from "@/components/visualizations/graph-visualizer";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { AlertCircle, CheckCircle2, Clock, Lightbulb, GitBranch, ArrowDown, ArrowUp, CircuitBoard, Target, Network, Layers, Hash, Zap, Flame, Map, TreePine, Droplets, Puzzle, Binary, Copy, Link, Route, TrendingUp, Share2, Lock, Unlock } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function GraphAlgorithmsPage() {
  const algorithmCategories = [
    {
      title: "Traversal Algorithms",
      icon: Network,
      topics: ["BFS", "DFS", "BFS vs DFS", "Bidirectional BFS"]
    },
    {
      title: "Shortest Path",
      icon: Route,
      topics: ["Dijkstra's Algorithm", "Bellman-Ford", "Floyd-Warshall", "A* Search"]
    },
    {
      title: "Grid Problems",
      icon: Map,
      topics: ["Rotten Oranges", "Number of Islands", "Flood Fill", "Pacific Atlantic Water Flow", "Shortest Path in Binary Matrix"]
    },
    {
      title: "Path Finding",
      icon: Target,
      topics: ["Word Ladder", "Snakes and Ladders", "Water Jug"]
    },
    {
      title: "Graph Properties",
      icon: CircuitBoard,
      topics: ["Check for Bipartite", "Clone a Graph", "Transitive Closure", "Strongly Connected Components"]
    },
    {
      title: "Topological Sort",
      icon: TrendingUp,
      topics: ["Kahn's Algorithm", "DFS-based Topo Sort", "Course Schedule", "Alien Dictionary"]
    },
    {
      title: "Union-Find (Disjoint Set)",
      icon: Share2,
      topics: ["Union by Rank", "Path Compression", "Connected Components", "Cycle Detection"]
    },
    {
      title: "Advanced Topics",
      icon: Puzzle,
      topics: ["Minimum Spanning Tree", "Articulation Points", "Bridges", "Network Flow"]
    }
  ];

  const bfsVsDfsTable = [
    {
      aspect: "Data Structure",
      bfs: "Queue (FIFO)",
      dfs: "Stack/Recursion (LIFO)"
    },
    {
      aspect: "Traversal Order",
      bfs: "Level by level",
      dfs: "Depth first, then backtrack"
    },
    {
      aspect: "Shortest Path",
      bfs: "✓ Yes (unweighted graphs)",
      dfs: "✗ No guarantee"
    },
    {
      aspect: "Space Complexity",
      bfs: "O(V) worst case",
      dfs: "O(V) for recursion stack"
    },
    {
      aspect: "When to Use",
      bfs: "Shortest path, level order, web crawling",
      dfs: "Cycle detection, topological sort, path existence"
    }
  ];

  const algorithmImplementations = [
    {
      title: "BFS Implementation",
      language: "python",
      code: `from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

# For grid BFS (like shortest path)
def bfs_grid(grid, start, target):
    rows, cols = len(grid), len(grid[0])
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    queue = deque([(start[0], start[1], 0)])  # (row, col, distance)
    visited = set([(start[0], start[1])])
    
    while queue:
        r, c, dist = queue.popleft()
        
        if (r, c) == target:
            return dist
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and 
                grid[nr][nc] != 1 and  # Assuming 1 is obstacle
                (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))
    
    return -1  # Target not reachable`
    },
    {
      title: "DFS Implementation",
      language: "python",
      code: `# Iterative DFS
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    result = []
    
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result

# Recursive DFS
def dfs_recursive(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    result = [start]
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))
    
    return result

# DFS for cycle detection (undirected graph)
def has_cycle_undirected(graph):
    visited = set()
    
    def dfs(node, parent):
        visited.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        
        return False
    
    for node in graph:
        if node not in visited:
            if dfs(node, -1):
                return True
    
    return False

# DFS for cycle detection (directed graph)
def has_cycle_directed(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}
    
    def dfs(node):
        color[node] = GRAY
        
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:  # Back edge
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        
        color[node] = BLACK
        return False
    
    for node in graph:
        if color[node] == WHITE:
            if dfs(node):
                return True
    
    return False`
    },
    {
      title: "Dijkstra's Algorithm",
      language: "python",
      code: `import heapq

def dijkstra(graph, start):
    """
    Find shortest paths from start to all vertices
    graph: dict of {node: [(neighbor, weight), ...]}
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    # Priority queue: (distance, node)
    pq = [(0, start)]
    visited = set()
    
    while pq:
        curr_dist, node = heapq.heappop(pq)
        
        if node in visited:
            continue
        
        visited.add(node)
        
        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# With path reconstruction
def dijkstra_with_path(graph, start, end):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    parent = {node: None for node in graph}
    
    pq = [(0, start)]
    visited = set()
    
    while pq:
        curr_dist, node = heapq.heappop(pq)
        
        if node in visited:
            continue
        
        if node == end:
            # Reconstruct path
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return distances[end], path[::-1]
        
        visited.add(node)
        
        for neighbor, weight in graph[node]:
            distance = curr_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parent[neighbor] = node
                heapq.heappush(pq, (distance, neighbor))
    
    return float('inf'), []

# Network Delay Time (LeetCode 743)
def networkDelayTime(times, n, k):
    graph = {i: [] for i in range(1, n + 1)}
    for u, v, w in times:
        graph[u].append((v, w))
    
    distances = dijkstra(graph, k)
    max_time = max(distances.values())
    
    return max_time if max_time != float('inf') else -1`
    },
    {
      title: "Topological Sort (Kahn's Algorithm - BFS)",
      language: "python",
      code: `from collections import deque, defaultdict

def topological_sort_kahns(n, edges):
    """
    Kahn's Algorithm using BFS
    Returns topological order or [] if cycle exists
    """
    graph = defaultdict(list)
    in_degree = [0] * n
    
    # Build graph and calculate in-degrees
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    # Start with nodes having 0 in-degree
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # If all nodes processed, no cycle
    return result if len(result) == n else []

# Course Schedule (LeetCode 207)
def canFinish(numCourses, prerequisites):
    result = topological_sort_kahns(numCourses, prerequisites)
    return len(result) == numCourses

# Course Schedule II (LeetCode 210)
def findOrder(numCourses, prerequisites):
    return topological_sort_kahns(numCourses, prerequisites)`
    },
    {
      title: "Topological Sort (DFS-based)",
      language: "python",
      code: `def topological_sort_dfs(n, edges):
    """
    DFS-based topological sort
    Uses three colors: WHITE (unvisited), GRAY (processing), BLACK (done)
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
    
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    result = []
    
    def dfs(node):
        color[node] = GRAY
        
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:  # Cycle detected
                return False
            if color[neighbor] == WHITE:
                if not dfs(neighbor):
                    return False
        
        color[node] = BLACK
        result.append(node)  # Add after all descendants
        return True
    
    for i in range(n):
        if color[i] == WHITE:
            if not dfs(i):
                return []  # Cycle exists
    
    return result[::-1]  # Reverse to get correct order`
    },
    {
      title: "Union-Find (Disjoint Set)",
      language: "python",
      code: `class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x, y):
        """Check if x and y are in same set"""
        return self.find(x) == self.find(y)
    
    def count_components(self):
        """Return number of connected components"""
        return self.components

# Number of Connected Components (LeetCode 323)
def countComponents(n, edges):
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.count_components()

# Redundant Connection (LeetCode 684)
def findRedundantConnection(edges):
    n = len(edges)
    uf = UnionFind(n + 1)
    
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]  # This edge creates cycle
    
    return []

# Graph Valid Tree (LeetCode 261)
def validTree(n, edges):
    if len(edges) != n - 1:
        return False
    
    uf = UnionFind(n)
    for u, v in edges:
        if not uf.union(u, v):
            return False  # Cycle found
    
    return uf.count_components() == 1`
    },
    {
      title: "Number of Islands",
      language: "python",
      code: `def numIslands(grid):
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    def bfs(r, c):
        queue = deque([(r, c)])
        grid[r][c] = '0'  # Mark as visited
        
        while queue:
            cr, cc = queue.popleft()
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr][nc] == '1'):
                    grid[nr][nc] = '0'
                    queue.append((nr, nc))
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] == '0'):
            return
        
        grid[r][c] = '0'
        for dr, dc in directions:
            dfs(r + dr, c + dc)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                bfs(r, c)  # Can use dfs(r, c) instead
    
    return count`
    },
    {
      title: "Rotten Oranges",
      language: "python",
      code: `def orangesRotting(grid):
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh_count = 0
    
    # Initialize queue with all rotten oranges
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh_count += 1
    
    if fresh_count == 0:
        return 0
    
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    minutes = 0
    
    while queue and fresh_count > 0:
        minutes += 1
        for _ in range(len(queue)):
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr][nc] == 1):
                    grid[nr][nc] = 2
                    fresh_count -= 1
                    queue.append((nr, nc))
    
    return -1 if fresh_count > 0 else minutes`
    },
    {
      title: "Word Ladder",
      language: "python",
      code: `from collections import deque

def ladderLength(beginWord, endWord, wordList):
    wordSet = set(wordList)
    if endWord not in wordSet:
        return 0
    
    queue = deque([beginWord])
    visited = set([beginWord])
    changes = 1
    
    while queue:
        for _ in range(len(queue)):
            word = queue.popleft()
            if word == endWord:
                return changes
            
            # Try all possible transformations
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c != word[i]:
                        new_word = word[:i] + c + word[i+1:]
                        if new_word in wordSet and new_word not in visited:
                            visited.add(new_word)
                            queue.append(new_word)
        
        changes += 1
    
    return 0

# Bidirectional BFS (optimized)
def ladderLength_bidirectional(beginWord, endWord, wordList):
    wordSet = set(wordList)
    if endWord not in wordSet:
        return 0
    
    # Start from both ends
    front = {beginWord}
    back = {endWord}
    visited = set()
    changes = 1
    
    while front and back:
        # Always expand the smaller set
        if len(front) > len(back):
            front, back = back, front
        
        next_level = set()
        for word in front:
            if word in back:
                return changes
            
            visited.add(word)
            
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = word[:i] + c + word[i+1:]
                    if new_word in wordSet and new_word not in visited:
                        next_level.add(new_word)
        
        front = next_level
        changes += 1
    
    return 0`
    },
    {
      title: "Check for Bipartite",
      language: "python",
      code: `from collections import deque

def isBipartite(graph):
    n = len(graph)
    colors = [-1] * n  # -1: uncolored, 0: color A, 1: color B
    
    def bfs(start):
        queue = deque([start])
        colors[start] = 0
        
        while queue:
            node = queue.popleft()
            current_color = colors[node]
            
            for neighbor in graph[node]:
                if colors[neighbor] == -1:
                    colors[neighbor] = 1 - current_color
                    queue.append(neighbor)
                elif colors[neighbor] == current_color:
                    return False
        return True
    
    # Check all components
    for i in range(n):
        if colors[i] == -1:
            if not bfs(i):
                return False
    
    return True

# DFS version
def isBipartite_dfs(graph):
    n = len(graph)
    colors = [-1] * n
    
    def dfs(node, color):
        colors[node] = color
        
        for neighbor in graph[node]:
            if colors[neighbor] == -1:
                if not dfs(neighbor, 1 - color):
                    return False
            elif colors[neighbor] == color:
                return False
        
        return True
    
    for i in range(n):
        if colors[i] == -1:
            if not dfs(i, 0):
                return False
    
    return True`
    },
    {
      title: "Clone a Graph",
      language: "python",
      code: `class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

# DFS approach
def cloneGraph_dfs(node):
    if not node:
        return None
    
    clones = {}
    
    def dfs(original):
        if original in clones:
            return clones[original]
        
        clone = Node(original.val)
        clones[original] = clone
        
        for neighbor in original.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)

# BFS approach
def cloneGraph_bfs(node):
    if not node:
        return None
    
    clones = {node: Node(node.val)}
    queue = deque([node])
    
    while queue:
        curr = queue.popleft()
        
        for neighbor in curr.neighbors:
            if neighbor not in clones:
                clones[neighbor] = Node(neighbor.val)
                queue.append(neighbor)
            
            clones[curr].neighbors.append(clones[neighbor])
    
    return clones[node]`
    },
    {
      title: "Shortest Path in Binary Matrix",
      language: "python",
      code: `from collections import deque

def shortestPathBinaryMatrix(grid):
    n = len(grid)
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    
    # 8 directions for diagonal movement
    directions = [
        (1,0), (-1,0), (0,1), (0,-1),
        (1,1), (1,-1), (-1,1), (-1,-1)
    ]
    
    queue = deque([(0, 0, 1)])  # (row, col, distance)
    grid[0][0] = 1  # Mark as visited
    
    while queue:
        r, c, dist = queue.popleft()
        
        if r == n-1 and c == n-1:
            return dist
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < n and 0 <= nc < n and 
                grid[nr][nc] == 0):
                grid[nr][nc] = 1  # Mark as visited
                queue.append((nr, nc, dist + 1))
    
    return -1`
    },
    {
      title: "Surrounded Regions (LeetCode 130)",
      language: "python",
      code: `def solve(board):
    """
    Capture all regions surrounded by 'X'
    Key insight: Mark border-connected 'O's, then flip rest
    """
    if not board:
        return
    
    rows, cols = len(board), len(board[0])
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            board[r][c] != 'O'):
            return
        
        board[r][c] = 'T'  # Temporary mark
        for dr, dc in directions:
            dfs(r + dr, c + dc)
    
    # Mark all border-connected 'O's
    for r in range(rows):
        if board[r][0] == 'O':
            dfs(r, 0)
        if board[r][cols-1] == 'O':
            dfs(r, cols-1)
    
    for c in range(cols):
        if board[0][c] == 'O':
            dfs(0, c)
        if board[rows-1][c] == 'O':
            dfs(rows-1, c)
    
    # Flip captured 'O's to 'X', restore border-connected
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'O':
                board[r][c] = 'X'
            elif board[r][c] == 'T':
                board[r][c] = 'O'`
    },
    {
      title: "Alien Dictionary (LeetCode 269)",
      language: "python",
      code: `def alienOrder(words):
    """
    Derive alien language character order using topological sort
    """
    # Build initial graph with all characters
    graph = {c: set() for word in words for c in word}
    in_degree = {c: 0 for c in graph}
    
    # Build graph from word pairs
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        min_len = min(len(w1), len(w2))
        
        # Check for invalid case: "abc" comes before "ab"
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""
        
        # Find first differing character
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in graph[w1[j]]:
                    graph[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                break
    
    # Kahn's algorithm for topological sort
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []
    
    while queue:
        char = queue.popleft()
        result.append(char)
        
        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if valid (all characters processed)
    return "".join(result) if len(result) == len(graph) else ""`
    },
    {
      title: "Minimum Height Trees (LeetCode 310)",
      language: "python",
      code: `def findMinHeightTrees(n, edges):
    """
    Find root nodes that minimize tree height
    Key insight: Answer is center(s) of graph
    """
    if n <= 2:
        return list(range(n))
    
    # Build adjacency list
    graph = defaultdict(list)
    degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
        degree[u] += 1
        degree[v] += 1
    
    # Start with leaf nodes (degree 1)
    queue = deque([i for i in range(n) if degree[i] == 1])
    remaining = n
    
    # Peel off layers like an onion
    while remaining > 2:
        leaf_count = len(queue)
        remaining -= leaf_count
        
        for _ in range(leaf_count):
            leaf = queue.popleft()
            
            for neighbor in graph[leaf]:
                degree[neighbor] -= 1
                if degree[neighbor] == 1:
                    queue.append(neighbor)
    
    return list(queue)`
    }
  ];

  const advancedTopics = [
    {
      title: "Strongly Connected Components (Kosaraju's)",
      icon: CircuitBoard,
      code: `def strongly_connected_components(graph, n):
    """
    Find all strongly connected components in directed graph
    """
    # Step 1: Do DFS and store finish times
    visited = set()
    stack = []
    
    def dfs1(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs1(neighbor)
        stack.append(node)
    
    for i in range(n):
        if i not in visited:
            dfs1(i)
    
    # Step 2: Create transpose graph
    transpose = defaultdict(list)
    for node in graph:
        for neighbor in graph[node]:
            transpose[neighbor].append(node)
    
    # Step 3: DFS on transpose in reverse finish time order
    visited.clear()
    sccs = []
    
    def dfs2(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in transpose[node]:
            if neighbor not in visited:
                dfs2(neighbor, component)
    
    while stack:
        node = stack.pop()
        if node not in visited:
            component = []
            dfs2(node, component)
            sccs.append(component)
    
    return sccs`,
      explanation: "Used for finding cycles in directed graphs, analyzing connectivity"
    },
    {
      title: "Bellman-Ford Algorithm",
      icon: Route,
      code: `def bellman_ford(edges, n, source):
    """
    Shortest path with negative weights
    Can detect negative cycles
    Time: O(V*E)
    """
    dist = [float('inf')] * n
    dist[source] = 0
    
    # Relax all edges V-1 times
    for _ in range(n - 1):
        for u, v, weight in edges:
            if dist[u] != float('inf') and dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
    
    # Check for negative cycles
    for u, v, weight in edges:
        if dist[u] != float('inf') and dist[u] + weight < dist[v]:
            return None  # Negative cycle exists
    
    return dist

# Cheaper Flights Within K Stops (LeetCode 787)
def findCheapestPrice(n, flights, src, dst, k):
    prices = [float('inf')] * n
    prices[src] = 0
    
    for i in range(k + 1):
        temp = prices.copy()
        
        for u, v, price in flights:
            if prices[u] != float('inf'):
                temp[v] = min(temp[v], prices[u] + price)
        
        prices = temp
    
    return -1 if prices[dst] == float('inf') else prices[dst]`,
      explanation: "Handles negative weights, detects negative cycles"
    },
    {
      title: "Floyd-Warshall Algorithm",
      icon: Network,
      code: `def floyd_warshall(graph, n):
    """
    All-pairs shortest path
    Time: O(V³), Space: O(V²)
    """
    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for u in graph:
        for v, weight in graph[u]:
            dist[u][v] = weight
    
    # Try all intermediate vertices
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], 
                                dist[i][k] + dist[k][j])
    
    return dist

# Find the City (LeetCode 1334)
def findTheCity(n, edges, distanceThreshold):
    dist = [[float('inf')] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for u, v, w in edges:
        dist[u][v] = dist[v][u] = w
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], 
                                dist[i][k] + dist[k][j])
    
    # Find city with smallest number of reachable cities
    min_count = n
    result = 0
    
    for i in range(n):
        count = sum(1 for j in range(n) 
                   if i != j and dist[i][j] <= distanceThreshold)
        if count <= min_count:
            min_count = count
            result = i
    
    return result`,
      explanation: "Best for dense graphs, all-pairs shortest paths"
    },
    {
      title: "Prim's Algorithm (MST)",
      icon: TreePine,
      code: `import heapq

def prim_mst(n, edges):
    """
    Minimum Spanning Tree using Prim's
    Time: O(E log V) with heap
    """
    graph = defaultdict(list)
    for u, v, weight in edges:
        graph[u].append((v, weight))
        graph[v].append((u, weight))
    
    visited = set()
    # Start from node 0
    pq = [(0, 0, -1)]  # (weight, node, parent)
    mst_cost = 0
    mst_edges = []
    
    while pq and len(visited) < n:
        weight, node, parent = heapq.heappop(pq)
        
        if node in visited:
            continue
        
        visited.add(node)
        mst_cost += weight
        
        if parent != -1:
            mst_edges.append((parent, node, weight))
        
        for neighbor, edge_weight in graph[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (edge_weight, neighbor, node))
    
    return mst_cost, mst_edges

# Min Cost to Connect All Points (LeetCode 1584)
def minCostConnectPoints(points):
    n = len(points)
    visited = set([0])
    pq = []
    
    # Add all edges from point 0
    for i in range(1, n):
        dist = abs(points[0][0] - points[i][0]) + abs(points[0][1] - points[i][1])
        heapq.heappush(pq, (dist, i))
    
    cost = 0
    
    while len(visited) < n:
        dist, point = heapq.heappop(pq)
        
        if point in visited:
            continue
        
        visited.add(point)
        cost += dist
        
        for i in range(n):
            if i not in visited:
                new_dist = abs(points[point][0] - points[i][0]) + abs(points[point][1] - points[i][1])
                heapq.heappush(pq, (new_dist, i))
    
    return cost`,
      explanation: "Greedy algorithm for minimum spanning tree"
    }
  ];

  const problemPatterns = [
    {
      title: "Grid BFS/DFS Pattern",
      icon: Map,
      description: "Convert grid to graph, use BFS for shortest path, DFS for connected components",
      problems: ["Number of Islands", "Rotten Oranges", "Flood Fill", "Shortest Path in Binary Matrix"],
      template: `def grid_bfs(grid, start):
    rows, cols = len(grid), len(grid[0])
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    queue = deque([start])
    visited = set([start])
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and 
                (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc))`
    },
    {
      title: "Level Order BFS Pattern",
      icon: Layers,
      description: "Track levels/distance by processing queue in batches",
      problems: ["Word Ladder", "Shortest transformations", "Minimum steps problems"],
      template: `def level_order_bfs(start, target):
    queue = deque([start])
    visited = set([start])
    level = 0
    
    while queue:
        level += 1
        for _ in range(len(queue)):
            current = queue.popleft()
            if current == target:
                return level
            for neighbor in get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)`
    },
    {
      title: "DFS with Memoization",
      icon: Zap,
      description: "Cache results to avoid recomputation (DFS + DP)",
      problems: ["Longest Increasing Path", "Word Search", "Unique Paths"],
      template: `def dfs_with_memo(node, memo):
    if node in memo:
        return memo[node]
    
    result = 0
    for neighbor in get_neighbors(node):
        result = max(result, dfs_with_memo(neighbor, memo) + 1)
    
    memo[node] = result
    return result`
    },
    {
      title: "Multi-source BFS",
      icon: Network,
      description: "Initialize queue with multiple sources, process simultaneously",
      problems: ["Rotten Oranges", "01 Matrix", "Walls and Gates"],
      template: `def multi_source_bfs(sources):
    queue = deque(sources)
    visited = set(sources)
    
    while queue:
        current = queue.popleft()
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)`
    },
    {
      title: "Cycle Detection Pattern",
      icon: CircuitBoard,
      description: "Use colors or visited/recursion stack for cycle detection",
      problems: ["Course Schedule", "Detect Cycle in Graph"],
      template: `# For directed graph
def has_cycle(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}
    
    def dfs(node):
        color[node] = GRAY
        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                return True  # Back edge
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False
    
    return any(color[node] == WHITE and dfs(node) 
               for node in graph)`
    },
    {
      title: "Topological Sort Pattern",
      icon: TrendingUp,
      description: "Process nodes with no dependencies first",
      problems: ["Course Schedule", "Alien Dictionary", "Sequence Reconstruction"],
      template: `def topo_sort(n, edges):
    graph = defaultdict(list)
    in_degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == n else []`
    },
    {
      title: "Union-Find Pattern",
      icon: Share2,
      description: "Group elements and check connectivity efficiently",
      problems: ["Number of Connected Components", "Graph Valid Tree", "Redundant Connection"],
      template: `class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True`
    },
    {
      title: "Shortest Path with Dijkstra",
      icon: Route,
      description: "Find shortest paths in weighted graphs (non-negative weights)",
      problems: ["Network Delay Time", "Path with Maximum Probability", "Cheapest Flights"],
      template: `def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    pq = [(0, start)]
    visited = set()
    
    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        
        for neighbor, weight in graph[node]:
            if d + weight < dist[neighbor]:
                dist[neighbor] = d + weight
                heapq.heappush(pq, (dist[neighbor], neighbor))
    
    return dist`
    }
  ];

  const interviewTips = [
    {
      title: "Algorithm Selection",
      icon: Lightbulb,
      tips: [
        "BFS for shortest path in unweighted graphs",
        "Dijkstra for weighted graphs with non-negative weights",
        "Bellman-Ford when negative weights are present",
        "DFS for cycle detection, topological sort, connectivity",
        "Union-Find for dynamic connectivity and MST",
        "Topological sort for dependency resolution"
      ]
    },
    {
      title: "Optimization Techniques",
      icon: Zap,
      tips: [
        "Use visited set to avoid revisiting nodes (prevents infinite loops)",
        "For BFS level tracking: process queue by levels using for loop",
        "Path compression in Union-Find for O(α(n)) amortized time",
        "Bidirectional BFS for faster shortest path in large graphs",
        "Multi-source BFS when multiple starting points exist",
        "Memoization with DFS to avoid recomputation"
      ]
    },
    {
      title: "Common Mistakes to Avoid",
      icon: AlertCircle,
      tips: [
        "Forgetting to mark nodes as visited (causes infinite loops)",
        "Using DFS for shortest path problems (use BFS instead)",
        "Not handling disconnected components",
        "Incorrect cycle detection in directed vs undirected graphs",
        "Not checking for empty graph or single node edge cases",
        "Modifying input grid without copying (if immutability needed)"
      ]
    },
    {
      title: "Time/Space Complexity",
      icon: Clock,
      tips: [
        "BFS/DFS: O(V + E) time, O(V) space",
        "Dijkstra: O((V + E) log V) with heap",
        "Bellman-Ford: O(V × E) time",
        "Floyd-Warshall: O(V³) time, O(V²) space",
        "Union-Find: O(α(n)) per operation with optimizations",
        "Topological Sort: O(V + E) time"
      ]
    }
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "Which algorithm would you use to find the shortest path in an unweighted graph?",
      options: ["DFS", "BFS", "Dijkstra", "A*"],
      correctAnswer: 1,
      explanation: "BFS explores level by level, guaranteeing shortest path in terms of edges for unweighted graphs.",
    },
    {
      id: 2,
      question: "What is the time complexity of BFS/DFS on a graph with V vertices and E edges?",
      options: ["O(V)", "O(E)", "O(V+E)", "O(V*E)"],
      correctAnswer: 2,
      explanation: "Both BFS and DFS visit each vertex once and explore each edge once, resulting in O(V+E) time.",
    },
    {
      id: 3,
      question: "Which data structure is essential for BFS implementation?",
      options: ["Stack", "Queue", "Heap", "Hash Set"],
      correctAnswer: 1,
      explanation: "BFS uses a Queue (FIFO) to process vertices level by level.",
    },
    {
      id: 4,
      question: "When would you prefer DFS over BFS?",
      options: [
        "Finding shortest path",
        "Memory is limited",
        "Detecting cycles in directed graphs",
        "Web crawling"
      ],
      correctAnswer: 2,
      explanation: "DFS is better for cycle detection, topological sort, and path existence problems.",
    },
    {
      id: 5,
      question: "What is the space complexity of BFS in worst case?",
      options: ["O(1)", "O(V)", "O(E)", "O(V+E)"],
      correctAnswer: 1,
      explanation: "BFS queue can hold all vertices in worst case (complete graph), giving O(V) space complexity.",
    },
    {
      id: 6,
      question: "Which algorithm is best for detecting negative weight cycles?",
      options: ["BFS", "DFS", "Dijkstra", "Bellman-Ford"],
      correctAnswer: 3,
      explanation: "Bellman-Ford can detect negative cycles by checking if any edge can still be relaxed after V-1 iterations.",
    },
    {
      id: 7,
      question: "What is the time complexity of Union-Find with path compression and union by rank?",
      options: ["O(1)", "O(log n)", "O(α(n))", "O(n)"],
      correctAnswer: 2,
      explanation: "With both optimizations, Union-Find operations run in O(α(n)) amortized time, where α is the inverse Ackermann function (practically constant).",
    },
    {
      id: 8,
      question: "Which algorithm would you use for finding all-pairs shortest paths?",
      options: ["BFS from each vertex", "Dijkstra from each vertex", "Floyd-Warshall", "Bellman-Ford from each vertex"],
      correctAnswer: 2,
      explanation: "Floyd-Warshall is specifically designed for all-pairs shortest paths with O(V³) complexity, which is efficient for dense graphs.",
    },
    {
      id: 9,
      question: "What color scheme is used in DFS-based cycle detection for directed graphs?",
      options: ["Red-Black", "White-Gray-Black", "Blue-Green", "Primary-Secondary"],
      correctAnswer: 1,
      explanation: "White (unvisited), Gray (processing), Black (finished). A back edge to a Gray node indicates a cycle.",
    },
    {
      id: 10,
      question: "In Kahn's algorithm for topological sort, which nodes do we start with?",
      options: ["Nodes with maximum out-degree", "Nodes with zero in-degree", "Leaf nodes", "Root nodes"],
      correctAnswer: 1,
      explanation: "Kahn's algorithm starts with nodes that have zero in-degree (no dependencies) and processes them level by level.",
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <header className="mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Graph Algorithms Masterclass
          </h1>
          <p className="text-lg text-muted-foreground mb-4">
            Comprehensive guide to essential graph algorithms with implementations, patterns, and interview strategies.
          </p>
          
          <Alert className="mb-4">
            <Lightbulb className="h-4 w-4" />
            <AlertTitle>Interview-Ready Content</AlertTitle>
            <AlertDescription>
              This guide covers all major graph algorithms asked in FAANG interviews, from basic traversals to advanced topics like MST and strongly connected components.
            </AlertDescription>
          </Alert>
        </header>

        {/* Categories Overview */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground flex items-center gap-2">
            <Network className="h-6 w-6" />
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

        {/* Interview Tips */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Interview Success Tips</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {interviewTips.map((tip, idx) => (
              <Card key={idx}>
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <tip.icon className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold text-foreground">{tip.title}</h3>
                  </div>
                  <ul className="space-y-2">
                    {tip.tips.map((item, tIdx) => (
                      <li key={tIdx} className="flex items-start gap-2 text-sm">
                        <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                        <span className="text-muted-foreground">{item}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* BFS vs DFS Comparison */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">BFS vs DFS: Head-to-Head</h2>
          <Card>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-muted border-b">
                      <th className="p-4 text-left font-semibold text-foreground">Aspect</th>
                      <th className="p-4 text-left font-semibold text-foreground">BFS</th>
                      <th className="p-4 text-left font-semibold text-foreground">DFS</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bfsVsDfsTable.map((row, idx) => (
                      <tr key={idx} className="border-b hover:bg-muted/50">
                        <td className="p-4 font-medium text-foreground">{row.aspect}</td>
                        <td className="p-4">
                          <div className="flex items-center gap-2">
                            {row.aspect === "Shortest Path" ? (
                              <CheckCircle2 className="h-4 w-4 text-green-500" />
                            ) : null}
                            {row.bfs}
                          </div>
                        </td>
                        <td className="p-4">
                          <div className="flex items-center gap-2">
                            {row.aspect === "Shortest Path" ? (
                              <AlertCircle className="h-4 w-4 text-red-500" />
                            ) : null}
                            {row.dfs}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Interactive Visualization */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">
            Interactive Graph Visualization
          </h2>
          <GraphVisualizer />
        </section>

        {/* Core Algorithm Implementations */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Core Algorithm Implementations</h2>
          <Accordion type="multiple" className="space-y-4">
            {algorithmImplementations.map((algo, idx) => (
              <AccordionItem key={idx} value={`algo-${idx}`} className="border rounded-lg">
                <AccordionTrigger className="px-6 hover:no-underline">
                  <div className="flex items-center justify-between w-full pr-4">
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className="font-mono">
                        {algo.language}
                      </Badge>
                      <span className="font-semibold text-foreground">{algo.title}</span>
                    </div>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="px-6 pb-4">
                  <CodeBlock
                    language={algo.language}
                    code={algo.code}
                    className="mt-2"
                  />
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </section>

        {/* Advanced Topics */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Advanced Topics</h2>
          <div className="space-y-6">
            {advancedTopics.map((topic, idx) => (
              <Card key={idx}>
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <topic.icon className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold text-lg text-foreground">{topic.title}</h3>
                  </div>
                  <p className="text-sm text-muted-foreground mb-4">{topic.explanation}</p>
                  <CodeBlock
                    language="python"
                    code={topic.code}
                  />
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Problem Patterns */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Problem-Solving Patterns</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {problemPatterns.map((pattern, idx) => (
              <Card key={idx} className="hover:shadow-lg transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <pattern.icon className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold text-lg text-foreground">{pattern.title}</h3>
                  </div>
                  <p className="text-sm text-muted-foreground mb-4">
                    {pattern.description}
                  </p>
                  <div className="mb-4">
                    <h4 className="font-medium text-sm text-foreground mb-2">Applicable Problems:</h4>
                    <div className="flex flex-wrap gap-2">
                      {pattern.problems.map((problem, pIdx) => (
                        <Badge key={pIdx} variant="secondary" className="text-xs">
                          {problem}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div className="bg-muted p-4 rounded-lg overflow-x-auto">
                    <pre className="text-xs font-mono text-foreground">
                      <code>{pattern.template}</code>
                    </pre>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Time Complexity Reference */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Complexity Analysis</h2>
          <Card>
            <CardContent className="p-6">
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-muted">
                      <th className="border border-border p-3 text-left text-foreground font-semibold">Algorithm</th>
                      <th className="border border-border p-3 text-left text-foreground font-semibold">Time Complexity</th>
                      <th className="border border-border p-3 text-left text-foreground font-semibold">Space Complexity</th>
                      <th className="border border-border p-3 text-left text-foreground font-semibold">Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="border border-border p-3 text-foreground">BFS/DFS</td>
                      <td className="border border-border p-3 font-mono text-primary">O(V + E)</td>
                      <td className="border border-border p-3 font-mono text-primary">O(V)</td>
                      <td className="border border-border p-3 text-muted-foreground">Adjacency list representation</td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 text-foreground">Dijkstra (Binary Heap)</td>
                      <td className="border border-border p-3 font-mono text-primary">O((V+E) log V)</td>
                      <td className="border border-border p-3 font-mono text-primary">O(V)</td>
                      <td className="border border-border p-3 text-muted-foreground">Non-negative weights only</td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 text-foreground">Bellman-Ford</td>
                      <td className="border border-border p-3 font-mono text-warning">O(V × E)</td>
                      <td className="border border-border p-3 font-mono text-primary">O(V)</td>
                      <td className="border border-border p-3 text-muted-foreground">Handles negative weights</td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 text-foreground">Floyd-Warshall</td>
                      <td className="border border-border p-3 font-mono text-destructive">O(V³)</td>
                      <td className="border border-border p-3 font-mono text-warning">O(V²)</td>
                      <td className="border border-border p-3 text-muted-foreground">All-pairs shortest paths</td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 text-foreground">Topological Sort</td>
                      <td className="border border-border p-3 font-mono text-primary">O(V + E)</td>
                      <td className="border border-border p-3 font-mono text-primary">O(V)</td>
                      <td className="border border-border p-3 text-muted-foreground">Kahn's or DFS-based</td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 text-foreground">Union-Find (optimized)</td>
                      <td className="border border-border p-3 font-mono text-primary">O(α(n))</td>
                      <td className="border border-border p-3 font-mono text-primary">O(n)</td>
                      <td className="border border-border p-3 text-muted-foreground">Amortized per operation</td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 text-foreground">Prim's MST</td>
                      <td className="border border-border p-3 font-mono text-primary">O(E log V)</td>
                      <td className="border border-border p-3 font-mono text-primary">O(V)</td>
                      <td className="border border-border p-3 text-muted-foreground">With binary heap</td>
                    </tr>
                    <tr>
                      <td className="border border-border p-3 text-foreground">Kruskal's MST</td>
                      <td className="border border-border p-3 font-mono text-primary">O(E log E)</td>
                      <td className="border border-border p-3 font-mono text-primary">O(V)</td>
                      <td className="border border-border p-3 text-muted-foreground">Sort edges + Union-Find</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Common Mistakes */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Common Pitfalls & Solutions</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="border-destructive/20">
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <AlertCircle className="h-5 w-5 text-destructive" />
                  <h3 className="font-semibold text-foreground">Forgetting Visited Set</h3>
                </div>
                <p className="text-sm text-muted-foreground mb-3">
                  Not maintaining visited set causes infinite loops in graphs with cycles.
                </p>
                <div className="text-xs font-mono bg-destructive/5 p-3 rounded">
                  ❌ No visited tracking<br/>
                  ✅ visited.add(node) when enqueuing/processing
                </div>
              </CardContent>
            </Card>
            
            <Card className="border-warning/20">
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <AlertCircle className="h-5 w-5 text-warning" />
                  <h3 className="font-semibold text-foreground">Using DFS for Shortest Path</h3>
                </div>
                <p className="text-sm text-muted-foreground mb-3">
                  DFS doesn't guarantee shortest path in unweighted graphs.
                </p>
                <div className="text-xs font-mono bg-warning/5 p-3 rounded">
                  ❌ DFS for shortest path<br/>
                  ✅ BFS for unweighted, Dijkstra for weighted
                </div>
              </CardContent>
            </Card>
            
            <Card className="border-warning/20">
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <AlertCircle className="h-5 w-5 text-warning" />
                  <h3 className="font-semibold text-foreground">Wrong Cycle Detection</h3>
                </div>
                <p className="text-sm text-muted-foreground mb-3">
                  Directed and undirected graphs need different cycle detection approaches.
                </p>
                <div className="text-xs font-mono bg-warning/5 p-3 rounded">
                  Undirected: Check parent ≠ neighbor<br/>
                  Directed: Use WHITE-GRAY-BLACK colors
                </div>
              </CardContent>
            </Card>
            
            <Card className="border-destructive/20">
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <AlertCircle className="h-5 w-5 text-destructive" />
                  <h3 className="font-semibold text-foreground">Dijkstra with Negative Weights</h3>
                </div>
                <p className="text-sm text-muted-foreground mb-3">
                  Dijkstra fails with negative edge weights.
                </p>
                <div className="text-xs font-mono bg-destructive/5 p-3 rounded">
                  ❌ Dijkstra with negative weights<br/>
                  ✅ Use Bellman-Ford instead
                </div>
              </CardContent>
            </Card>
            
            <Card className="border-primary/20">
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <CheckCircle2 className="h-5 w-5 text-primary" />
                  <h3 className="font-semibold text-foreground">Multi-source BFS Optimization</h3>
                </div>
                <p className="text-sm text-muted-foreground mb-3">
                  Initialize queue with all sources for parallel processing.
                </p>
                <div className="text-xs font-mono bg-primary/5 p-3 rounded">
                  queue = deque(all_sources)<br/>
                  visited = set(all_sources)
                </div>
              </CardContent>
            </Card>
            
            <Card className="border-primary/20">
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <CheckCircle2 className="h-5 w-5 text-primary" />
                  <h3 className="font-semibold text-foreground">Level Tracking in BFS</h3>
                </div>
                <p className="text-sm text-muted-foreground mb-3">
                  Process nodes level by level for distance calculation.
                </p>
                <div className="text-xs font-mono bg-primary/5 p-3 rounded">
                  while queue:<br/>
                  &nbsp;&nbsp;for _ in range(len(queue)):<br/>
                  &nbsp;&nbsp;&nbsp;&nbsp;# Process current level
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Quiz Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Test Your Knowledge</h2>
          <Quiz questions={quizQuestions} title="Graph Algorithms Quiz" />
        </section>

        {/* Practice Problems Grid */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Practice Problems by Category</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Flame className="h-5 w-5 text-green-500" />
                  <h3 className="font-semibold text-foreground">Easy</h3>
                </div>
                <ul className="space-y-3">
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Flood Fill</span>
                    <Badge variant="outline">LC 733</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Number of Islands</span>
                    <Badge variant="outline">LC 200</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Clone Graph</span>
                    <Badge variant="outline">LC 133</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Find Center of Star</span>
                    <Badge variant="outline">LC 1791</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Zap className="h-5 w-5 text-yellow-500" />
                  <h3 className="font-semibold text-foreground">Medium</h3>
                </div>
                <ul className="space-y-3">
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Rotten Oranges</span>
                    <Badge variant="outline">LC 994</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Course Schedule</span>
                    <Badge variant="outline">LC 207</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Network Delay Time</span>
                    <Badge variant="outline">LC 743</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Graph Valid Tree</span>
                    <Badge variant="outline">LC 261</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Surrounded Regions</span>
                    <Badge variant="outline">LC 130</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Min Height Trees</span>
                    <Badge variant="outline">LC 310</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Puzzle className="h-5 w-5 text-red-500" />
                  <h3 className="font-semibold text-foreground">Hard</h3>
                </div>
                <ul className="space-y-3">
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Word Ladder</span>
                    <Badge variant="outline">LC 127</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Alien Dictionary</span>
                    <Badge variant="outline">LC 269</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Cheapest Flights K Stops</span>
                    <Badge variant="outline">LC 787</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Find City</span>
                    <Badge variant="outline">LC 1334</Badge>
                  </li>
                  <li className="flex justify-between items-center">
                    <span className="text-sm">Reconstruct Itinerary</span>
                    <Badge variant="outline">LC 332</Badge>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Quick Reference */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold mb-6 text-foreground">Quick Reference</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">When to use BFS</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    Shortest path (unweighted)
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    Level order traversal
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    Finding closest nodes
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    Multi-source problems
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">When to use DFS</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    Cycle detection
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    Topological sort
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    Path existence
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    Connected components
                  </li>
                </ul>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-6">
                <h3 className="font-semibold mb-4 text-foreground">Algorithm Selection</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <Route className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
                    <span><strong>Dijkstra:</strong> Weighted, non-negative</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Share2 className="h-4 w-4 text-purple-500 mt-0.5 shrink-0" />
                    <span><strong>Union-Find:</strong> Connectivity queries</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <TrendingUp className="h-4 w-4 text-green-500 mt-0.5 shrink-0" />
                    <span><strong>Topo Sort:</strong> Dependency resolution</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <TreePine className="h-4 w-4 text-orange-500 mt-0.5 shrink-0" />
                    <span><strong>MST:</strong> Minimum cost connection</span>
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