"use client"

import { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line as DreiLine, Html } from '@react-three/drei';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, RotateCcw, Plus, Trash2, Download, Upload, Info, ChevronRight } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import * as THREE from 'three';

// Graph data structures
interface Vertex {
  id: number;
  x: number;
  y: number;
  z: number;
  label: string;
  color: string;
  size: number;
}

interface Edge {
  from: number;
  to: number;
  weight: number;
  directed: boolean;
  color: string;
  width: number;
}

interface PathInfo {
  distance: number;
  path: number[];
}

type Algorithm = 'bfs' | 'dfs' | 'dijkstra' | 'bellman-ford' | 'prim' | 'kruskal' | 'none';
type GraphType = 'undirected' | 'directed' | 'weighted';
type Layout = 'circular' | 'force' | 'grid' | 'random';

// Animated vertex component
function AnimatedVertex({ vertex, isCurrent, isVisited, distance }: { 
  vertex: Vertex; 
  isCurrent: boolean; 
  isVisited: boolean;
  distance?: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current && isCurrent) {
      meshRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 4) * 0.15);
    } else if (meshRef.current) {
      meshRef.current.scale.setScalar(1);
    }
  });
  
  let color = vertex.color;
  if (isCurrent) color = '#f59e0b';
  else if (isVisited) color = '#10b981';
  
  return (
    <group position={[vertex.x, vertex.y, vertex.z]}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[vertex.size * 0.4, 32, 32]} />
        <meshStandardMaterial 
          color={color}
          emissive={color}
          emissiveIntensity={isCurrent ? 0.6 : isVisited ? 0.3 : 0}
          metalness={0.3}
          roughness={0.4}
        />
      </mesh>
      
      <Text
        position={[0, 0, vertex.size * 0.4 + 0.4]}
        fontSize={0.5}
        color="white"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.05}
        outlineColor="#000000"
      >
        {vertex.label}
      </Text>
      
      {distance !== undefined && distance !== Infinity && (
        <Html position={[0, -vertex.size * 0.6, 0]} center>
          <div className="bg-black/80 text-white px-2 py-1 rounded text-xs font-mono whitespace-nowrap">
            d={distance}
          </div>
        </Html>
      )}
    </group>
  );
}

// Animated edge component
function AnimatedEdge({ edge, fromVertex, toVertex, isVisited, isInPath }: { 
  edge: Edge; 
  fromVertex: Vertex; 
  toVertex: Vertex;
  isVisited: boolean;
  isInPath: boolean;
}) {
  const start = new THREE.Vector3(fromVertex.x, fromVertex.y, fromVertex.z);
  const end = new THREE.Vector3(toVertex.x, toVertex.y, toVertex.z);
  
  let color = edge.color;
  if (isInPath) color = '#ef4444'; // Red for final path
  else if (isVisited) color = '#f59e0b'; // Orange for visited
  
  const direction = new THREE.Vector3().subVectors(end, start).normalize();
  const arrowPos = new THREE.Vector3().copy(end).sub(direction.clone().multiplyScalar(0.5));
  const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
  
  return (
    <group>
      <DreiLine
        points={[start, end]}
        color={color}
        lineWidth={isInPath ? 4 : isVisited ? 3 : 2}
        opacity={isInPath ? 1 : isVisited ? 0.8 : 0.6}
        transparent
      />
      
      {edge.directed && (
        <mesh position={arrowPos.toArray()} rotation={[
          0,
          Math.atan2(direction.x, direction.z),
          Math.acos(direction.y)
        ]}>
          <coneGeometry args={[0.15, 0.4, 8]} />
          <meshStandardMaterial color={color} />
        </mesh>
      )}
      
      {edge.weight > 1 && (
        <Html position={midpoint.toArray()} center>
          <div className="bg-blue-600 text-white px-2 py-0.5 rounded-full text-xs font-bold">
            {edge.weight}
          </div>
        </Html>
      )}
    </group>
  );
}

// Main component wrapped for client-side only rendering
function GraphVisualizerCore() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="text-center space-y-2">
            <h1 className="text-4xl font-bold text-white">Graph Algorithm Visualizer</h1>
            <p className="text-slate-400">Loading interactive visualization...</p>
          </div>
          <Card className="bg-slate-800/50 border-slate-700">
            <CardContent className="p-20 text-center">
              <div className="animate-pulse space-y-4">
                <div className="h-8 bg-slate-700 rounded w-1/4 mx-auto"></div>
                <div className="h-64 bg-slate-700 rounded"></div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return <GraphVisualizerMain />;
}

function GraphVisualizerMain() {
  const [vertices, setVertices] = useState<Vertex[]>([
    { id: 0, x: -3, y: 2, z: 0, label: 'Home', color: '#3b82f6', size: 1 },
    { id: 1, x: 0, y: 3, z: 0, label: 'School', color: '#3b82f6', size: 1 },
    { id: 2, x: 3, y: 2, z: 0, label: 'Park', color: '#3b82f6', size: 1 },
    { id: 3, x: 3, y: -1, z: 0, label: 'Store', color: '#3b82f6', size: 1 },
    { id: 4, x: 0, y: -2, z: 0, label: 'Library', color: '#3b82f6', size: 1 },
    { id: 5, x: -3, y: -1, z: 0, label: 'Gym', color: '#3b82f6', size: 1 },
  ]);

  const [edges, setEdges] = useState<Edge[]>([
    { from: 0, to: 1, weight: 5, directed: false, color: '#64748b', width: 2 }, // Home to School: 5 min
    { from: 0, to: 5, weight: 3, directed: false, color: '#64748b', width: 2 }, // Home to Gym: 3 min
    { from: 1, to: 2, weight: 4, directed: false, color: '#64748b', width: 2 }, // School to Park: 4 min
    { from: 2, to: 3, weight: 2, directed: false, color: '#64748b', width: 2 }, // Park to Store: 2 min
    { from: 3, to: 4, weight: 3, directed: false, color: '#64748b', width: 2 }, // Store to Library: 3 min
    { from: 4, to: 5, weight: 4, directed: false, color: '#64748b', width: 2 }, // Library to Gym: 4 min
    { from: 1, to: 4, weight: 6, directed: false, color: '#64748b', width: 2 }, // School to Library: 6 min
    { from: 2, to: 5, weight: 7, directed: false, color: '#64748b', width: 2 }, // Park to Gym: 7 min (alternate path)
  ]);

  const [algorithm, setAlgorithm] = useState<Algorithm>('none');
  const [graphType, setGraphType] = useState<GraphType>('weighted');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [speed, setSpeed] = useState(1000);
  const [startVertex, setStartVertex] = useState(0); // Home
  const [endVertex, setEndVertex] = useState(3); // Store
  
  const [traversalOrder, setTraversalOrder] = useState<number[]>([]);
  const [visitedVertices, setVisitedVertices] = useState<Set<number>>(() => new Set());
  const [visitedEdges, setVisitedEdges] = useState<Set<string>>(() => new Set());
  const [distances, setDistances] = useState<Map<number, number>>(() => new Map());
  const [finalPath, setFinalPath] = useState<number[]>([]);
  const [mstEdges, setMstEdges] = useState<Set<string>>(() => new Set());
  const [algorithmSteps, setAlgorithmSteps] = useState<string[]>([]);
  
  const animationRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [layout, setLayout] = useState<Layout>('circular');

  // Build adjacency structures
  const buildGraph = () => {
    const adjList = new Map<number, Array<{to: number, weight: number}>>();
    vertices.forEach(v => adjList.set(v.id, []));
    
    edges.forEach(edge => {
      adjList.get(edge.from)?.push({ to: edge.to, weight: edge.weight });
      if (!edge.directed) {
        adjList.get(edge.to)?.push({ to: edge.from, weight: edge.weight });
      }
    });
    
    return adjList;
  };

  // BFS Algorithm
  const bfs = (start: number) => {
    const order: number[] = [];
    const visited = new Set<number>();
    const queue: number[] = [start];
    const steps: string[] = [];
    const edgeVisits: string[] = [];
    const parent = new Map<number, number>();
    
    visited.add(start);
    steps.push(`Start BFS from ${vertices[start]?.label}`);
    
    while (queue.length > 0) {
      const current = queue.shift()!;
      order.push(current);
      steps.push(`Visit ${vertices[current]?.label}`);
      
      const adjList = buildGraph();
      const neighbors = adjList.get(current) || [];
      
      for (const {to: neighbor} of neighbors) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          queue.push(neighbor);
          parent.set(neighbor, current);
          const edgeKey = getEdgeKey(current, neighbor);
          edgeVisits.push(edgeKey);
          steps.push(`Discover ${vertices[neighbor]?.label} from ${vertices[current]?.label}`);
        }
      }
    }
    
    steps.push(`BFS complete! Visited ${order.length} locations`);
    
    return { order, steps, edgeVisits };
  };

  // DFS Algorithm
  const dfs = (start: number) => {
    const order: number[] = [];
    const visited = new Set<number>();
    const steps: string[] = [];
    const edgeVisits: string[] = [];
    const parent = new Map<number, number>();
    
    const dfsRecursive = (node: number, parentNode: number | null = null) => {
      visited.add(node);
      order.push(node);
      
      if (parentNode !== null) {
        const edgeKey = getEdgeKey(parentNode, node);
        edgeVisits.push(edgeKey);
        steps.push(`Visit ${vertices[node]?.label} from ${vertices[parentNode]?.label}`);
      } else {
        steps.push(`Start DFS from ${vertices[node]?.label}`);
      }
      
      const adjList = buildGraph();
      const neighbors = adjList.get(node) || [];
      
      for (const {to: neighbor} of neighbors) {
        if (!visited.has(neighbor)) {
          parent.set(neighbor, node);
          dfsRecursive(neighbor, node);
        }
      }
      
      if (parentNode !== null) {
        steps.push(`Backtrack from ${vertices[node]?.label} to ${vertices[parentNode]?.label}`);
      }
    };
    
    dfsRecursive(start);
    steps.push(`DFS complete! Visited ${order.length} locations`);
    
    return { order, steps, edgeVisits };
  };

  // Dijkstra's Algorithm
  const dijkstra = (start: number, end: number) => {
    const dist = new Map<number, number>();
    const prev = new Map<number, number | null>();
    const visited = new Set<number>();
    const order: number[] = [];
    const steps: string[] = [];
    const edgeVisits: string[] = [];
    
    vertices.forEach(v => {
      dist.set(v.id, Infinity);
      prev.set(v.id, null);
    });
    dist.set(start, 0);
    
    steps.push(`🎯 Finding shortest path from ${vertices[start]?.label} to ${vertices[end]?.label}`);
    steps.push(`Set distance to ${vertices[start]?.label} = 0 minutes`);
    
    const pq: Array<{node: number, dist: number}> = [{node: start, dist: 0}];
    
    while (pq.length > 0) {
      pq.sort((a, b) => a.dist - b.dist);
      const {node: current} = pq.shift()!;
      
      if (visited.has(current)) continue;
      
      visited.add(current);
      order.push(current);
      
      const currentDist = dist.get(current)!;
      if (currentDist === Infinity) {
        steps.push(`⚠️ Cannot reach ${vertices[current]?.label} from ${vertices[start]?.label}`);
        continue;
      }
      
      steps.push(`📍 Visit ${vertices[current]?.label} (current shortest: ${currentDist} min)`);
      
      if (current === end) {
        steps.push(`🎉 Reached destination ${vertices[end]?.label}!`);
        break;
      }
      
      const adjList = buildGraph();
      const neighbors = adjList.get(current) || [];
      
      for (const {to: neighbor, weight} of neighbors) {
        if (!visited.has(neighbor)) {
          const newDist = currentDist + weight;
          const oldDist = dist.get(neighbor)!;
          
          if (newDist < oldDist) {
            dist.set(neighbor, newDist);
            prev.set(neighbor, current);
            pq.push({node: neighbor, dist: newDist});
            edgeVisits.push(getEdgeKey(current, neighbor));
            
            if (oldDist === Infinity) {
              steps.push(`  ➜ Found route to ${vertices[neighbor]?.label}: ${newDist} min via ${vertices[current]?.label}`);
            } else {
              steps.push(`  ➜ Better route to ${vertices[neighbor]?.label}: ${newDist} min (was ${oldDist} min)`);
            }
          }
        }
      }
    }
    
    // Reconstruct path
    const path: number[] = [];
    let curr: number | null = end;
    
    while (curr !== null && prev.get(curr) !== undefined) {
      path.unshift(curr);
      curr = prev.get(curr)!;
    }
    
    const finalDist = dist.get(end)!;
    
    if (path.length > 0 && path[0] === start) {
      steps.push(`✅ Shortest path: ${path.map(id => vertices[id]?.label).join(' → ')} (${finalDist} min)`);
    } else {
      steps.push(`❌ No path exists from ${vertices[start]?.label} to ${vertices[end]?.label}`);
    }
    
    return { order, steps, edgeVisits, distances: dist, path: path[0] === start ? path : [] };
  };

  // Prim's MST Algorithm
  const prim = (start: number) => {
    const visited = new Set<number>();
    const mst: string[] = [];
    const order: number[] = [];
    const steps: string[] = [];
    const edgeVisits: string[] = [];
    let totalWeight = 0;
    
    const pq: Array<{from: number, to: number, weight: number}> = [];
    visited.add(start);
    order.push(start);
    
    steps.push(`🌳 Building Minimum Spanning Tree from ${vertices[start]?.label}`);
    steps.push(`Add ${vertices[start]?.label} to tree`);
    
    const adjList = buildGraph();
    const neighbors = adjList.get(start) || [];
    neighbors.forEach(({to, weight}) => {
      pq.push({from: start, to, weight});
    });
    
    while (pq.length > 0 && visited.size < vertices.length) {
      pq.sort((a, b) => a.weight - b.weight);
      const edge = pq.shift()!;
      
      if (visited.has(edge.to)) continue;
      
      visited.add(edge.to);
      order.push(edge.to);
      const edgeKey = getEdgeKey(edge.from, edge.to);
      mst.push(edgeKey);
      edgeVisits.push(edgeKey);
      totalWeight += edge.weight;
      
      steps.push(`➕ Add edge: ${vertices[edge.from]?.label} ↔ ${vertices[edge.to]?.label} (${edge.weight} min) | Total: ${totalWeight} min`);
      
      const newNeighbors = adjList.get(edge.to) || [];
      newNeighbors.forEach(({to, weight}) => {
        if (!visited.has(to)) {
          pq.push({from: edge.to, to, weight});
        }
      });
    }
    
    if (visited.size === vertices.length) {
      steps.push(`✅ MST complete! Connected all ${vertices.length} locations with total distance: ${totalWeight} min`);
    } else {
      steps.push(`⚠️ Graph is disconnected. Connected ${visited.size}/${vertices.length} locations`);
    }
    
    return { order, steps, edgeVisits, mst };
  };

  // Kruskal's MST Algorithm
  const kruskal = () => {
    const parent = new Map<number, number>();
    const rank = new Map<number, number>();
    
    vertices.forEach(v => {
      parent.set(v.id, v.id);
      rank.set(v.id, 0);
    });
    
    const find = (x: number): number => {
      if (parent.get(x) !== x) {
        parent.set(x, find(parent.get(x)!));
      }
      return parent.get(x)!;
    };
    
    const union = (x: number, y: number): boolean => {
      const px = find(x);
      const py = find(y);
      
      if (px === py) return false;
      
      if (rank.get(px)! < rank.get(py)!) {
        parent.set(px, py);
      } else if (rank.get(px)! > rank.get(py)!) {
        parent.set(py, px);
      } else {
        parent.set(py, px);
        rank.set(px, rank.get(px)! + 1);
      }
      
      return true;
    };
    
    const sortedEdges = edges
      .map((edge, idx) => ({...edge, idx}))
      .sort((a, b) => a.weight - b.weight);
    
    const mst: string[] = [];
    const order: number[] = [];
    const steps: string[] = [];
    const edgeVisits: string[] = [];
    let totalWeight = 0;
    let edgesAdded = 0;
    
    steps.push(`🔗 Kruskal's MST - Sort ${edges.length} edges by weight`);
    steps.push(`Need ${vertices.length - 1} edges to connect ${vertices.length} locations`);
    
    for (const edge of sortedEdges) {
      const fromLabel = vertices[edge.from]?.label;
      const toLabel = vertices[edge.to]?.label;
      
      if (union(edge.from, edge.to)) {
        const edgeKey = getEdgeKey(edge.from, edge.to);
        mst.push(edgeKey);
        edgeVisits.push(edgeKey);
        totalWeight += edge.weight;
        edgesAdded++;
        
        if (!order.includes(edge.from)) order.push(edge.from);
        if (!order.includes(edge.to)) order.push(edge.to);
        
        steps.push(`✅ Edge ${edgesAdded}: ${fromLabel} ↔ ${toLabel} (${edge.weight} min) | Total: ${totalWeight} min`);
        
        if (edgesAdded === vertices.length - 1) {
          steps.push(`🎉 MST complete! All locations connected with minimum total distance`);
          break;
        }
      } else {
        steps.push(`❌ Skip: ${fromLabel} ↔ ${toLabel} (${edge.weight} min) - would create a cycle`);
      }
    }
    
    if (edgesAdded < vertices.length - 1) {
      steps.push(`⚠️ Graph is disconnected. Only ${edgesAdded} edges added, need ${vertices.length - 1}`);
    } else {
      steps.push(`Total MST weight: ${totalWeight} minutes`);
    }
    
    return { order, steps, edgeVisits, mst };
  };

  const getEdgeKey = (from: number, to: number) => {
    return `${Math.min(from, to)}-${Math.max(from, to)}`;
  };

  const runAlgorithm = () => {
    let result: any;
    
    // Reset all states before running
    setVisitedVertices(() => new Set());
    setVisitedEdges(() => new Set());
    setDistances(() => new Map());
    setFinalPath([]);
    setMstEdges(() => new Set());
    setCurrentStep(0);
    
    switch (algorithm) {
      case 'bfs':
        result = bfs(startVertex);
        setTraversalOrder(result.order);
        setAlgorithmSteps(result.steps);
        // For BFS/DFS, edges will be animated during playback
        break;
      case 'dfs':
        result = dfs(startVertex);
        setTraversalOrder(result.order);
        setAlgorithmSteps(result.steps);
        // For BFS/DFS, edges will be animated during playback
        break;
      case 'dijkstra':
        result = dijkstra(startVertex, endVertex);
        setTraversalOrder(result.order);
        setAlgorithmSteps(result.steps);
        setDistances(result.distances);
        setFinalPath(result.path);
        // Show all explored edges immediately for Dijkstra
        setVisitedEdges(new Set(result.edgeVisits));
        break;
      case 'prim':
        result = prim(startVertex);
        setTraversalOrder(result.order);
        setAlgorithmSteps(result.steps);
        setMstEdges(new Set(result.mst));
        // Show MST edges immediately
        setVisitedEdges(new Set(result.edgeVisits));
        break;
      case 'kruskal':
        result = kruskal();
        setTraversalOrder(result.order);
        setAlgorithmSteps(result.steps);
        setMstEdges(new Set(result.mst));
        // Show MST edges immediately
        setVisitedEdges(new Set(result.edgeVisits));
        break;
      default:
        return;
    }
  };

  useEffect(() => {
    if (animationRef.current) {
      clearInterval(animationRef.current);
      animationRef.current = null;
    }
    
    if (isPlaying && traversalOrder.length > 0) {
      animationRef.current = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= traversalOrder.length) {
            if (animationRef.current) {
              clearInterval(animationRef.current);
              animationRef.current = null;
            }
            setIsPlaying(false);
            return prev;
          }
          
          const currentVertex = traversalOrder[prev];
          setVisitedVertices(prevVisited => new Set([...prevVisited, currentVertex]));
          
          // For algorithms that track edges, mark the edge from previous to current
          if (prev > 0 && (algorithm === 'bfs' || algorithm === 'dfs')) {
            const prevVertex = traversalOrder[prev - 1];
            const edgeKey = getEdgeKey(prevVertex, currentVertex);
            
            // Check if this edge actually exists in the graph
            const edgeExists = edges.some(e => 
              (e.from === prevVertex && e.to === currentVertex) ||
              (e.from === currentVertex && e.to === prevVertex && !e.directed)
            );
            
            if (edgeExists) {
              setVisitedEdges(prevEdges => new Set([...prevEdges, edgeKey]));
            }
          }
          
          return prev + 1;
        });
      }, speed);
    }
    
    return () => {
      if (animationRef.current) {
        clearInterval(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [isPlaying, speed, traversalOrder, algorithm, edges]);

  const handlePlayPause = () => {
    if (traversalOrder.length === 0) {
      runAlgorithm();
    }
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    if (animationRef.current) {
      clearInterval(animationRef.current);
      animationRef.current = null;
    }
    setIsPlaying(false);
    setCurrentStep(0);
    setTraversalOrder([]);
    setVisitedVertices(() => new Set());
    setVisitedEdges(() => new Set());
    setDistances(() => new Map());
    setFinalPath([]);
    setMstEdges(() => new Set());
    setAlgorithmSteps([]);
  };

  const addVertex = () => {
    const id = vertices.length;
    const label = String.fromCharCode(65 + id);
    
    let x = 0, y = 0, z = 0;
    
    if (layout === 'circular') {
      const angle = (id * 2 * Math.PI) / (vertices.length + 1);
      const radius = 3;
      x = Math.cos(angle) * radius;
      y = Math.sin(angle) * radius;
    } else if (layout === 'random') {
      x = (Math.random() - 0.5) * 8;
      y = (Math.random() - 0.5) * 8;
      z = (Math.random() - 0.5) * 2;
    }
    
    setVertices([...vertices, { id, x, y, z, label, color: '#3b82f6', size: 1 }]);
  };

  const removeVertex = (id: number) => {
    setVertices(vertices.filter(v => v.id !== id));
    setEdges(edges.filter(e => e.from !== id && e.to !== id));
    handleReset();
  };

  const addEdge = (from: number, to: number, weight: number = 1) => {
    if (from === to) return;
    const exists = edges.some(e => 
      (e.from === from && e.to === to) || 
      (!e.directed && e.from === to && e.to === from)
    );
    if (exists) return;
    
    setEdges([...edges, {
      from,
      to,
      weight,
      directed: graphType === 'directed',
      color: '#64748b',
      width: 2
    }]);
  };

  const removeEdge = (from: number, to: number) => {
    setEdges(edges.filter(e => !(e.from === from && e.to === to)));
  };

  const applyLayout = (newLayout: Layout) => {
    setLayout(newLayout);
    const newVertices = vertices.map((v, idx) => {
      let x = v.x, y = v.y, z = 0;
      
      switch (newLayout) {
        case 'circular':
          const angle = (idx * 2 * Math.PI) / vertices.length;
          const radius = 3;
          x = Math.cos(angle) * radius;
          y = Math.sin(angle) * radius;
          break;
        case 'grid':
          const cols = Math.ceil(Math.sqrt(vertices.length));
          x = (idx % cols) * 2 - cols;
          y = Math.floor(idx / cols) * 2 - Math.floor(vertices.length / cols);
          break;
        case 'random':
          x = (Math.random() - 0.5) * 8;
          y = (Math.random() - 0.5) * 8;
          z = (Math.random() - 0.5) * 2;
          break;
      }
      
      return { ...v, x, y, z };
    });
    setVertices(newVertices);
  };

  const exportGraph = () => {
    const data = { vertices, edges, graphType };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'graph.json';
    a.click();
  };

  const generateRandomGraph = (numVertices: number) => {
    const newVertices: Vertex[] = [];
    for (let i = 0; i < numVertices; i++) {
      const angle = (i * 2 * Math.PI) / numVertices;
      const radius = 3;
      newVertices.push({
        id: i,
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        z: 0,
        label: String.fromCharCode(65 + i),
        color: '#3b82f6',
        size: 1
      });
    }
    
    const newEdges: Edge[] = [];
    for (let i = 0; i < numVertices; i++) {
      const numConnections = Math.floor(Math.random() * 3) + 1;
      for (let j = 0; j < numConnections; j++) {
        const to = Math.floor(Math.random() * numVertices);
        if (to !== i && !newEdges.some(e => e.from === i && e.to === to)) {
          newEdges.push({
            from: i,
            to,
            weight: Math.floor(Math.random() * 10) + 1,
            directed: graphType === 'directed',
            color: '#64748b',
            width: 2
          });
        }
      }
    }
    
    setVertices(newVertices);
    setEdges(newEdges);
    handleReset();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-white">Graph Algorithm Visualizer</h1>
          <p className="text-slate-400">Watch how computers find paths and solve problems step-by-step!</p>
          <div className="flex justify-center gap-2 text-sm text-slate-500">
            <Badge variant="outline" className="border-blue-500 text-blue-400">Real-world Example</Badge>
            <Badge variant="outline" className="border-green-500 text-green-400">Step-by-Step</Badge>
            <Badge variant="outline" className="border-purple-500 text-purple-400">Interactive</Badge>
          </div>
        </div>

        {/* Main Control Panel */}
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="p-6">
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">Algorithm</label>
                <Select value={algorithm} onValueChange={(v: Algorithm) => { setAlgorithm(v); handleReset(); }}>
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">Select Algorithm</SelectItem>
                    <SelectItem value="bfs"> BFS - Explore Nearby First</SelectItem>
                    <SelectItem value="dfs"> DFS - Go Deep First</SelectItem>
                    <SelectItem value="dijkstra"> Dijkstra - Shortest Path</SelectItem>
                    <SelectItem value="prim"> Prim - Build Tree</SelectItem>
                    <SelectItem value="kruskal"> Kruskal - Connect All</SelectItem>
                  </SelectContent>
                </Select>
                {algorithm !== 'none' && (
                  <p className="text-xs text-slate-400 mt-1">
                    {algorithm === 'bfs' && ' Visits places level by level, like exploring nearby locations first'}
                    {algorithm === 'dfs' && ' Goes as far as possible before backtracking, like solving a maze'}
                    {algorithm === 'dijkstra' && ' Finds the fastest route between two locations'}
                    {algorithm === 'prim' && ' Builds a network connecting all places with minimum total distance'}
                    {algorithm === 'kruskal' && ' Connects all locations using shortest roads, avoiding loops'}
                  </p>
                )}
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">Graph Type</label>
                <Select value={graphType} onValueChange={(v: GraphType) => setGraphType(v)}>
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="undirected">Undirected</SelectItem>
                    <SelectItem value="directed">Directed</SelectItem>
                    <SelectItem value="weighted">Weighted</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">Start Vertex</label>
                <Select value={startVertex.toString()} onValueChange={(v) => setStartVertex(parseInt(v))}>
                  <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {vertices.map(v => (
                      <SelectItem key={v.id} value={v.id.toString()}>{v.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              {algorithm === 'dijkstra' && (
                <div className="space-y-2">
                  <label className="text-sm font-medium text-slate-300">End Vertex</label>
                  <Select value={endVertex.toString()} onValueChange={(v) => setEndVertex(parseInt(v))}>
                    <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {vertices.map(v => (
                        <SelectItem key={v.id} value={v.id.toString()}>{v.label}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
            </div>
            
            <Separator className="my-4 bg-slate-700" />
            
            <div className="flex flex-wrap gap-3">
              <Button onClick={handlePlayPause} className="gap-2 bg-blue-600 hover:bg-blue-700">
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                {isPlaying ? 'Pause' : traversalOrder.length > 0 ? 'Resume' : 'Run'}
              </Button>
              
              <Button onClick={handleReset} variant="outline" className="gap-2 border-slate-600 text-slate-300">
                <RotateCcw className="h-4 w-4" />
                Reset
              </Button>
              
              <Button onClick={addVertex} variant="outline" className="gap-2 border-slate-600 text-slate-300">
                <Plus className="h-4 w-4" />
                Add Vertex
              </Button>
              
              <Button onClick={exportGraph} variant="outline" className="gap-2 border-slate-600 text-slate-300">
                <Download className="h-4 w-4" />
                Export
              </Button>
              
              <Select value={layout} onValueChange={(v: Layout) => applyLayout(v)}>
                <SelectTrigger className="w-40 bg-slate-700 border-slate-600 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="circular">Circular Layout</SelectItem>
                  <SelectItem value="grid">Grid Layout</SelectItem>
                  <SelectItem value="random">Random Layout</SelectItem>
                </SelectContent>
              </Select>
              
              <div className="ml-auto flex items-center gap-3">
                <span className="text-sm text-slate-400">Speed:</span>
                <Slider
                  value={[2000 - speed]}
                  onValueChange={([v]) => setSpeed(2000 - v)}
                  min={100}
                  max={1900}
                  step={100}
                  className="w-32"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* 3D Visualization */}
          <div className="lg:col-span-2">
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-white">City Map Visualization</CardTitle>
                    <p className="text-sm text-slate-400 mt-1">
                      🏙️ Example: Finding routes between locations in your neighborhood
                    </p>
                  </div>
                  <Badge variant="outline" className="border-green-500 text-green-400">
                    Live Preview
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <div className="h-[600px] relative rounded-lg overflow-hidden">
                  <Canvas camera={{ position: [0, 0, 12], fov: 50 }}>
                    <ambientLight intensity={0.4} />
                    <pointLight position={[10, 10, 10]} intensity={0.8} />
                    <pointLight position={[-10, -10, -10]} intensity={0.3} />
                    
                    <gridHelper args={[20, 20, '#475569', '#334155']} />
                    
                    {edges.map((edge, idx) => {
                      const fromV = vertices.find(v => v.id === edge.from);
                      const toV = vertices.find(v => v.id === edge.to);
                      if (!fromV || !toV) return null;
                      
                      const edgeKey = getEdgeKey(edge.from, edge.to);
                      const isInPath = finalPath.length > 1 && (
                        (finalPath.includes(edge.from) && finalPath.includes(edge.to) &&
                         Math.abs(finalPath.indexOf(edge.from) - finalPath.indexOf(edge.to)) === 1)
                      );
                      const isInMST = mstEdges.has(edgeKey);
                      
                      return (
                        <AnimatedEdge
                          key={idx}
                          edge={edge}
                          fromVertex={fromV}
                          toVertex={toV}
                          isVisited={visitedEdges.has(edgeKey) || isInMST}
                          isInPath={isInPath}
                        />
                      );
                    })}
                    
                    {vertices.map(vertex => (
                      <AnimatedVertex
                        key={vertex.id}
                        vertex={vertex}
                        isCurrent={traversalOrder[currentStep - 1] === vertex.id}
                        isVisited={visitedVertices.has(vertex.id)}
                        distance={distances.get(vertex.id)}
                      />
                    ))}
                    
                    <OrbitControls enablePan enableZoom enableRotate />
                  </Canvas>
                  
                  <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg p-3 space-y-2">
                    <div className="text-white font-semibold text-sm mb-2">Legend</div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                      <span className="text-white text-xs">Not visited yet</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-green-500"></div>
                      <span className="text-white text-xs">Already visited ✓</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-orange-500 animate-pulse"></div>
                      <span className="text-white text-xs">Visiting now...</span>
                    </div>
                    {finalPath.length > 0 && (
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-red-500"></div>
                        <span className="text-white text-xs">Fastest path 🎯</span>
                      </div>
                    )}
                  </div>
                  
                  {algorithm !== 'none' && (
                    <div className="absolute bottom-4 left-4 right-4 bg-black/70 backdrop-blur-sm rounded-lg p-3">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-white font-semibold">Progress</span>
                        <span className="text-slate-300 text-sm">{currentStep} / {traversalOrder.length}</span>
                      </div>
                      <div className="w-full bg-slate-700 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-full rounded-full transition-all duration-300"
                          style={{ width: `${(currentStep / Math.max(traversalOrder.length, 1)) * 100}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Side Panel */}
          <div className="space-y-6">
            <Tabs defaultValue="steps" className="w-full">
              <TabsList className="grid grid-cols-3 bg-slate-700">
                <TabsTrigger value="steps">Steps</TabsTrigger>
                <TabsTrigger value="graph">Graph</TabsTrigger>
                <TabsTrigger value="tools">Tools</TabsTrigger>
              </TabsList>
              
              <TabsContent value="steps" className="space-y-4">
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="text-white">Algorithm Steps</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 max-h-[500px] overflow-y-auto">
                    {algorithmSteps.length === 0 ? (
                      <p className="text-slate-400 text-sm italic">Run an algorithm to see step-by-step execution</p>
                    ) : (
                      algorithmSteps.map((step, idx) => (
                        <div
                          key={idx}
                          className={`p-2 rounded text-sm ${
                            idx === currentStep - 1
                              ? 'bg-blue-600 text-white'
                              : idx < currentStep
                              ? 'bg-slate-700 text-slate-300'
                              : 'bg-slate-800 text-slate-500'
                          }`}
                        >
                          <div className="flex items-start gap-2">
                            <span className="font-mono text-xs mt-0.5">{idx + 1}.</span>
                            <span>{step}</span>
                          </div>
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>
                
                {finalPath.length > 0 && (
                  <Alert className="bg-green-900/20 border-green-700">
                    <Info className="h-4 w-4 text-green-400" />
                    <AlertDescription className="text-green-300">
                      <strong>Fastest Route Found! 🎉</strong>
                      <br />
                      Path: {finalPath.map(id => vertices[id]?.label).join(' → ')}
                      <br />
                      <strong>Total Time:</strong> {distances.get(endVertex)} minutes
                    </AlertDescription>
                  </Alert>
                )}
              </TabsContent>
              
              <TabsContent value="graph">
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="text-white">Graph Information</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-slate-700 rounded-lg p-3">
                        <div className="text-slate-400 text-xs">Vertices</div>
                        <div className="text-white text-2xl font-bold">{vertices.length}</div>
                      </div>
                      <div className="bg-slate-700 rounded-lg p-3">
                        <div className="text-slate-400 text-xs">Edges</div>
                        <div className="text-white text-2xl font-bold">{edges.length}</div>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="text-white font-semibold mb-2">Vertices</h4>
                      <div className="space-y-2 max-h-60 overflow-y-auto">
                        {vertices.map(v => (
                          <div key={v.id} className="flex items-center justify-between bg-slate-700 rounded p-2">
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                              <span className="text-white text-sm font-mono">{v.label} (ID: {v.id})</span>
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => removeVertex(v.id)}
                              className="h-6 w-6 p-0 text-red-400 hover:text-red-300"
                            >
                              <Trash2 className="h-3 w-3" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
              
              <TabsContent value="tools">
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader>
                    <CardTitle className="text-white">Quick Tools</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <label className="text-sm font-medium text-slate-300 mb-2 block">Generate Random Graph</label>
                      <div className="flex gap-2">
                        <Input
                          type="number"
                          placeholder="Number of vertices"
                          className="bg-slate-700 border-slate-600 text-white"
                          id="numVertices"
                        />
                        <Button
                          onClick={() => {
                            const input = document.getElementById('numVertices') as HTMLInputElement;
                            const num = parseInt(input.value) || 8;
                            generateRandomGraph(Math.min(num, 26));
                          }}
                          className="bg-blue-600 hover:bg-blue-700"
                        >
                          Generate
                        </Button>
                      </div>
                    </div>
                    
                    <div>
                      <label className="text-sm font-medium text-slate-300 mb-2 block">Add Edge</label>
                      <div className="space-y-2">
                        <div className="flex gap-2">
                          <Select onValueChange={(v) => (window as any).tmpFrom = parseInt(v)}>
                            <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                              <SelectValue placeholder="From" />
                            </SelectTrigger>
                            <SelectContent>
                              {vertices.map(v => (
                                <SelectItem key={v.id} value={v.id.toString()}>{v.label}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <Select onValueChange={(v) => (window as any).tmpTo = parseInt(v)}>
                            <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                              <SelectValue placeholder="To" />
                            </SelectTrigger>
                            <SelectContent>
                              {vertices.map(v => (
                                <SelectItem key={v.id} value={v.id.toString()}>{v.label}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="flex gap-2">
                          <Input
                            type="number"
                            placeholder="Weight"
                            className="bg-slate-700 border-slate-600 text-white"
                            id="edgeWeight"
                            defaultValue="1"
                          />
                          <Button
                            onClick={() => {
                              const from = (window as any).tmpFrom;
                              const to = (window as any).tmpTo;
                              const weight = parseInt((document.getElementById('edgeWeight') as HTMLInputElement).value) || 1;
                              if (from !== undefined && to !== undefined) {
                                addEdge(from, to, weight);
                              }
                            }}
                            className="bg-blue-600 hover:bg-blue-700"
                          >
                            <Plus className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                    
                    <Separator className="bg-slate-700" />
                    
                    <div className="space-y-2">
                      <Button
                        onClick={() => generateRandomGraph(8)}
                        variant="outline"
                        className="w-full border-slate-600 text-slate-300"
                      >
                        Load Sample Graph
                      </Button>
                      <Button
                        onClick={() => {
                          setVertices([]);
                          setEdges([]);
                          handleReset();
                        }}
                        variant="outline"
                        className="w-full border-red-600 text-red-400 hover:bg-red-900/20"
                      >
                        <Trash2 className="h-4 w-4 mr-2" />
                        Clear Graph
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </div>

        {/* Instructions */}
        <Card className="bg-blue-900/20 border-blue-700">
          <CardContent className="p-6">
            <div className="flex gap-4">
              <Info className="h-6 w-6 text-blue-400 shrink-0" />
              <div className="space-y-3">
                <h3 className="font-semibold text-white text-lg">How It Works - Simple Guide 🎓</h3>
                <div className="grid md:grid-cols-2 gap-3 text-sm text-slate-300">
                  <div className="flex items-start gap-2">
                    <ChevronRight className="h-4 w-4 mt-0.5 text-blue-400" />
                    <span><strong>The Graph:</strong> Shows places (circles) connected by roads (lines). Numbers = travel time in minutes</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <ChevronRight className="h-4 w-4 mt-0.5 text-blue-400" />
                    <span><strong>BFS:</strong> Like searching room by room - checks nearby places first before going far</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <ChevronRight className="h-4 w-4 mt-0.5 text-blue-400" />
                    <span><strong>DFS:</strong> Like exploring a cave - goes deep into one path before trying others</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <ChevronRight className="h-4 w-4 mt-0.5 text-blue-400" />
                    <span><strong>Dijkstra:</strong> GPS navigator - finds the fastest route considering all travel times</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <ChevronRight className="h-4 w-4 mt-0.5 text-blue-400" />
                    <span><strong>MST:</strong> Building roads to connect all places using the least total distance</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <ChevronRight className="h-4 w-4 mt-0.5 text-blue-400" />
                    <span><strong>Controls:</strong> 🖱️ Drag to rotate view, scroll to zoom, right-click to move camera</span>
                  </div>
                </div>
                <div className="mt-4 p-3 bg-blue-900/30 rounded-lg border border-blue-700">
                  <p className="text-sm text-blue-200">
                    💡 <strong>Try this:</strong> Select "Dijkstra" algorithm, choose Home → Store, and press Run to see the fastest route!
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// Export the wrapper component
export function GraphVisualizer() {
  return <GraphVisualizerCore />;
}