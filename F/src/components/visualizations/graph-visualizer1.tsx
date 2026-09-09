"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { 
  Play, 
  RotateCw, 
  Plus, 
  Trash2, 
  GitMerge, 
  GitBranch,
  ChevronRight,
  AlertCircle,
  CheckCircle,
  X,
  Move,
  Minus,
  GitGraph,
  Network,
  Target
} from "lucide-react";
import { Slider } from "@/components/ui/slider";

interface Node {
  id: number;
  x: number;
  y: number;
  label: string;
  color: string;
  radius: number;
}

interface Edge {
  id: number;
  from: number;
  to: number;
  weight?: number;
  color: string;
  directed: boolean;
}

interface DetectionStep {
  description: string;
  nodes: number[];
  edges: number[];
  message: string;
}

type GraphType = 'undirected' | 'directed' | 'weighted';

export function GraphVisualizer() {
  const [nodes, setNodes] = useState<Node[]>([
    { id: 0, x: 100, y: 100, label: 'A', color: '#3b82f6', radius: 25 },
    { id: 1, x: 250, y: 100, label: 'B', color: '#10b981', radius: 25 },
    { id: 2, x: 100, y: 250, label: 'C', color: '#8b5cf6', radius: 25 },
    { id: 3, x: 250, y: 250, label: 'D', color: '#f59e0b', radius: 25 },
  ]);
  
  const [edges, setEdges] = useState<Edge[]>([
    { id: 0, from: 0, to: 1, color: '#6b7280', directed: true },
    { id: 1, from: 1, to: 2, color: '#6b7280', directed: true },
    { id: 2, from: 2, to: 3, color: '#6b7280', directed: true },
  ]);
  
  const [graphType, setGraphType] = useState<GraphType>('directed');
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [draggingNode, setDraggingNode] = useState<number | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [detectionSteps, setDetectionSteps] = useState<DetectionStep[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [cycleFound, setCycleFound] = useState<number[] | null>(null);
  const [algorithm, setAlgorithm] = useState<'dfs' | 'union-find' | 'topological'>('dfs');
  const [showWeights, setShowWeights] = useState(false);
  const [nodeRadius, setNodeRadius] = useState(25);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Initialize canvas
  useEffect(() => {
    drawGraph();
  }, [nodes, edges, selectedNode, cycleFound, currentStep, detectionSteps]);

  const drawGraph = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw edges
    edges.forEach(edge => {
      const fromNode = nodes.find(n => n.id === edge.from);
      const toNode = nodes.find(n => n.id === edge.to);
      
      if (!fromNode || !toNode) return;
      
      ctx.beginPath();
      ctx.strokeStyle = edge.color;
      ctx.lineWidth = 2;
      
      // Draw edge line
      ctx.moveTo(fromNode.x, fromNode.y);
      ctx.lineTo(toNode.x, toNode.y);
      ctx.stroke();
      
      // Draw arrow for directed edges
      if (edge.directed) {
        drawArrow(ctx, fromNode.x, fromNode.y, toNode.x, toNode.y, edge.color);
      }
      
      // Draw weight
      if (showWeights && edge.weight !== undefined) {
        const midX = (fromNode.x + toNode.x) / 2;
        const midY = (fromNode.y + toNode.y) / 2;
        ctx.fillStyle = '#000';
        ctx.font = '12px Arial';
        ctx.fillText(edge.weight.toString(), midX, midY - 10);
      }
    });
    
    // Draw nodes
    nodes.forEach(node => {
      ctx.beginPath();
      ctx.fillStyle = node.color;
      ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
      ctx.fill();
      
      // Node border
      ctx.strokeStyle = selectedNode === node.id ? '#000' : '#374151';
      ctx.lineWidth = selectedNode === node.id ? 3 : 2;
      ctx.stroke();
      
      // Node label
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 14px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node.label, node.x, node.y);
    });
    
    // Highlight cycle if found
    if (cycleFound && cycleFound.length > 0) {
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 3;
      ctx.setLineDash([5, 5]);
      
      for (let i = 0; i < cycleFound.length; i++) {
        const node1 = nodes.find(n => n.id === cycleFound[i]);
        const node2 = nodes.find(n => n.id === cycleFound[(i + 1) % cycleFound.length]);
        
        if (node1 && node2) {
          ctx.beginPath();
          ctx.moveTo(node1.x, node1.y);
          ctx.lineTo(node2.x, node2.y);
          ctx.stroke();
        }
      }
      
      ctx.setLineDash([]);
    }
    
    // Highlight current step
    if (detectionSteps.length > 0 && currentStep < detectionSteps.length) {
      const step = detectionSteps[currentStep];
      
      step.nodes.forEach(nodeId => {
        const node = nodes.find(n => n.id === nodeId);
        if (node) {
          ctx.beginPath();
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 4;
          ctx.arc(node.x, node.y, node.radius + 2, 0, Math.PI * 2);
          ctx.stroke();
        }
      });
    }
  };

  const drawArrow = (ctx: CanvasRenderingContext2D, fromX: number, fromY: number, toX: number, toY: number, color: string) => {
    const headlen = 15;
    const dx = toX - fromX;
    const dy = toY - fromY;
    const angle = Math.atan2(dy, dx);
    
    // Adjust arrow position to not overlap node
    const adjustedToX = toX - (nodeRadius * Math.cos(angle));
    const adjustedToY = toY - (nodeRadius * Math.sin(angle));
    
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(adjustedToX, adjustedToY);
    ctx.stroke();
    
    // Draw arrow head
    ctx.beginPath();
    ctx.moveTo(adjustedToX, adjustedToY);
    ctx.lineTo(
      adjustedToX - headlen * Math.cos(angle - Math.PI / 6),
      adjustedToY - headlen * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      adjustedToX - headlen * Math.cos(angle + Math.PI / 6),
      adjustedToY - headlen * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Check if clicked on a node
    const clickedNode = nodes.find(node => {
      const distance = Math.sqrt((node.x - x) ** 2 + (node.y - y) ** 2);
      return distance <= node.radius;
    });
    
    if (clickedNode) {
      if (selectedNode === null) {
        // Select node for edge creation
        setSelectedNode(clickedNode.id);
        highlightNode(clickedNode.id, '#f59e0b');
      } else {
        // Create edge between selected node and clicked node
        if (selectedNode !== clickedNode.id) {
          const newEdge: Edge = {
            id: edges.length,
            from: selectedNode,
            to: clickedNode.id,
            color: '#6b7280',
            directed: graphType === 'undirected' ? false : true,
            weight: showWeights ? Math.floor(Math.random() * 10) + 1 : undefined
          };
          setEdges([...edges, newEdge]);
        }
        setSelectedNode(null);
        resetNodeColors();
      }
    } else {
      // Add new node
      if (selectedNode) {
        setSelectedNode(null);
        resetNodeColors();
      } else {
        const newNode: Node = {
          id: nodes.length,
          x: x,
          y: y,
          label: String.fromCharCode(65 + nodes.length),
          color: '#3b82f6',
          radius: nodeRadius
        };
        setNodes([...nodes, newNode]);
      }
    }
  };

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const clickedNode = nodes.find(node => {
      const distance = Math.sqrt((node.x - x) ** 2 + (node.y - y) ** 2);
      return distance <= node.radius;
    });
    
    if (clickedNode) {
      setDraggingNode(clickedNode.id);
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (draggingNode === null) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setNodes(prev => prev.map(node => 
      node.id === draggingNode ? { ...node, x, y } : node
    ));
  };

  const handleCanvasMouseUp = () => {
    setDraggingNode(null);
  };

  const highlightNode = (nodeId: number, color: string) => {
    setNodes(prev => prev.map(node => 
      node.id === nodeId ? { ...node, color } : node
    ));
  };

  const resetNodeColors = () => {
    const colors = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444', '#8b5cf6'];
    setNodes(prev => prev.map((node, idx) => ({
      ...node,
      color: colors[idx % colors.length]
    })));
  };

  const resetEdgeColors = () => {
    setEdges(prev => prev.map(edge => ({
      ...edge,
      color: '#6b7280'
    })));
  };

  const addRandomEdge = () => {
    if (nodes.length < 2) return;
    
    const from = Math.floor(Math.random() * nodes.length);
    let to = Math.floor(Math.random() * nodes.length);
    
    // Avoid self-loops for now
    while (to === from) {
      to = Math.floor(Math.random() * nodes.length);
    }
    
    // Check if edge already exists
    const edgeExists = edges.some(e => 
      (e.from === from && e.to === to) || (!e.directed && e.from === to && e.to === from)
    );
    
    if (!edgeExists) {
      const newEdge: Edge = {
        id: edges.length,
        from,
        to,
        color: '#6b7280',
        directed: graphType === 'undirected' ? false : true,
        weight: showWeights ? Math.floor(Math.random() * 10) + 1 : undefined
      };
      setEdges([...edges, newEdge]);
    }
  };

  const removeSelectedNode = () => {
    if (selectedNode === null) return;
    
    // Remove node
    setNodes(prev => prev.filter(node => node.id !== selectedNode));
    
    // Remove connected edges
    setEdges(prev => prev.filter(edge => 
      edge.from !== selectedNode && edge.to !== selectedNode
    ));
    
    setSelectedNode(null);
  };

  const clearGraph = () => {
    setNodes([
      { id: 0, x: 100, y: 100, label: 'A', color: '#3b82f6', radius: nodeRadius },
      { id: 1, x: 250, y: 100, label: 'B', color: '#10b981', radius: nodeRadius },
      { id: 2, x: 100, y: 250, label: 'C', color: '#8b5cf6', radius: nodeRadius },
      { id: 3, x: 250, y: 250, label: 'D', color: '#f59e0b', radius: nodeRadius },
    ]);
    setEdges([
      { id: 0, from: 0, to: 1, color: '#6b7280', directed: true },
      { id: 1, from: 1, to: 2, color: '#6b7280', directed: true },
      { id: 2, from: 2, to: 3, color: '#6b7280', directed: true },
    ]);
    setCycleFound(null);
    setDetectionSteps([]);
    setCurrentStep(0);
    setIsRunning(false);
  };

  const runCycleDetection = () => {
    setIsRunning(true);
    setCycleFound(null);
    setDetectionSteps([]);
    setCurrentStep(0);
    
    const steps: DetectionStep[] = [];
    const visited = new Set<number>();
    const recStack = new Set<number>();
    let cycle: number[] = [];
    
    // Build adjacency list
    const adjList: Record<number, number[]> = {};
    nodes.forEach(node => adjList[node.id] = []);
    edges.forEach(edge => {
      adjList[edge.from].push(edge.to);
      if (!edge.directed) {
        adjList[edge.to].push(edge.from);
      }
    });
    
    const dfs = (node: number, parent: number | null, path: number[]): boolean => {
      visited.add(node);
      recStack.add(node);
      path.push(node);
      
      steps.push({
        description: `Visiting node ${nodes[node].label}`,
        nodes: [node],
        edges: [],
        message: `Exploring ${nodes[node].label}...`
      });
      
      for (const neighbor of adjList[node]) {
        steps.push({
          description: `Checking edge ${nodes[node].label} → ${nodes[neighbor].label}`,
          nodes: [node, neighbor],
          edges: [edges.findIndex(e => 
            (e.from === node && e.to === neighbor) || 
            (!e.directed && e.from === neighbor && e.to === node)
          )],
          message: `Checking connection to ${nodes[neighbor].label}`
        });
        
        if (!visited.has(neighbor)) {
          if (dfs(neighbor, node, path)) {
            return true;
          }
        } else if (recStack.has(neighbor) && (graphType === 'directed' || neighbor !== parent)) {
          // Cycle found
          const cycleStart = path.indexOf(neighbor);
          cycle = path.slice(cycleStart);
          cycle.push(neighbor); // Close the cycle
          
          steps.push({
            description: `Cycle detected! ${cycle.map(id => nodes[id].label).join(' → ')}`,
            nodes: cycle,
            edges: [],
            message: `Found cycle: ${cycle.map(id => nodes[id].label).join(' → ')}`
          });
          
          return true;
        }
      }
      
      recStack.delete(node);
      path.pop();
      
      steps.push({
        description: `Backtracking from ${nodes[node].label}`,
        nodes: [node],
        edges: [],
        message: `Finished exploring ${nodes[node].label}`
      });
      
      return false;
    };
    
    for (const node of nodes) {
      if (!visited.has(node.id)) {
        if (dfs(node.id, null, [])) {
          break;
        }
      }
    }
    
    setDetectionSteps(steps);
    
    if (cycle.length > 0) {
      setCycleFound(cycle);
    } else {
      steps.push({
        description: "No cycles found in the graph",
        nodes: [],
        edges: [],
        message: "Graph is acyclic"
      });
    }
  };

  const nextStep = () => {
    if (currentStep < detectionSteps.length - 1) {
      setCurrentStep(prev => prev + 1);
    } else {
      setIsRunning(false);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  };

  return (
    <Card className="w-full">
      <CardContent className="p-6">
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Left Panel - Graph Canvas */}
          <div className="lg:w-2/3">
            <div className="mb-4 flex flex-wrap gap-2 items-center justify-between">
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant={graphType === 'directed' ? "default" : "outline"}
                  onClick={() => setGraphType('directed')}
                >
                  <GitMerge className="h-4 w-4 mr-2" />
                  Directed
                </Button>
                <Button
                  size="sm"
                  variant={graphType === 'undirected' ? "default" : "outline"}
                  onClick={() => setGraphType('undirected')}
                >
                  <GitBranch className="h-4 w-4 mr-2" />
                  Undirected
                </Button>
              </div>
              
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={runCycleDetection}
                  disabled={isRunning}
                >
                  <Play className="h-4 w-4 mr-2" />
                  Detect Cycles
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={clearGraph}
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Clear
                </Button>
              </div>
            </div>
            
            <div className="relative">
              <div
                ref={containerRef}
                className="border rounded-lg bg-gray-50 relative overflow-hidden"
              >
                <canvas
                  ref={canvasRef}
                  width={600}
                  height={400}
                  onClick={handleCanvasClick}
                  onMouseDown={handleCanvasMouseDown}
                  onMouseMove={handleCanvasMouseMove}
                  onMouseUp={handleCanvasMouseUp}
                  onMouseLeave={handleCanvasMouseUp}
                  className="cursor-pointer w-full"
                />
                
                {/* Instructions Overlay */}
                {nodes.length === 0 && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center p-8 bg-white/80 rounded-lg">
                      <Network className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                      <h3 className="font-semibold mb-2">Create Your Graph</h3>
                      <p className="text-sm text-gray-600 mb-4">
                        Click anywhere to add nodes, then select nodes to create edges
                      </p>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Legend */}
              <div className="mt-4 flex flex-wrap gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-blue-500"></div>
                  <span>Normal Node</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-amber-500"></div>
                  <span>Selected Node</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-emerald-500"></div>
                  <span>Currently Visiting</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-red-500"></div>
                  <span>Cycle Node</span>
                </div>
              </div>
            </div>
            
            {/* Control Panel */}
            <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
              <Button
                variant="outline"
                size="sm"
                onClick={addRandomEdge}
              >
                <Plus className="h-4 w-4 mr-2" />
                Add Random Edge
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowWeights(!showWeights)}
              >
                <Target className="h-4 w-4 mr-2" />
                {showWeights ? 'Hide Weights' : 'Show Weights'}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={removeSelectedNode}
                disabled={selectedNode === null}
              >
                <Minus className="h-4 w-4 mr-2" />
                Remove Node
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  const newNodes = nodes.map(node => ({
                    ...node,
                    x: Math.random() * 500 + 50,
                    y: Math.random() * 300 + 50
                  }));
                  setNodes(newNodes);
                }}
              >
                <Move className="h-4 w-4 mr-2" />
                Random Layout
              </Button>
            </div>
            
            {/* Node Size Control */}
            <div className="mt-6">
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium">Node Size</label>
                <span className="text-sm">{nodeRadius}px</span>
              </div>
              <Slider
                value={[nodeRadius]}
                onValueChange={([value]) => {
                  setNodeRadius(value);
                  setNodes(prev => prev.map(node => ({ ...node, radius: value })));
                }}
                min={15}
                max={40}
                step={1}
                className="w-full"
              />
            </div>
          </div>
          
          {/* Right Panel - Controls & Steps */}
          <div className="lg:w-1/3">
            <Tabs defaultValue="steps">
              <TabsList className="grid grid-cols-2">
                <TabsTrigger value="steps">Detection Steps</TabsTrigger>
                <TabsTrigger value="info">Graph Info</TabsTrigger>
              </TabsList>
              
              <TabsContent value="steps" className="space-y-4">
                {isRunning && detectionSteps.length > 0 && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={prevStep}
                        disabled={currentStep === 0}
                      >
                        Previous
                      </Button>
                      <Badge variant="secondary">
                        Step {currentStep + 1} of {detectionSteps.length}
                      </Badge>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={nextStep}
                        disabled={currentStep === detectionSteps.length - 1}
                      >
                        Next
                      </Button>
                    </div>
                    
                    <Separator />
                  </div>
                )}
                
                <div className="space-y-3">
                  {detectionSteps.slice(0, currentStep + 1).map((step, idx) => (
                    <div
                      key={idx}
                      className={`p-3 rounded-lg border ${
                        idx === currentStep 
                          ? 'bg-primary/10 border-primary' 
                          : 'bg-gray-50'
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        <div className={`h-2 w-2 rounded-full mt-2 ${
                          step.description.includes('Cycle detected') 
                            ? 'bg-red-500' 
                            : step.description.includes('No cycles')
                            ? 'bg-gray-500'
                            : idx === currentStep
                            ? 'bg-primary'
                            : 'bg-gray-400'
                        }`}></div>
                        <div>
                          <p className="text-sm font-medium">{step.description}</p>
                          <p className="text-xs text-gray-600 mt-1">{step.message}</p>
                          {step.nodes.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-1">
                              <span className="text-xs text-gray-500">Nodes:</span>
                              {step.nodes.map(nodeId => (
                                <Badge key={nodeId} variant="outline" className="text-xs">
                                  {nodes[nodeId]?.label}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {!isRunning && detectionSteps.length === 0 && (
                    <div className="text-center p-8 text-gray-500">
                      <RotateCw className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                      <p>Run cycle detection to see step-by-step visualization</p>
                    </div>
                  )}
                </div>
                
                {/* Cycle Result */}
                {cycleFound && (
                  <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertCircle className="h-5 w-5 text-red-600" />
                      <h4 className="font-semibold text-red-700">Cycle Found!</h4>
                    </div>
                    <p className="text-sm text-red-600">
                      Cycle: {cycleFound.map(id => nodes[id]?.label).join(' → ')}
                    </p>
                  </div>
                )}
                
                {!cycleFound && detectionSteps.length > 0 && !isRunning && (
                  <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <CheckCircle className="h-5 w-5 text-green-600" />
                      <h4 className="font-semibold text-green-700">No Cycles</h4>
                    </div>
                    <p className="text-sm text-green-600">
                      The graph is acyclic
                    </p>
                  </div>
                )}
              </TabsContent>
              
              <TabsContent value="info">
                <div className="space-y-4">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h4 className="font-semibold mb-2">Graph Statistics</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Nodes:</span>
                        <Badge variant="secondary">{nodes.length}</Badge>
                      </div>
                      <div className="flex justify-between">
                        <span>Edges:</span>
                        <Badge variant="secondary">{edges.length}</Badge>
                      </div>
                      <div className="flex justify-between">
                        <span>Type:</span>
                        <Badge variant="outline">
                          {graphType === 'directed' ? 'Directed' : 'Undirected'}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span>Density:</span>
                        <span>
                          {((edges.length / (nodes.length * (nodes.length - 1))) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="p-4 bg-amber-50 rounded-lg">
                    <h4 className="font-semibold mb-2">How to Use</h4>
                    <ul className="space-y-2 text-sm">
                      <li className="flex items-start gap-2">
                        <Plus className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
                        <span>Click empty space to add nodes</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <GitGraph className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
                        <span>Click node, then another to create edge</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Move className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
                        <span>Drag nodes to rearrange</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Play className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
                        <span>Click "Detect Cycles" to visualize algorithm</span>
                      </li>
                    </ul>
                  </div>
                  
                  <div className="p-4 bg-purple-50 rounded-lg">
                    <h4 className="font-semibold mb-2">Algorithm Details</h4>
                    <p className="text-sm mb-2">
                      Currently using Depth-First Search (DFS) for cycle detection.
                    </p>
                    <div className="text-xs space-y-1">
                      <p>• Time Complexity: O(V + E)</p>
                      <p>• Space Complexity: O(V)</p>
                      <p>• Works for both directed and undirected graphs</p>
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
            
            {/* Quick Actions */}
            <div className="mt-6 space-y-3">
              <h4 className="font-semibold">Quick Graph Templates</h4>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    // Create a simple cycle
                    setNodes([
                      { id: 0, x: 150, y: 100, label: 'A', color: '#3b82f6', radius: nodeRadius },
                      { id: 1, x: 250, y: 150, label: 'B', color: '#10b981', radius: nodeRadius },
                      { id: 2, x: 150, y: 200, label: 'C', color: '#8b5cf6', radius: nodeRadius },
                      { id: 3, x: 50, y: 150, label: 'D', color: '#f59e0b', radius: nodeRadius },
                    ]);
                    setEdges([
                      { id: 0, from: 0, to: 1, color: '#6b7280', directed: true },
                      { id: 1, from: 1, to: 2, color: '#6b7280', directed: true },
                      { id: 2, from: 2, to: 3, color: '#6b7280', directed: true },
                      { id: 3, from: 3, to: 0, color: '#6b7280', directed: true },
                    ]);
                    setCycleFound(null);
                  }}
                >
                  Simple Cycle
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    // Create a tree (acyclic)
                    setNodes([
                      { id: 0, x: 200, y: 50, label: 'A', color: '#3b82f6', radius: nodeRadius },
                      { id: 1, x: 100, y: 150, label: 'B', color: '#10b981', radius: nodeRadius },
                      { id: 2, x: 300, y: 150, label: 'C', color: '#8b5cf6', radius: nodeRadius },
                      { id: 3, x: 50, y: 250, label: 'D', color: '#f59e0b', radius: nodeRadius },
                      { id: 4, x: 150, y: 250, label: 'E', color: '#ef4444', radius: nodeRadius },
                    ]);
                    setEdges([
                      { id: 0, from: 0, to: 1, color: '#6b7280', directed: false },
                      { id: 1, from: 0, to: 2, color: '#6b7280', directed: false },
                      { id: 2, from: 1, to: 3, color: '#6b7280', directed: false },
                      { id: 3, from: 1, to: 4, color: '#6b7280', directed: false },
                    ]);
                    setCycleFound(null);
                  }}
                >
                  Tree (Acyclic)
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    // Create a complex graph
                    setNodes([
                      { id: 0, x: 100, y: 100, label: 'A', color: '#3b82f6', radius: nodeRadius },
                      { id: 1, x: 250, y: 100, label: 'B', color: '#10b981', radius: nodeRadius },
                      { id: 2, x: 400, y: 100, label: 'C', color: '#8b5cf6', radius: nodeRadius },
                      { id: 3, x: 100, y: 250, label: 'D', color: '#f59e0b', radius: nodeRadius },
                      { id: 4, x: 250, y: 250, label: 'E', color: '#ef4444', radius: nodeRadius },
                      { id: 5, x: 400, y: 250, label: 'F', color: '#8b5cf6', radius: nodeRadius },
                    ]);
                    setEdges([
                      { id: 0, from: 0, to: 1, color: '#6b7280', directed: true },
                      { id: 1, from: 1, to: 2, color: '#6b7280', directed: true },
                      { id: 2, from: 2, to: 5, color: '#6b7280', directed: true },
                      { id: 3, from: 5, to: 4, color: '#6b7280', directed: true },
                      { id: 4, from: 4, to: 3, color: '#6b7280', directed: true },
                      { id: 5, from: 3, to: 0, color: '#6b7280', directed: true },
                      { id: 6, from: 1, to: 4, color: '#6b7280', directed: true },
                      { id: 7, from: 2, to: 3, color: '#6b7280', directed: true },
                    ]);
                    setCycleFound(null);
                  }}
                >
                  Complex Graph
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    // Create a bipartite graph
                    setNodes([
                      { id: 0, x: 100, y: 100, label: 'A', color: '#3b82f6', radius: nodeRadius },
                      { id: 1, x: 100, y: 200, label: 'B', color: '#10b981', radius: nodeRadius },
                      { id: 2, x: 100, y: 300, label: 'C', color: '#8b5cf6', radius: nodeRadius },
                      { id: 3, x: 300, y: 100, label: 'D', color: '#f59e0b', radius: nodeRadius },
                      { id: 4, x: 300, y: 200, label: 'E', color: '#ef4444', radius: nodeRadius },
                      { id: 5, x: 300, y: 300, label: 'F', color: '#8b5cf6', radius: nodeRadius },
                    ]);
                    setEdges([
                      { id: 0, from: 0, to: 3, color: '#6b7280', directed: false },
                      { id: 1, from: 0, to: 4, color: '#6b7280', directed: false },
                      { id: 2, from: 1, to: 4, color: '#6b7280', directed: false },
                      { id: 3, from: 1, to: 5, color: '#6b7280', directed: false },
                      { id: 4, from: 2, to: 5, color: '#6b7280', directed: false },
                    ]);
                    setCycleFound(null);
                  }}
                >
                  Bipartite Graph
                </Button>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}