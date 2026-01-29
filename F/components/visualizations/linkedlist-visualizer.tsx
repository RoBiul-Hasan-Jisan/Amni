"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Play, Pause, RotateCcw, Plus, Trash2, Search } from "lucide-react";
import { cn } from "@/lib/utils";

interface ListNode {
  value: number;
  id: string;
}

export function LinkedListVisualizer() {
  const [nodes, setNodes] = React.useState<ListNode[]>([
    { value: 10, id: "1" },
    { value: 20, id: "2" },
    { value: 30, id: "3" },
    { value: 40, id: "4" },
  ]);
  const [inputValue, setInputValue] = React.useState("");
  const [positionValue, setPositionValue] = React.useState("");
  const [highlightedIndex, setHighlightedIndex] = React.useState<number | null>(null);
  const [traversingIndex, setTraversingIndex] = React.useState<number | null>(null);
  const [isAnimating, setIsAnimating] = React.useState(false);
  const [message, setMessage] = React.useState("Click on operations to visualize linked list");

  const generateId = () => Math.random().toString(36).substr(2, 9);

  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  const insertAtHead = async () => {
    const value = parseInt(inputValue);
    if (isNaN(value)) {
      setMessage("Please enter a valid number");
      return;
    }
    setIsAnimating(true);
    setMessage(`Inserting ${value} at head - O(1) operation`);
    
    const newNode: ListNode = { value, id: generateId() };
    setNodes((prev) => [newNode, ...prev]);
    setHighlightedIndex(0);
    
    await sleep(1000);
    setHighlightedIndex(null);
    setIsAnimating(false);
    setInputValue("");
    setMessage(`Inserted ${value} at head successfully`);
  };

  const insertAtTail = async () => {
    const value = parseInt(inputValue);
    if (isNaN(value)) {
      setMessage("Please enter a valid number");
      return;
    }
    setIsAnimating(true);
    setMessage(`Traversing to tail to insert ${value} - O(n) operation`);

    // Traverse animation
    for (let i = 0; i < nodes.length; i++) {
      setTraversingIndex(i);
      await sleep(400);
    }
    setTraversingIndex(null);

    const newNode: ListNode = { value, id: generateId() };
    setNodes((prev) => [...prev, newNode]);
    setHighlightedIndex(nodes.length);

    await sleep(800);
    setHighlightedIndex(null);
    setIsAnimating(false);
    setInputValue("");
    setMessage(`Inserted ${value} at tail successfully`);
  };

  const insertAtPosition = async () => {
    const value = parseInt(inputValue);
    const position = parseInt(positionValue);
    if (isNaN(value) || isNaN(position)) {
      setMessage("Please enter valid number and position");
      return;
    }
    if (position < 0 || position > nodes.length) {
      setMessage(`Position must be between 0 and ${nodes.length}`);
      return;
    }
    setIsAnimating(true);
    setMessage(`Traversing to position ${position} to insert ${value}`);

    // Traverse animation
    for (let i = 0; i < position; i++) {
      setTraversingIndex(i);
      await sleep(400);
    }
    setTraversingIndex(null);

    const newNode: ListNode = { value, id: generateId() };
    setNodes((prev) => [...prev.slice(0, position), newNode, ...prev.slice(position)]);
    setHighlightedIndex(position);

    await sleep(800);
    setHighlightedIndex(null);
    setIsAnimating(false);
    setInputValue("");
    setPositionValue("");
    setMessage(`Inserted ${value} at position ${position} successfully`);
  };

  const deleteAtHead = async () => {
    if (nodes.length === 0) {
      setMessage("List is empty");
      return;
    }
    setIsAnimating(true);
    setMessage(`Deleting head node - O(1) operation`);
    setHighlightedIndex(0);

    await sleep(800);
    const deletedValue = nodes[0].value;
    setNodes((prev) => prev.slice(1));
    setHighlightedIndex(null);
    setIsAnimating(false);
    setMessage(`Deleted ${deletedValue} from head successfully`);
  };

  const deleteAtTail = async () => {
    if (nodes.length === 0) {
      setMessage("List is empty");
      return;
    }
    setIsAnimating(true);
    setMessage(`Traversing to tail to delete - O(n) operation`);

    // Traverse animation
    for (let i = 0; i < nodes.length; i++) {
      setTraversingIndex(i);
      await sleep(400);
    }
    setTraversingIndex(null);
    setHighlightedIndex(nodes.length - 1);

    await sleep(800);
    const deletedValue = nodes[nodes.length - 1].value;
    setNodes((prev) => prev.slice(0, -1));
    setHighlightedIndex(null);
    setIsAnimating(false);
    setMessage(`Deleted ${deletedValue} from tail successfully`);
  };

  const searchValue = async () => {
    const value = parseInt(inputValue);
    if (isNaN(value)) {
      setMessage("Please enter a valid number to search");
      return;
    }
    setIsAnimating(true);
    setMessage(`Searching for ${value} - O(n) operation`);

    for (let i = 0; i < nodes.length; i++) {
      setTraversingIndex(i);
      await sleep(500);
      if (nodes[i].value === value) {
        setTraversingIndex(null);
        setHighlightedIndex(i);
        setMessage(`Found ${value} at position ${i}!`);
        await sleep(1500);
        setHighlightedIndex(null);
        setIsAnimating(false);
        setInputValue("");
        return;
      }
    }

    setTraversingIndex(null);
    setIsAnimating(false);
    setInputValue("");
    setMessage(`${value} not found in the list`);
  };

  const reverseList = async () => {
    if (nodes.length <= 1) {
      setMessage("Need at least 2 nodes to reverse");
      return;
    }
    setIsAnimating(true);
    setMessage("Reversing linked list - O(n) operation");

    const reversed: ListNode[] = [];
    for (let i = nodes.length - 1; i >= 0; i--) {
      setHighlightedIndex(i);
      await sleep(500);
      reversed.push(nodes[i]);
    }

    setNodes(reversed);
    setHighlightedIndex(null);
    setIsAnimating(false);
    setMessage("List reversed successfully!");
  };

  const reset = () => {
    setNodes([
      { value: 10, id: "1" },
      { value: 20, id: "2" },
      { value: 30, id: "3" },
      { value: 40, id: "4" },
    ]);
    setHighlightedIndex(null);
    setTraversingIndex(null);
    setIsAnimating(false);
    setInputValue("");
    setPositionValue("");
    setMessage("List reset to initial state");
  };

  return (
    <div className="space-y-6">
      {/* Visualization Area */}
      <div className="bg-muted/50 rounded-lg p-6 min-h-[180px] overflow-x-auto">
        <div className="flex items-center gap-1 min-w-max">
          <div className="text-xs font-mono text-muted-foreground mr-2">HEAD</div>
          {nodes.length === 0 ? (
            <div className="text-muted-foreground italic">Empty list (NULL)</div>
          ) : (
            nodes.map((node, index) => (
              <React.Fragment key={node.id}>
                <div
                  className={cn(
                    "flex items-center transition-all duration-300",
                    highlightedIndex === index && "scale-110",
                  )}
                >
                  <div
                    className={cn(
                      "relative flex border-2 rounded-lg overflow-hidden transition-all duration-300",
                      highlightedIndex === index
                        ? "border-primary bg-primary/20 shadow-lg"
                        : traversingIndex === index
                          ? "border-warning bg-warning/20"
                          : "border-border bg-card",
                    )}
                  >
                    {/* Data part */}
                    <div className="px-4 py-3 border-r border-border min-w-[60px] text-center">
                      <div className="text-xs text-muted-foreground mb-1">data</div>
                      <div className="font-mono font-semibold text-foreground">{node.value}</div>
                    </div>
                    {/* Next pointer part */}
                    <div className="px-3 py-3 min-w-[50px] text-center bg-muted/30">
                      <div className="text-xs text-muted-foreground mb-1">next</div>
                      <div className="font-mono text-xs text-muted-foreground">
                        {index < nodes.length - 1 ? "ptr" : "NULL"}
                      </div>
                    </div>
                  </div>
                </div>
                {/* Arrow */}
                {index < nodes.length - 1 && (
                  <div className="flex items-center mx-1">
                    <div className="w-6 h-0.5 bg-muted-foreground" />
                    <div className="w-0 h-0 border-t-4 border-b-4 border-l-6 border-transparent border-l-muted-foreground" />
                  </div>
                )}
              </React.Fragment>
            ))
          )}
          {nodes.length > 0 && (
            <>
              <div className="flex items-center mx-1">
                <div className="w-4 h-0.5 bg-muted-foreground" />
              </div>
              <div className="text-xs font-mono text-muted-foreground">NULL</div>
            </>
          )}
        </div>
      </div>

      {/* Message */}
      <div className="text-sm text-center p-3 bg-secondary/50 rounded-lg text-secondary-foreground">
        {message}
      </div>

      {/* Controls */}
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Input
            type="number"
            placeholder="Value"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            className="w-24"
            disabled={isAnimating}
          />
          <Input
            type="number"
            placeholder="Position"
            value={positionValue}
            onChange={(e) => setPositionValue(e.target.value)}
            className="w-24"
            disabled={isAnimating}
          />
        </div>

        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            onClick={insertAtHead}
            disabled={isAnimating}
            className="gap-1"
          >
            <Plus className="h-3 w-3" /> Insert Head
          </Button>
          <Button
            size="sm"
            onClick={insertAtTail}
            disabled={isAnimating}
            className="gap-1"
          >
            <Plus className="h-3 w-3" /> Insert Tail
          </Button>
          <Button
            size="sm"
            onClick={insertAtPosition}
            disabled={isAnimating}
            className="gap-1"
          >
            <Plus className="h-3 w-3" /> Insert at Pos
          </Button>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="destructive"
            onClick={deleteAtHead}
            disabled={isAnimating}
            className="gap-1"
          >
            <Trash2 className="h-3 w-3" /> Delete Head
          </Button>
          <Button
            size="sm"
            variant="destructive"
            onClick={deleteAtTail}
            disabled={isAnimating}
            className="gap-1"
          >
            <Trash2 className="h-3 w-3" /> Delete Tail
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={searchValue}
            disabled={isAnimating}
            className="gap-1"
          >
            <Search className="h-3 w-3" /> Search
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={reverseList}
            disabled={isAnimating}
            className="gap-1"
          >
            <RotateCcw className="h-3 w-3" /> Reverse
          </Button>
        </div>

        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={reset} disabled={isAnimating}>
            <RotateCcw className="h-4 w-4 mr-1" /> Reset
          </Button>
        </div>
      </div>
    </div>
  );
}
