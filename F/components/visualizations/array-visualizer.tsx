"use client";

import * as React from "react";
import { Play, Pause, RotateCcw, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface ArrayVisualizerProps {
  initialArray?: number[];
}

export function ArrayVisualizer({ initialArray = [64, 34, 25, 12, 22, 11, 90] }: ArrayVisualizerProps) {
  const [array, setArray] = React.useState(initialArray);
  const [highlightedIndices, setHighlightedIndices] = React.useState<number[]>([]);
  const [sortingIndices, setSortingIndices] = React.useState<number[]>([]);
  const [isSorting, setIsSorting] = React.useState(false);
  const [newValue, setNewValue] = React.useState("");
  const [insertIndex, setInsertIndex] = React.useState("");
  const [message, setMessage] = React.useState("Click on operations to see the array in action");

  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  const reset = () => {
    setArray(initialArray);
    setHighlightedIndices([]);
    setSortingIndices([]);
    setIsSorting(false);
    setMessage("Array reset to initial state");
  };

  const insertElement = () => {
    const value = parseInt(newValue);
    const index = insertIndex ? parseInt(insertIndex) : array.length;
    
    if (isNaN(value)) {
      setMessage("Please enter a valid number");
      return;
    }

    if (index < 0 || index > array.length) {
      setMessage("Invalid index position");
      return;
    }

    const newArray = [...array.slice(0, index), value, ...array.slice(index)];
    setArray(newArray);
    setHighlightedIndices([index]);
    setMessage(`Inserted ${value} at index ${index}`);
    setNewValue("");
    setInsertIndex("");

    setTimeout(() => setHighlightedIndices([]), 1500);
  };

  const deleteElement = async (index: number) => {
    setHighlightedIndices([index]);
    setMessage(`Deleting element at index ${index}`);
    
    await sleep(500);
    
    const newArray = array.filter((_, i) => i !== index);
    setArray(newArray);
    setHighlightedIndices([]);
    setMessage(`Deleted element. Array shifted left.`);
  };

  const bubbleSort = async () => {
    if (isSorting) return;
    setIsSorting(true);
    setMessage("Starting Bubble Sort...");

    const arr = [...array];
    const n = arr.length;

    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < n - i - 1; j++) {
        setSortingIndices([j, j + 1]);
        setMessage(`Comparing indices ${j} and ${j + 1}`);
        
        await sleep(400);

        if (arr[j] > arr[j + 1]) {
          [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
          setArray([...arr]);
          setMessage(`Swapped ${arr[j + 1]} and ${arr[j]}`);
          await sleep(400);
        }
      }
    }

    setSortingIndices([]);
    setIsSorting(false);
    setMessage("Bubble Sort complete!");
  };

  const linearSearch = async (target: number) => {
    if (isSorting) return;
    setIsSorting(true);
    setMessage(`Searching for ${target}...`);

    for (let i = 0; i < array.length; i++) {
      setHighlightedIndices([i]);
      setMessage(`Checking index ${i}: ${array[i]}`);
      await sleep(500);

      if (array[i] === target) {
        setMessage(`Found ${target} at index ${i}!`);
        setIsSorting(false);
        return;
      }
    }

    setHighlightedIndices([]);
    setMessage(`${target} not found in array`);
    setIsSorting(false);
  };

  return (
    <div className="space-y-6 p-6 bg-card rounded-lg border border-border">
      <div className="flex flex-wrap gap-4 items-end">
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Value</label>
          <Input
            type="number"
            placeholder="Value"
            value={newValue}
            onChange={(e) => setNewValue(e.target.value)}
            className="w-24"
          />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Index (optional)</label>
          <Input
            type="number"
            placeholder="Index"
            value={insertIndex}
            onChange={(e) => setInsertIndex(e.target.value)}
            className="w-24"
          />
        </div>
        <Button onClick={insertElement} disabled={isSorting}>
          <Plus className="h-4 w-4 mr-2" />
          Insert
        </Button>
        <Button variant="outline" onClick={bubbleSort} disabled={isSorting}>
          <Play className="h-4 w-4 mr-2" />
          Bubble Sort
        </Button>
        <Button variant="outline" onClick={() => linearSearch(25)} disabled={isSorting}>
          Search (25)
        </Button>
        <Button variant="ghost" onClick={reset} disabled={isSorting}>
          <RotateCcw className="h-4 w-4 mr-2" />
          Reset
        </Button>
      </div>

      <div className="p-4 bg-muted/30 rounded-lg min-h-[120px]">
        <div className="flex flex-wrap gap-2 items-end justify-center">
          {array.map((value, index) => (
            <div key={index} className="flex flex-col items-center gap-1">
              <div
                className={cn(
                  "relative w-14 h-14 flex items-center justify-center rounded-lg border-2 font-mono font-semibold transition-all duration-300",
                  highlightedIndices.includes(index)
                    ? "bg-primary text-primary-foreground border-primary scale-110"
                    : sortingIndices.includes(index)
                    ? "bg-warning/20 border-warning text-foreground"
                    : "bg-card border-border text-foreground"
                )}
              >
                {value}
                <button
                  onClick={() => deleteElement(index)}
                  disabled={isSorting}
                  className="absolute -top-2 -right-2 h-5 w-5 rounded-full bg-destructive text-destructive-foreground flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
              <span className="text-xs text-muted-foreground">[{index}]</span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-2 p-3 bg-muted/50 rounded-lg">
        <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
        <span className="text-sm text-foreground">{message}</span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center text-sm">
        <div className="p-3 bg-muted/30 rounded-lg">
          <div className="font-semibold text-foreground">Length</div>
          <div className="text-muted-foreground">{array.length}</div>
        </div>
        <div className="p-3 bg-muted/30 rounded-lg">
          <div className="font-semibold text-foreground">First</div>
          <div className="text-muted-foreground">{array[0] ?? "N/A"}</div>
        </div>
        <div className="p-3 bg-muted/30 rounded-lg">
          <div className="font-semibold text-foreground">Last</div>
          <div className="text-muted-foreground">{array[array.length - 1] ?? "N/A"}</div>
        </div>
        <div className="p-3 bg-muted/30 rounded-lg">
          <div className="font-semibold text-foreground">Sum</div>
          <div className="text-muted-foreground">{array.reduce((a, b) => a + b, 0)}</div>
        </div>
      </div>
    </div>
  );
}
