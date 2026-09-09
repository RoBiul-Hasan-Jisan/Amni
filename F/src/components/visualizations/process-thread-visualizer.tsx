"use client";

import * as React from "react";
import { Play, Pause, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface Process {
  id: number;
  name: string;
  threads: Thread[];
  memory: { heap: number; stack: number };
  status: "running" | "waiting" | "blocked";
}

interface Thread {
  id: number;
  name: string;
  status: "running" | "ready" | "waiting";
}

export function ProcessThreadVisualizer() {
  const [isAnimating, setIsAnimating] = React.useState(false);
  const [currentStep, setCurrentStep] = React.useState(0);
  const [activeProcess, setActiveProcess] = React.useState(0);
  const [activeThread, setActiveThread] = React.useState(0);

  const processes: Process[] = [
    {
      id: 1,
      name: "Process A",
      threads: [
        { id: 1, name: "Main Thread", status: "running" },
        { id: 2, name: "Worker 1", status: "ready" },
      ],
      memory: { heap: 256, stack: 64 },
      status: "running",
    },
    {
      id: 2,
      name: "Process B",
      threads: [
        { id: 1, name: "Main Thread", status: "ready" },
        { id: 2, name: "Worker 1", status: "waiting" },
        { id: 3, name: "Worker 2", status: "ready" },
      ],
      memory: { heap: 512, stack: 128 },
      status: "waiting",
    },
  ];

  const steps = [
    { process: 0, thread: 0, message: "Process A - Main Thread is running" },
    { process: 0, thread: 1, message: "Context switch to Worker 1 in Process A" },
    { process: 1, thread: 0, message: "Process switch to Process B - Main Thread" },
    { process: 1, thread: 2, message: "Context switch to Worker 2 in Process B" },
    { process: 0, thread: 0, message: "Back to Process A - Main Thread" },
  ];

  React.useEffect(() => {
    if (isAnimating) {
      const interval = setInterval(() => {
        setCurrentStep((prev) => {
          const next = (prev + 1) % steps.length;
          setActiveProcess(steps[next].process);
          setActiveThread(steps[next].thread);
          return next;
        });
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [isAnimating, steps.length]);

  const reset = () => {
    setIsAnimating(false);
    setCurrentStep(0);
    setActiveProcess(0);
    setActiveThread(0);
  };

  return (
    <div className="space-y-6 p-6 bg-card rounded-lg border border-border">
      <div className="flex gap-4">
        <Button
          onClick={() => setIsAnimating(!isAnimating)}
          variant={isAnimating ? "secondary" : "default"}
        >
          {isAnimating ? (
            <>
              <Pause className="h-4 w-4 mr-2" /> Pause
            </>
          ) : (
            <>
              <Play className="h-4 w-4 mr-2" /> Animate
            </>
          )}
        </Button>
        <Button variant="outline" onClick={reset}>
          <RotateCcw className="h-4 w-4 mr-2" /> Reset
        </Button>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {processes.map((process, pIndex) => (
          <div
            key={process.id}
            className={cn(
              "p-4 rounded-lg border-2 transition-all duration-300",
              activeProcess === pIndex
                ? "border-primary bg-primary/5"
                : "border-border bg-muted/30"
            )}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-foreground">{process.name}</h3>
              <span
                className={cn(
                  "px-2 py-1 text-xs rounded-full",
                  process.status === "running"
                    ? "bg-success/20 text-success"
                    : process.status === "waiting"
                    ? "bg-warning/20 text-warning"
                    : "bg-destructive/20 text-destructive"
                )}
              >
                {process.status}
              </span>
            </div>

            <div className="space-y-4">
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-2">
                  Threads
                </h4>
                <div className="space-y-2">
                  {process.threads.map((thread, tIndex) => (
                    <div
                      key={thread.id}
                      className={cn(
                        "flex items-center justify-between p-2 rounded transition-all duration-300",
                        activeProcess === pIndex && activeThread === tIndex
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted/50"
                      )}
                    >
                      <span className="text-sm">{thread.name}</span>
                      <span className="text-xs opacity-70">{thread.status}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-2">
                  Memory Allocation
                </h4>
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2 bg-muted/50 rounded text-center">
                    <div className="text-xs text-muted-foreground">Heap</div>
                    <div className="font-mono text-sm text-foreground">{process.memory.heap} MB</div>
                  </div>
                  <div className="p-2 bg-muted/50 rounded text-center">
                    <div className="text-xs text-muted-foreground">Stack</div>
                    <div className="font-mono text-sm text-foreground">{process.memory.stack} MB</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2 p-3 bg-muted/50 rounded-lg">
        <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
        <span className="text-sm text-foreground">{steps[currentStep].message}</span>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="p-4 bg-muted/30 rounded-lg">
          <h4 className="font-semibold mb-2 text-foreground">Process vs Thread</h4>
          <div className="space-y-2 text-sm text-muted-foreground">
            <p><strong className="text-foreground">Process:</strong> Independent execution unit with its own memory space</p>
            <p><strong className="text-foreground">Thread:</strong> Lightweight unit within a process, shares memory with other threads</p>
          </div>
        </div>
        <div className="p-4 bg-muted/30 rounded-lg">
          <h4 className="font-semibold mb-2 text-foreground">Context Switching</h4>
          <div className="space-y-2 text-sm text-muted-foreground">
            <p><strong className="text-foreground">Thread switch:</strong> Fast, same address space</p>
            <p><strong className="text-foreground">Process switch:</strong> Slower, requires memory remapping</p>
          </div>
        </div>
      </div>
    </div>
  );
}
