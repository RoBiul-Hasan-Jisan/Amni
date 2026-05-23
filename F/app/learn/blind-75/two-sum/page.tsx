"use client";
import Link from "next/link";
import { AlertCircle, CheckCircle2, Lightbulb, Code, Clock, HardDrive, Target, Eye, Play, RefreshCw, ArrowRight, Hash, Github, Star, TrendingUp, Battery, Zap, Shield, Sparkles, Layers, GitBranch, Braces, FileCheck, Sliders, Wand2, Plus, Trash2 } from "lucide-react";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { useState, useEffect, useCallback, useMemo } from "react";

// Type definitions
interface SeenMap {
  [key: number]: number;
}

interface ResultState {
  found: boolean;
  indices?: [number, number];
  values?: [number, number];
}

interface Example {
  nums: number[];
  target: number;
}

interface TestCase {
  nums: number[];
  target: number;
  expected: [number, number];
  description: string;
}

// Flowchart Component
function TwoSumFlowchart() {
  return (
    <div className="bg-card border border-border rounded-lg p-6 mb-6">
      <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
        <GitBranch className="h-5 w-5 text-primary" />
        Algorithm Flowchart
      </h3>
      
      <div className="w-full rounded-4xl border border-border bg-background p-4 sm:p-6 lg:p-8">

  {/* Header */}
  <div className="mb-10 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">

    <div>
      <div className="inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium text-muted-foreground">
        LeetCode #1
      </div>

      <h2 className="mt-4 text-3xl font-black tracking-tight sm:text-4xl">
        Two Sum Flowchart
      </h2>

      <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground sm:text-base">
        Visual breakdown of the optimal HashMap approach with O(n) time complexity.
      </p>
    </div>

    <div className="grid grid-cols-2 gap-3 sm:flex">

      <div className="rounded-2xl border bg-card px-5 py-4">
        <p className="text-xs text-muted-foreground">
          Time
        </p>

        <p className="mt-1 text-xl font-black text-green-500">
          O(n)
        </p>
      </div>

      <div className="rounded-2xl border bg-card px-5 py-4">
        <p className="text-xs text-muted-foreground">
          Space
        </p>

        <p className="mt-1 text-xl font-black text-blue-500">
          O(n)
        </p>
      </div>

    </div>
  </div>

  {/* Flowchart */}
  <div className="relative flex flex-col items-center">

    {/* STEP 1 */}
    <div className="w-full max-w-xl rounded-3xl border bg-card p-6 shadow-sm transition-all hover:shadow-md">

      <div className="flex items-start gap-4">

        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-green-500 text-lg font-black text-white">
          1
        </div>

        <div className="flex-1">

          <p className="text-sm font-medium text-green-500">
            START
          </p>

          <h3 className="mt-1 text-xl font-bold">
            Initialize HashMap
          </h3>

          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            Create an empty HashMap to store previously visited numbers and their indices.
          </p>

          <div className="mt-5 overflow-x-auto rounded-2xl bg-muted px-4 py-3">
            <code className="font-mono text-sm">
              const seen = new Map()
            </code>
          </div>

        </div>

      </div>
    </div>

    {/* Connector */}
    <div className="flex h-16 items-center justify-center">
      <div className="h-full w-0.75 rounded-full bg-border" />
    </div>

    {/* STEP 2 */}
    <div className="w-full max-w-xl rounded-3xl border bg-card p-6 shadow-sm transition-all hover:shadow-md">

      <div className="flex items-start gap-4">

        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-blue-500 text-lg font-black text-white">
          2
        </div>

        <div className="flex-1">

          <p className="text-sm font-medium text-blue-500">
            ITERATE
          </p>

          <h3 className="mt-1 text-xl font-bold">
            Traverse Array
          </h3>

          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            Visit each number only once while checking if its complement already exists.
          </p>

          <div className="mt-5 overflow-x-auto rounded-2xl bg-muted px-4 py-3">
            <code className="font-mono text-sm">
              for (let i = 0; i &lt; nums.length; i++)
            </code>
          </div>

        </div>

      </div>
    </div>

    {/* Connector */}
    <div className="flex h-16 items-center justify-center">
      <div className="h-full w-0.75 rounded-full bg-border" />
    </div>

    {/* STEP 3 DECISION */}
    <div className="relative w-full max-w-xl overflow-hidden rounded-3xl border-2 border-amber-500/30 bg-amber-500/5 p-6">

      <div className="absolute right-0 top-0 h-32 w-32 rounded-full bg-amber-500/10 blur-3xl" />

      <div className="relative flex items-start gap-4">

        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-amber-500 text-lg font-black text-white">
          ?
        </div>

        <div className="flex-1">

          <p className="text-sm font-medium text-amber-600">
            DECISION
          </p>

          <h3 className="mt-1 text-xl font-bold">
            Does Complement Exist?
          </h3>

          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            Calculate complement and check whether it already exists inside the HashMap.
          </p>

          <div className="mt-5 overflow-x-auto rounded-2xl bg-background px-4 py-3">
            <code className="font-mono text-sm">
              complement = target - nums[i]
            </code>
          </div>

        </div>

      </div>
    </div>

    {/* Connector */}
    <div className="flex h-16 items-center justify-center">
      <div className="h-full w-0.75 rounded-full bg-border" />
    </div>

    {/* Branches */}
    <div className="grid w-full max-w-6xl gap-6 lg:grid-cols-2">

      {/* YES */}
      <div className="relative overflow-hidden rounded-3xl border-2 border-green-500/30 bg-green-500/5 p-6">

        <div className="absolute right-0 top-0 h-28 w-28 rounded-full bg-green-500/10 blur-3xl" />

        <div className="relative">

          <div className="inline-flex items-center rounded-full bg-green-500 px-3 py-1 text-xs font-bold text-white">
            YES PATH
          </div>

          <h3 className="mt-4 text-2xl font-black">
            Return Answer
          </h3>

          <p className="mt-3 text-sm leading-6 text-muted-foreground">
            Complement already exists, meaning we found the two indices.
          </p>

          <div className="mt-6 overflow-x-auto rounded-2xl bg-background px-4 py-3">
            <code className="font-mono text-sm">
              return [seen.get(complement), i]
            </code>
          </div>

        </div>
      </div>

      {/* NO */}
      <div className="relative overflow-hidden rounded-3xl border-2 border-rose-500/30 bg-rose-500/5 p-6">

        <div className="absolute right-0 top-0 h-28 w-28 rounded-full bg-rose-500/10 blur-3xl" />

        <div className="relative">

          <div className="inline-flex items-center rounded-full bg-rose-500 px-3 py-1 text-xs font-bold text-white">
            NO PATH
          </div>

          <h3 className="mt-4 text-2xl font-black">
            Store Current Number
          </h3>

          <p className="mt-3 text-sm leading-6 text-muted-foreground">
            Save current value inside the HashMap and continue traversal.
          </p>

          <div className="mt-6 overflow-x-auto rounded-2xl bg-background px-4 py-3">
            <code className="font-mono text-sm">
              seen.set(nums[i], i)
            </code>
          </div>

        </div>
      </div>

    </div>

    {/* Final Connector */}
    <div className="mt-10 flex h-16 items-center justify-center">
      <div className="h-full w-0.75 rounded-full bg-border" />
    </div>

    {/* END */}
    <div className="w-full max-w-xl rounded-3xl bg-linear-to-br from-emerald-500 to-green-600 p-6 text-white shadow-lg">

      <div className="flex items-center gap-4">

        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-white/20 text-2xl font-black">
          ✓
        </div>

        <div>

          <p className="text-sm font-medium text-white/80">
            COMPLETED
          </p>

          <h3 className="mt-1 text-2xl font-black">
            Two Sum Solved Efficiently
          </h3>

        </div>

      </div>
    </div>

  </div>
</div>
      
      <div className="mt-4 p-3 bg-primary/10 rounded-lg">
        <p className="text-sm text-foreground">
          💡 <span className="font-semibold">Flow Explanation:</span> The algorithm makes a single pass through the array. 
          For each element, it calculates the complement needed to reach the target, checks if that complement has been seen before,
          and either returns the pair or stores the current element for future lookups.
        </p>
      </div>
    </div>
  );
}

// Enhanced Interactive Two Sum Visualizer with Custom Data Support
function TwoSumVisualizer() {
  const [nums, setNums] = useState<number[]>([2, 7, 11, 15]);
  const [target, setTarget] = useState<number>(9);
  const [currentStep, setCurrentStep] = useState<number>(-1);
  const [seen, setSeen] = useState<SeenMap>({});
  const [result, setResult] = useState<ResultState | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [speed, setSpeed] = useState<number>(1000);
  const [animationFrame, setAnimationFrame] = useState<NodeJS.Timeout | null>(null);
  
  // Custom input states
  const [customNumsInput, setCustomNumsInput] = useState<string>("");
  const [customTargetInput, setCustomTargetInput] = useState<string>("");
  const [savedDataSets, setSavedDataSets] = useState<{ id: string; name: string; nums: number[]; target: number }[]>([]);
  const [selectedDataSet, setSelectedDataSet] = useState<string>("");

  const defaultExamples: Record<string, Example> = {
    "Example 1": { nums: [2, 7, 11, 15], target: 9 },
    "Example 2": { nums: [3, 2, 4], target: 6 },
    "Example 3": { nums: [3, 3], target: 6 },
    "Edge Case: Negatives": { nums: [-1, -2, -3, -4, -5], target: -8 },
    "Edge Case: Zero": { nums: [0, 4, 3, 0], target: 0 },
    "Custom": { nums: [1, 8, 10, 14], target: 9 }
  };

  // Load saved data sets from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("twoSumDataSets");
    if (saved) {
      try {
        setSavedDataSets(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to load saved data sets");
      }
    }
  }, []);

  // Save data sets to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem("twoSumDataSets", JSON.stringify(savedDataSets));
  }, [savedDataSets]);

  const loadExample = (exampleName: string) => {
    const example = defaultExamples[exampleName];
    if (example) {
      setNums([...example.nums]);
      setTarget(example.target);
      resetVisualization();
    }
  };

  const loadSavedDataSet = (dataSetId: string) => {
    const dataSet = savedDataSets.find(ds => ds.id === dataSetId);
    if (dataSet) {
      setNums([...dataSet.nums]);
      setTarget(dataSet.target);
      setSelectedDataSet(dataSetId);
      resetVisualization();
    }
  };

  const addCustomDataSet = () => {
    if (!customNumsInput.trim() || !customTargetInput.trim()) {
      alert("Please enter both numbers array and target value");
      return;
    }

    const numsArray = customNumsInput.split(",").map(n => {
      const parsed = parseInt(n.trim());
      if (isNaN(parsed)) throw new Error("Invalid number");
      return parsed;
    });
    
    const targetValue = parseInt(customTargetInput.trim());
    
    if (isNaN(targetValue)) {
      alert("Please enter a valid target number");
      return;
    }

    const newDataSet = {
      id: Date.now().toString(),
      name: `Custom: [${numsArray.join(", ")}], target=${targetValue}`,
      nums: numsArray,
      target: targetValue
    };

    setSavedDataSets(prev => [...prev, newDataSet]);
    setNums(numsArray);
    setTarget(targetValue);
    setSelectedDataSet(newDataSet.id);
    resetVisualization();
    
    // Clear inputs
    setCustomNumsInput("");
    setCustomTargetInput("");
  };

  const deleteSavedDataSet = (id: string) => {
    setSavedDataSets(prev => prev.filter(ds => ds.id !== id));
    if (selectedDataSet === id) {
      setSelectedDataSet("");
    }
  };

  const resetVisualization = () => {
    if (animationFrame) clearInterval(animationFrame);
    setCurrentStep(-1);
    setSeen({});
    setResult(null);
    setIsPlaying(false);
    setAnimationFrame(null);
  };

  const stepThrough = useCallback(() => {
    if (result) {
      resetVisualization();
      return;
    }

    const nextStep = currentStep + 1;
    if (nextStep >= nums.length) {
      setResult({ found: false });
      setIsPlaying(false);
      return;
    }

    const currentNum = nums[nextStep];
    const needed = target - currentNum;
    
    const newSeen = { ...seen };
    
    if (newSeen[needed] !== undefined) {
      setResult({ found: true, indices: [newSeen[needed], nextStep], values: [needed, currentNum] });
      setCurrentStep(nextStep);
      setIsPlaying(false);
      if (animationFrame) clearInterval(animationFrame);
    } else {
      newSeen[currentNum] = nextStep;
      setSeen(newSeen);
      setCurrentStep(nextStep);
      
      if (nextStep + 1 >= nums.length) {
        setResult({ found: false });
        setIsPlaying(false);
        if (animationFrame) clearInterval(animationFrame);
      }
    }
  }, [currentStep, nums, target, seen, result, animationFrame]);

  const playThrough = useCallback(() => {
    if (result) {
      resetVisualization();
    }
    if (animationFrame) clearInterval(animationFrame);
    setIsPlaying(true);
    
    const interval = setInterval(() => {
      setCurrentStep(prev => {
        const next = prev + 1;
        if (next >= nums.length) {
          clearInterval(interval);
          setIsPlaying(false);
          setResult({ found: false });
          return prev;
        }
        
        const currentNum = nums[next];
        const needed = target - currentNum;
        
        setSeen(prevSeen => {
          if (prevSeen[needed] !== undefined) {
            setResult({ found: true, indices: [prevSeen[needed], next], values: [needed, currentNum] });
            clearInterval(interval);
            setIsPlaying(false);
            return prevSeen;
          } else {
            return { ...prevSeen, [currentNum]: next };
          }
        });
        
        return next;
      });
    }, speed);
    
    setAnimationFrame(interval);
  }, [nums, target, speed, result, animationFrame]);

  useEffect(() => {
    return () => {
      if (animationFrame) clearInterval(animationFrame);
    };
  }, [animationFrame]);

  const getCellColor = (index: number, value: number): string => {
    if (result?.found && result.indices?.includes(index)) {
      return "bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg ring-2 ring-green-300 dark:ring-green-700";
    }
    if (currentStep >= index) {
      return seen[value] !== undefined || (result?.found && result.indices?.[0] === index) 
        ? "bg-gradient-to-r from-yellow-500 to-amber-500 text-white shadow-md" 
        : "bg-gradient-to-r from-blue-500 to-indigo-500 text-white shadow-md";
    }
    return "bg-gray-300 dark:bg-gray-600 hover:bg-gray-400 dark:hover:bg-gray-500 transition-colors";
  };

  return (
    <div className="bg-card border border-border rounded-lg p-6 mb-6 shadow-lg">
      <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
        <Eye className="h-5 w-5 text-primary" />
        Interactive Two Sum Visualizer
      </h3>
      
      {/* Custom Data Input Section */}
      <div className="mb-6 p-4 bg-linear-to-r from-primary/5 to-primary/10 rounded-lg border border-primary/20">
        <h4 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <Plus className="h-4 w-4 text-primary" />
          Add Your Own Data
        </h4>
        <div className="grid md:grid-cols-2 gap-4 mb-3">
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Numbers Array (comma-separated)</label>
            <input
              type="text"
              value={customNumsInput}
              onChange={(e) => setCustomNumsInput(e.target.value)}
              placeholder="e.g., 5, 10, 15, 20"
              className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Target Value</label>
            <input
              type="text"
              value={customTargetInput}
              onChange={(e) => setCustomTargetInput(e.target.value)}
              placeholder="e.g., 25"
              className="w-full px-3 py-2 bg-background border border-border rounded-lg text-foreground font-mono text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            />
          </div>
        </div>
        <button
          onClick={addCustomDataSet}
          className="w-full px-4 py-2 bg-primary/20 text-primary rounded-lg hover:bg-primary/30 transition-all flex items-center justify-center gap-2 text-sm font-medium"
        >
          <Plus className="h-4 w-4" />
          Add & Visualize
        </button>
      </div>

      {/* Saved Data Sets Section */}
      {savedDataSets.length > 0 && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
            <HardDrive className="h-4 w-4 text-primary" />
            Saved Data Sets
          </h4>
          <div className="flex flex-wrap gap-2">
            {savedDataSets.map(ds => (
              <div key={ds.id} className="flex items-center gap-1 bg-muted rounded-lg overflow-hidden">
                <button
                  onClick={() => loadSavedDataSet(ds.id)}
                  className={`px-3 py-1.5 text-xs transition-colors ${
                    selectedDataSet === ds.id 
                      ? "bg-primary text-primary-foreground" 
                      : "text-foreground hover:bg-primary/20"
                  }`}
                >
                  {ds.name.length > 30 ? ds.name.slice(0, 27) + "..." : ds.name}
                </button>
                <button
                  onClick={() => deleteSavedDataSet(ds.id)}
                  className="px-2 py-1.5 text-red-500 hover:bg-red-500/10 transition-colors"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Controls */}
      <div className="flex flex-wrap gap-3 mb-6">
        <select 
          onChange={(e) => loadExample(e.target.value)}
          className="px-3 py-2 bg-muted border border-border rounded-lg text-foreground text-sm cursor-pointer hover:bg-muted/80 transition-colors"
          defaultValue=""
        >
          <option value="" disabled>📋 Load Example</option>
          <option value="Example 1">Example 1: [2,7,11,15], target=9</option>
          <option value="Example 2">Example 2: [3,2,4], target=6</option>
          <option value="Example 3">Example 3: [3,3], target=6</option>
          <option value="Edge Case: Negatives">Edge: Negatives [-1,-2,-3,-4,-5], target=-8</option>
          <option value="Edge Case: Zero">Edge: Zeros [0,4,3,0], target=0</option>
          <option value="Custom">Custom: [1,8,10,14], target=9</option>
        </select>
        
        <div className="flex items-center gap-2 px-3 py-2 bg-muted rounded-lg">
          <Zap className="h-4 w-4 text-primary" />
          <input
            type="range"
            min="200"
            max="2000"
            step="100"
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="w-24"
          />
          <span className="text-xs text-muted-foreground">{speed}ms</span>
        </div>
        
        <button
          onClick={playThrough}
          disabled={isPlaying || !!result}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 flex items-center gap-2 transition-all transform hover:scale-105"
        >
          <Play className="h-4 w-4" /> Auto Play
        </button>
        
        <button
          onClick={stepThrough}
          disabled={isPlaying || result?.found === true}
          className="px-4 py-2 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 flex items-center gap-2 transition-all"
        >
          <ArrowRight className="h-4 w-4" /> Step
        </button>
        
        <button
          onClick={resetVisualization}
          className="px-4 py-2 bg-muted text-muted-foreground rounded-lg hover:bg-muted/80 flex items-center gap-2 transition-all"
        >
          <RefreshCw className="h-4 w-4" /> Reset
        </button>
      </div>

      {/* Array Visualization */}
      <div className="mb-6">
        <div className="text-sm text-muted-foreground mb-2">Array nums:</div>
        <div className="flex gap-3 flex-wrap justify-center">
          {nums.map((num, idx) => (
            <div key={idx} className="text-center transform transition-all hover:scale-105">
              <div className={`w-16 h-16 rounded-xl flex items-center justify-center font-bold text-lg transition-all ${getCellColor(idx, num)}`}>
                {num}
              </div>
              <div className="text-xs text-muted-foreground mt-1 font-mono">index {idx}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Current Step Info */}
      <div className="bg-linear-to-r from-muted/30 to-muted/10 rounded-lg p-4 mb-4 border border-border">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <span className="text-sm text-muted-foreground">Target:</span>
            <span className="ml-2 font-mono font-bold text-primary text-lg">{target}</span>
          </div>
          {currentStep >= 0 && currentStep < nums.length && !result && (
            <div className="animate-pulse">
              <span className="text-sm text-muted-foreground">🔍 Checking index {currentStep}:</span>
              <span className="ml-2 font-mono font-semibold">nums[{currentStep}] = {nums[currentStep]}</span>
              <span className="ml-2 font-mono">→ need {target - nums[currentStep]}</span>
            </div>
          )}
          {result?.found && result.values && (
            <div className="text-green-600 dark:text-green-400 font-semibold animate-bounce">
              ✓ Found! {result.values[0]} + {result.values[1]} = {target}
            </div>
          )}
          {result?.found === false && (
            <div className="text-red-600 dark:text-red-400 font-semibold">
              ✗ No solution found
            </div>
          )}
        </div>
      </div>

      {/* Hash Map Visualization */}
      <div>
        <div className="text-sm text-muted-foreground mb-2 flex items-center gap-2">
          <Hash className="h-4 w-4 text-primary" />
          Hash Map (seen numbers → index):
        </div>
        <div className="bg-muted/20 rounded-lg p-3 border border-border">
          {Object.keys(seen).length === 0 ? (
            <div className="text-muted-foreground text-sm italic">📭 Empty map - no numbers processed yet</div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {Object.entries(seen).map(([num, idx]) => (
                <div key={num} className="bg-primary/10 rounded px-3 py-2 text-sm border border-primary/20 hover:bg-primary/20 transition-colors">
                  <span className="font-mono font-semibold text-primary">{num}</span>
                  <span className="text-muted-foreground"> → index {idx}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Algorithm Performance Dashboard
function PerformanceDashboard() {
  const [inputSize, setInputSize] = useState<number>(10000);
  
  const bruteForceOps = useMemo(() => inputSize * inputSize, [inputSize]);
  const hashMapOps = useMemo(() => inputSize, [inputSize]);
  const speedup = useMemo(() => bruteForceOps / hashMapOps, [bruteForceOps, hashMapOps]);
  
  return (
    <div className="bg-card border border-border rounded-lg p-6 mb-6">
      <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
        <TrendingUp className="h-5 w-5 text-primary" />
        Performance Dashboard
      </h3>
      
      <div className="mb-4">
        <label className="text-sm text-muted-foreground block mb-2">
          Input Size (n): {inputSize.toLocaleString()} elements
        </label>
        <input
          type="range"
          min="100"
          max="100000"
          step="100"
          value={inputSize}
          onChange={(e) => setInputSize(Number(e.target.value))}
          className="w-full"
        />
      </div>
      
      <div className="grid md:grid-cols-3 gap-4 mb-4">
        <div className="bg-red-50 dark:bg-red-950/20 rounded-lg p-3">
          <p className="text-xs text-muted-foreground">Brute Force O(n²)</p>
          <p className="text-lg font-bold text-red-600 dark:text-red-400">{bruteForceOps.toLocaleString()}</p>
          <p className="text-xs text-muted-foreground">operations</p>
        </div>
        <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-3">
          <p className="text-xs text-muted-foreground">Hash Map O(n)</p>
          <p className="text-lg font-bold text-green-600 dark:text-green-400">{hashMapOps.toLocaleString()}</p>
          <p className="text-xs text-muted-foreground">operations</p>
        </div>
        <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-3">
          <p className="text-xs text-muted-foreground">Speed Improvement</p>
          <p className="text-lg font-bold text-blue-600 dark:text-blue-400">{speedup.toLocaleString()}x</p>
          <p className="text-xs text-muted-foreground">faster!</p>
        </div>
      </div>
      
      <div className="relative pt-1">
        <div className="flex mb-2 items-center justify-between">
          <div>
            <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-red-600 bg-red-200 dark:bg-red-900/30">
              Brute Force
            </span>
          </div>
          <div className="text-right">
            <span className="text-xs font-semibold inline-block text-red-600">
              100%
            </span>
          </div>
        </div>
        <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-red-200 dark:bg-red-900/30">
          <div style={{ width: "100%" }} className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-red-500"></div>
        </div>
        
        <div className="flex mb-2 items-center justify-between">
          <div>
            <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-green-600 bg-green-200 dark:bg-green-900/30">
              Hash Map
            </span>
          </div>
          <div className="text-right">
            <span className="text-xs font-semibold inline-block text-green-600">
              {(hashMapOps / bruteForceOps * 100).toFixed(2)}%
            </span>
          </div>
        </div>
        <div className="overflow-hidden h-2 mb-4 text-xs flex rounded bg-green-200 dark:bg-green-900/30">
          <div style={{ width: `${(hashMapOps / bruteForceOps * 100)}%` }} className="shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center bg-green-500"></div>
        </div>
      </div>
    </div>
  );
}

// Code Playground - Test different inputs
function CodePlayground() {
  const [userNums, setUserNums] = useState<string>("2,7,11,15");
  const [userTarget, setUserTarget] = useState<string>("9");
  const [output, setOutput] = useState<string>("");
  
  const testSolution = () => {
    try {
      const nums = userNums.split(",").map(n => parseInt(n.trim()));
      const target = parseInt(userTarget);
      
      const seen: Record<number, number> = {};
      let result: number[] = [];
      
      for (let i = 0; i < nums.length; i++) {
        const complement = target - nums[i];
        if (seen[complement] !== undefined) {
          result = [seen[complement], i];
          break;
        }
        seen[nums[i]] = i;
      }
      
      if (result.length === 2) {
        setOutput(`✓ Found! Indices: [${result[0]}, ${result[1]}] (${nums[result[0]]} + ${nums[result[1]]} = ${target})`);
      } else {
        setOutput(`✗ No solution found for target ${target} in [${nums.join(", ")}]`);
      }
    } catch (error) {
      setOutput(" Invalid input format. Please use comma-separated numbers (e.g., 2,7,11,15)");
    }
  };
  
  return (
    <div className="bg-card border border-border rounded-lg p-6 mb-6">
      <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
        <Wand2 className="h-5 w-5 text-primary" />
        Test Your Understanding
      </h3>
      
      <div className="space-y-4">
        <div>
          <label className="text-sm text-muted-foreground block mb-2">nums =</label>
          <input
            type="text"
            value={userNums}
            onChange={(e) => setUserNums(e.target.value)}
            className="w-full px-3 py-2 bg-muted border border-border rounded-lg text-foreground font-mono"
            placeholder="2,7,11,15"
          />
        </div>
        
        <div>
          <label className="text-sm text-muted-foreground block mb-2">target =</label>
          <input
            type="text"
            value={userTarget}
            onChange={(e) => setUserTarget(e.target.value)}
            className="w-full px-3 py-2 bg-muted border border-border rounded-lg text-foreground font-mono"
            placeholder="9"
          />
        </div>
        
        <button
          onClick={testSolution}
          className="w-full px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-all"
        >
          Run Solution
        </button>
        
        {output && (
          <div className={`p-3 rounded-lg ${output.includes("✓") ? "bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-800" : "bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800"}`}>
            <p className={`font-mono text-sm ${output.includes("✓") ? "text-green-700 dark:text-green-300" : "text-red-700 dark:text-red-300"}`}>
              {output}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// Additional Solutions (Multiple Approaches)
function MultipleSolutions() {
  const [activeTab, setActiveTab] = useState<"brute" | "twoPass" | "onePass">("onePass");
  
  const solutions = {
    brute: {
      name: "Brute Force",
      time: "O(n²)",
      space: "O(1)",
      code: `def twoSum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []`,
      pros: "Simple, no extra space",
      cons: "Very slow for large inputs"
    },
    twoPass: {
      name: "Two-Pass Hash Map",
      time: "O(n)",
      space: "O(n)",
      code: `def twoSum_twoPass(nums, target):
    # First pass: build hash map
    num_map = {}
    for i, num in enumerate(nums):
        num_map[num] = i
    
    # Second pass: find complement
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map and num_map[complement] != i:
            return [i, num_map[complement]]
    return []`,
      pros: "O(n) time, easy to understand",
      cons: "Two passes through array"
    },
    onePass: {
      name: "One-Pass Hash Map (Optimal)",
      time: "O(n)",
      space: "O(n)",
      code: `def twoSum_onePass(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []`,
      pros: "O(n) time, single pass",
      cons: "Slightly harder to understand initially"
    }
  };
  
  const current = solutions[activeTab];
  
  return (
    <div className="bg-card border border-border rounded-lg p-6 mb-6">
      <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
        <Layers className="h-5 w-5 text-primary" />
        Multiple Approaches Comparison
      </h3>
      
      <div className="flex gap-2 mb-4 border-b border-border">
        <button
          onClick={() => setActiveTab("brute")}
          className={`px-4 py-2 transition-all ${activeTab === "brute" ? "border-b-2 border-primary text-primary" : "text-muted-foreground hover:text-foreground"}`}
        >
           Brute Force
        </button>
        <button
          onClick={() => setActiveTab("twoPass")}
          className={`px-4 py-2 transition-all ${activeTab === "twoPass" ? "border-b-2 border-primary text-primary" : "text-muted-foreground hover:text-foreground"}`}
        >
           Two-Pass Hash
        </button>
        <button
          onClick={() => setActiveTab("onePass")}
          className={`px-4 py-2 transition-all ${activeTab === "onePass" ? "border-b-2 border-primary text-primary" : "text-muted-foreground hover:text-foreground"}`}
        >
           One-Pass Hash
        </button>
      </div>
      
      <div className="grid md:grid-cols-2 gap-4 mb-4">
        <div>
          <p className="text-sm text-muted-foreground mb-1">Time Complexity</p>
          <p className="font-mono text-lg font-bold text-primary">{current.time}</p>
        </div>
        <div>
          <p className="text-sm text-muted-foreground mb-1">Space Complexity</p>
          <p className="font-mono text-lg font-bold text-primary">{current.space}</p>
        </div>
      </div>
      
      <pre className="bg-muted/50 p-4 rounded-lg text-primary font-mono text-sm overflow-x-auto mb-3">
        {current.code}
      </pre>
      
      <div className="grid md:grid-cols-2 gap-3">
        <div className="bg-green-50 dark:bg-green-950/20 rounded-lg p-3">
          <p className="text-xs font-semibold text-green-700 dark:text-green-300 mb-1">✓ Pros</p>
          <p className="text-sm text-muted-foreground">{current.pros}</p>
        </div>
        <div className="bg-red-50 dark:bg-red-950/20 rounded-lg p-3">
          <p className="text-xs font-semibold text-red-700 dark:text-red-300 mb-1">✗ Cons</p>
          <p className="text-sm text-muted-foreground">{current.cons}</p>
        </div>
      </div>
    </div>
  );
}

// Interview Tips & Tricks
function InterviewTips() {
  const tips = [
    "Always clarify if there's exactly one solution or multiple",
    "Ask about handling duplicate numbers",
    "Discuss the possibility of negative numbers",
    "Mention the space-time tradeoff upfront",
    "Start with brute force, then optimize",
    "Explain why hash map gives O(1) lookups"
  ];
  
  return (
    <div className="bg-linear-to-r from-yellow-50 to-amber-50 dark:from-yellow-950/20 dark:to-amber-950/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6 mb-6">
      <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
        <Star className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />
        Interview Pro Tips
      </h3>
      
      <div className="grid md:grid-cols-2 gap-3">
        {tips.map((tip, idx) => (
          <div key={idx} className="flex items-start gap-2">
            <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400 mt-0.5 shrink-0" />
            <span className="text-sm text-foreground">{tip}</span>
          </div>
        ))}
      </div>
      
      <div className="mt-4 p-3 bg-primary/10 rounded-lg">
        <p className="text-sm font-semibold text-primary mb-1"> Common Follow-up Questions:</p>
        <p className="text-sm text-muted-foreground">"What if the array is sorted?" → Use two-pointer technique (O(n) time, O(1) space)</p>
        <p className="text-sm text-muted-foreground mt-1">"What if there are multiple solutions?" → Return all pairs or the first one?</p>
      </div>
    </div>
  );
}

// Complexity Analysis with visualizations (enhanced)
function EnhancedComplexityAnalysis() {
  return (
    <section>
      <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2 flex items-center gap-2">
        <TrendingUp className="h-5 w-5 text-primary" />
        Complexity Analysis
      </h2>
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="h-5 w-5 text-primary" />
            <h3 className="font-semibold text-foreground">Time Complexity: O(n)</h3>
          </div>
          <ul className="list-disc list-inside text-muted-foreground text-sm ml-4 space-y-2">
            <li>Single pass through array: n iterations</li>
            <li>Hash map lookup (average): O(1) per operation</li>
            <li className="font-semibold text-primary">Total: O(n) - Linear time!</li>
          </ul>
          <div className="mt-3 p-2 bg-green-50 dark:bg-green-950/20 rounded">
            <p className="text-xs text-green-700 dark:text-green-300">✓ For n=10,000 elements → ~10,000 operations</p>
            <p className="text-xs text-green-700 dark:text-green-300 mt-1">✓ For n=1,000,000 elements → ~1,000,000 operations (still fast!)</p>
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <HardDrive className="h-5 w-5 text-primary" />
            <h3 className="font-semibold text-foreground">Space Complexity: O(n)</h3>
          </div>
          <ul className="list-disc list-inside text-muted-foreground text-sm ml-4 space-y-2">
            <li>Hash map stores at most n-1 elements</li>
            <li>Each entry: integer key + integer value</li>
            <li className="font-semibold text-primary">Total: O(n) - Linear space</li>
          </ul>
          <div className="mt-3 p-2 bg-yellow-50 dark:bg-yellow-950/20 rounded">
            <p className="text-xs text-yellow-700 dark:text-yellow-300">Memory: ~8 bytes × n for integers + hash table overhead</p>
            <p className="text-xs text-yellow-700 dark:text-yellow-300 mt-1">For n=1,000,000 → ~16 MB + overhead (acceptable)</p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function TwoSumPage() {
  const result = getSubtopicBySlug("blind-75", "two-sum");
  if (!result) return null;

  const { topic, subtopic } = result;

  const examples = [
    { nums: [2, 7, 11, 15], target: 9, answer: [0, 1] as [number, number], explanation: "2 + 7 = 9" },
    { nums: [3, 2, 4], target: 6, answer: [1, 2] as [number, number], explanation: "2 + 4 = 6" },
    { nums: [3, 3], target: 6, answer: [0, 1] as [number, number], explanation: "3 + 3 = 6" }
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Problem Header with badges */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4 flex-wrap">
            <span className="bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 px-3 py-1 rounded-full text-sm font-semibold flex items-center gap-1">
              <Battery className="h-3 w-3" /> Easy
            </span>
            <span className="bg-primary/10 text-primary px-3 py-1 rounded-full text-sm font-semibold flex items-center gap-1">
              <Layers className="h-3 w-3" /> Array • Hash Table
            </span>
            
           
          </div>
         
          <h1 className="text-4xl font-bold text-foreground mb-3 flex items-center gap-3">
            Two Sum
            <a 
              href="https://leetcode.com/problems/two-sum/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-primary transition-colors"
            >
             <Target className="h-5 w-5 text-primary shrink-0 mt-0.5" />
            </a>
          </h1>
          <p className="text-muted-foreground text-lg">
            Given an integer array <code className="bg-muted px-2 py-0.5 rounded font-mono">nums</code> and an integer <code className="bg-muted px-2 py-0.5 rounded font-mono">target</code>, return the indices of two different elements such that their sum is <code className="bg-muted px-2 py-0.5 rounded font-mono">target</code>.
          </p>
        </div>

        {/* Flowchart - NEW COMPONENT ADDED HERE */}
        <TwoSumFlowchart />

        {/* Interactive Visualizer */}
        <TwoSumVisualizer />
        
        {/* Performance Dashboard */}
        <PerformanceDashboard />
        
        {/* Multiple Solutions */}
        <MultipleSolutions />
        
        {/* Code Playground */}
        <CodePlayground />
        
        {/* Interview Tips */}
        <InterviewTips />
        
        {/* Algorithm Walkthrough */}
        <AlgorithmWalkthrough />

        {/* Problem Link */}
        <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
          <div className="flex gap-3">
            <Target className="h-5 w-5 text-primary shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-foreground">Problem Link</p>
              <a 
                href="https://leetcode.com/problems/two-sum/" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-primary hover:underline break-all flex items-center gap-1"
              >
                https://leetcode.com/problems/two-sum/
                <ArrowRight className="h-3 w-3" />
              </a>
            </div>
          </div>
        </div>

        {/* Visual Examples with highlighting */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Examples with Visualization
          </h2>
          
          <div className="space-y-4">
            {examples.map((ex, idx) => (
              <div key={idx} className="bg-card border border-border rounded-lg p-4 hover:shadow-lg transition-shadow">
                <p className="font-semibold text-foreground mb-2">Example {idx + 1}</p>
                <pre className="bg-muted/50 p-3 rounded text-primary font-mono text-sm overflow-x-auto mb-3">
                  {`Input: nums = [${ex.nums.join(", ")}], target = ${ex.target}
Output: [${ex.answer.join(", ")}]
Explanation: ${ex.explanation}`}
                </pre>
                <div className="flex justify-around text-center">
                  {ex.nums.map((num, i) => (
                    <div key={i} className="text-center">
                      <div className={`w-12 h-12 rounded-lg flex items-center justify-center text-white font-bold text-lg transition-all ${
                        ex.answer.includes(i) ? "bg-linear-to-r from-green-500 to-emerald-500 ring-4 ring-green-300 dark:ring-green-700 shadow-lg" : "bg-primary/10 text-primary"
                      }`}>
                        {num}
                      </div>
                      <div className="text-muted-foreground text-sm mt-1 font-mono">{i}</div>
                    </div>
                  ))}
                </div>
                {ex.answer && (
                  <div className="mt-3 text-center text-sm text-green-600 dark:text-green-400 animate-pulse">
                    ✓ {ex.nums[ex.answer[0]]} + {ex.nums[ex.answer[1]]} = {ex.target}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* Solutions Comparison with visual bars */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Solutions Comparison
          </h2>
          
          <div className="overflow-x-auto">
           <table className="min-w-200">
               <thead>
    <tr className="border-b border-border bg-muted/30">
      <th className="text-left p-3 font-semibold text-foreground">
        Approach
      </th>
      <th className="text-left p-3 font-semibold text-foreground">
        Time
      </th>
      <th className="text-left p-3 font-semibold text-foreground">
        Space
      </th>
      <th className="text-left p-3 font-semibold text-foreground">
        Visual
      </th>
    </tr>
  </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">Brute Force</td>
                  <td className="p-3 font-mono">O(n²)</td>
                  <td className="p-3 font-mono">O(1)</td>
                  <td className="p-3">
                    <div className="w-32 bg-red-200 rounded h-2">
                      <div className="bg-red-500 h-2 rounded" style={{ width: "100%" }}></div>
                    </div>
                  </td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">Two-pass Hash</td>
                  <td className="p-3 font-mono">O(n)</td>
                  <td className="p-3 font-mono">O(n)</td>
                  <td className="p-3">
                    <div className="w-32 bg-green-200 rounded h-2">
                      <div className="bg-green-500 h-2 rounded" style={{ width: "10%" }}></div>
                    </div>
                   </td>
                 </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">One-pass Hash </td>
                  <td className="p-3 font-mono">O(n)</td>
                  <td className="p-3 font-mono">O(n)</td>
                  <td className="p-3">
                    <div className="w-32 bg-green-200 rounded h-2">
                      <div className="bg-green-500 h-2 rounded" style={{ width: "10%" }}></div>
                    </div>
                    <span className="text-primary font-semibold ml-2">Optimal</span>
                   </td>
                 </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Solution Idea with visual flow */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Solution Idea
          </h2>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="text-foreground font-semibold mb-2">The Key Insight:</p>
                <p className="text-muted-foreground mb-3">Instead of checking every pair (which is slow), we can use a hash map to remember numbers we've seen.</p>
                
                <div className="bg-muted/30 rounded-lg p-3 mb-3">
                  <p className="font-mono text-sm mb-2">For each number at index i:</p>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold">1</div>
                      <span>Calculate <code className="bg-muted px-2 py-0.5 rounded font-mono">complement = target - nums[i]</code></span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold">2</div>
                      <span>Check if <code className="bg-muted px-2 py-0.5 rounded font-mono">complement</code> exists in our hash map</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold">3</div>
                      <span>If yes → return [map[complement], i] (we found our pair!)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold">4</div>
                      <span>If no → store <code className="bg-muted px-2 py-0.5 rounded font-mono">nums[i] → i</code> in map for future checks</span>
                    </div>
                  </div>
                </div>
                
                <p className="text-muted-foreground">This works in a single pass because when we find the complement, it must have been from an earlier index!</p>
              </div>
            </div>
          </div>
        </section>

        {/* Code Section with syntax highlighting */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Code Implementation
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <div className="bg-muted px-4 py-2 border-b border-border">
              <div className="flex items-center gap-2">
                <Code className="h-4 w-4 text-primary" />
                <span className="text-foreground font-mono text-sm">Python 3 Solution (Optimal)</span>
              </div>
            </div>
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`from typing import List
 
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # Dictionary to store number -> index
        seen = {}
        
        # Enumerate gives us both index and value
        for i, num in enumerate(nums):
            # What number do we need to reach target?
            complement = target - num
            
            # Have we seen this number before?
            if complement in seen:
                # Found it! Return the indices
                return [seen[complement], i]
            
            # Store current number for future checks
            seen[num] = i
        
        # Problem guarantees a solution exists
        return []`}
            </pre>
          </div>
          
          {/* Code explanation */}
          <div className="bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mt-3">
            <p className="text-sm text-foreground">
               <span className="font-semibold">Key Line:</span> <code className="bg-blue-100 dark:bg-blue-900/50 px-2 py-0.5 rounded font-mono">if complement in seen:</code> - This O(1) lookup is what makes the solution fast!
            </p>
          </div>
        </section>

        {/* Enhanced Complexity Analysis */}
        <EnhancedComplexityAnalysis />

        {/* Common Mistakes with visual examples */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Common Mistakes & How to Fix Them
          </h2>
          <div className="space-y-3">
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="font-semibold text-foreground mb-2"> Using same element twice</p>
                  <pre className="bg-muted/50 p-2 rounded text-sm mb-2 overflow-x-auto"># Wrong: This could use the same element twice!
if num in seen:  # Checking before calculating complement
    return [seen[num], i]</pre>
                  <div className="bg-green-50 dark:bg-green-950/20 rounded p-2">
                    <p className="text-sm text-green-700 dark:text-green-300">✓ Fix: Always calculate complement first, then check</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="font-semibold text-foreground mb-2"> Using list for lookups (inefficient)</p>
                  <pre className="bg-muted/50 p-2 rounded text-sm mb-2 overflow-x-auto"># Wrong: O(n) lookup makes total O(n²)
for i in range(len(nums)):
    for j in range(i+1, len(nums)):
        if nums[i] + nums[j] == target:
            return [i, j]</pre>
                  <div className="bg-green-50 dark:bg-green-950/20 rounded p-2">
                    <p className="text-sm text-green-700 dark:text-green-300">✓ Fix: Use hash map for O(1) lookups</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Practice Tips */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
             Practice Tips
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground mb-2">Before Writing Code:</h3>
              <ul className="list-disc list-inside text-muted-foreground text-sm space-y-1">
                <li>Verify problem assumptions (exactly one solution)</li>
                <li>Consider edge cases (negative numbers, duplicates)</li>
                <li>Think about space-time tradeoffs</li>
                <li>Mention hash map approach early</li>
              </ul>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <h3 className="font-semibold text-foreground mb-2">During Interview:</h3>
              <ul className="list-disc list-inside text-muted-foreground text-sm space-y-1">
                <li>Start with brute force, then optimize</li>
                <li>Explain why hash map works</li>
                <li>Walk through an example step-by-step</li>
                <li>Discuss space-time tradeoffs</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Quick Reference Card */}
        <section>
          <div className="bg-linear-to-r from-primary/10 to-primary/5 rounded-lg p-6 border border-primary/20">
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle2 className="h-5 w-5 text-primary" />
              <h3 className="font-semibold text-foreground">Interview Quick Reference</h3>
            </div>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div>
                <p className="font-semibold text-foreground">Pattern:</p>
                <p className="text-muted-foreground">Hash Map for O(1) lookups</p>
              </div>
              <div>
                <p className="font-semibold text-foreground">Key Insight:</p>
                <p className="text-muted-foreground">Complement = target - current</p>
              </div>
              <div>
                <p className="font-semibold text-foreground">Time:</p>
                <p className="text-muted-foreground">O(n) - Single pass</p>
              </div>
              <div>
                <p className="font-semibold text-foreground">Space:</p>
                <p className="text-muted-foreground">O(n) - Hash map storage</p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </TopicContent>
  );
}

// Keep the AlgorithmWalkthrough component from the original
function AlgorithmWalkthrough() {
  const steps = [
    {
      title: "Step 1: Initialize empty hash map",
      description: "Create a dictionary to store numbers we've seen and their indices.",
      code: "seen = {}  # value → index"
    },
    {
      title: "Step 2: Iterate through array",
      description: "For each element, calculate what number we need to reach the target.",
      code: "needed = target - nums[i]"
    },
    {
      title: "Step 3: Check if needed number is in map",
      description: "If we've seen the complement before, we found our pair!",
      code: "if needed in seen: return [seen[needed], i]"
    },
    {
      title: "Step 4: Store current number",
      description: "If not found, add current number to map for future checks.",
      code: "seen[nums[i]] = i"
    }
  ];

  return (
    <div className="bg-card border border-border rounded-lg p-6 mb-6">
      <h3 className="text-lg font-semibold text-foreground mb-4">How the Hash Map Solution Works</h3>
      <div className="space-y-4">
        {steps.map((step, idx) => (
          <div key={idx} className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-primary/10 text-primary flex items-center justify-center font-bold shrink-0">
              {idx + 1}
            </div>
            <div>
              <div className="font-semibold text-foreground mb-1">{step.title}</div>
              <p className="text-muted-foreground text-sm mb-2">{step.description}</p>
              <pre className="bg-muted/50 p-2 rounded text-primary font-mono text-xs">{step.code}</pre>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}