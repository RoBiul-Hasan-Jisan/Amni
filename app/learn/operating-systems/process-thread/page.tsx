import { TopicContent } from "@/components/topic-content";
import { ProcessThreadVisualizer } from "@/components/visualizations/process-thread-visualizer";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb, ArrowRight } from "lucide-react";

export default function ProcessThreadPage() {
  const result = getSubtopicBySlug("operating-systems", "process-thread");
  if (!result) return null;

  const { topic, subtopic } = result;

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "Which of the following is true about threads within the same process?",
      options: [
        "They have separate memory spaces",
        "They share the same memory space",
        "They cannot communicate with each other",
        "They run on different computers"
      ],
      correctAnswer: 1,
      explanation: "Threads within the same process share the same memory space (heap, global variables), which makes communication between them fast but requires careful synchronization.",
    },
    {
      id: 2,
      question: "What is the main advantage of using threads over processes?",
      options: [
        "Better security isolation",
        "Lower overhead for creation and context switching",
        "Completely independent execution",
        "Separate memory allocation"
      ],
      correctAnswer: 1,
      explanation: "Threads have lower overhead because they share resources with their parent process. Creating a thread is faster than creating a process, and context switching between threads is quicker.",
    },
    {
      id: 3,
      question: "Which resource is NOT shared between threads of the same process?",
      options: [
        "Heap memory",
        "Global variables",
        "Stack",
        "Open files"
      ],
      correctAnswer: 2,
      explanation: "Each thread has its own stack for local variables and function call information. This allows threads to execute different functions independently.",
    },
    {
      id: 4,
      question: "What happens during a context switch from Process A to Process B?",
      options: [
        "Only register values are saved",
        "Memory mapping and page tables must be switched",
        "Nothing needs to be changed",
        "Only the program counter is updated"
      ],
      correctAnswer: 1,
      explanation: "A process context switch requires saving the entire process state, switching memory mappings (page tables), and loading the new process state. This is why process switches are slower than thread switches.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Introduction */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Process vs Thread: Understanding the Difference</h2>
          <p className="text-muted-foreground mb-4">
            In operating systems, <strong className="text-foreground">processes</strong> and <strong className="text-foreground">threads</strong> are 
            fundamental concepts for program execution. Understanding the difference between them is crucial 
            for writing efficient software and acing technical interviews.
          </p>
          
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 my-6">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Real-world Analogy</h4>
                <p className="text-sm text-muted-foreground">
                  Think of a <strong>process</strong> as a house with its own address, utilities, and resources. 
                  <strong> Threads</strong> are like family members living in that house - they share the kitchen, 
                  living room (shared memory), but each has their own bedroom (stack).
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Definitions */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">What is a Process?</h2>
          <div className="p-4 bg-card border border-border rounded-lg mb-6">
            <p className="text-muted-foreground mb-4">
              A <strong className="text-foreground">process</strong> is an independent program in execution. It has its own:
            </p>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary mt-1 shrink-0" />
                <span><strong className="text-foreground">Memory Space:</strong> Code, data, heap, and stack segments</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary mt-1 shrink-0" />
                <span><strong className="text-foreground">Resources:</strong> File handles, sockets, device access</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary mt-1 shrink-0" />
                <span><strong className="text-foreground">Process ID (PID):</strong> Unique identifier assigned by the OS</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary mt-1 shrink-0" />
                <span><strong className="text-foreground">Security Context:</strong> Permissions and access rights</span>
              </li>
            </ul>
          </div>

          <h2 className="text-2xl font-bold mb-4 text-foreground">What is a Thread?</h2>
          <div className="p-4 bg-card border border-border rounded-lg">
            <p className="text-muted-foreground mb-4">
              A <strong className="text-foreground">thread</strong> is the smallest unit of execution within a process. Multiple threads share:
            </p>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary mt-1 shrink-0" />
                <span><strong className="text-foreground">Code Section:</strong> Same program instructions</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary mt-1 shrink-0" />
                <span><strong className="text-foreground">Data Section:</strong> Global variables and heap memory</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary mt-1 shrink-0" />
                <span><strong className="text-foreground">Resources:</strong> Open files and signals</span>
              </li>
            </ul>
            <p className="text-muted-foreground mt-4">
              But each thread has its own: <strong className="text-foreground">Stack</strong>, <strong className="text-foreground">Registers</strong>, and <strong className="text-foreground">Program Counter</strong>
            </p>
          </div>
        </section>

        {/* Interactive Visualization */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Interactive Visualization</h2>
          <p className="text-muted-foreground mb-4">
            Watch how processes and threads are scheduled and how context switching works. 
            Click Animate to see the CPU switching between different execution units.
          </p>
          <ProcessThreadVisualizer />
        </section>

        {/* Comparison Table */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Process vs Thread Comparison</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-muted">
                  <th className="border border-border p-3 text-left text-foreground">Aspect</th>
                  <th className="border border-border p-3 text-left text-foreground">Process</th>
                  <th className="border border-border p-3 text-left text-foreground">Thread</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border border-border p-3 font-medium text-foreground">Memory</td>
                  <td className="border border-border p-3 text-muted-foreground">Separate memory space</td>
                  <td className="border border-border p-3 text-muted-foreground">Shared memory space</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 font-medium text-foreground">Creation Time</td>
                  <td className="border border-border p-3 text-muted-foreground">Slower (heavyweight)</td>
                  <td className="border border-border p-3 text-muted-foreground">Faster (lightweight)</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 font-medium text-foreground">Context Switch</td>
                  <td className="border border-border p-3 text-muted-foreground">Expensive (memory remapping)</td>
                  <td className="border border-border p-3 text-muted-foreground">Cheap (same address space)</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 font-medium text-foreground">Communication</td>
                  <td className="border border-border p-3 text-muted-foreground">IPC mechanisms needed</td>
                  <td className="border border-border p-3 text-muted-foreground">Direct via shared memory</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 font-medium text-foreground">Isolation</td>
                  <td className="border border-border p-3 text-muted-foreground">Complete isolation</td>
                  <td className="border border-border p-3 text-muted-foreground">No isolation</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 font-medium text-foreground">Failure Impact</td>
                  <td className="border border-border p-3 text-muted-foreground">Crash affects only itself</td>
                  <td className="border border-border p-3 text-muted-foreground">Can crash entire process</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Context Switching */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Context Switching</h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">Context switching</strong> is the process of saving the state of a currently 
            running process/thread and loading the state of another. This allows multiple processes to share a single CPU.
          </p>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-3 text-foreground">Process Context Switch</h4>
              <ol className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">1</span>
                  <span>Save CPU registers and program counter</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">2</span>
                  <span>Save memory management info (page tables)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">3</span>
                  <span>Update process state in PCB</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">4</span>
                  <span>Load new process page tables</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">5</span>
                  <span>Flush TLB (Translation Lookaside Buffer)</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">6</span>
                  <span>Load new process registers</span>
                </li>
              </ol>
            </div>
            <div className="p-4 bg-card border border-border rounded-lg">
              <h4 className="font-semibold mb-3 text-foreground">Thread Context Switch</h4>
              <ol className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">1</span>
                  <span>Save CPU registers and program counter</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">2</span>
                  <span>Save stack pointer</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">3</span>
                  <span>Load new thread registers</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">4</span>
                  <span>Load new stack pointer</span>
                </li>
              </ol>
              <p className="text-xs text-muted-foreground mt-4 italic">
                No memory remapping needed - same address space!
              </p>
            </div>
          </div>
        </section>

        {/* When to Use What */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">When to Use Processes vs Threads</h2>
          
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use Processes When:</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>You need strong isolation (security-critical applications)</li>
                  <li>Failures should not affect other tasks</li>
                  <li>Different programming languages are involved</li>
                  <li>Running on multiple machines (distributed systems)</li>
                </ul>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Use Threads When:</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>Tasks need to share data frequently</li>
                  <li>Low latency communication is required</li>
                  <li>Memory efficiency is important</li>
                  <li>Tasks are part of the same logical application</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Common Pitfalls */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Interview Traps to Avoid</h2>
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Confusing Thread Safety</h4>
                <p className="text-sm text-muted-foreground">
                  Just because threads can share memory does not mean they should access it without synchronization. 
                  Race conditions occur when multiple threads access shared data simultaneously.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Assuming More Threads = Better Performance</h4>
                <p className="text-sm text-muted-foreground">
                  Too many threads can cause overhead from context switching and contention. 
                  The optimal number depends on the nature of the work (CPU-bound vs I/O-bound).
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Forgetting About Deadlocks</h4>
                <p className="text-sm text-muted-foreground">
                  When threads wait for resources held by each other, the system can freeze. 
                  Always acquire locks in a consistent order.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Test Your Knowledge</h2>
          <Quiz questions={quizQuestions} title="Process vs Thread Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}
