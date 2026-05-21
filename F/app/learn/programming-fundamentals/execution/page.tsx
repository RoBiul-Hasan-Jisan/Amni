import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function ExecutionPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "execution");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Execution Basics
          </h1>
          <p className="text-muted-foreground text-lg">
            What it means that Python is interpreted, what happens when you run a .py file, and how source code becomes executed instructions.
          </p>
        </div>

        {/* Note Box */}
        <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
          <div className="flex gap-3">
            <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
            <p className="text-sm text-foreground">
              In interviews, you'll often be asked: "What does interpreted mean?", "Is Python compiled?", and "What exactly happens when I run <code className="bg-muted px-1.5 py-0.5 rounded">python file.py</code>?" This lesson answers those clearly.
            </p>
          </div>
        </div>

        {/* What Does "Interpreted Language" Mean? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            What Does "Interpreted Language" Mean?
          </h2>
          <p className="text-muted-foreground mb-4">
            An interpreted language is one where code is executed by an interpreter at runtime, rather than being compiled ahead of time into a standalone native binary that runs directly on the CPU.
          </p>
          <p className="text-muted-foreground mb-4">
            In practice for Python:
          </p>
          <ul className="list-disc list-inside text-muted-foreground mb-4 ml-4 space-y-1">
            <li>You write source code in <code className="bg-muted px-1.5 py-0.5 rounded">.py</code> files.</li>
            <li>The Python interpreter implementation, most often CPython, reads and processes that code.</li>
            <li>Execution happens through Python's virtual machine layer.</li>
          </ul>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
            <div className="flex gap-3">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                This is different from the common flow in C or C++, where source code is compiled and linked to produce a platform-specific executable.
              </p>
            </div>
          </div>
        </section>

        {/* Is Python Compiled or Interpreted? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Is Python Compiled or Interpreted?
          </h2>
          <p className="text-muted-foreground mb-4">
            For interviews, the accurate answer is:
          </p>
          <ul className="list-disc list-inside text-muted-foreground mb-4 ml-4 space-y-1">
            <li>Python is <strong className="text-foreground">interpreted at runtime</strong>, and</li>
            <li>Python code is also <strong className="text-foreground">compiled to bytecode</strong> before execution.</li>
          </ul>
          <p className="text-muted-foreground">
            So Python is not "purely interpreted" in the simplistic sense. CPython first compiles <code className="bg-muted px-1.5 py-0.5 rounded">.py</code> source to bytecode (<code className="bg-muted px-1.5 py-0.5 rounded">.pyc</code>), then executes that bytecode on the Python Virtual Machine (PVM).
          </p>
        </section>

        {/* What Is CPython? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            What Is CPython?
          </h2>
          <p className="text-muted-foreground mb-4">
            CPython is the reference implementation of the Python programming language and the one most developers use.
          </p>
          <p className="text-muted-foreground mb-4">
            It is not a separate programming language. It is also not only a compiler. CPython is an interpreter runtime that includes a compiler stage from source code to bytecode.
          </p>
          <ul className="list-disc list-inside text-muted-foreground mb-4 ml-4 space-y-1">
            <li>It is written mainly in the C language.</li>
            <li>It compiles Python source to bytecode.</li>
            <li>It executes bytecode on the Python Virtual Machine.</li>
            <li>It includes the default memory manager and garbage collector used in mainstream Python.</li>
          </ul>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                When people say "Python does this," they usually mean CPython behavior unless stated otherwise.
              </p>
            </div>
          </div>
        </section>

        {/* What Happens When You Run python script.py? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            What Happens When You Run <code className="bg-muted px-2 py-0.5 rounded">python script.py</code>?
          </h2>
          <p className="text-muted-foreground mb-4">
            High-level flow:
          </p>
          <ol className="list-decimal list-inside text-muted-foreground mb-4 ml-4 space-y-1">
            <li>Python reads <code className="bg-muted px-1.5 py-0.5 rounded">script.py</code>.</li>
            <li>Source code is tokenized and parsed into an internal syntax tree.</li>
            <li>Python compiles it to bytecode instructions.</li>
            <li>Bytecode is executed by the Python Virtual Machine.</li>
            <li>Objects are created, functions run, imports are resolved, and output is produced.</li>
          </ol>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# script.py
x = 2
y = 4
print(x + y)`}
            </pre>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mt-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`$ python script.py
6`}
            </pre>
          </div>
        </section>

        {/* Bytecode and __pycache__ */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Bytecode and __pycache__
          </h2>
          <p className="text-muted-foreground mb-4">
            CPython may write compiled bytecode to a <code className="bg-muted px-1.5 py-0.5 rounded">.pyc</code> file inside <code className="bg-muted px-1.5 py-0.5 rounded">__pycache__</code> to speed up future imports.
          </p>
          <ul className="list-disc list-inside text-muted-foreground mb-4 ml-4 space-y-1">
            <li>Source: <code className="bg-muted px-1.5 py-0.5 rounded">module.py</code></li>
            <li>Cached bytecode: inside <code className="bg-muted px-1.5 py-0.5 rounded">__pycache__</code> as <code className="bg-muted px-1.5 py-0.5 rounded">module.cpython-3x.pyc</code></li>
          </ul>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
            <div className="flex gap-3">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground">Important points:</p>
                <ul className="list-disc list-inside text-sm text-muted-foreground ml-4">
                  <li><code className="bg-muted px-1.5 py-0.5 rounded">__pycache__</code> is a cache optimization, not required for correctness.</li>
                  <li>The presence or absence of <code className="bg-muted px-1.5 py-0.5 rounded">.pyc</code> files does not change program behavior.</li>
                  <li>Cache reuse depends on matching source timestamp, source hash, and interpreter version.</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Python Virtual Machine (PVM) */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Python Virtual Machine (PVM)
          </h2>
          <p className="text-muted-foreground mb-4">
            The PVM executes Python bytecode instruction by instruction. It is part of the Python runtime, not your operating system's CPU instruction set.
          </p>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                This design helps Python stay portable: the same source code can run on different platforms, as long as a compatible Python interpreter is available.
              </p>
            </div>
          </div>
        </section>

        {/* GIL in CPython */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            GIL in CPython
          </h2>
          <p className="text-muted-foreground mb-4">
            The <strong className="text-foreground">GIL (Global Interpreter Lock)</strong> is a lock in CPython that allows only one thread at a time to execute Python bytecode inside one process.
          </p>
          <p className="text-muted-foreground mb-2 font-semibold">Why it exists:</p>
          <ul className="list-disc list-inside text-muted-foreground mb-4 ml-4 space-y-1">
            <li>It simplifies parts of CPython memory management, especially reference counting.</li>
            <li>It keeps interpreter internals safer in multi-threaded execution.</li>
          </ul>
          <p className="text-muted-foreground mb-2 font-semibold">How it affects performance:</p>
          <ul className="list-disc list-inside text-muted-foreground mb-4 ml-4 space-y-1">
            <li>For <strong className="text-foreground">CPU-bound</strong> work, Python threads in one CPython process do not run Python bytecode in true parallel across multiple CPU cores.</li>
            <li>For <strong className="text-foreground">I/O bound</strong> work, threading is still useful because threads can run while others wait on network, disk, or sleep calls.</li>
          </ul>
          <p className="mb-2 font-semibold text-foreground">What to use for CPU-heavy parallelism:</p>
          <ul className="list-disc list-inside text-muted-foreground ml-4 space-y-1">
            <li><code className="bg-muted px-1.5 py-0.5 rounded">multiprocessing</code> with separate processes</li>
            <li>Native extensions that release the GIL</li>
            <li>Distributed execution systems when needed</li>
          </ul>
        </section>

        {/* Execution of Top-Level Code */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Execution of Top-Level Code
          </h2>
          <p className="text-muted-foreground mb-4">
            When a file is run directly, all top-level statements execute from top to bottom.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`print("start")
x = 2
print("end")`}
            </pre>
          </div>
          <p className="text-muted-foreground mt-4">
            Function definitions and class definitions are also executable statements at import time and run time. They create function objects and class objects, then bind names. Function bodies run only when called.
          </p>
        </section>

        {/* Script Execution vs Import Execution */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Script Execution vs Import Execution
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# script.py
print("Module loaded")

def greet():
    print("Hello, World!")

if __name__ == "__main__":
    greet()`}
            </pre>
          </div>
          <p className="text-muted-foreground mb-2">
            <code className="bg-muted px-1.5 py-0.5 rounded">python script.py</code>:
          </p>
          <ul className="list-disc list-inside text-muted-foreground mb-4 ml-4 space-y-1">
            <li>The file is treated as the entry point.</li>
            <li><code className="bg-muted px-1.5 py-0.5 rounded">__name__</code> is set to <code className="bg-muted px-1.5 py-0.5 rounded">"__main__"</code>.</li>
            <li>Top-level code executes.</li>
          </ul>
          <p className="text-muted-foreground mb-2">
            <code className="bg-muted px-1.5 py-0.5 rounded">import script</code>:
          </p>
          <ul className="list-disc list-inside text-muted-foreground ml-4 space-y-1">
            <li><code className="bg-muted px-1.5 py-0.5 rounded">script.py</code> executes once per process on first import.</li>
            <li><code className="bg-muted px-1.5 py-0.5 rounded">__name__</code> is <code className="bg-muted px-1.5 py-0.5 rounded">"script"</code> (module name), not <code className="bg-muted px-1.5 py-0.5 rounded">"__main__"</code>.</li>
            <li>Module object is cached in <code className="bg-muted px-1.5 py-0.5 rounded">sys.modules</code>.</li>
          </ul>
        </section>

        {/* The if __name__ == "__main__" Guard */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            The <code className="bg-muted px-2 py-0.5 rounded">if __name__ == "__main__"</code> Guard
          </h2>
          <p className="text-muted-foreground mb-4">
            Code inside this guard runs only when the file is executed directly. It does not run when the file is imported.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def greet():
    print("Hello, World!")

if __name__ == "__main__":
    greet()`}
            </pre>
          </div>
          <p className="text-muted-foreground mt-4">
            <strong className="text-foreground">Direct execution:</strong> <code className="bg-muted px-1.5 py-0.5 rounded">__name__</code> is <code className="bg-muted px-1.5 py-0.5 rounded">"__main__"</code> → guard runs → <code className="bg-muted px-1.5 py-0.5 rounded">greet()</code> runs.
          </p>
          <p className="text-muted-foreground">
            <strong className="text-foreground">Import as module:</strong> <code className="bg-muted px-1.5 py-0.5 rounded">__name__</code> is <code className="bg-muted px-1.5 py-0.5 rounded">"script"</code> → guard does not run → <code className="bg-muted px-1.5 py-0.5 rounded">greet()</code> does not run automatically.
          </p>
        </section>

        {/* Common Guard Patterns */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Common Guard Patterns
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Minimal guarded entry point
if __name__ == "__main__":
    print("Hello, World!")`}
            </pre>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Use a main function for cleaner structure
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()`}
            </pre>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                Using <code className="bg-muted px-1.5 py-0.5 rounded">main()</code> makes script logic easier to test and easier to read.
              </p>
            </div>
          </div>
        </section>

        {/* Why This Matters in Interviews */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Why This Matters in Interviews
          </h2>
          <p className="text-muted-foreground mb-4">
            Interviewers often test whether you can explain runtime behavior, not just syntax:
          </p>
          <ul className="list-disc list-inside text-muted-foreground ml-4 space-y-1">
            <li>Why import side effects happen.</li>
            <li>Why top-level code runs on import.</li>
            <li>Why Python startup and execution are slower than precompiled native binaries.</li>
            <li>Why deleting <code className="bg-muted px-1.5 py-0.5 rounded">__pycache__</code> is usually safe.</li>
          </ul>
        </section>

        {/* Tricky Points */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Tricky Points
          </h2>
          <div className="space-y-3">
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">"Python is interpreted" is incomplete</p>
                  <p className="text-muted-foreground text-sm">Python source is first compiled to bytecode, then interpreted and executed by the PVM.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">GIL limits Python bytecode parallelism</p>
                  <p className="text-muted-foreground text-sm">Threads are excellent for I/O-bound workloads, but CPU-bound Python bytecode in one process does not scale linearly across cores because of the GIL.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Import executes module top-level code</p>
                  <p className="text-muted-foreground text-sm">If module code has prints, network calls, or file writes at top level, importing that module triggers them.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Wrong <code className="bg-muted px-1.5 py-0.5 rounded">__name__</code> comparison silently breaks guard logic</p>
                  <p className="text-muted-foreground text-sm">Typos such as <code className="bg-muted px-1.5 py-0.5 rounded">_name_</code> or <code className="bg-muted px-1.5 py-0.5 rounded">__main_</code> cause the guard condition to fail. The exact comparison is <code className="bg-muted px-1.5 py-0.5 rounded">__name__ == "__main__"</code>.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Interview Questions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Interview Questions
          </h2>
          <div className="space-y-4">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What does it mean that Python is an interpreted language?</p>
                  <p className="text-muted-foreground">It means Python code is executed through an interpreter at runtime rather than as a standalone native executable produced ahead of time.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Is Python compiled?</p>
                  <p className="text-muted-foreground">Yes. CPython compiles source code to bytecode (<code className="bg-muted px-1.5 py-0.5 rounded">.pyc</code>) and then executes that bytecode on the Python Virtual Machine.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is CPython?</p>
                  <p className="text-muted-foreground">CPython is the reference implementation of Python. It is an interpreter runtime with a compiler stage that converts source code to bytecode before execution.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the GIL in Python?</p>
                  <p className="text-muted-foreground">The GIL is a CPython lock that allows only one thread at a time to execute Python bytecode in one process.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the difference between running a file and importing it?</p>
                  <p className="text-muted-foreground">Running sets <code className="bg-muted px-1.5 py-0.5 rounded">__name__ = "__main__"</code> for the entry file. Importing sets <code className="bg-muted px-1.5 py-0.5 rounded">__name__</code> to module name, executes top-level code once, and caches the module.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What does <code className="bg-muted px-1.5 py-0.5 rounded">if __name__ == "__main__"</code> do?</p>
                  <p className="text-muted-foreground">It ensures guarded code runs only when the file is executed directly, not when the file is imported as a module.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

   
      </div>
    </TopicContent>
  );
}