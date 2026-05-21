import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function RangePage() {
  const result = getSubtopicBySlug("programming-fundamentals", "range");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            range
          </h1>
          <p className="text-muted-foreground text-lg">
            <strong className="text-foreground">range</strong> produces a sequence of integers without storing them all. It computes each value on demand, so it stays cheap even for huge spans like 0 to 10⁸. Typical uses: <strong className="text-foreground">for</strong> loops by index, repeating something N times, or building lists/tuples of numbers.
          </p>
        </div>

        {/* Note Box */}
        <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
          <div className="flex gap-3">
            <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
            <p className="text-sm text-foreground">
              <strong>Note:</strong> <strong className="text-foreground">stop</strong> is never included. <code className="bg-muted px-1.5 py-0.5 rounded">range(2, 8)</code> gives 2 through 7, not 8. Step can't be zero, and if the step goes the wrong way (e.g. counting up when start &gt; stop), the result is an empty range or a <strong>TypeError</strong> for step 0.
            </p>
          </div>
        </div>

        {/* What is range? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            What is range?
          </h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">range</strong> is a built-in type. It takes <strong className="text-foreground">start</strong>, <strong className="text-foreground">stop</strong>, and optionally <strong className="text-foreground">step</strong>; it behaves like a sequence (loop over it, index it, slice it, use <code className="bg-muted px-1.5 py-0.5 rounded">len()</code> and <code className="bg-muted px-1.5 py-0.5 rounded">in</code>), but it doesn't build the whole list in memory. So <code className="bg-muted px-1.5 py-0.5 rounded">range(0, 10**8)</code> is fine; a list of that many integers would not be.
          </p>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
            <div className="flex gap-3">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <strong>Type:</strong> <code className="bg-muted px-1.5 py-0.5 rounded">type(range(4))</code> → <code className="bg-muted px-1.5 py-0.5 rounded">&lt;class 'range'&gt;</code>. A range is immutable: it cannot be changed after creation.
              </p>
            </div>
          </div>
        </section>

        {/* Creating a Range */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Creating a Range
          </h2>
          <p className="text-muted-foreground mb-4">
            Three forms, and all arguments must be integers (no floats).
          </p>
          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Form</th>
                  <th className="text-left p-3 font-semibold text-foreground">Meaning</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">range(stop)</td><td className="p-3">0 up to but not including <strong>stop</strong></td><td className="p-3 font-mono">range(6) → 0, 1, 2, 3, 4, 5</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">range(start, stop)</td><td className="p-3"><strong>start</strong> up to but not including <strong>stop</strong></td><td className="p-3 font-mono">range(2, 8) → 2, 3, 4, 5, 6, 7</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">range(start, stop, step)</td><td className="p-3"><strong>start</strong> by <strong>step</strong> until <strong>stop</strong> is reached or passed</td><td className="p-3 font-mono">range(0, 10, 2) → 0, 2, 4, 6, 8</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# One argument: 0 through stop - 1
list(range(6))           # [0, 1, 2, 3, 4, 5]

# Two arguments: start through stop - 1
list(range(2, 8))       # [2, 3, 4, 5, 6, 7]

# Three: start, stop, step (stop still not included)
list(range(0, 10, 2))   # [0, 2, 4, 6, 8] - evens in 0..9
list(range(2, 12, 2))   # [2, 4, 6, 8, 10]
list(range(8, 0, -2))   # [8, 6, 4, 2] - count down by 2`}
            </pre>
          </div>
        </section>

        {/* Empty and Single-Value Ranges */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Empty and Single-Value Ranges
          </h2>
          <p className="text-muted-foreground mb-4">
            A range is empty when the step doesn't move from <strong>start</strong> toward <strong>stop</strong>: positive step with start ≥ stop, or negative step with start ≤ stop. No error; the range is simply empty.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`list(range(0))         # []  - 0 to 0 (exclusive) = nothing
list(range(2, 2))      # []  - start equals stop
list(range(4, 2))      # []  - start > stop with step 1 (wrong direction)
list(range(2, 8, -2))  # []  - negative step but start < stop (wrong direction)
list(range(4, 6, 2))   # [4] - only 4; 6 not included
len(range(2, 2))       # 0
len(range(4, 6, 2))    # 1`}
            </pre>
          </div>
        </section>

        {/* Length and Membership */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Length and Membership
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`r = range(0, 10, 2)
len(r)    # 5  - 0, 2, 4, 6, 8
4 in r    # True
5 in r    # False  - 5 isn't 0 + k×2
10 in r   # False  - stop is never included`}
            </pre>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <code className="bg-muted px-1.5 py-0.5 rounded">len(r)</code> and <code className="bg-muted px-1.5 py-0.5 rounded">x in r</code> are <strong>O(1)</strong> - computed from start, stop, and step without iterating.
              </p>
            </div>
          </div>
        </section>

        {/* Indexing and Slicing */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Indexing and Slicing
          </h2>
          <p className="text-muted-foreground mb-4">
            A range supports indexing and slicing like any sequence. Slicing returns another <strong>range</strong>, not a list.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`r = range(2, 12, 2)   # 2, 4, 6, 8, 10
r[0]      # 2
r[2]      # 6
r[-1]     # 10
r[-2]     # 8
r[1:4]    # range(4, 10, 2)  - 4, 6, 8
r[::2]    # range(2, 12, 4)  - 2, 6, 10
r[::-1]   # range(10, 0, -2) - 10, 8, 6, 4, 2; reversed`}
            </pre>
          </div>
        </section>

        {/* Iteration and Conversion */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Iteration and Conversion
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`for i in range(0, 8, 2):
    print(i)   # 0, 2, 4, 6

evens = list(range(0, 10, 2))   # [0, 2, 4, 6, 8]
t = tuple(range(2, 8, 2))       # (2, 4, 6)

# reversed() works directly
list(reversed(range(0, 8, 2)))   # [6, 4, 2, 0]
for i in reversed(range(2, 12, 2)):
    print(i)   # 10, 8, 6, 4, 2`}
            </pre>
          </div>
        </section>

        {/* Common Use Cases */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Common Use Cases
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Use case</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3">Repeat N times</td><td className="p-3 font-mono">for _ in range(4): ...</td></tr>
                <tr className="border-b border-border"><td className="p-3">Loop with index over a sequence</td><td className="p-3 font-mono">for i in range(len(items)): ... items[i]</td></tr>
                <tr className="border-b border-border"><td className="p-3">Every 2nd index</td><td className="p-3 font-mono">for i in range(0, len(items), 2): ...</td></tr>
                <tr className="border-b border-border"><td className="p-3">Count down</td><td className="p-3 font-mono">for i in range(8, 0, -2): ...</td></tr>
                <tr className="border-b border-border"><td className="p-3">Build a list of integers</td><td className="p-3 font-mono">list(range(0, 10, 2))</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                Use <code className="bg-muted px-1.5 py-0.5 rounded">range(len(...))</code> only when the index is actually needed. When both index and value are needed, <code className="bg-muted px-1.5 py-0.5 rounded">enumerate(items)</code> is clearer and less error-prone.
              </p>
            </div>
          </div>
        </section>

        {/* Range vs List of Integers */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Range vs List of Integers
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Aspect</th>
                  <th className="text-left p-3 font-semibold text-foreground">range</th>
                  <th className="text-left p-3 font-semibold text-foreground">list of integers</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3">Memory</td><td className="p-3 text-primary font-semibold">O(1)</td><td className="p-3 text-destructive">O(n)</td></tr>
                <tr className="border-b border-border"><td className="p-3">Creation time</td><td className="p-3 text-primary font-semibold">O(1)</td><td className="p-3 text-destructive">O(n)</td></tr>
                <tr className="border-b border-border"><td className="p-3">Indexing</td><td className="p-3">O(1)</td><td className="p-3">O(1)</td></tr>
                <tr className="border-b border-border"><td className="p-3">Iteration</td><td className="p-3">O(n)</td><td className="p-3">O(n)</td></tr>
                <tr><td className="p-3">Mutability</td><td className="p-3">Immutable</td><td className="p-3">Mutable</td></tr>
              </tbody>
            </table>
          </div>
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
                  <p className="font-semibold text-foreground">Creating - Integer-only arguments</p>
                  <p className="text-muted-foreground"><strong>range</strong> only takes integers. Floats raise <strong>TypeError</strong>.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Parameters - stop is exclusive</p>
                  <p className="text-muted-foreground"><strong>stop</strong> is never included. <code className="bg-muted px-1.5 py-0.5 rounded">range(2, 8)</code> gives 2 through 7, not 8.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Parameters - step must not be zero</p>
                  <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">range(2, 10, 0)</code> raises <strong>TypeError</strong>.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">range is not a list</p>
                  <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">.append</code>, <code className="bg-muted px-1.5 py-0.5 rounded">.sort</code>, and other list methods are not available.</p>
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
                  <p className="font-semibold text-foreground">What is the type of range(5)? Can it be used in a set or as a dict key?</p>
                  <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">type(range(5))</code> is <strong>range</strong>. Ranges are immutable and hashable, so they can go in sets and work as dict keys.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What are the three forms of range()?</p>
                  <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">range(stop)</code>, <code className="bg-muted px-1.5 py-0.5 rounded">range(start, stop)</code>, and <code className="bg-muted px-1.5 py-0.5 rounded">range(start, stop, step)</code>. All arguments must be integers.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the time complexity of len(r) and x in r?</p>
                  <p className="text-muted-foreground">Both O(1). <code className="bg-muted px-1.5 py-0.5 rounded">len(r)</code> is computed from start, stop, and step. <code className="bg-muted px-1.5 py-0.5 rounded">x in r</code> checks if the value lies on the grid without iterating.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Why use range instead of a list of integers?</p>
                  <p className="text-muted-foreground"><strong>range</strong> uses O(1) memory and is O(1) to create. A list uses O(n) memory and takes O(n) time to build. For large N, <code className="bg-muted px-1.5 py-0.5 rounded">range(N)</code> is the right choice unless an actual list is required.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

      
      </div>
    </TopicContent>
  );
}