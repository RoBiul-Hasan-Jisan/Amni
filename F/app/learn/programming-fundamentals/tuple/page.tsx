import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function TuplePage() {
  const result = getSubtopicBySlug("programming-fundamentals", "tuple");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8 border-b border-border pb-4">
          <h1 className="text-4xl font-bold text-foreground mb-2">
            Tuples
          </h1>
          <p className="text-muted-foreground text-lg">
            Tuples are ordered, immutable sequences. They can hold mixed types and allow duplicate values.
          </p>
        </div>

        {/* Creating Tuples */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Creating Tuples
          </h2>
          <p className="text-muted-foreground mb-4">
            You can build a tuple in several ways:
          </p>

          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Approach</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                  <th className="text-left p-3 font-semibold text-foreground">Use when</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">Literal `()`</td>
                  <td className="p-3 font-mono">row = (2, 4, 6)</td>
                  <td className="p-3">You know the elements</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">Empty tuple</td>
                  <td className="p-3 font-mono">seq = ()</td>
                  <td className="p-3">Empty placeholder</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">tuple()</td>
                  <td className="p-3 font-mono">tuple() → ()</td>
                  <td className="p-3">Empty tuple (same as ())</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">tuple(iterable)</td>
                  <td className="p-3 font-mono">tuple([2, 4, 6]) → (2, 4, 6)</td>
                  <td className="p-3">Convert from list, range, str, set, etc.</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3 mb-4">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <span className="font-semibold">Single-element tuple:</span> Requires a trailing comma: <code className="bg-muted px-1.5 py-0.5 rounded">(2,)</code> - without the comma, <code className="bg-muted px-1.5 py-0.5 rounded">(2)</code> is just an integer in parentheses.
              </p>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`nums = (2, 4, 6, 8, 10)
single = (2,)           # tuple
not_tuple = (2)         # int
built = tuple([2, 4, 6])  # (2, 4, 6)
empty = ()
from_list = tuple([2, 4, 6, 8])`}
            </pre>
          </div>

          <div className="mt-4 space-y-2">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <p className="text-sm text-foreground">
                  <span className="font-semibold">Length:</span> Use <code className="bg-muted px-1.5 py-0.5 rounded">len(t)</code> to get the number of elements. For tuples this is O(1); the length is stored.
                </p>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <p className="text-sm text-foreground">
                  <span className="font-semibold">Truthiness:</span> An empty tuple <code className="bg-muted px-1.5 py-0.5 rounded">()</code> is falsy; a non-empty tuple is truthy. Use <code className="bg-muted px-1.5 py-0.5 rounded">if t:</code> to mean "if the tuple has at least one element" and <code className="bg-muted px-1.5 py-0.5 rounded">if not t:</code> for "if the tuple is empty."
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Tuple Methods */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Tuple Methods
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Method</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">count(x)</td>
                  <td className="p-3">Return the number of occurrences of x</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">index(x)</td>
                  <td className="p-3">Return the index of the first occurrence of x; raises ValueError if missing</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="text-muted-foreground mt-4 text-sm">
            Tuples have only these two methods because they are immutable. No <code className="bg-muted px-1.5 py-0.5 rounded">append</code>, <code className="bg-muted px-1.5 py-0.5 rounded">remove</code>, <code className="bg-muted px-1.5 py-0.5 rounded">sort</code>, or <code className="bg-muted px-1.5 py-0.5 rounded">reverse</code>.
          </p>
        </section>

        {/* Access Tuples */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Access Tuples
          </h2>
          <p className="text-muted-foreground mb-4">
            Use <span className="font-semibold text-foreground">indexing</span> and <span className="font-semibold text-foreground">slicing</span> the same way as lists. Indices start at 0; negative indices count from the end.
          </p>

          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Pattern</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">t[i]</td>
                  <td className="p-3">Single element at index i</td>
                  <td className="p-3 font-mono">nums[2] → 6</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">t[-1]</td>
                  <td className="p-3">Last element</td>
                  <td className="p-3 font-mono">nums[-1] → 10</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">t[i:j]</td>
                  <td className="p-3">Slice from i to j (excl.)</td>
                  <td className="p-3 font-mono">nums[2:5] → (6, 8, 10)</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">t[:n]</td>
                  <td className="p-3">First n elements</td>
                  <td className="p-3 font-mono">nums[:4] → (2, 4, 6, 8)</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">t[n:]</td>
                  <td className="p-3">From index n to end</td>
                  <td className="p-3 font-mono">nums[4:] → (10,)</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">t[-m:-n]</td>
                  <td className="p-3">Slice with negative indices</td>
                  <td className="p-3 font-mono">nums[-4:-2] → (4, 6)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mb-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <span className="font-semibold">Slicing with step:</span> The full form is <code className="bg-muted px-1.5 py-0.5 rounded">t[start:stop:step]</code>. <code className="bg-muted px-1.5 py-0.5 rounded">step</code> is the stride; omit it and it defaults to 1. Use a negative step to walk backward (e.g. <code className="bg-muted px-1.5 py-0.5 rounded">t[::-1]</code> for a reversed copy).
              </p>
            </div>
          </div>

          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Pattern</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">t[::k]</td>
                  <td className="p-3">Every k-th element</td>
                  <td className="p-3 font-mono">nums[::2] → (2, 6, 10, 14)</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">t[i::k]</td>
                  <td className="p-3">Every k-th from index i</td>
                  <td className="p-3 font-mono">nums[1::2] → (4, 8, 12, 16)</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">t[::-1]</td>
                  <td className="p-3">Reversed (step -1)</td>
                  <td className="p-3 font-mono">nums[::-1] → reversed copy</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`nums = (2, 4, 6, 8, 10, 12, 14, 16)
nums[2]         # 6
nums[-1]        # 16
nums[2:6]       # (6, 8, 10, 12)
nums[:4]        # (2, 4, 6, 8)
nums[4:]        # (10, 12, 14, 16)
nums[-4:-2]     # (12, 14)
4 in nums       # True
nums.count(6)   # 1
nums.index(8)   # 3`}
            </pre>
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <span className="font-semibold">Value lookup:</span> Use <code className="bg-muted px-1.5 py-0.5 rounded">count(x)</code> to count occurrences and <code className="bg-muted px-1.5 py-0.5 rounded">index(x)</code> to get the first index. <span className="font-semibold">Note:</span> <code className="bg-muted px-1.5 py-0.5 rounded">index(x)</code> raises <code className="bg-muted px-1.5 py-0.5 rounded">ValueError</code> when the item is not present. Check with <code className="bg-muted px-1.5 py-0.5 rounded">x in t</code> first, or wrap in <code className="bg-muted px-1.5 py-0.5 rounded">try</code>/<code className="bg-muted px-1.5 py-0.5 rounded">except</code>, if you need to handle the missing case.
              </p>
            </div>
          </div>
        </section>

        {/* Reversing a Tuple */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Reversing a Tuple
          </h2>
          <p className="text-muted-foreground mb-4">
            Create a new tuple with elements in reverse order. The original tuple is unchanged.
          </p>

          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Approach</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                  <th className="text-left p-3 font-semibold text-foreground">Returns</th>
                  <th className="text-left p-3 font-semibold text-foreground">Mutates original?</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">tuple(reversed(t))</td>
                  <td className="p-3 font-mono">tuple(reversed(nums))</td>
                  <td className="p-3">New tuple</td>
                  <td className="p-3">No</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">Slice [::-1]</td>
                  <td className="p-3 font-mono">nums[::-1]</td>
                  <td className="p-3">New tuple</td>
                  <td className="p-3">No</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`original = (4, 2, 8, 6, 10)
# reversed() returns a reverse iterator; wrap in tuple() to get a tuple
nums = tuple(reversed(original))  # nums = (10, 6, 8, 2, 4); original = (4, 2, 8, 6, 10)

original = (4, 2, 8, 6, 10)
nums = original[::-1]  # nums = (10, 6, 8, 2, 4); original = (4, 2, 8, 6, 10)`}
            </pre>
          </div>

          <ul className="list-disc list-inside text-muted-foreground space-y-1 mt-4 text-sm">
            <li><code className="bg-muted px-1.5 py-0.5 rounded">tuple(reversed(t))</code> - <code className="bg-muted px-1.5 py-0.5 rounded">reversed()</code> returns a reverse iterator; wrap in <code className="bg-muted px-1.5 py-0.5 rounded">tuple()</code> to get a tuple. O(n) time, O(n) space.</li>
            <li><code className="bg-muted px-1.5 py-0.5 rounded">t[::-1]</code> - Slice with step -1; creates a new tuple. O(n) time, O(n) space.</li>
          </ul>

          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                Tuples have no <code className="bg-muted px-1.5 py-0.5 rounded">reverse()</code> method (unlike lists) because they are immutable.
              </p>
            </div>
          </div>
        </section>

        {/* Unpack Tuples */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Unpack Tuples
          </h2>
          <p className="text-muted-foreground mb-4">
            Assign tuple elements to variables in one statement.
          </p>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`nums = (2, 4, 6)
a, b, c = nums  # a=2, b=4, c=6`}
            </pre>
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mb-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <span className="font-semibold">Asterisk `*`:</span> Collect remaining elements into a list.
              </p>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`nums = (2, 4, 6, 8, 10)
first, second, *rest = nums  # first=2, second=4, rest=[6, 8, 10]
print(type(rest))            # <class 'list'>`}
            </pre>
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mb-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <span className="font-semibold">Middle unpacking:</span> Use <code className="bg-muted px-1.5 py-0.5 rounded">*</code> at any position to capture the rest.
              </p>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`nums = (2, 4, 6, 8, 10)
a, *mid, z = nums  # a=2, mid=[4, 6, 8], z=10`}
            </pre>
          </div>

          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3 mb-4">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <span className="font-semibold">Mismatched count:</span> The number of variables must match the number of elements, unless <code className="bg-muted px-1.5 py-0.5 rounded">*</code> is used to absorb extras.
              </p>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`nums = (2, 4, 6)
a, b = nums        # ValueError: too many values to unpack
a, b, c, d = nums  # ValueError: not enough values to unpack

a, b, c, *d = (2, 4, 6)  # a=2, b=4, c=6, d=[]`}
            </pre>
          </div>

          <div className="mt-4">
            <p className="text-muted-foreground mb-2">
              <span className="font-semibold text-foreground">Index with loop:</span> Use <code className="bg-muted px-1.5 py-0.5 rounded">enumerate(t)</code> when you need both index and value:
            </p>
            <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
              <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                {`row = (2, 4, 6, 8)
for i, x in enumerate(row):
    print(i, x)  # 0 2, 1 4, 2 6, 3 8`}
              </pre>
            </div>
          </div>
        </section>

        {/* Update Tuples */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Update Tuples
          </h2>
          <p className="text-muted-foreground mb-4">
            Tuples are <span className="font-semibold text-foreground">immutable</span> - you cannot change, add, or remove elements in place. To "update" a tuple, create a new one.
          </p>

          <div className="mb-4">
            <p className="text-muted-foreground mb-2">
              <span className="font-semibold text-foreground">Replace an element:</span> Convert to list, modify, convert back.
            </p>
            <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
              <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8, 10)
temp = list(nums)
temp[1] = 0
nums = tuple(temp)  # (2, 0, 6, 8, 10)`}
              </pre>
            </div>
          </div>

          <div className="mb-4">
            <p className="text-muted-foreground mb-2">
              <span className="font-semibold text-foreground">Add an element:</span> Convert to list, append, convert back - or concatenate with a single-element tuple.
            </p>
            <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
              <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8, 10)
temp = list(nums)
temp.append(12)
nums = tuple(temp)  # (2, 4, 6, 8, 10, 12)
print(nums)

nums = (2, 4, 6, 8, 10)
nums += (12,)  # (2, 4, 6, 8, 10, 12)`}
              </pre>
            </div>
          </div>

          <div>
            <p className="text-muted-foreground mb-2">
              <span className="font-semibold text-foreground">Remove an element:</span> Convert to list, remove or filter, convert back.
            </p>
            <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
              <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8, 10)
temp = list(nums)
temp.remove(6)
nums = tuple(temp)  # (2, 4, 8, 10)`}
              </pre>
            </div>
          </div>
        </section>

        {/* Join Tuples */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Join Tuples
          </h2>

          <div className="mb-4">
            <p className="text-muted-foreground mb-2">
              <span className="font-semibold text-foreground">Concatenation with `+`:</span> Both operands must be tuples. Returns a new tuple.
            </p>
            <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
              <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                {`a = (2, 4, 6)
b = (8, 10, 12)
c = a + b  # (2, 4, 6, 8, 10, 12)`}
              </pre>
            </div>
          </div>

          <div>
            <p className="text-muted-foreground mb-2">
              <span className="font-semibold text-foreground">Repetition with `*`:</span> Repeat the tuple a given number of times.
            </p>
            <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
              <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6)
doubled = nums * 2  # (2, 4, 6, 2, 4, 6)`}
              </pre>
            </div>
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <span className="font-semibold">Note:</span> <code className="bg-muted px-1.5 py-0.5 rounded">+</code> and <code className="bg-muted px-1.5 py-0.5 rounded">*</code> create new tuples; they do not modify the original.
              </p>
            </div>
          </div>
        </section>

        {/* Loop Tuples */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Loop Tuples
          </h2>
          <p className="text-muted-foreground mb-4">
            Three common patterns: direct iteration, index-based, and while loop.
          </p>

          <div className="mb-4">
            <p className="text-muted-foreground mb-2">
              <span className="font-semibold text-foreground">Direct iteration:</span>
            </p>
            <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
              <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8)
for x in nums:
    print(x)  # 2, 4, 6, 8`}
              </pre>
            </div>
          </div>

          <div className="mb-4">
            <p className="text-muted-foreground mb-2">
              <span className="font-semibold text-foreground">Index-based:</span>
            </p>
            <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
              <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8)
for i in range(len(nums)):
    print(nums[i])  # 2, 4, 6, 8`}
              </pre>
            </div>
          </div>

          <div className="mb-4">
            <p className="text-muted-foreground mb-2">
              <span className="font-semibold text-foreground">While loop:</span>
            </p>
            <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
              <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8)
i = 0
while i < len(nums):
    print(nums[i])
    i += 2  # 2, 6 - step by 2`}
              </pre>
            </div>
          </div>

          <div>
            <p className="text-muted-foreground mb-2">
              <span className="font-semibold text-foreground">Loop in reverse:</span>
            </p>
            <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
              <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8)

for x in reversed(nums):  # O(n)
    print(x)              # 8, 6, 4, 2

for x in nums[::-1]:  # O(n)
    print(x)          # 8, 6, 4, 2 - slice with step -1

for i in range(len(nums) - 1, -1, -1):  # O(n)
    print(nums[i])                      # 8, 6, 4, 2 - index-based reverse`}
              </pre>
            </div>
          </div>

          <ul className="list-disc list-inside text-muted-foreground space-y-1 mt-4 text-sm">
            <li><code className="bg-muted px-1.5 py-0.5 rounded">reversed(t)</code> - Returns a reverse iterator; does not create a new tuple.</li>
            <li><code className="bg-muted px-1.5 py-0.5 rounded">t[::-1]</code> - Slice with step -1; creates a new tuple in reverse order.</li>
            <li><code className="bg-muted px-1.5 py-0.5 rounded">range(len(t)-1, -1, -1)</code> - Index-based; iterate from last index down to 0.</li>
          </ul>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <span className="font-semibold">Why LeetCode often prefers the index-based reverse:</span> When you need to <span className="font-semibold">modify</span> elements by index (swap, overwrite, remove), you must have the index. <code className="bg-muted px-1.5 py-0.5 rounded">reversed()</code> and <code className="bg-muted px-1.5 py-0.5 rounded">t[::-1]</code> give values, not indices. The <code className="bg-muted px-1.5 py-0.5 rounded">range(len(t)-1, -1, -1)</code> pattern also avoids index shifting when removing elements in place, and it translates directly to other languages (C, Java, etc.).
              </p>
            </div>
          </div>
        </section>

        {/* Sort Tuples */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Sort Tuples
          </h2>
          <p className="text-muted-foreground mb-4">
            Tuples have no <code className="bg-muted px-1.5 py-0.5 rounded">sort()</code> method because they are immutable. To get a sorted tuple, create a new one using <code className="bg-muted px-1.5 py-0.5 rounded">sorted()</code>.
          </p>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mb-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <span className="font-semibold">Note:</span> <code className="bg-muted px-1.5 py-0.5 rounded">sorted(t)</code> returns a <span className="font-semibold">list</span>, not a tuple. Wrap in <code className="bg-muted px-1.5 py-0.5 rounded">tuple()</code> to get a tuple.
              </p>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`nums = (10, 4, 8, 2, 6)
tuple(sorted(nums))                # (2, 4, 6, 8, 10)
tuple(sorted(nums, reverse=True))  # (10, 8, 6, 4, 2)`}
            </pre>
          </div>

          <div className="mb-4">
            <p className="text-muted-foreground mb-2">
              <span className="font-semibold text-foreground">With `key`:</span> Use <code className="bg-muted px-1.5 py-0.5 rounded">key</code> for custom sort order, same as for lists.
            </p>
            <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
              <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                {`nums = (2, 4, 6, 8, 10, 12, 14, 16)
tuple(sorted(nums, key=lambda n: abs(n - 10)))  # sort by distance from 10
# (10, 8, 12, 6, 14, 4, 16, 2)`}
              </pre>
            </div>
          </div>

          <div>
            <p className="text-muted-foreground mb-2">
              <span className="font-semibold text-foreground">Alternative:</span> Convert to list, sort in place, convert back. <code className="bg-muted px-1.5 py-0.5 rounded">sorted(t)</code> is simpler and preferred.
            </p>
            <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
              <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                {`nums = (10, 4, 8, 2, 6)
temp = list(nums)
temp.sort()
tuple(temp)  # (2, 4, 6, 8, 10)`}
              </pre>
            </div>
          </div>
        </section>

        {/* Tuple vs Other Collections */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Tuple vs Other Collections
          </h2>
          <p className="text-muted-foreground mb-4">
            Python has other built-in types for sequences and collections. Choosing the right one depends on order, mutability, and whether you need hashability.
          </p>

          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Type</th>
                  <th className="text-left p-3 font-semibold text-foreground">Ordered?</th>
                  <th className="text-left p-3 font-semibold text-foreground">Mutable?</th>
                  <th className="text-left p-3 font-semibold text-foreground">Duplicates?</th>
                  <th className="text-left p-3 font-semibold text-foreground">Hashable?</th>
                  <th className="text-left p-3 font-semibold text-foreground">Typical use</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">tuple</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">No</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">Yes (if elements are)</td>
                  <td className="p-3">Fixed sequences; dict keys; return values</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">list</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">No</td>
                  <td className="p-3">Ordered sequences you change</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">set</td>
                  <td className="p-3">No</td>
                  <td className="p-3">Yes</td>
                  <td className="p-3">No</td>
                  <td className="p-3">N/A</td>
                  <td className="p-3">Unique items, membership, set math</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">dict</td>
                  <td className="p-3">Yes (ins.)</td>
                  <td className="p-3">Yes (vals)</td>
                  <td className="p-3">No (keys)</td>
                  <td className="p-3">Keys only</td>
                  <td className="p-3">Key–value mapping</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <span className="font-semibold">Note:</span> A tuple is hashable only if every element is hashable (e.g. no lists or dicts inside). Lists and sets are not hashable.
              </p>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mt-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Hashable tuple (all elements hashable) - valid as dict key
key_ok = (2, 4)
d = {key_ok: "point"}   # {(2, 4): 'point'}

# Tuple with a list inside - not hashable
key_bad = (2, [4, 6])
# d2 = {key_bad: "x"}  # TypeError: unhashable type: 'list'

# List cannot be a dict key
# d3 = {[2, 4]: "x"}   # TypeError: unhashable type: 'list'`}
            </pre>
          </div>

          <div className="mt-4">
            <p className="text-muted-foreground">
              <span className="font-semibold text-foreground">When to use a tuple:</span> Fixed sequence (e.g. coordinates, record-like data), dict keys, set elements, or multiple return values from a function. Use a <span className="font-semibold text-foreground">list</span> when you need to append, remove, or reorder. Use a <span className="font-semibold text-foreground">set</span> when you care only about uniqueness or fast membership.
            </p>
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-foreground mb-1">Common use cases:</p>
                <ul className="list-disc list-inside text-sm text-muted-foreground">
                  <li><span className="font-semibold text-foreground">Return values:</span> A function that "returns multiple values" actually returns a tuple. You can unpack it: <code className="bg-muted px-1.5 py-0.5 rounded">def min_max(vals): return min(vals), max(vals)</code> → <code className="bg-muted px-1.5 py-0.5 rounded">lo, hi = min_max([2, 4, 6, 8])</code> gives <code className="bg-muted px-1.5 py-0.5 rounded">lo=2</code>, <code className="bg-muted px-1.5 py-0.5 rounded">hi=8</code>.</li>
                  <li><span className="font-semibold text-foreground">Dict keys and set elements:</span> Tuples are hashable when all elements are hashable, so they can be dict keys or set members. Lists cannot: <code className="bg-muted px-1.5 py-0.5 rounded">{'{(2, 4): "point"}'}</code> is valid; <code className="bg-muted px-1.5 py-0.5 rounded">{'{[2, 4]: "point"}'}</code> raises <code className="bg-muted px-1.5 py-0.5 rounded">TypeError: unhashable type: 'list'</code>.</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Tricky Points */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Tricky Points
          </h2>
          <div className="space-y-3">
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">Creating - Single-element tuple</p>
                  <p className="text-sm text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">(2)</code> is an int; <code className="bg-muted px-1.5 py-0.5 rounded">(2,)</code> is a tuple. The comma is required so Python parses it as a tuple.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">Access - <code className="bg-muted px-1.5 py-0.5 rounded">index(x)</code> when missing</p>
                  <p className="text-sm text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">t.index(x)</code> raises <code className="bg-muted px-1.5 py-0.5 rounded">ValueError</code> if <code className="bg-muted px-1.5 py-0.5 rounded">x</code> is not in the tuple. Check with <code className="bg-muted px-1.5 py-0.5 rounded">x in t</code> first, or use <code className="bg-muted px-1.5 py-0.5 rounded">try</code>/<code className="bg-muted px-1.5 py-0.5 rounded">except</code>, or something like <code className="bg-muted px-1.5 py-0.5 rounded">next((i for i, v in enumerate(t) if v == x), None)</code> for a safe "index or None."</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">Update - Immutability and nested mutables</p>
                  <p className="text-sm text-muted-foreground">The tuple itself cannot be changed, but if an element is mutable (e.g. a list), that object's contents can change. <code className="bg-muted px-1.5 py-0.5 rounded">t = (1, [2, 4])</code>; <code className="bg-muted px-1.5 py-0.5 rounded">t[1].append(6)</code> is allowed and makes <code className="bg-muted px-1.5 py-0.5 rounded">t</code> equal to <code className="bg-muted px-1.5 py-0.5 rounded">(1, [2, 4, 6])</code>.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">Join - <code className="bg-muted px-1.5 py-0.5 rounded">t + other</code> with non-tuple</p>
                  <p className="text-sm text-muted-foreground">Concatenation requires both sides to be tuples. <code className="bg-muted px-1.5 py-0.5 rounded">(2, 4) + [6, 8]</code> raises <code className="bg-muted px-1.5 py-0.5 rounded">TypeError</code>. Use <code className="bg-muted px-1.5 py-0.5 rounded">(2, 4) + tuple([6, 8])</code> or convert the other iterable to a tuple first.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">Sort - <code className="bg-muted px-1.5 py-0.5 rounded">sorted(t)</code> returns a list</p>
                  <p className="text-sm text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">sorted(t)</code> gives a list, not a tuple. Use <code className="bg-muted px-1.5 py-0.5 rounded">tuple(sorted(t))</code> when you need a sorted tuple. Forgetting the wrapper is a common mistake when coming from other languages or when you assume <code className="bg-muted px-1.5 py-0.5 rounded">sorted()</code> returns the same type.</p>
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
                  <p className="font-semibold text-foreground mb-1">How do you create a single-element tuple?</p>
                  <p className="text-muted-foreground">Use a trailing comma: <code className="bg-muted px-1.5 py-0.5 rounded">(2,)</code>. Without the comma, <code className="bg-muted px-1.5 py-0.5 rounded">(2)</code> is just an integer in parentheses.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">Why does a tuple have only <code className="bg-muted px-1.5 py-0.5 rounded">count</code> and <code className="bg-muted px-1.5 py-0.5 rounded">index</code>?</p>
                  <p className="text-muted-foreground">Tuples are immutable, so they have no mutating methods. <code className="bg-muted px-1.5 py-0.5 rounded">count</code> and <code className="bg-muted px-1.5 py-0.5 rounded">index</code> are the only read-only operations needed; no <code className="bg-muted px-1.5 py-0.5 rounded">append</code>, <code className="bg-muted px-1.5 py-0.5 rounded">remove</code>, <code className="bg-muted px-1.5 py-0.5 rounded">sort</code>, or <code className="bg-muted px-1.5 py-0.5 rounded">reverse</code>.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">How do you reverse a tuple?</p>
                  <p className="text-muted-foreground">Use <code className="bg-muted px-1.5 py-0.5 rounded">tuple(reversed(t))</code> or <code className="bg-muted px-1.5 py-0.5 rounded">t[::-1]</code>. Both create a <span className="font-semibold">new</span> tuple in reverse order; the original is unchanged. Tuples have no <code className="bg-muted px-1.5 py-0.5 rounded">reverse()</code> method (unlike lists) because they are immutable.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">What does <code className="bg-muted px-1.5 py-0.5 rounded">*</code> do in tuple unpacking?</p>
                  <p className="text-muted-foreground">Collects remaining elements into a list: <code className="bg-muted px-1.5 py-0.5 rounded">first, *rest = nums</code> → <code className="bg-muted px-1.5 py-0.5 rounded">rest = [4, 6, 8, 10]</code>. Use at any position: <code className="bg-muted px-1.5 py-0.5 rounded">a, *mid, z = nums</code>.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

      
        
      </div>
    </TopicContent>
  );
}