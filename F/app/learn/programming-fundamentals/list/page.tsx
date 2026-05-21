import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function ListPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "list");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Lists
          </h1>
          <p className="text-muted-foreground text-lg">
            Lists are ordered, mutable sequences that can hold mixed types. Lists show up everywhere in Python - and in interviews.
          </p>
        </div>

        {/* Creating Lists */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Creating Lists
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Approach</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                  <th className="text-left p-3 font-semibold text-foreground">Use when</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">Literal []</td><td className="p-3 font-mono">row = [2, 4, 6]</td><td className="p-3">You know the elements</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">Empty list</td><td className="p-3 font-mono">seq = []</td><td className="p-3">Start empty, append or extend later</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">list()</td><td className="p-3 font-mono">list() → []</td><td className="p-3">Empty list (same as [])</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">list(iterable)</td><td className="p-3 font-mono">list(range(4)) → [0, 1, 2, 3]</td><td className="p-3">Convert from another iterable</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`row = [2, 4, 6, 8]
empty = []
from_range = list(range(0, 6, 2))   # [0, 2, 4]
from_string = list("ab")             # ['a', 'b']
 
len(row)    # 4
len(empty)  # 0
if row:     # True
if not empty:  # True`}
            </pre>
          </div>
        </section>

        {/* List Methods Reference */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            List Methods
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
                <tr className="border-b border-border"><td className="p-3 font-mono">count()</td><td className="p-3">Count occurrences of a value</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">index()</td><td className="p-3">Return index of first occurrence; raises if missing</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">append()</td><td className="p-3">Add element at the end</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">insert()</td><td className="p-3">Insert element at index</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">extend()</td><td className="p-3">Append all items from an iterable</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">remove()</td><td className="p-3">Remove first occurrence of value</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">pop()</td><td className="p-3">Remove and return element at index (default last)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">clear()</td><td className="p-3">Remove all elements</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">copy()</td><td className="p-3">Return a shallow copy</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">reverse()</td><td className="p-3">Reverse order in place</td></tr>
                <tr><td className="p-3 font-mono">sort()</td><td className="p-3">Sort in place</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Access List Items */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Access List Items
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Pattern</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">lst[i]</td><td className="p-3">Single element at index i</td><td className="p-3 font-mono">vals[2] → element at index 2</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">lst[-1]</td><td className="p-3">Last element</td><td className="p-3 font-mono">vals[-1] → last element</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">lst[i:j]</td><td className="p-3">Slice from i to j (excl.)</td><td className="p-3 font-mono">vals[2:6] → indices 2, 3, 4, 5</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">lst[:n]</td><td className="p-3">First n elements</td><td className="p-3 font-mono">vals[:4] → indices 0–3</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">lst[n:]</td><td className="p-3">From index n to end</td><td className="p-3 font-mono">vals[4:] → indices 4 onward</td></tr>
                <tr><td className="p-3 font-mono">lst[::-1]</td><td className="p-3">Reversed (step -1)</td><td className="p-3 font-mono">vals[::-1] → reversed copy</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`vals = [2, 4, 6, 8, 10, 12, 14, 16]
vals[2]         # 6
vals[-1]        # 16
vals[2:6]       # [6, 8, 10, 12]
vals[:4]        # [2, 4, 6, 8]
vals[4:]        # [10, 12, 14, 16]
vals[::2]       # [2, 6, 10, 14]
vals[::-1]      # [16, 14, 12, 10, 8, 6, 4, 2]

4 in vals       # True
vals.count(6)   # 1
vals.index(8)   # 3`}
            </pre>
          </div>
        </section>

        {/* Add List Items */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Add List Items
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Method</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">lst.append(x)</td><td className="p-3">Add x at the end</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">lst.insert(i, x)</td><td className="p-3">Insert x at index i</td></tr>
                <tr><td className="p-3 font-mono">lst.extend(iter)</td><td className="p-3">Append all items from an iterable</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`evens = [2, 4, 6]
evens.append(8)         # [2, 4, 6, 8]
evens.insert(1, 0)      # [2, 0, 4, 6, 8]
more = [10, 12]
evens.extend(more)      # [2, 0, 4, 6, 8, 10, 12]`}
            </pre>
          </div>
        </section>

        {/* Remove List Items */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Remove List Items
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Method / Keyword</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">lst.remove(x)</td><td className="p-3">Remove first occurrence of x; raises ValueError if missing</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">lst.pop(i)</td><td className="p-3">Remove and return element at index i; default -1 (last)</td></tr>
                <tr><td className="p-3 font-mono">del lst[i]</td><td className="p-3">Delete element at index i (or slice i:j)</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`data = [2, 4, 6, 4, 8]
data.remove(4)     # [2, 6, 4, 8] - first 4 removed
data.pop(2)        # returns 4; data = [2, 6, 8]
data.pop()         # returns 8; data = [2, 6]
del data[0]        # data = [6]`}
            </pre>
          </div>
        </section>

        {/* Comprehension */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            List Comprehension
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Syntax: [expr for item in iterable if condition]

evens = [2, 4, 6, 8, 10]
doubled = [x * 2 for x in evens]                   # [4, 8, 12, 16, 20]
filtered = [x for x in evens if x > 4]             # [6, 8, 10]
from_range = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
squared = [x ** 2 for x in evens]                  # [4, 16, 36, 64, 100]`}
            </pre>
          </div>
        </section>

        {/* Copy Lists */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Copy Lists
          </h2>
          
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mb-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground">Shallow Copy Approaches</p>
                <div className="overflow-x-auto mt-2">
                  <table className="min-w-full">
                    <thead><tr className="text-muted-foreground"><th className="text-left p-2">Approach</th><th className="text-left p-2">Example</th></tr></thead>
                    <tbody className="text-muted-foreground text-sm">
                      <tr><td className="p-2 font-mono">lst.copy()</td><td className="p-2 font-mono">new = orig.copy()</td></tr>
                      <tr><td className="p-2 font-mono">list(lst)</td><td className="p-2 font-mono">new = list(orig)</td></tr>
                      <tr><td className="p-2 font-mono">Slice [:]</td><td className="p-2 font-mono">new = orig[:]</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import copy
 
# Shallow copy - fine for flat lists
a = [2, 4, 6]
b = a.copy() # b = [2, 4, 6]
a.append(8)  # a = [2, 4, 6, 8]; b unchanged
 
# Deep copy - fully independent for nested lists
original = [[1, 2], [3, 4]]
deep = copy.deepcopy(original)
deep[0].append(99)  # original unchanged`}
            </pre>
          </div>
        </section>

        {/* Sort Lists */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Sort Lists
          </h2>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mb-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <code className="bg-muted px-1.5 py-0.5 rounded">lst.sort()</code> sorts in place, returns None. <code className="bg-muted px-1.5 py-0.5 rounded">sorted(lst)</code> returns new sorted list, original unchanged.
              </p>
            </div>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`nums = [22, 4, 16, 8, 10]
nums.sort()                 # [4, 8, 10, 16, 22]
nums.sort(reverse=True)     # [22, 16, 10, 8, 4]

# sort() vs sorted()
row = [6, 2, 8, 4]
row.sort()              # row is now [2, 4, 6, 8]
out = sorted(row)       # out = [2, 4, 6, 8]; row unchanged

# Sort by key function
nums.sort(key=lambda n: abs(n - 12))  # sort by distance from 12`}
            </pre>
          </div>
        </section>

        {/* List vs Other Collections */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            List vs Other Collections
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Type</th>
                  <th className="text-left p-3 font-semibold text-foreground">Ordered?</th>
                  <th className="text-left p-3 font-semibold text-foreground">Mutable?</th>
                  <th className="text-left p-3 font-semibold text-foreground">Duplicates?</th>
                  <th className="text-left p-3 font-semibold text-foreground">Typical use</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">list</td><td className="p-3">Yes</td><td className="p-3">Yes</td><td className="p-3">Yes</td><td className="p-3">Ordered sequences you change</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">tuple</td><td className="p-3">Yes</td><td className="p-3">No</td><td className="p-3">Yes</td><td className="p-3">Fixed sequences (dict keys)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">set</td><td className="p-3">No</td><td className="p-3">Yes</td><td className="p-3">No</td><td className="p-3">Unique items, membership tests</td></tr>
                <tr><td className="p-3 font-mono">dict</td><td className="p-3">Yes (insertion)</td><td className="p-3">Yes (values)</td><td className="p-3">No (keys)</td><td className="p-3">Key–value mapping</td></tr>
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
                  <p className="font-semibold text-foreground">Loop - Modifying while iterating</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`# WRONG - skip elements
lst = [1, 2, 3, 4]
for i, item in enumerate(lst):
    if item % 2 == 0:
        del lst[i]

# CORRECT - iterate over copy
for item in lst[:]:
    if item % 2 == 0:
        lst.remove(item)

# CORRECT - list comprehension
lst = [item for item in lst if item % 2 != 0]`}
                  </pre>
                </div>
              </div>
            </div>

            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Mutable default argument</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`# WRONG - list persists across calls
def add_item(item, lst=[]):
    lst.append(item)
    return lst

# CORRECT
def add_item(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst`}
                  </pre>
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
                  <p className="font-semibold text-foreground">What is the time complexity of list append, insert, and index lookup?</p>
                  <p className="text-muted-foreground">Append O(1) amortized; insert O(n); index lookup O(1); in O(n). List is a dynamic array.</p>
                </div>
              </div>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">sort() vs sorted() - in place vs new list?</p>
                  <p className="text-muted-foreground">sort() mutates the list in place and returns None. sorted() returns a new list and leaves the original unchanged.</p>
                </div>
              </div>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the difference between lst.clear() and del lst?</p>
                  <p className="text-muted-foreground">lst.clear() empties the list in place; the list object still exists. del lst removes the name lst; the list may be garbage-collected.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

       
      </div>
    </TopicContent>
  );
}