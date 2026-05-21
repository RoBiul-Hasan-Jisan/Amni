import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function IteratorsPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "iterators");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Iterators
          </h1>
          <p className="text-muted-foreground text-lg">
            An <strong className="text-foreground">iterator</strong> is an object that yields values one after another. When something is looped over with <code className="bg-muted px-1.5 py-0.5 rounded">for</code>, or when <code className="bg-muted px-1.5 py-0.5 rounded">next()</code> is called on it, an iterator is doing the work behind the scenes.
          </p>
        </div>

        {/* Note Box */}
        <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
          <div className="flex gap-3">
            <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
            <p className="text-sm text-foreground">
              Many built-in types (lists, tuples, strings, sets, dictionaries) are <strong>iterable</strong>: they can produce an iterator. The iterator is the object that actually steps through the values. That distinction (iterable vs iterator) is where a lot of confusion comes from.
            </p>
          </div>
        </div>

        {/* What is an Iterator? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            What is an Iterator?
          </h2>
          <p className="text-muted-foreground mb-4">
            An iterator is an object that supports two operations:
          </p>
          <ul className="list-disc list-inside text-muted-foreground mb-4 space-y-1 ml-4">
            <li><code className="bg-muted px-1.5 py-0.5 rounded">__iter__</code> – returns the iterator itself (so iterators are also iterables)</li>
            <li><code className="bg-muted px-1.5 py-0.5 rounded">__next__</code> – returns the next value, or raises <code className="bg-muted px-1.5 py-0.5 rounded">StopIteration</code> when there are no more items</li>
          </ul>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
            <div className="flex gap-3">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <strong>Lazy evaluation:</strong> Values are produced when requested, not up front. <code className="bg-muted px-1.5 py-0.5 rounded">range(0, 10**8)</code> does not build 10⁸ integers; it produces them as you iterate.
              </p>
            </div>
          </div>
        </section>

        {/* Iterables vs Iterators */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Iterables vs Iterators
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Concept</th>
                  <th className="text-left p-3 font-semibold text-foreground">Meaning</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">Iterable</td>
                  <td className="p-3">Any object that can be passed to <code className="bg-muted px-1.5 py-0.5 rounded">iter()</code> and yields an iterator</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">Iterator</td>
                  <td className="p-3">An object with <code className="bg-muted px-1.5 py-0.5 rounded">__iter__</code> (returns self) and <code className="bg-muted px-1.5 py-0.5 rounded">__next__</code> (next item or <code className="bg-muted px-1.5 py-0.5 rounded">StopIteration</code>)</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mt-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`nums = [2, 4, 6, 8]
it = iter(nums)
type(it)   # <class 'list_iterator'>
next(it)   # 2
next(it)   # 4
next(it)   # 6
next(it)   # 8
# next(it)  # StopIteration`}
            </pre>
          </div>
        </section>

        {/* The Iterator Protocol */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            The Iterator Protocol
          </h2>
          <p className="text-muted-foreground mb-4">
            The protocol is minimal:
          </p>
          <ol className="list-decimal list-inside text-muted-foreground mb-4 space-y-1 ml-4">
            <li><code className="bg-muted px-1.5 py-0.5 rounded">iter(obj)</code> calls <code className="bg-muted px-1.5 py-0.5 rounded">obj.__iter__()</code></li>
            <li><code className="bg-muted px-1.5 py-0.5 rounded">next(obj)</code> calls <code className="bg-muted px-1.5 py-0.5 rounded">obj.__next__()</code> - returns next value or raises <code className="bg-muted px-1.5 py-0.5 rounded">StopIteration</code></li>
          </ol>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# What "for x in seq:" does under the hood
seq = [2, 4, 6]
it = iter(seq)
while True:
    try:
        x = next(it)
        print(x)  # 2, 4, 6
    except StopIteration:
        break`}
            </pre>
          </div>
        </section>

        {/* Creating and Obtaining Iterators */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Creating and Obtaining Iterators
          </h2>
          
          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">iter()</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`iter([2, 4, 6])     # list_iterator
iter((2, 4, 6))     # tuple_iterator
iter("246")         # str_iterator
iter({2, 4, 6})     # set_iterator
iter({2: "a", 4: "b"})  # dict_keyiterator`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">next()</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`it = iter([2, 4, 6, 8])
next(it)         # 2
next(it)         # 4
next(it, 0)      # 6
next(it, 0)      # 8
next(it, 0)      # 0 - exhausted, default used`}
            </pre>
          </div>
        </section>

        {/* Consuming an Iterator */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Consuming an Iterator
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Approach</th>
                  <th className="text-left p-3 font-semibold text-foreground">Effect</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">for x in it:</td><td className="p-3">Loop over remaining items; stops when exhausted</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">list(it)</td><td className="p-3">Build a list from all remaining items; iterator exhausted</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">tuple(it)</td><td className="p-3">Build a tuple from all remaining items; iterator exhausted</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">sum(it), max(it)</td><td className="p-3">Consume the iterator; return one value</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">next(it)</td><td className="p-3">Take one item; advance the iterator by one</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Built-in Iterators */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Built-in Iterators and Tools
          </h2>
          
          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">enumerate()</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`list(enumerate([2, 4, 6]))           # [(0, 2), (1, 4), (2, 6)]
list(enumerate([2, 4, 6], start=2))   # [(2, 2), (3, 4), (4, 6)]`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">zip()</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`list(zip([2, 4, 6], [8, 10, 12]))           # [(2, 8), (4, 10), (6, 12)]
list(zip([2, 4], [6, 8, 10]))               # [(2, 6), (4, 8)] - stops at shortest
list(zip([2, 4, 6]))                        # [(2,), (4,), (6,)]
list(zip())                                 # []`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">map()</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`list(map(lambda x: x * 2, [2, 4, 6]))           # [4, 8, 12]
list(map(lambda a, b: a + b, [2, 4], [6, 8]))   # [8, 12]`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">filter()</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`list(filter(lambda x: x > 4, [2, 4, 6, 8]))   # [6, 8]
list(filter(None, [0, 2, "", 4, None, 6]))    # [2, 4, 6] - removes falsy`}
            </pre>
          </div>
        </section>

        {/* Custom Iterators */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Custom Iterators (Class-Based)
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class EvensUpTo:
    def __init__(self, limit):
        self.limit = limit
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.limit:
            raise StopIteration
        value = self.current
        self.current += 2
        return value

list(EvensUpTo(10))   # [0, 2, 4, 6, 8]`}
            </pre>
          </div>
        </section>

        {/* Generators as Iterators */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Generators as Iterators
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def evens_up_to(limit):
    n = 0
    while n < limit:
        yield n
        n += 2

g = evens_up_to(10)
next(g)   # 0
next(g)   # 2
list(g)   # [4, 6, 8] - consumes the rest

# Generator expression
squares = (x * x for x in [2, 4, 6, 8])
next(squares)   # 4
next(squares)   # 16
list(squares)   # [36, 64]`}
            </pre>
          </div>
        </section>

        {/* itertools */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            itertools Module
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Tool</th>
                  <th className="text-left p-3 font-semibold text-foreground">What it does</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">chain(it1, it2, ...)</td><td className="p-3">Concatenate iterables without building a list</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">islice(it, stop)</td><td className="p-3">Take the first stop items from an iterator</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">zip_longest(*its, fillvalue)</td><td className="p-3">Like zip but pads shorter iterables</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">tee(it, n=2)</td><td className="p-3">Return n independent iterators (at a memory cost)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">count(start, step)</td><td className="p-3">Infinite counter</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`from itertools import chain, islice, zip_longest, tee

list(chain([2, 4], [6, 8], [10]))          # [2, 4, 6, 8, 10]
list(islice(range(100), 4))               # [0, 1, 2, 3]
list(zip_longest([2, 4], [6], fillvalue=0)) # [(2, 6), (4, 0)]`}
            </pre>
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
                  <p className="font-semibold text-foreground">iter() on a non-iterable</p>
                  <p className="text-muted-foreground">Passing a non-iterable to <code className="bg-muted px-1.5 py-0.5 rounded">iter()</code> (e.g. <code className="bg-muted px-1.5 py-0.5 rounded">iter(42)</code>) raises <code className="bg-muted px-1.5 py-0.5 rounded">TypeError</code>.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Exhaustion: second loop does nothing</p>
                  <p className="text-muted-foreground">After a <code className="bg-muted px-1.5 py-0.5 rounded">for</code> loop over an iterator, that iterator is exhausted. Create a new iterator with <code className="bg-muted px-1.5 py-0.5 rounded">iter(iterable)</code> if you need to loop again.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">range is not an iterator</p>
                  <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">range</code> is an iterable. <code className="bg-muted px-1.5 py-0.5 rounded">next(range(4))</code> is invalid; use <code className="bg-muted px-1.5 py-0.5 rounded">next(iter(range(4)))</code>.</p>
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
                  <p className="font-semibold text-foreground">What is the difference between an iterable and an iterator?</p>
                  <p className="text-muted-foreground">An <strong>iterable</strong> is any object that can be passed to <code className="bg-muted px-1.5 py-0.5 rounded">iter()</code> and yields an iterator. An <strong>iterator</strong> is an object with <code className="bg-muted px-1.5 py-0.5 rounded">__iter__</code> and <code className="bg-muted px-1.5 py-0.5 rounded">__next__</code> methods.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">How does a for loop use the iterator protocol?</p>
                  <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">for x in obj:</code> calls <code className="bg-muted px-1.5 py-0.5 rounded">iter(obj)</code> to get an iterator, then repeatedly calls <code className="bg-muted px-1.5 py-0.5 rounded">next()</code> until <code className="bg-muted px-1.5 py-0.5 rounded">StopIteration</code> is raised.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What happens when you iterate over the same iterator twice?</p>
                  <p className="text-muted-foreground">The first iteration consumes the iterator. The second iteration has no items left, so the loop body does not run.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

      
      </div>
    </TopicContent>
  );
}