import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function FunctionsPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "functions");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Functions
          </h1>
          <p className="text-muted-foreground text-lg">
            Functions are reusable blocks of code that accept inputs, perform a task, and optionally return a result.
          </p>
          <p className="text-muted-foreground mt-2">
            Functions are the backbone of every Python program. Understanding how arguments, scope, closures, and decorators work will separate a confident Python developer from someone who just knows the syntax.
          </p>
        </div>

        {/* Defining and Calling Functions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Defining and Calling Functions
          </h2>
          <p className="text-muted-foreground mb-4">
            A function is defined with <code className="bg-muted px-1.5 py-0.5 rounded">def</code>, a name, parentheses, and a colon. The body is indented. Call it by writing its name followed by <code className="bg-muted px-1.5 py-0.5 rounded">()</code>.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def greet():
    print("Hello")
 
greet()  # Hello`}
            </pre>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mb-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground">Key Points:</p>
                <ul className="text-muted-foreground text-sm list-disc ml-4 mt-1">
                  <li>Use lowercase with underscores (snake_case) for function names</li>
                  <li>Names should describe what the function does, not how</li>
                  <li>A function without an explicit <code className="bg-muted px-1.5 py-0.5 rounded">return</code> returns <code className="bg-muted px-1.5 py-0.5 rounded">None</code> implicitly</li>
                </ul>
              </div>
            </div>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def add(a, b):
    return a + b
 
result = add(2, 4)  # 6`}
            </pre>
          </div>
        </section>

        {/* Parameters vs Arguments */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Parameters vs Arguments
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Term</th>
                  <th className="text-left p-3 font-semibold text-foreground">Definition</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">Parameter</td>
                  <td className="p-3">The variable name listed in the function definition</td>
                  <td className="p-3 font-mono">def add(a, b): - a and b</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">Argument</td>
                  <td className="p-3">The actual value passed when calling the function</td>
                  <td className="p-3 font-mono">add(2, 4) - 2 and 4</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Positional and Keyword Arguments */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Positional and Keyword Arguments
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def subtract(x, y):
    return x - y
 
subtract(8, 2)  # 6  (x=8, y=2)
subtract(2, 8)  # -6 (x=2, y=8) - order matters

# Keyword arguments
def describe(name, age):
    print(f"{name} is {age}")
 
describe(age=20, name="Alice")  # keyword - order doesn't matter
describe("Alice", age=20)       # positional first, then keyword`}
            </pre>
          </div>
          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <span className="font-semibold">Rule:</span> Positional arguments must come before keyword arguments in a call.
              </p>
            </div>
          </div>
        </section>

        {/* Default Parameter Values */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Default Parameter Values
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def power(base, exp=2):
    return base ** exp
 
power(4)     # 16 - exp defaults to 2
power(2, 4)  # 16 - exp overridden`}
            </pre>
          </div>
          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground">Mutable Default Trap</p>
                <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                  {`# WRONG - list persists across calls
def append_to(x, lst=[]):
    lst.append(x)
    return lst
 
append_to(2)   # [2]
append_to(4)   # [2, 4] - same list reused!

# CORRECT
def append_to(x, lst=None):
    if lst is None:
        lst = []
    lst.append(x)
    return lst`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* *args and **kwargs */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            *args and **kwargs
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# *args - variable positional arguments (tuple)
def total(*args):
    return sum(args)
 
total(2, 4)         # 6
total(2, 4, 6, 8)   # 20

# **kwargs - variable keyword arguments (dict)
def display(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")
 
display(x=2, y=4, z=6)
# x: 2
# y: 4
# z: 6

# Spread with * and **
nums = [2, 4, 6]
total(*nums)  # same as total(2, 4, 6) -> 12

data = {"x": 2, "y": 4}
display(**data)  # same as display(x=2, y=4)`}
            </pre>
          </div>
        </section>

        {/* Parameter Order */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Parameter Order
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Position</th>
                  <th className="text-left p-3 font-semibold text-foreground">Type</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3">1</td><td className="p-3">Regular positional</td><td className="p-3 font-mono">a, b</td></tr>
                <tr className="border-b border-border"><td className="p-3">2</td><td className="p-3">*args</td><td className="p-3 font-mono">*args</td></tr>
                <tr className="border-b border-border"><td className="p-3">3</td><td className="p-3">Keyword-only (after *args)</td><td className="p-3 font-mono">c, d</td></tr>
                <tr><td className="p-3">4</td><td className="p-3">**kwargs</td><td className="p-3 font-mono">**kwargs</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def full(a, b, *args, c=0, **kwargs):
    print(a, b, args, c, kwargs)
 
full(2, 4, 6, 8, c=10, x=12)
# 2 4 (6, 8) 10 {'x': 12}`}
            </pre>
          </div>
        </section>

        {/* Return Values */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Return Values
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Single return
def square(n):
    return n * n
 
result = square(4)  # 16

# Multiple return values (tuple packing/unpacking)
def min_max(nums):
    return min(nums), max(nums)   # returns tuple (min, max)
 
lo, hi = min_max([2, 8, 4, 6])   # lo=2, hi=8

# Early return
def first_even(nums):
    for n in nums:
        if n % 2 == 0:
            return n
    return None  # none found`}
            </pre>
          </div>
        </section>

        {/* Scope and LEGB Rule */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Scope and the LEGB Rule
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Scope</th>
                  <th className="text-left p-3 font-semibold text-foreground">Where</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3">Local</td><td className="p-3">Inside the current function</td><td className="p-3 font-mono">Variable declared in def f():</td></tr>
                <tr className="border-b border-border"><td className="p-3">Enclosing</td><td className="p-3">Inside an outer function</td><td className="p-3 font-mono">Outer function's variables</td></tr>
                <tr className="border-b border-border"><td className="p-3">Global</td><td className="p-3">Module level (top of the file)</td><td className="p-3 font-mono">x = 2 at module level</td></tr>
                <tr><td className="p-3">Built-in</td><td className="p-3">Python's built-in namespace</td><td className="p-3 font-mono">len, range, print</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`x = 2  # global
 
def outer():
    x = 4  # enclosing (from inner's perspective)
 
    def inner():
        x = 6  # local
        print(x)  # 6 - local wins
 
    inner()
    print(x)  # 4 - outer's local
 
outer()
print(x)  # 2 - global unchanged`}
            </pre>
          </div>
        </section>

        {/* Lambda Functions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Lambda Functions
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Lambda is a single-expression anonymous function
square = lambda n: n * n
square(4)  # 16
 
add = lambda a, b: a + b
add(2, 6)  # 8

# Used as callbacks
pairs = [(2, "b"), (4, "a"), (6, "c")]
pairs.sort(key=lambda p: p[1])  # sort by second element
# [(4, 'a'), (2, 'b'), (6, 'c')]`}
            </pre>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                Limit: Only use lambdas for genuinely simple cases. A named <code className="bg-muted px-1.5 py-0.5 rounded">def</code> is clearer and easier to test whenever the logic is more than a line.
              </p>
            </div>
          </div>
        </section>

        {/* Closures */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Closures
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def multiplier(factor):
    def multiply(n):
        return n * factor  # factor is "closed over"
    return multiply
 
double = multiplier(2)
quadruple = multiplier(4)
double(6)      # 12
quadruple(6)   # 24

# Late binding in closures - common pitfall
funcs = [lambda: i for i in range(4)]
[f() for f in funcs]  # [3, 3, 3, 3] - all see i=3 at call time
 
# Fix: capture the current value as a default argument
funcs = [lambda i=i: i for i in range(4)]
[f() for f in funcs]  # [0, 1, 2, 3]`}
            </pre>
          </div>
        </section>

        {/* Decorators */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Decorators
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import functools

def shout(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return str(result).upper()
    return wrapper
 
@shout
def greet(name):
    return f"hello {name}"
 
greet("world")  # 'HELLO WORLD'

# Decorator with arguments
def repeat(n):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for _ in range(n):
                fn(*args, **kwargs)
        return wrapper
    return decorator
 
@repeat(2)
def say(msg):
    print(msg)`}
            </pre>
          </div>
        </section>

        {/* Recursion */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Recursion
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`from functools import lru_cache

def factorial(n):
    if n == 0:        # base case
        return 1
    return n * factorial(n - 1)  # recursive case
 
factorial(4)   # 24

# Fibonacci with caching for performance
@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n - 2) + fib(n - 1)
 
fib(6)  # 8`}
            </pre>
          </div>
        </section>

        {/* Generator Functions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Generator Functions
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def even_up_to(limit):
    n = 0
    while n <= limit:
        yield n
        n += 2
 
gen = even_up_to(8)
next(gen)  # 0
next(gen)  # 2
next(gen)  # 4
 
list(even_up_to(8))  # [0, 2, 4, 6, 8]

# Generator expression (lazy)
gen = (x * 2 for x in [2, 4, 6])
list(gen)  # [4, 8, 12]`}
            </pre>
          </div>
        </section>

        {/* Pure Functions and Side Effects */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Pure Functions and Side Effects
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Pure function - same output for same input, no side effects
def add(a, b):
    return a + b

# Impure - has side effects
results = []
def collect(x):
    results.append(x)   # side effect: mutates external list
    return x

# Mutable argument side-effect trap
def double_items(lst):
    for i in range(len(lst)):
        lst[i] *= 2   # mutates caller's list!

# CORRECT - return a new object
def double_items(lst):
    return [x * 2 for x in lst]`}
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
                  <p className="font-semibold text-foreground">Calling - Positional before keyword</p>
                  <p className="text-muted-foreground">Positional arguments must come before keyword arguments in any call.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Return - Implicit None</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`lst = [3, 1, 2]
result = lst.sort()    # bug: sort() returns None
print(result)          # None
 
result = sorted(lst)   # correct: sorted() returns new sorted list`}
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
                  <p className="font-semibold text-foreground">What is the difference between a parameter and an argument?</p>
                  <p className="text-muted-foreground">A parameter is the name listed in the function definition. An argument is the value passed when the function is called. Parameters are placeholders; arguments are the actual values that fill them.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What happens when a mutable object is used as a default argument?</p>
                  <p className="text-muted-foreground">The default is created once when the function is defined. Every call that omits the argument shares the same mutable object. Use None as the default and create the mutable object inside the function body.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the LEGB rule?</p>
                  <p className="text-muted-foreground">Python searches for names in this order: Local, Enclosing, Global, Built-in. The first match wins.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

 
      </div>
    </TopicContent>
  );
}