import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function LambdaPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "lambda");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Lambda Functions
          </h1>
          <p className="text-muted-foreground text-lg">
            A <strong className="text-foreground">lambda</strong> is a small, anonymous function that can be defined inline. Unlike functions defined with <code className="bg-muted px-1.5 py-0.5 rounded">def</code>, a lambda is a <strong>single expression</strong> and returns its value implicitly.
          </p>
        </div>

        {/* Note Box */}
        <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
          <div className="flex gap-3">
            <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
            <p className="text-sm text-foreground">
              Lambdas shine where you need a quick, throwaway function for a short period (often as an argument to higher-order functions like <code className="bg-muted px-1.5 py-0.5 rounded">sorted()</code>, <code className="bg-muted px-1.5 py-0.5 rounded">map()</code>, or <code className="bg-muted px-1.5 py-0.5 rounded">filter()</code>). If the logic needs statements, multiple lines, or a docstring, use a regular <code className="bg-muted px-1.5 py-0.5 rounded">def</code>.
            </p>
          </div>
        </div>

        {/* What is a Lambda? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            What is a Lambda?
          </h2>
          <p className="text-muted-foreground mb-4">
            A lambda is an anonymous function. It has no name (unless you assign it to a variable, which is often pointless—just use <code className="bg-muted px-1.5 py-0.5 rounded">def</code> if you need a name). The body is a single expression that is evaluated and returned.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Syntax: lambda parameters: expression

# Lambda that squares a number
square = lambda x: x * x
square(4)  # 16

# Lambda with two parameters
add = lambda a, b: a + b
add(2, 6)  # 8`}
            </pre>
          </div>
        </section>

        {/* Lambda Syntax */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Lambda Syntax
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Component</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">lambda</td>
                  <td className="p-3">Keyword that creates the function</td>
                  <td className="p-3 font-mono">lambda</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">parameters</td>
                  <td className="p-3">Comma-separated parameter names (optional)</td>
                  <td className="p-3 font-mono">x, y</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">:</td>
                  <td className="p-3">Separator between parameters and expression</td>
                  <td className="p-3 font-mono">:</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">expression</td>
                  <td className="p-3">Single expression that is evaluated and returned</td>
                  <td className="p-3 font-mono">x + y</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Lambda vs Regular Function */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Lambda vs Regular Function
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Aspect</th>
                  <th className="text-left p-3 font-semibold text-foreground">Lambda</th>
                  <th className="text-left p-3 font-semibold text-foreground">def Function</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3">Name</td><td className="p-3">Anonymous</td><td className="p-3">Has a name</td></tr>
                <tr className="border-b border-border"><td className="p-3">Body</td><td className="p-3">Single expression</td><td className="p-3">Multiple statements</td></tr>
                <tr className="border-b border-border"><td className="p-3">Return</td><td className="p-3">Implicit (expression result)</td><td className="p-3">Explicit <code className="bg-muted px-1.5 py-0.5 rounded">return</code></td></tr>
                <tr className="border-b border-border"><td className="p-3">Docstring</td><td className="p-3">Not possible</td><td className="p-3">Supported</td></tr>
                <tr className="border-b border-border"><td className="p-3">Statements</td><td className="p-3">Not allowed</td><td className="p-3">Any statements (if, for, while, etc.)</td></tr>
                <tr className="border-b border-border"><td className="p-3">Use case</td><td className="p-3">Short, throwaway functions</td><td className="p-3">Anything complex or reusable</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Regular function
def add(a, b):
    \"\"\"Return the sum of a and b.\"\"\"
    return a + b

# Lambda equivalent
add_lambda = lambda a, b: a + b

# Both work the same way
add(2, 4)        # 6
add_lambda(2, 4) # 6`}
            </pre>
          </div>
        </section>

        {/* Common Use Cases */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Common Use Cases
          </h2>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">1. Sorting with key</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Sort by second element of tuple
pairs = [(2, "b"), (4, "a"), (6, "c")]
pairs.sort(key=lambda p: p[1])
# [(4, 'a'), (2, 'b'), (6, 'c')]

# Sort strings by length
words = ["apple", "fig", "banana", "date"]
sorted_words = sorted(words, key=lambda w: len(w))
# ['fig', 'date', 'apple', 'banana']

# Sort by distance from a number
nums = [22, 4, 16, 8, 10]
nums.sort(key=lambda n: abs(n - 12))
# [10, 8, 16, 4, 22]`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">2. map() - Transform each element</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`nums = [2, 4, 6, 8]
doubled = list(map(lambda x: x * 2, nums))
# [4, 8, 12, 16]

# Convert strings to uppercase
words = ["hello", "world"]
upper = list(map(lambda w: w.upper(), words))
# ['HELLO', 'WORLD']`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">3. filter() - Keep elements that satisfy condition</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`nums = [1, 2, 3, 4, 5, 6, 7, 8]
evens = list(filter(lambda x: x % 2 == 0, nums))
# [2, 4, 6, 8]

# Filter words longer than 3 characters
words = ["hi", "hello", "hey", "greetings"]
long = list(filter(lambda w: len(w) > 3, words))
# ['hello', 'greetings']`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">4. reduce() - Fold a sequence</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`from functools import reduce

nums = [2, 4, 6, 8]
product = reduce(lambda a, b: a * b, nums)
# 2 * 4 * 6 * 8 = 384

# Find maximum
max_val = reduce(lambda a, b: a if a > b else b, nums)
# 8`}
            </pre>
          </div>
        </section>

        {/* Lambda with Conditional Expression */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Lambda with Conditional Expressions
          </h2>
          <p className="text-muted-foreground mb-4">
            Lambdas can use Python's ternary operator (<code className="bg-muted px-1.5 py-0.5 rounded">value_if_true if condition else value_if_false</code>) for conditional logic.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Absolute value
abs_val = lambda x: x if x >= 0 else -x
abs_val(-5)  # 5

# Max of two numbers
max_val = lambda a, b: a if a > b else b
max_val(10, 20)  # 20

# Sign function (-1, 0, 1)
sign = lambda x: -1 if x < 0 else (1 if x > 0 else 0)
sign(-5)  # -1
sign(0)   # 0
sign(10)  # 1

# Using conditional in map
nums = [-2, -1, 0, 1, 2]
abs_nums = list(map(lambda x: x if x >= 0 else -x, nums))
# [2, 1, 0, 1, 2]`}
            </pre>
          </div>
        </section>

        {/* Lambda with Default Arguments */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Lambda with Default Arguments
          </h2>
          <p className="text-muted-foreground mb-4">
            Like regular functions, lambdas can have default parameter values.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Power with default exponent
power = lambda base, exp=2: base ** exp
power(4)    # 16
power(2, 3) # 8

# Greeting with default salutation
greet = lambda name, prefix="Hello": f"{prefix}, {name}!"
greet("Alice")           # "Hello, Alice!"
greet("Bob", "Hi")       # "Hi, Bob!"`}
            </pre>
          </div>
          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground">Mutable Default Arguments Trap</p>
                <p className="text-muted-foreground text-sm">
                  The same mutable default trap applies to lambdas. Avoid mutable defaults like <code className="bg-muted px-1.5 py-0.5 rounded">[]</code> or <code className="bg-muted px-1.5 py-0.5 rounded">{}</code>.
                </p>
                <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                  {`# WRONG - list persists across calls
bad = lambda x, lst=[]: lst.append(x) or lst
bad(2)  # [2]
bad(4)  # [2, 4]

# CORRECT - use None and create inside
good = lambda x, lst=None: (lst if lst is not None else []).append(x) or lst or [x]`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Lambda with *args and **kwargs */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Lambda with *args and **kwargs
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Variable positional arguments
sum_all = lambda *args: sum(args)
sum_all(2, 4, 6, 8)  # 20

# Variable keyword arguments
make_dict = lambda **kwargs: kwargs
make_dict(a=2, b=4, c=6)  # {'a': 2, 'b': 4, 'c': 6}

# Combine positional and keyword
process = lambda x, y, *args, **kwargs: (x, y, args, kwargs)
process(2, 4, 6, 8, a=10, b=12)
# (2, 4, (6, 8), {'a': 10, 'b': 12})`}
            </pre>
          </div>
        </section>

        {/* Lambda in List Comprehensions and Loops */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Lambda in List Comprehensions
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Create a list of functions
functions = [lambda x: x * n for n in range(5)]

# All functions see the same n (4) - Late binding!
[f(2) for f in functions]  # [8, 8, 8, 8, 8] - not [0, 2, 4, 6, 8]

# Fix: capture n as default argument
functions = [lambda x, n=n: x * n for n in range(5)]
[f(2) for f in functions]  # [0, 2, 4, 6, 8]`}
            </pre>
          </div>
        </section>

        {/* When NOT to Use Lambda */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            When NOT to Use Lambda
          </h2>
          <div className="space-y-4">
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">❌ Complex logic</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`# BAD - too complex for lambda
fib = lambda n: n if n <= 1 else fib(n-1) + fib(n-2)  # Works but unreadable

# GOOD - use def
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)`}
                  </pre>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">❌ Need statements (if/else blocks, loops, try/except)</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`# BAD - can't use print or assignment
# lambda x: print(x) or x  # print returns None, works but hacky

# GOOD - use def
def log(x):
    print(x)
    return x`}
                  </pre>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">❌ Need a docstring</p>
                  <p className="text-muted-foreground">Lambdas cannot have docstrings. If the function needs documentation, use <code className="bg-muted px-1.5 py-0.5 rounded">def</code>.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Performance Considerations */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Performance Considerations
          </h2>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground mb-1">Lambda vs def Performance</p>
                <p className="text-muted-foreground text-sm">
                  Lambdas and regular functions have nearly identical performance. The choice should be based on readability, not speed. Use lambdas for short, simple operations where they make the code clearer.
                </p>
                <div className="overflow-x-auto mt-2">
                  <table className="min-w-full">
                    <thead>
                      <tr className="text-muted-foreground">
                        <th className="text-left p-2">Use Case</th>
                        <th className="text-left p-2">Recommendation</th>
                        <th className="text-left p-2">Example</th>
                      </tr>
                    </thead>
                    <tbody className="text-muted-foreground text-sm">
                      <tr><td className="p-2">Simple transformation</td><td className="p-2">Lambda</td><td className="p-2 font-mono">lambda x: x * 2</td></tr>
                      <tr><td className="p-2">Complex logic</td><td className="p-2">def</td><td className="p-2 font-mono">def process(x): ...</td></tr>
                      <tr><td className="p-2">Reusable helper</td><td className="p-2">def</td><td className="p-2 font-mono">def sort_key(item): ...</td></tr>
                    </tbody>
                  </table>
                </div>
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
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Lambda can't contain statements</p>
                  <p className="text-muted-foreground">No <code className="bg-muted px-1.5 py-0.5 rounded">print()</code> (unless as expression), no <code className="bg-muted px-1.5 py-0.5 rounded">return</code>, no <code className="bg-muted px-1.5 py-0.5 rounded">if</code> blocks (but ternary works), no loops.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Late binding in loops</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`# All functions see the final value of i
funcs = [lambda: i for i in range(5)]
[f() for f in funcs]  # [4, 4, 4, 4, 4]

# Fix: capture current value as default
funcs = [lambda i=i: i for i in range(5)]
[f() for f in funcs]  # [0, 1, 2, 3, 4]`}
                  </pre>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Lambdas are first-class objects</p>
                  <p className="text-muted-foreground">Lambdas can be stored in variables, passed as arguments, and returned from functions - just like regular functions.</p>
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
                  <p className="font-semibold text-foreground">What is a lambda function? When would you use it?</p>
                  <p className="text-muted-foreground">A lambda is an anonymous, single-expression function. Use it for short, throwaway operations like sorting keys, <code className="bg-muted px-1.5 py-0.5 rounded">map()</code> callbacks, or <code className="bg-muted px-1.5 py-0.5 rounded">filter()</code> predicates where the logic is simple and a full function definition would be overkill.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What are the limitations of lambda functions?</p>
                  <p className="text-muted-foreground">Lambdas can only contain a single expression (no statements like <code className="bg-muted px-1.5 py-0.5 rounded">return</code>, <code className="bg-muted px-1.5 py-0.5 rounded">if</code> blocks, loops, or <code className="bg-muted px-1.5 py-0.5 rounded">try</code>/<code className="bg-muted px-1.5 py-0.5 rounded">except</code>), can't have docstrings, and their body is implicitly returned.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Can lambda functions have default arguments? Can they use *args and **kwargs?</p>
                  <p className="text-muted-foreground">Yes to both. <code className="bg-muted px-1.5 py-0.5 rounded">lambda x, y=2: x * y</code> and <code className="bg-muted px-1.5 py-0.5 rounded">lambda *args, **kwargs: ...</code> work exactly like regular functions.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the output of <code className="bg-muted px-1.5 py-0.5 rounded">[lambda: i for i in range(3)][0]()</code>? Why?</p>
                  <p className="text-muted-foreground">It returns <code className="bg-muted px-1.5 py-0.5 rounded">2</code> (or <code className="bg-muted px-1.5 py-0.5 rounded">i</code>'s final value). The lambda captures the variable <code className="bg-muted px-1.5 py-0.5 rounded">i</code> by reference, not its current value. When called, <code className="bg-muted px-1.5 py-0.5 rounded">i</code> is <code className="bg-muted px-1.5 py-0.5 rounded">2</code> (the last value from the loop).</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">How do you create a list of lambdas that capture different values from a loop?</p>
                  <p className="text-muted-foreground">Use a default argument to capture the current value: <code className="bg-muted px-1.5 py-0.5 rounded">[lambda i=i: i for i in range(5)]</code>.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

    
      </div>
    </TopicContent>
  );
}