import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function RecursionPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "recursion");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Recursion
          </h1>
          <p className="text-muted-foreground text-lg">
            <strong className="text-foreground">Recursion</strong> is a programming technique where a function calls itself to solve a problem by breaking it down into smaller, similar subproblems.
          </p>
        </div>

        {/* Note Box */}
        <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
          <div className="flex gap-3">
            <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
            <p className="text-sm text-foreground">
              <strong>Every recursive function needs:</strong> A <strong className="text-foreground">base case</strong> (stops recursion) and a <strong className="text-foreground">recursive case</strong> (calls itself with modified arguments).
            </p>
          </div>
        </div>

        {/* What is Recursion? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            What is Recursion?
          </h2>
          <p className="text-muted-foreground mb-4">
            Recursion solves problems by reducing them to smaller instances of the same problem. It's like Russian dolls — each recursive call works on a smaller piece until reaching the simplest case.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def countdown(n):
    \"\"\"Count down from n to 0\"\"\"
    if n < 0:                # Base case
        return
    print(n)                 # Work
    countdown(n - 1)         # Recursive call with smaller input

countdown(5)
# Output: 5, 4, 3, 2, 1, 0`}
            </pre>
          </div>
        </section>

        {/* Anatomy of Recursion */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Anatomy of Recursion
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
                  <td className="p-3 font-semibold text-foreground">Base Case</td>
                  <td className="p-3">Stops recursion; simplest possible input</td>
                  <td className="p-3 font-mono">if n == 0: return 1</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">Recursive Case</td>
                  <td className="p-3">Breaks problem into smaller piece + recursive call</td>
                  <td className="p-3 font-mono">return n * factorial(n-1)</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-semibold text-foreground">Progress</td>
                  <td className="p-3">Each call moves closer to base case</td>
                  <td className="p-3 font-mono">n-1, n-2, n-3...</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Classic Examples */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Classic Recursion Examples
          </h2>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">1. Factorial</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def factorial(n):
    \"\"\"Calculate n! = n * (n-1) * ... * 1\"\"\"
    if n <= 1:          # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

# factorial(5) = 5 * 4 * 3 * 2 * 1 = 120
print(factorial(5))  # 120

# How it works:
# factorial(5) = 5 * factorial(4)
# factorial(4) = 4 * factorial(3)
# factorial(3) = 3 * factorial(2)
# factorial(2) = 2 * factorial(1)
# factorial(1) = 1  (base case)
# Then unwinds: 1 → 2 → 6 → 24 → 120`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">2. Fibonacci Sequence</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def fibonacci(n):
    \"\"\"Return the nth Fibonacci number (0, 1, 1, 2, 3, 5, 8, ...)\"\"\"
    if n <= 1:          # Base cases
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)  # Recursive case

print(fibonacci(7))  # 13

# Tree of recursive calls (inefficient without memoization!)
# fibonacci(5) calls fibonacci(4) and fibonacci(3)
# fibonacci(4) calls fibonacci(3) and fibonacci(2)
# ... many repeated calculations`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">3. Sum of List</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def sum_list(numbers):
    \"\"\"Return sum of all numbers in list recursively\"\"\"
    if not numbers:              # Base case: empty list
        return 0
    return numbers[0] + sum_list(numbers[1:])  # First + rest

print(sum_list([2, 4, 6, 8]))  # 20`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">4. Power Function</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def power(base, exp):
    \"\"\"Calculate base raised to exponent (exp >= 0)\"\"\"
    if exp == 0:                # Base case: any number^0 = 1
        return 1
    return base * power(base, exp - 1)  # Recursive case

print(power(2, 5))  # 32 (2 * 2 * 2 * 2 * 2)

# Optimization: fast exponentiation O(log n)
def power_fast(base, exp):
    if exp == 0:
        return 1
    if exp % 2 == 0:  # Even exponent
        half = power_fast(base, exp // 2)
        return half * half
    else:              # Odd exponent
        return base * power_fast(base, exp - 1)`}
            </pre>
          </div>
        </section>

        {/* Recursion vs Iteration */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Recursion vs Iteration
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Aspect</th>
                  <th className="text-left p-3 font-semibold text-foreground">Recursion</th>
                  <th className="text-left p-3 font-semibold text-foreground">Iteration</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3">Code readability</td><td className="p-3">Often cleaner for recursive problems</td><td className="p-3">Clean for simple loops</td></tr>
                <tr className="border-b border-border"><td className="p-3">Memory usage</td><td className="p-3">Uses stack (each call adds frame)</td><td className="p-3">Constant (O(1) extra memory)</td></tr>
                <tr className="border-b border-border"><td className="p-3">Performance</td><td className="p-3">Function call overhead</td><td className="p-3">Faster (no call overhead)</td></tr>
                <tr className="border-b border-border"><td className="p-3">Risk</td><td className="p-3">Stack overflow for deep recursion</td><td className="p-3">None (can run indefinitely)</td></tr>
                <tr className="border-b border-border"><td className="p-3">Best for</td><td className="p-3">Tree/graph traversal, divide-and-conquer</td><td className="p-3">Simple repetition, performance-critical code</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Memoization (Caching) */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Memoization - Making Recursion Efficient
          </h2>
          <p className="text-muted-foreground mb-4">
            Many recursive functions (like Fibonacci) recompute the same values repeatedly. <strong className="text-foreground">Memoization</strong> caches results to avoid redundant calculations.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`from functools import lru_cache

# Without memoization: O(2^n) - very slow
def fib_slow(n):
    if n <= 1:
        return n
    return fib_slow(n - 1) + fib_slow(n - 2)

# With memoization: O(n) - linear!
@lru_cache(maxsize=None)
def fib_fast(n):
    if n <= 1:
        return n
    return fib_fast(n - 1) + fib_fast(n - 2)

print(fib_fast(40))  # Fast! Results are cached

# Manual memoization
memo = {}
def fib_manual(n):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_manual(n - 1) + fib_manual(n - 2)
    return memo[n]`}
            </pre>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <code className="bg-muted px-1.5 py-0.5 rounded">@lru_cache</code> is Python's built-in memoization decorator. It caches results based on arguments.
              </p>
            </div>
          </div>
        </section>

        {/* Recursion Limit */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Recursion Limit
          </h2>
          <p className="text-muted-foreground mb-4">
            Python has a default recursion limit (usually 1000) to prevent stack overflow.
          </p>
          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4 mb-4">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground">⚠️ RecursionError</p>
                <p className="text-muted-foreground text-sm">
                  Exceeding the recursion limit raises <code className="bg-muted px-1.5 py-0.5 rounded">RecursionError</code>. You can increase it, but deep recursion may crash Python or cause stack overflow.
                </p>
              </div>
            </div>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import sys

# Check current limit
print(sys.getrecursionlimit())  # 1000 (default)

# Increase limit (use cautiously!)
sys.setrecursionlimit(5000)

# Better: convert to iteration for deep recursion
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# Or use tail recursion (Python doesn't optimize tail calls)
# Tail recursion still adds stack frames in Python!`}
            </pre>
          </div>
        </section>

        {/* Tail Recursion */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Tail Recursion
          </h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">Tail recursion</strong> occurs when the recursive call is the last operation in the function. Some languages optimize this (tail call optimization), but <strong className="text-destructive">Python does NOT</strong>.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# NOT tail recursive (multiplication happens after recursive call)
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # Recursion not last - multiplication after

# Tail recursive version (still not optimized in Python)
def factorial_tail(n, accumulator=1):
    if n <= 1:
        return accumulator
    return factorial_tail(n - 1, n * accumulator)  # Recursive call is last

# Both cause stack growth in Python
# For deep recursion, use iteration`}
            </pre>
          </div>
        </section>

        {/* Common Recursion Patterns */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Common Recursion Patterns
          </h2>
          
          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">1. Divide and Conquer</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def binary_search(arr, target, left, right):
    \"\"\"Binary search using divide and conquer\"\"\"
    if left > right:
        return -1  # Not found
    
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr, target, mid + 1, right)
    else:
        return binary_search(arr, target, left, mid - 1)

arr = [2, 4, 6, 8, 10, 12, 14]
print(binary_search(arr, 10, 0, len(arr) - 1))  # 4`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">2. Tree Traversal</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def inorder_traversal(node):
    \"\"\"In-order traversal: left, root, right\"\"\"
    if node is None:
        return []
    return inorder_traversal(node.left) + [node.val] + inorder_traversal(node.right)

# Build tree: root=2, left=4, right=6
root = TreeNode(2)
root.left = TreeNode(4)
root.right = TreeNode(6)

print(inorder_traversal(root))  # [4, 2, 6]`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">3. Backtracking</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`def find_paths(node, target, path=None, paths=None):
    \"\"\"Find all root-to-leaf paths that sum to target\"\"\"
    if path is None:
        path = []
    if paths is None:
        paths = []
    
    if node is None:
        return paths
    
    path.append(node.val)
    
    if node.left is None and node.right is None and sum(path) == target:
        paths.append(path.copy())
    else:
        find_paths(node.left, target, path, paths)
        find_paths(node.right, target, path, paths)
    
    path.pop()  # Backtrack
    return paths`}
            </pre>
          </div>
        </section>

        {/* Problems Well-Suited for Recursion */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Problems Well-Suited for Recursion
          </h2>
          <div className="space-y-4">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <p className="font-semibold text-foreground mb-1">Tree and Graph Traversal</p>
              <p className="text-muted-foreground text-sm">DFS, tree traversal (inorder, preorder, postorder)</p>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <p className="font-semibold text-foreground mb-1">Divide and Conquer Algorithms</p>
              <p className="text-muted-foreground text-sm">Merge sort, quick sort, binary search, Tower of Hanoi</p>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <p className="font-semibold text-foreground mb-1">Backtracking</p>
              <p className="text-muted-foreground text-sm">N-Queens, Sudoku solver, maze solving, permutations</p>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <p className="font-semibold text-foreground mb-1">Mathematical Definitions</p>
              <p className="text-muted-foreground text-sm">Factorial, Fibonacci, GCD, exponentiation</p>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <p className="font-semibold text-foreground mb-1">Recursive Data Structures</p>
              <p className="text-muted-foreground text-sm">Linked lists, JSON parsing, expression evaluation</p>
            </div>
          </div>
        </section>

        {/* Problems NOT Well-Suited for Recursion */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Problems NOT Well-Suited for Recursion
          </h2>
          <div className="space-y-4">
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1"> Simple repetition (use loops)</p>
                  <p className="text-muted-foreground text-sm">Print 1 to N, sum of N numbers, array traversal</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1"> Deep recursion (risk of stack overflow)</p>
                  <p className="text-muted-foreground text-sm">Problems requiring &gt;1000 recursive calls</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1"> Performance-critical code</p>
                  <p className="text-muted-foreground text-sm">Function call overhead can be significant</p>
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
                  <p className="font-semibold text-foreground">Missing base case → infinite recursion</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`def infinite(n):
    return infinite(n + 1)  # No base case!

# RecursionError: maximum recursion depth exceeded`}
                  </pre>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Not making progress toward base case</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`def no_progress(n):
    if n == 0:
        return
    return no_progress(n)  # Same argument, never reaches 0!`}
                  </pre>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Python doesn't optimize tail recursion</p>
                  <p className="text-muted-foreground text-sm">Tail-recursive functions still use stack frames. Convert to iteration for deep recursion.</p>
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
                  <p className="font-semibold text-foreground">What are the two essential parts of a recursive function?</p>
                  <p className="text-muted-foreground">Base case (stops recursion) and recursive case (calls itself with modified arguments).</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the default recursion limit in Python? Can you change it?</p>
                  <p className="text-muted-foreground">Default is 1000. Use <code className="bg-muted px-1.5 py-0.5 rounded">sys.setrecursionlimit()</code> to change, but deep recursion may crash Python.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is memoization and why is it useful for recursion?</p>
                  <p className="text-muted-foreground">Memoization caches recursive results. It prevents exponential recomputation (like in Fibonacci), making it O(n) instead of O(2^n).</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Does Python optimize tail recursion? Why or why not?</p>
                  <p className="text-muted-foreground">No. Python intentionally does not implement tail call optimization to preserve full stack traces for debugging.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">When would you choose recursion over iteration?</p>
                  <p className="text-muted-foreground">For naturally recursive problems like tree traversal, divide-and-conquer, backtracking, or when the recursive solution is significantly clearer.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

       
   
      </div>
    </TopicContent>
  );
}