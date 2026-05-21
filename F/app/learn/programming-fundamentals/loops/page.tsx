import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function LoopsPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "loops");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Loops
          </h1>
          <p className="text-muted-foreground text-lg">
            Loops repeat a block of code multiple times. Python provides <code className="bg-muted px-1.5 py-0.5 rounded font-mono">for</code> loops for iteration and <code className="bg-muted px-1.5 py-0.5 rounded font-mono">while</code> loops for condition-based repetition.
          </p>
        </div>

        {/* for Loop */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            for Loop
          </h2>
          <p className="text-muted-foreground mb-4">
            The <code className="bg-muted px-1.5 py-0.5 rounded font-mono">for</code> loop iterates over any iterable (list, tuple, string, range, etc.).
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
                <div className="bg-muted px-4 py-2 text-foreground font-mono text-sm border-b border-border">Iterating over a list</div>
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">range() function:</span> Generate sequences of numbers.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden mt-2 border border-border">
                <div className="bg-muted px-4 py-2 text-foreground font-mono text-sm border-b border-border">range(stop), range(start, stop), range(start, stop, step)</div>
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`# range(stop) - 0 to stop-1
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# range(start, stop) - start to stop-1
for i in range(2, 6):
    print(i)  # 2, 3, 4, 5

# range(start, stop, step) - with step value
for i in range(0, 10, 2):
    print(i)  # 0, 2, 4, 6, 8

# Count backwards
for i in range(5, 0, -1):
    print(i)  # 5, 4, 3, 2, 1`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">enumerate():</span> Get both index and value while iterating.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden mt-2 border border-border">
                <div className="bg-muted px-4 py-2 text-foreground font-mono text-sm border-b border-border">Using enumerate</div>
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`colors = ["red", "green", "blue"]
for index, color in enumerate(colors):
    print(f"{index}: {color}")

# Start index from 1
for index, color in enumerate(colors, start=1):
    print(f"{index}: {color}")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">zip():</span> Iterate over multiple sequences in parallel.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden mt-2 border border-border">
                <div className="bg-muted px-4 py-2 text-foreground font-mono text-sm border-b border-border">Parallel iteration</div>
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# zip stops at shortest iterable
a = [1, 2, 3]
b = ["x", "y"]
for x, y in zip(a, b):
    print(x, y)  # (1, 'x'), (2, 'y')`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">Iterating over dictionaries:</span> Keys, values, or items.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden mt-2 border border-border">
                <div className="bg-muted px-4 py-2 text-foreground font-mono text-sm border-b border-border">Dictionary iteration</div>
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`person = {"name": "Alice", "age": 30, "city": "New York"}

# Iterate over keys
for key in person:
    print(key)

# Iterate over values
for value in person.values():
    print(value)

# Iterate over key-value pairs
for key, value in person.items():
    print(f"{key}: {value}")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* while Loop */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            while Loop
          </h2>
          <p className="text-muted-foreground mb-4">
            The <code className="bg-muted px-1.5 py-0.5 rounded font-mono">while</code> loop repeats as long as a condition is <code className="bg-muted px-1.5 py-0.5 rounded font-mono">True</code>.
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
                <div className="bg-muted px-4 py-2 text-foreground font-mono text-sm border-b border-border">Basic while loop</div>
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`count = 0
while count < 5:
    print(count)
    count += 1`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">User input validation:</span> Common use case for while loops.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden mt-2 border border-border">
                <div className="bg-muted px-4 py-2 text-foreground font-mono text-sm border-b border-border">Input validation</div>
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`while True:
    age = input("Enter your age: ")
    if age.isdigit() and 0 < int(age) < 120:
        age = int(age)
        break
    print("Invalid input. Please enter a valid age.")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Loop Control Statements */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Loop Control Statements
          </h2>

          <div className="space-y-5">
            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">break:</span> Exit the loop immediately.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden mt-2 border border-border">
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`for i in range(10):
    if i == 5:
        break
    print(i)  # 0, 1, 2, 3, 4`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">continue:</span> Skip the rest of the current iteration and move to the next.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden mt-2 border border-border">
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`for i in range(5):
    if i == 2:
        continue
    print(i)  # 0, 1, 3, 4`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">else with loops:</span> Executes only if loop completes normally (no break).
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden mt-2 border border-border">
                <div className="bg-muted px-4 py-2 text-foreground font-mono text-sm border-b border-border">Loop else clause</div>
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`# Example: Check if number is prime
num = 7
for i in range(2, num):
    if num % i == 0:
        print(f"{num} is not prime")
        break
else:
    print(f"{num} is prime")  # Executes when no break

# else does NOT run if break occurs
for i in range(3):
    if i == 1:
        break
else:
    print("This won't print because loop broke")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">pass:</span> Null statement - does nothing. Placeholder for future code.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden mt-2 border border-border">
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`for i in range(5):
    if i == 2:
        pass  # TODO: Add logic here
    print(i)`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Nested Loops */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Nested Loops
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Multiplication table
for i in range(1, 4):
    for j in range(1, 4):
        print(f"{i} x {j} = {i*j}")
    print("-" * 10)

# Matrix iteration
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for row in matrix:
    for element in row:
        print(element, end=" ")
    print()  # New line after each row`}
            </pre>
          </div>
        </section>

        {/* Common Patterns */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Common Loop Patterns
          </h2>
          <div className="space-y-4">
            <div>
              <p className="text-muted-foreground mb-2 font-semibold">Summing values:</p>
              <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`numbers = [1, 2, 3, 4, 5]
total = 0
for num in numbers:
    total += num
print(f"Sum: {total}")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2 font-semibold">Finding maximum value:</p>
              <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`numbers = [3, 7, 2, 9, 5]
max_value = numbers[0]
for num in numbers:
    if num > max_value:
        max_value = num
print(f"Max: {max_value}")`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2 font-semibold">Filtering with condition:</p>
              <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = []
for num in numbers:
    if num % 2 == 0:
        evens.append(num)
print(f"Even numbers: {evens}")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Infinite Loops */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Infinite Loops
          </h2>
          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4 mb-4">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground mb-1"> Warning</p>
                <p className="text-muted-foreground text-sm">
                  Infinite loops never terminate and will freeze your program. Always ensure your loop condition becomes False at some point.
                </p>
              </div>
            </div>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Infinite loop (don't run without break)
# while True:
#     print("This runs forever")

# Controlled infinite loop with break
while True:
    user_input = input("Type 'quit' to exit: ")
    if user_input == 'quit':
        break
    print(f"You typed: {user_input}")`}
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
                  <p className="font-semibold text-foreground mb-1">Don't modify a list while iterating over it</p>
                  <p className="text-muted-foreground text-sm">
                    Modifying a list while iterating causes skipped or repeated elements. Iterate over a copy instead: <code className="bg-muted px-1.5 py-0.5 rounded">for item in list[:]:</code>
                  </p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">range() doesn't create a list in Python 3</p>
                  <p className="text-muted-foreground text-sm">
                    <code className="bg-muted px-1.5 py-0.5 rounded">range()</code> returns a lazy sequence (range object), not a list. Use <code className="bg-muted px-1.5 py-0.5 rounded">list(range(10))</code> if you need an actual list.
                  </p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">else clause executes after normal completion</p>
                  <p className="text-muted-foreground text-sm">
                    The <code className="bg-muted px-1.5 py-0.5 rounded">else</code> block runs only if the loop wasn't terminated by <code className="bg-muted px-1.5 py-0.5 rounded">break</code>. Useful for search operations.
                  </p>
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
                  <p className="font-semibold text-foreground mb-1">
                    What is the difference between <code className="bg-muted px-1.5 py-0.5 rounded">for</code> and <code className="bg-muted px-1.5 py-0.5 rounded">while</code> loops?
                  </p>
                  <p className="text-muted-foreground">
                    <code className="bg-muted px-1.5 py-0.5 rounded">for</code> is used when the number of iterations is known or when iterating over a sequence. <code className="bg-muted px-1.5 py-0.5 rounded">while</code> is used when the number of iterations depends on a condition that may change during execution.
                  </p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">
                    When does the <code className="bg-muted px-1.5 py-0.5 rounded">else</code> clause in a loop execute?
                  </p>
                  <p className="text-muted-foreground">
                    The <code className="bg-muted px-1.5 py-0.5 rounded">else</code> block executes when the loop completes normally (without encountering a <code className="bg-muted px-1.5 py-0.5 rounded">break</code> statement).
                  </p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">
                    What's the difference between <code className="bg-muted px-1.5 py-0.5 rounded">pass</code>, <code className="bg-muted px-1.5 py-0.5 rounded">continue</code>, and <code className="bg-muted px-1.5 py-0.5 rounded">break</code>?
                  </p>
                  <p className="text-muted-foreground">
                    <code className="bg-muted px-1.5 py-0.5 rounded">break</code> exits the loop entirely. <code className="bg-muted px-1.5 py-0.5 rounded">continue</code> skips to the next iteration. <code className="bg-muted px-1.5 py-0.5 rounded">pass</code> does nothing (placeholder).
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

      
      </div>
    </TopicContent>
  );
}