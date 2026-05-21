import { TopicContent } from "@/components/topic-content";
import { CodeBlock, MultiLanguageCode } from "@/components/code-block";
import { Quiz, QuizQuestion } from "@/components/quiz";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Clock, Lightbulb, Terminal, GitBranch, Code2 } from "lucide-react";

export default function ControlFlowPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "control-flow");
  if (!result) return null;

  const { topic, subtopic } = result;

  const codeExamples = [
    {
      language: "python",
      label: "Python",
      code: `# Basic if statement
age = 18
if age >= 18:
    print("You can vote")

# if-else
temperature = 30
if temperature > 25:
    print("It's hot outside")
else:
    print("It's cool outside")

# if-elif-else
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"
print(f"Grade: {grade}")

# match/case (Python 3.10+)
command = "start"
match command:
    case "start":
        print("Starting...")
    case "stop":
        print("Stopping...")
    case "restart":
        print("Restarting...")
    case _:
        print("Unknown command")`,
    },
    {
      language: "javascript",
      label: "JavaScript",
      code: `// Basic if statement
let age = 18;
if (age >= 18) {
    console.log("You can vote");
}

// if-else
let temperature = 30;
if (temperature > 25) {
    console.log("It's hot outside");
} else {
    console.log("It's cool outside");
}

// if-else if-else
let score = 85;
let grade;
if (score >= 90) {
    grade = "A";
} else if (score >= 80) {
    grade = "B";
} else if (score >= 70) {
    grade = "C";
} else {
    grade = "F";
}
console.log(\`Grade: \${grade}\`);

// switch statement
let command = "start";
switch(command) {
    case "start":
        console.log("Starting...");
        break;
    case "stop":
        console.log("Stopping...");
        break;
    case "restart":
        console.log("Restarting...");
        break;
    default:
        console.log("Unknown command");
}`,
    },
    {
      language: "java",
      label: "Java",
      code: `// Basic if statement
int age = 18;
if (age >= 18) {
    System.out.println("You can vote");
}

// if-else
int temperature = 30;
if (temperature > 25) {
    System.out.println("It's hot outside");
} else {
    System.out.println("It's cool outside");
}

// if-else if-else
int score = 85;
String grade;
if (score >= 90) {
    grade = "A";
} else if (score >= 80) {
    grade = "B";
} else if (score >= 70) {
    grade = "C";
} else {
    grade = "F";
}
System.out.println("Grade: " + grade);

// switch statement
String command = "start";
switch(command) {
    case "start":
        System.out.println("Starting...");
        break;
    case "stop":
        System.out.println("Stopping...");
        break;
    case "restart":
        System.out.println("Restarting...");
        break;
    default:
        System.out.println("Unknown command");
}`,
    },
  ];

  const quizQuestions: QuizQuestion[] = [
    {
      id: 1,
      question: "What is the output of: `print('A' if 5 > 3 else 'B')`?",
      options: ["A", "B", "True", "False"],
      correctAnswer: 0,
      explanation: "Since 5 > 3 is True, the ternary operator returns the first value 'A'.",
    },
    {
      id: 2,
      question: "Which of the following is falsy in Python?",
      options: ["1", "'False'", "[]", "True"],
      correctAnswer: 2,
      explanation: "Empty list [] is falsy. Non-empty strings and non-zero numbers are truthy.",
    },
    {
      id: 3,
      question: "What does the `elif` keyword stand for?",
      options: ["Else if", "Else in if", "Else loop if", "None of the above"],
      correctAnswer: 0,
      explanation: "`elif` is Python's shorthand for 'else if', allowing multiple condition checks.",
    },
    {
      id: 4,
      question: "What happens if no case matches in Python's match/case?",
      options: [
        "Syntax error",
        "Runs the default case (_)",
        "Skips the entire block",
        "Repeats the match"
      ],
      correctAnswer: 1,
      explanation: "The underscore `_` serves as the default/wildcard case that matches anything.",
    },
  ];

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Introduction */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">What is Control Flow?</h2>
          <p className="text-muted-foreground mb-4">
            <strong className="text-foreground">Control flow</strong> determines the order in which statements are executed. 
            Think of it as the traffic signals of your code - telling Python which path to take based on conditions.
          </p>
          
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 my-6">
            <div className="flex gap-3">
              <GitBranch className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Real-world Analogy</h4>
                <p className="text-sm text-muted-foreground">
                  Imagine a GPS navigation system. At each intersection, it checks conditions 
                  (traffic, distance, user preference) and decides which route to take. 
                  Control flow works the same way - making decisions at decision points!
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* if Statement */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">The if Statement</h2>
          <p className="text-muted-foreground mb-4">
            The <code className="bg-muted px-1.5 py-0.5 rounded font-mono">if</code> statement is the simplest form of control flow. 
            It executes a block of code only when a condition is <code className="bg-muted px-1.5 py-0.5 rounded font-mono">True</code>.
          </p>

          <div className="space-y-6">
            <div className="bg-card border border-border rounded-lg overflow-hidden">
              <div className="bg-muted px-4 py-2 border-b border-border">
                <h4 className="font-semibold text-foreground">Basic if</h4>
              </div>
              <div className="p-4">
                <CodeBlock 
                  code={`age = 18
if age >= 18:
    print("You can vote")`}
                  language="python"
                />
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg overflow-hidden">
              <div className="bg-muted px-4 py-2 border-b border-border">
                <h4 className="font-semibold text-foreground">if-else</h4>
              </div>
              <div className="p-4">
                <CodeBlock 
                  code={`temperature = 30
if temperature > 25:
    print("It's hot outside")
else:
    print("It's cool outside")`}
                  language="python"
                />
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg overflow-hidden">
              <div className="bg-muted px-4 py-2 border-b border-border">
                <h4 className="font-semibold text-foreground">if-elif-else (Multiple Conditions)</h4>
              </div>
              <div className="p-4">
                <CodeBlock 
                  code={`score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"
print(f"Grade: {grade}")  # Output: B`}
                  language="python"
                />
              </div>
            </div>
          </div>
        </section>

        {/* Comparison Operators */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Comparison Operators</h2>
          <p className="text-muted-foreground mb-4">
            These operators compare values and return <code className="bg-muted px-1.5 py-0.5 rounded">True</code> or <code className="bg-muted px-1.5 py-0.5 rounded">False</code>.
          </p>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <code className="text-lg font-mono text-primary">==</code>
              <p className="text-sm text-muted-foreground mt-1">Equal to</p>
              <p className="text-xs text-muted-foreground mt-2"><code>5 == 5</code> → True</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <code className="text-lg font-mono text-primary">!=</code>
              <p className="text-sm text-muted-foreground mt-1">Not equal to</p>
              <p className="text-xs text-muted-foreground mt-2"><code>5 != 3</code> → True</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <code className="text-lg font-mono text-primary">&gt;</code>
              <p className="text-sm text-muted-foreground mt-1">Greater than</p>
              <p className="text-xs text-muted-foreground mt-2"><code>5 &gt; 3</code> → True</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <code className="text-lg font-mono text-primary">&lt;</code>
              <p className="text-sm text-muted-foreground mt-1">Less than</p>
              <p className="text-xs text-muted-foreground mt-2"><code>3 &lt; 5</code> → True</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <code className="text-lg font-mono text-primary">&gt;=</code>
              <p className="text-sm text-muted-foreground mt-1">Greater than or equal</p>
              <p className="text-xs text-muted-foreground mt-2"><code>5 &gt;= 5</code> → True</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <code className="text-lg font-mono text-primary">&lt;=</code>
              <p className="text-sm text-muted-foreground mt-1">Less than or equal</p>
              <p className="text-xs text-muted-foreground mt-2"><code>3 &lt;= 5</code> → True</p>
            </div>
          </div>
        </section>

        {/* Logical Operators */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Logical Operators</h2>
          <p className="text-muted-foreground mb-4">
            Combine multiple conditions using <code className="bg-muted px-1.5 py-0.5 rounded">and</code>, <code className="bg-muted px-1.5 py-0.5 rounded">or</code>, and <code className="bg-muted px-1.5 py-0.5 rounded">not</code>.
          </p>

          <div className="grid md:grid-cols-3 gap-4 mb-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <code className="text-lg font-mono text-primary">and</code>
              <p className="text-sm text-muted-foreground mt-1">Both must be True</p>
              <p className="text-xs text-muted-foreground mt-2"><code>True and True</code> → True</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <code className="text-lg font-mono text-primary">or</code>
              <p className="text-sm text-muted-foreground mt-1">At least one must be True</p>
              <p className="text-xs text-muted-foreground mt-2"><code>True or False</code> → True</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <code className="text-lg font-mono text-primary">not</code>
              <p className="text-sm text-muted-foreground mt-1">Reverses the truth value</p>
              <p className="text-xs text-muted-foreground mt-2"><code>not True</code> → False</p>
            </div>
          </div>

          <div className="bg-card border border-border rounded-lg overflow-hidden">
            <div className="bg-muted px-4 py-2 border-b border-border">
              <h4 className="font-semibold text-foreground">Example: Using Logical Operators</h4>
            </div>
            <div className="p-4">
              <CodeBlock 
                code={`age = 25
has_license = True

# AND - both conditions must be True
if age >= 18 and has_license:
    print("You can drive")

# OR - at least one condition must be True
if age < 18 or age > 65:
    print("Discount applies")

# NOT - reverses the condition
if not has_license:
    print("Get a license first")`}
                language="python"
              />
            </div>
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Short-circuit Evaluation</h4>
                <p className="text-sm text-muted-foreground">
                  Python stops evaluating as soon as the result is determined. 
                  With <code className="bg-muted px-1.5 py-0.5 rounded">and</code>, if first condition is False, it doesn't check the second. 
                  With <code className="bg-muted px-1.5 py-0.5 rounded">or</code>, if first is True, it stops.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Truthiness and Falsy Values */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Truthiness & Falsy Values</h2>
          <p className="text-muted-foreground mb-4">
            In Python, every value has an inherent truth value when used in conditions.
          </p>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <h4 className="font-semibold text-destructive mb-2 flex items-center gap-2">
                <AlertCircle className="h-4 w-4" />
                Falsy Values
              </h4>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li><code className="bg-muted px-1.5 py-0.5 rounded">False</code></li>
                <li><code className="bg-muted px-1.5 py-0.5 rounded">None</code></li>
                <li><code className="bg-muted px-1.5 py-0.5 rounded">0</code>, <code className="bg-muted px-1.5 py-0.5 rounded">0.0</code></li>
                <li><code className="bg-muted px-1.5 py-0.5 rounded">""</code> (empty string)</li>
                <li><code className="bg-muted px-1.5 py-0.5 rounded">[]</code> (empty list)</li>
                <li><code className="bg-muted px-1.5 py-0.5 rounded">{}</code> (empty dict)</li>
                <li><code className="bg-muted px-1.5 py-0.5 rounded">set()</code> (empty set)</li>
              </ul>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <h4 className="font-semibold text-primary mb-2 flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4" />
                Truthy Values
              </h4>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li><code className="bg-muted px-1.5 py-0.5 rounded">True</code></li>
                <li>Non-zero numbers (<code className="bg-muted px-1.5 py-0.5 rounded">1</code>, <code className="bg-muted px-1.5 py-0.5 rounded">-5</code>, <code className="bg-muted px-1.5 py-0.5 rounded">3.14</code>)</li>
                <li>Non-empty strings (<code className="bg-muted px-1.5 py-0.5 rounded">"Hello"</code>)</li>
                <li>Non-empty containers (<code className="bg-muted px-1.5 py-0.5 rounded">[1, 2]</code>, <code className="bg-muted px-1.5 py-0.5 rounded">{'{"a": 1}'}</code>)</li>
              </ul>
            </div>
          </div>

          <div className="bg-card border border-border rounded-lg overflow-hidden mt-4">
            <div className="bg-muted px-4 py-2 border-b border-border">
              <h4 className="font-semibold text-foreground">Practical Example: Checking Empty Collections</h4>
            </div>
            <div className="p-4">
              <CodeBlock 
                code={`# Instead of: if len(items) > 0:
items = []
if items:  # False for empty list
    print("Has items")
else:
    print("Empty")  # This will print`}
                language="python"
              />
            </div>
          </div>
        </section>

        {/* Ternary Operator */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Ternary Operator (Conditional Expression)</h2>
          <p className="text-muted-foreground mb-4">
            A one-liner for simple if-else assignments.
          </p>

          <div className="bg-card border border-border rounded-lg overflow-hidden">
            <div className="bg-muted px-4 py-2 border-b border-border">
              <h4 className="font-semibold text-foreground">Syntax: value_if_true if condition else value_if_false</h4>
            </div>
            <div className="p-4">
              <CodeBlock 
                code={`age = 20
status = "Adult" if age >= 18 else "Minor"
print(status)  # Adult

# Nested ternary (use sparingly for readability)
score = 85
result = "A" if score >= 90 else "B" if score >= 80 else "C"
print(result)  # B`}
                language="python"
              />
            </div>
          </div>
        </section>

        {/* match/case Statement */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">match/case Statement (Python 3.10+)</h2>
          <p className="text-muted-foreground mb-4">
            A powerful pattern matching feature that goes beyond traditional switch statements.
          </p>

          <div className="space-y-6">
            <div className="bg-card border border-border rounded-lg overflow-hidden">
              <div className="bg-muted px-4 py-2 border-b border-border">
                <h4 className="font-semibold text-foreground">Basic Pattern Matching</h4>
              </div>
              <div className="p-4">
                <CodeBlock 
                  code={`command = "start"
match command:
    case "start":
        print("Starting...")
    case "stop":
        print("Stopping...")
    case "restart":
        print("Restarting...")
    case _:  # Default case (underscore)
        print("Unknown command")`}
                  language="python"
                />
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg overflow-hidden">
              <div className="bg-muted px-4 py-2 border-b border-border">
                <h4 className="font-semibold text-foreground">Matching with Guards (Conditions)</h4>
              </div>
              <div className="p-4">
                <CodeBlock 
                  code={`value = 10
match value:
    case x if x < 0:
        print(f"Negative: {x}")
    case x if x == 0:
        print("Zero")
    case x if x > 0:
        print(f"Positive: {x}")`}
                  language="python"
                />
              </div>
            </div>

            <div className="bg-card border border-border rounded-lg overflow-hidden">
              <div className="bg-muted px-4 py-2 border-b border-border">
                <h4 className="font-semibold text-foreground">Matching Sequences (Tuples/Lists)</h4>
              </div>
              <div className="p-4">
                <CodeBlock 
                  code={`point = (2, 3)
match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"On Y-axis at {y}")
    case (x, 0):
        print(f"On X-axis at {x}")
    case (x, y):
        print(f"Point at ({x}, {y})")`}
                  language="python"
                />
              </div>
            </div>
          </div>
        </section>

        {/* Time Complexity */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Time Complexity</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-muted">
                  <th className="border border-border p-3 text-left text-foreground">Operation</th>
                  <th className="border border-border p-3 text-left text-foreground">Time Complexity</th>
                  <th className="border border-border p-3 text-left text-foreground">Explanation</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="border border-border p-3 text-foreground">Single if/else</td>
                  <td className="border border-border p-3 font-mono text-primary">O(1)</td>
                  <td className="border border-border p-3 text-muted-foreground">Constant time - one condition check</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 text-foreground">if-elif-else chain</td>
                  <td className="border border-border p-3 font-mono text-warning">O(n)</td>
                  <td className="border border-border p-3 text-muted-foreground">May check up to n conditions</td>
                </tr>
                <tr>
                  <td className="border border-border p-3 text-foreground">match/case</td>
                  <td className="border border-border p-3 font-mono text-warning">O(n)</td>
                  <td className="border border-border p-3 text-muted-foreground">Worst-case checks all patterns</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Code Examples */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Code Examples (Multiple Languages)</h2>
          <MultiLanguageCode codes={codeExamples} />
        </section>

        {/* Common Mistakes */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Common Mistakes to Avoid</h2>
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Assignment Instead of Comparison</h4>
                <p className="text-sm text-muted-foreground">
                  Using <code className="bg-muted px-1.5 py-0.5 rounded">=</code> instead of <code className="bg-muted px-1.5 py-0.5 rounded">==</code> in conditions. 
                  <code className="bg-muted px-1.5 py-0.5 rounded">if x = 5:</code> is a syntax error in Python.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Indentation Errors</h4>
                <p className="text-sm text-muted-foreground">
                  Inconsistent spacing within blocks causes <code className="bg-muted px-1.5 py-0.5 rounded">IndentationError</code>. 
                  Always use 4 spaces consistently.
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Dangling Else</h4>
                <p className="text-sm text-muted-foreground">
                  <code className="bg-muted px-1.5 py-0.5 rounded">else</code> attaches to the nearest <code className="bg-muted px-1.5 py-0.5 rounded">if</code>. 
                  Use <code className="bg-muted px-1.5 py-0.5 rounded">elif</code> for explicit chaining.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Interview Patterns */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Interview Tips & Patterns</h2>
          <div className="space-y-4">
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Short-circuit Evaluation</h4>
                <p className="text-sm text-muted-foreground">
                  Use <code className="bg-muted px-1.5 py-0.5 rounded">and</code> for safe property access: 
                  <code className="bg-muted px-1.5 py-0.5 rounded ml-2">if obj is not None and obj.value &gt; 0:</code>
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Truthiness for Clean Code</h4>
                <p className="text-sm text-muted-foreground">
                  Use implicit truthiness checks: <code className="bg-muted px-1.5 py-0.5 rounded">if items:</code> instead of 
                  <code className="bg-muted px-1.5 py-0.5 rounded">if len(items) &gt; 0:</code>
                </p>
              </div>
            </div>
            <div className="flex gap-3 p-4 bg-primary/5 border border-primary/20 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-foreground mb-1">Chained Comparisons</h4>
                <p className="text-sm text-muted-foreground">
                  Write cleaner range checks: <code className="bg-muted px-1.5 py-0.5 rounded">if 0 &lt;= x &lt;= 100:</code> instead of 
                  <code className="bg-muted px-1.5 py-0.5 rounded">if x &gt;= 0 and x &lt;= 100:</code>
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Quiz */}
        <section>
          <h2 className="text-2xl font-bold mb-4 text-foreground">Test Your Knowledge</h2>
          <Quiz questions={quizQuestions} title="Control Flow Quiz" />
        </section>
      </div>
    </TopicContent>
  );
}