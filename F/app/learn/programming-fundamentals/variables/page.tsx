import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function VariablesPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "variables");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Variables & Data Types
          </h1>
          <p className="text-muted-foreground text-lg">
            Variables store data in memory. Python is dynamically typed — variables can hold any type, and types are checked at runtime.
          </p>
        </div>

        {/* Variable Assignment */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Variable Assignment
          </h2>
          <p className="text-muted-foreground mb-4">
            Use <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">=</code> to assign a value to a variable. Variable names can contain letters, numbers, and underscores, but cannot start with a number.
          </p>

          <div className="space-y-5">
            <div>
              <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
                <div className="bg-muted px-4 py-2 text-foreground font-mono text-sm border-b border-border">Basic assignment</div>
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`x = 10
name = "Alice"
is_valid = True`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">Multiple assignment:</span> Assign multiple variables in one line.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden mt-2 border border-border">
                <div className="bg-muted px-4 py-2 text-foreground font-mono text-sm border-b border-border">Multiple assignment</div>
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`a, b, c = 1, 2, 3
x = y = z = 0  # All three reference 0`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">Dynamic typing:</span> Variables can change type at runtime.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden mt-2 border border-border">
                <div className="bg-muted px-4 py-2 text-foreground font-mono text-sm border-b border-border">Dynamic typing</div>
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`value = 42        # int
value = "hello"  # now str
value = [1, 2, 3] # now list`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Basic Data Types */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Basic Data Types
          </h2>

          <div className="space-y-6">
            {/* Integers */}
            <div>
              <h3 className="text-xl font-semibold text-foreground mb-2">int — Integer</h3>
              <p className="text-muted-foreground mb-2">
                Whole numbers, unbounded in Python 3.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`age = 25
count = -10
binary = 0b1010   # 10 in decimal
hex_num = 0xFF    # 255 in decimal`}
                </pre>
              </div>
            </div>

            {/* Floats */}
            <div>
              <h3 className="text-xl font-semibold text-foreground mb-2">float — Floating Point</h3>
              <p className="text-muted-foreground mb-2">
                Numbers with decimal points. Based on IEEE 754 double-precision.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`price = 19.99
pi = 3.14159
scientific = 1.5e-3  # 0.0015`}
                </pre>
              </div>
            </div>

            {/* Strings */}
            <div>
              <h3 className="text-xl font-semibold text-foreground mb-2">str — String</h3>
              <p className="text-muted-foreground mb-2">
                Sequence of Unicode characters. Use single, double, or triple quotes.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`name = "Python"
single = 'Hello'
multi_line = """This spans
multiple lines"""`}
                </pre>
              </div>
            </div>

            {/* Booleans */}
            <div>
              <h3 className="text-xl font-semibold text-foreground mb-2">bool — Boolean</h3>
              <p className="text-muted-foreground mb-2">
                Represents truth values. <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">True</code> and <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">False</code> (capitalized).
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`is_active = True
is_done = False
is_greater = 10 > 5  # True`}
                </pre>
              </div>
            </div>

            {/* None */}
            <div>
              <h3 className="text-xl font-semibold text-foreground mb-2">NoneType — None</h3>
              <p className="text-muted-foreground mb-2">
                Represents absence of a value. Similar to <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">null</code> in other languages.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`result = None
if result is None:
    print("No value")`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Type Inspection */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Type Inspection
          </h2>
          <div className="space-y-4">
            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">type() function:</span> Returns the type of an object.
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`print(type(42))      # <class 'int'>
print(type("hello")) # <class 'str'>
print(type(3.14))    # <class 'float'>`}
                </pre>
              </div>
            </div>

            <div>
              <p className="text-muted-foreground mb-2">
                <span className="font-semibold text-foreground">isinstance() function:</span> Checks if an object is of a specific type (preferred over <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">type()</code> for type checking).
              </p>
              <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
                <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                  {`print(isinstance(42, int))     # True
print(isinstance("hi", str))   # True
print(isinstance(3.14, int))   # False`}
                </pre>
              </div>
            </div>
          </div>
        </section>

        {/* Type Conversion */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Type Conversion (Casting)
          </h2>
          <div className="space-y-4">
            <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
              <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
                {`# String to int
num = int("123")      # 123

# Int to float
pi = float(3)         # 3.0

# Number to string
text = str(456)       # "456"

# String to list
chars = list("abc")   # ['a', 'b', 'c']`}
              </pre>
            </div>
            
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm text-foreground">
                    <span className="font-semibold">Caution:</span> Invalid conversion raises <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">ValueError</code>.
                    Example: <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">int("hello")</code> fails.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Naming Conventions */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Naming Conventions (PEP 8)
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-mono text-foreground font-semibold">snake_case</p>
              <p className="text-muted-foreground text-sm">Variables and functions</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-mono text-foreground font-semibold">PascalCase</p>
              <p className="text-muted-foreground text-sm">Classes</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-mono text-foreground font-semibold">UPPER_SNAKE_CASE</p>
              <p className="text-muted-foreground text-sm">Constants</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <p className="font-mono text-foreground font-semibold">_leading_underscore</p>
              <p className="text-muted-foreground text-sm">Protected/internal (convention)</p>
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
                  <p className="font-semibold text-foreground mb-1">Dynamic typing can cause bugs</p>
                  <p className="text-muted-foreground text-sm">
                    Variables can change type unexpectedly. Use type hints and mypy to catch errors early.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">Float precision issues</p>
                  <p className="text-muted-foreground text-sm">
                    <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">0.1 + 0.2 == 0.3</code> is <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">False</code> due to binary representation. Use <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">decimal.Decimal</code> for exact arithmetic.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">Strings are immutable</p>
                  <p className="text-muted-foreground text-sm">
                    <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">s[0] = 'a'</code> raises <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">TypeError</code>. Create new strings instead.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">Assignment copies references, not values</p>
                  <p className="text-muted-foreground text-sm">
                    For mutable objects, <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">b = a</code> makes both reference the same object. Use <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">copy()</code> or <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">deepcopy()</code> for independent copies.
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
          <div className="space-y-5">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">
                    What is dynamic typing in Python?
                  </p>
                  <p className="text-muted-foreground">
                    Variables don't have fixed types. A single variable can reference an integer, then a string, then a list. Type is checked at runtime.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">
                    What is the difference between <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">type()</code> and <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">isinstance()</code>?
                  </p>
                  <p className="text-muted-foreground">
                    <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">isinstance()</code> considers inheritance (returns True for subclasses). <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">type()</code> does not. <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">isinstance()</code> is preferred for type checking.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">
                    Why does <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">0.1 + 0.2 != 0.3</code>?
                  </p>
                  <p className="text-muted-foreground">
                    Floating point numbers use binary representation. Some decimals cannot be represented exactly. The result is <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">0.30000000000000004</code>.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">
                    What are Python's immutable data types?
                  </p>
                  <p className="text-muted-foreground">
                    <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">int</code>, <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">float</code>, <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">str</code>, <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">tuple</code>, <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">frozenset</code>, <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">bool</code>, <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-sm">None</code>.
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