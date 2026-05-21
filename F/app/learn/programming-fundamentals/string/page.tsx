import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function StringPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "string");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            Strings
          </h1>
          <p className="text-muted-foreground text-lg">
            Strings are immutable sequences of Unicode characters. They support concatenation, formatting, searching, slicing, and type-checking methods. Mastery of these operations is essential for text processing and interview problems.
          </p>
        </div>

        {/* Creating Strings */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Creating Strings
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Quote Style</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                  <th className="text-left p-3 font-semibold text-foreground">Use when</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">Single quotes</td>
                  <td className="p-3 font-mono">'hello'</td>
                  <td className="p-3">Most common, contains double quotes inside</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">Double quotes</td>
                  <td className="p-3 font-mono">"world"</td>
                  <td className="p-3">Contains single quotes inside (e.g., "It's")</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">Triple quotes</td>
                  <td className="p-3 font-mono">"""multiline"""</td>
                  <td className="p-3">Multiline strings, docstrings</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Different quote styles
s1 = "hello"
s2 = 'world'
s3 = "It's a beautiful day"
s4 = 'He said "Hello"'
s5 = """This is
a multiline
string"""

# Empty string
empty = ""

# String from other types
s6 = str(123)           # "123"
s7 = str(3.14)          # "3.14"
s8 = str([1, 2, 3])     # "[1, 2, 3]"

# Important: Strings are IMMUTABLE
s = "hello"
# s[0] = "H"           # TypeError: 'str' object does not support item assignment
s = "Hello"              # This creates a new string`}
            </pre>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground">Immutability</p>
                <p className="text-muted-foreground">Strings cannot be changed in place. Operations such as <code className="bg-muted px-1.5 py-0.5 rounded">replace()</code> and <code className="bg-muted px-1.5 py-0.5 rounded">upper()</code> return new strings; the original is unchanged.</p>
              </div>
            </div>
          </div>
        </section>

        {/* Slicing and Indexing */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Slicing and Indexing
          </h2>
          
          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">Indexing</h3>
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
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">s[i]</td>
                  <td className="p-3">Single character at index i</td>
                  <td className="p-3 font-mono">"Python"[2] → "t"</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">s[-1]</td>
                  <td className="p-3">Last character</td>
                  <td className="p-3 font-mono">"Python"[-1] → "n"</td>
                </tr>
                <tr className="border-b border-border">
                  <td className="p-3 font-mono">s[-2]</td>
                  <td className="p-3">Second last character</td>
                  <td className="p-3 font-mono">"Python"[-2] → "o"</td>
                </tr>
              </tbody>
            </table>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">Slicing</h3>
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
                <tr className="border-b border-border"><td className="p-3 font-mono">s[:n]</td><td className="p-3">First n characters</td><td className="p-3 font-mono">"Python"[:2] → "Py"</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s[n:]</td><td className="p-3">From index n to end</td><td className="p-3 font-mono">"Python"[2:] → "thon"</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s[i:j]</td><td className="p-3">Slice from i to j (exclusive)</td><td className="p-3 font-mono">"Python"[1:4] → "yth"</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s[-m:-n]</td><td className="p-3">Slice with negative indices</td><td className="p-3 font-mono">"Python"[-4:-2] → "th"</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s[::k]</td><td className="p-3">Every k-th character</td><td className="p-3 font-mono">"Python"[::2] → "Pto"</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s[::-1]</td><td className="p-3">Reversed string</td><td className="p-3 font-mono">"Python"[::-1] → "nohtyP"</td></tr>
              </tbody>
            </table>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`s = "Python"

# Indexing (0-based)
s[0]    # "P"
s[2]    # "t"
s[-1]   # "n" (last character)
s[-2]   # "o" (second last)

# Slicing [start:stop:step]
s[:2]       # "Py"
s[2:]       # "thon"
s[1:4]      # "yth"
s[-4:-2]    # "th"
s[::2]      # "Pto" (every 2nd character)
s[::-1]     # "nohtyP" (reverse)

# Length
len(s)      # 6

# Out of range - raises IndexError
# s[20]     # IndexError: string index out of range

# reversed() and join()
"".join(reversed(s))    # "nohtyP"`}
            </pre>
          </div>
        </section>

        {/* Concatenation */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Concatenation
          </h2>
          
          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">The + Operator</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Join two strings into a new string
"Hello" + " " + "World"    # "Hello World"
"2" + "4" + "6"            # "246"

# Type error - both operands must be strings
# "2" + 4                  # TypeError: can only concatenate str to str

# Convert numbers to strings
"2" + str(4)               # "24"`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">The join() Method</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Method</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">sep.join(iterable)</td><td className="p-3">Joins strings in iterable with separator</td><td className="p-3 font-mono">"-".join(["a","b","c"]) → "a-b-c"</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">"".join(iterable)</td><td className="p-3">Concatenates with no separator</td><td className="p-3 font-mono">"".join(["a","b","c"]) → "abc"</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# join() with different separators
"-".join(["a", "b", "c"])      # "a-b-c"
"".join(["2", "4", "6"])       # "246"
" | ".join(["x", "y", "z"])    # "x | y | z"
", ".join(["apple", "banana", "cherry"])  # "apple, banana, cherry"

# All elements must be strings
# ",".join([1, 2, 3])          # TypeError

# Convert to strings first
",".join(str(x) for x in [1, 2, 3])  # "1,2,3"

# join() with other iterables
"".join("abc")                 # "abc"
"-".join(("a", "b", "c"))      # "a-b-c"`}
            </pre>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground">Key Points about join()</p>
                <ul className="text-muted-foreground text-sm space-y-1 ml-4 list-disc mt-2">
                  <li><strong>Order:</strong> <code className="bg-muted px-1.5 py-0.5 rounded">sep.join(parts)</code> produces <code className="bg-muted px-1.5 py-0.5 rounded">parts[0] + sep + parts[1] + sep + ...</code></li>
                  <li><strong>Non-string elements:</strong> All elements must be strings</li>
                  <li><strong>Empty separator:</strong> <code className="bg-muted px-1.5 py-0.5 rounded">"".join(parts)</code> concatenates with no separator</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Formatting */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Formatting
          </h2>
          
          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">F-Strings (Python 3.6+)</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`name = "Alice"
age = 25
score = 86.8

# Basic f-string
f"Name: {name}, Age: {age}"           # "Name: Alice, Age: 25"

# Format specifiers
f"Score: {score:.2f}"                 # "Score: 86.80"
f"Percentage: {score:.1%}"            # "Percentage: 8680.0%"

# Expressions inside braces
x, y = 4, 6
f"{x} * {y} = {x * y}"                # "4 * 6 = 24"

# Alignment and padding
f"{'hi':<10}"                         # "hi        " (left align)
f"{'hi':>10}"                         # "        hi" (right align)
f"{'hi':^10}"                         # "    hi    " (center)
f"{42:06d}"                           # "000042" (zero pad)`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">The format() Method</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Positional arguments
"{} and {}".format("a", "b")          # "a and b"
"{0} and {1}".format("x", "y")        # "x and y"
"{1} and {0}".format("x", "y")        # "y and x"

# Keyword arguments
"{name} is {age} years old".format(name="Alice", age=25)  # "Alice is 25 years old"

# Mixed
"{0} is {name}".format("Alice", name="Smith")  # "Alice is Smith"

# Format specifiers
"{:.2f}".format(3.14159)              # "3.14"
"{:06d}".format(42)                   # "000042"`}
            </pre>
          </div>
        </section>

        {/* Modify Strings */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Modify Strings
          </h2>
          <p className="text-muted-foreground mb-4">Strings are immutable; these methods return new strings.</p>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">Case Conversion</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Method</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">s.upper()</td><td className="p-3">All characters to uppercase</td><td className="p-3 font-mono">"hello".upper() → "HELLO"</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s.lower()</td><td className="p-3">All characters to lowercase</td><td className="p-3 font-mono">"HELLO".lower() → "hello"</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s.casefold()</td><td className="p-3">Aggressive lowercase for caseless comparison</td><td className="p-3 font-mono">"Straße".casefold() → "strasse"</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s.capitalize()</td><td className="p-3">First character uppercase; rest lowercase</td><td className="p-3 font-mono">"hello".capitalize() → "Hello"</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s.title()</td><td className="p-3">First letter of each word uppercase</td><td className="p-3 font-mono">"hello world".title() → "Hello World"</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s.swapcase()</td><td className="p-3">Swap uppercase and lowercase</td><td className="p-3 font-mono">"HeLLo".swapcase() → "hEllO"</td></tr>
              </tbody>
            </table>
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 mb-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground">casefold() vs lower()</p>
                <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">casefold()</code> handles more Unicode cases (e.g., German ß → ss). Use it for caseless comparison when <code className="bg-muted px-1.5 py-0.5 rounded">lower()</code> is insufficient.</p>
              </div>
            </div>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">Replace</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Replace all occurrences
"aabb".replace("a", "x")                     # "xxbb"

# Replace with count limit
"one one one".replace("one", "two", 2)       # "two two one"

# Remove characters by replacing with empty
"hello world".replace(" ", "")               # "helloworld"

# Replace can remove substrings
"file.txt".replace(".txt", "")               # "file"`}
            </pre>
          </div>
        </section>

        {/* Search */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Search
          </h2>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">find() and index()</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Method</th>
                  <th className="text-left p-3 font-semibold text-foreground">When found</th>
                  <th className="text-left p-3 font-semibold text-foreground">When not found</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">s.find(sub)</td><td className="p-3">Index (int)</td><td className="p-3">-1</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s.index(sub)</td><td className="p-3">Index (int)</td><td className="p-3 text-destructive">ValueError</td></tr>
              </tbody>
            </table>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">count(), in, startswith(), endswith()</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Method</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">s.count(sub)</td><td className="p-3">Number of non-overlapping occurrences</td><td className="p-3 font-mono">"hello".count("l") → 2</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">sub in s</td><td className="p-3">True if substring exists</td><td className="p-3 font-mono">"ll" in "hello" → True</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s.startswith(prefix)</td><td className="p-3">True if string starts with prefix</td><td className="p-3 font-mono">"hello.py".startswith("hello") → True</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s.endswith(suffix)</td><td className="p-3">True if string ends with suffix</td><td className="p-3 font-mono">"hello.py".endswith(".py") → True</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Type-Checking Methods */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Type-Checking Methods
          </h2>
          <p className="text-muted-foreground mb-4">These methods return <code className="bg-muted px-1.5 py-0.5 rounded">True</code> or <code className="bg-muted px-1.5 py-0.5 rounded">False</code> based on character properties.</p>

          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Method</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">s.isalpha()</td><td className="p-3">All characters are alphabetic</td><td className="p-3 font-mono">"hello".isalpha() → True</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s.isalnum()</td><td className="p-3">All characters are alphanumeric</td><td className="p-3 font-mono">"abc123".isalnum() → True</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s.isdigit()</td><td className="p-3">All characters are digits</td><td className="p-3 font-mono">"468".isdigit() → True</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s.islower()</td><td className="p-3">All cased characters are lowercase</td><td className="p-3 font-mono">"hello".islower() → True</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s.isupper()</td><td className="p-3">All cased characters are uppercase</td><td className="p-3 font-mono">"HELLO".isupper() → True</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">s.isspace()</td><td className="p-3">All characters are whitespace</td><td className="p-3 font-mono">"   ".isspace() → True</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Performance Tips */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Performance Tips
          </h2>
          <div className="space-y-3">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">String concatenation in loops</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`# SLOW - creates many intermediate strings
result = ""
for s in list_of_strings:
    result += s

# FAST - use join()
result = "".join(list_of_strings)`}
                  </pre>
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
                  <p className="font-semibold text-foreground">Strings are immutable</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`# WRONG - strings cannot be modified in place
s = "hello"
# s[0] = "H"        # TypeError

# CORRECT - assign the result of the method
s = s.upper()        # s becomes "HELLO"`}
                  </pre>
                </div>
              </div>
            </div>

            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">find() vs index()</p>
                  <p className="text-muted-foreground">Use <code className="bg-muted px-1.5 py-0.5 rounded">find()</code> when absence is expected; it returns -1. Use <code className="bg-muted px-1.5 py-0.5 rounded">index()</code> when absent is an error; it raises <code className="bg-muted px-1.5 py-0.5 rounded">ValueError</code>.</p>
                </div>
              </div>
            </div>

            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">strip() removes characters, not substrings</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`# strip() removes any characters in the argument string
"hello".strip("ho")    # "ell" - removes 'h' and 'o'

# To remove a specific substring, use replace
"hello".replace("he", "")  # "llo"`}
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
                  <p className="font-semibold text-foreground mb-1">How do you reverse a string?</p>
                  <pre className="mt-2 text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                    {`s[::-1]                          # Slicing approach
"".join(reversed(s))               # join with reversed iterator`}
                  </pre>
                </div>
              </div>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">What does "".join(parts) do?</p>
                  <p className="text-muted-foreground">Concatenates all strings in <code className="bg-muted px-1.5 py-0.5 rounded">parts</code> with no separator.</p>
                </div>
              </div>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">What is the difference between find() and index()?</p>
                  <p className="text-muted-foreground">Both return the lowest index of the substring. <code className="bg-muted px-1.5 py-0.5 rounded">find()</code> returns -1 when not found; <code className="bg-muted px-1.5 py-0.5 rounded">index()</code> raises <code className="bg-muted px-1.5 py-0.5 rounded">ValueError</code>.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

      
    
      </div>
    </TopicContent>
  );
}