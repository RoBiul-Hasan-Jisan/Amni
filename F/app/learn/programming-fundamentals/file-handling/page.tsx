import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function FileHandlingPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "file-handling");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            File I/O
          </h1>
          <p className="text-muted-foreground text-lg">
            <strong className="text-foreground">File handling</strong> enables reading from and writing to files on disk. Python provides built-in functions like <code className="bg-muted px-1.5 py-0.5 rounded">open()</code> and the <code className="bg-muted px-1.5 py-0.5 rounded">with</code> statement for safe resource management.
          </p>
        </div>

        {/* Note Box */}
        <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
          <div className="flex gap-3">
            <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
            <p className="text-sm text-foreground">
              <strong>Always close files!</strong> Unclosed files can cause data loss, memory leaks, and "too many open files" errors. The <code className="bg-muted px-1.5 py-0.5 rounded">with</code> statement closes files automatically — use it whenever possible.
            </p>
          </div>
        </div>

        {/* Opening Files */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Opening Files
          </h2>
          <p className="text-muted-foreground mb-4">
            Use <code className="bg-muted px-1.5 py-0.5 rounded">open()</code> to open a file. It returns a file object with methods for reading and writing.
          </p>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Mode</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                  <th className="text-left p-3 font-semibold text-foreground">File pointer</th>
                  <th className="text-left p-3 font-semibold text-foreground">Creates file?</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">'r'</td><td className="p-3">Read (default)</td><td className="p-3">Beginning</td><td className="p-3">No (raises error if missing)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">'w'</td><td className="p-3">Write (overwrites)</td><td className="p-3">Beginning</td><td className="p-3">Yes (truncates existing)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">'a'</td><td className="p-3">Append</td><td className="p-3">End</td><td className="p-3">Yes</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">'x'</td><td className="p-3">Exclusive creation</td><td className="p-3">Beginning</td><td className="p-3">Only if doesn't exist (raises error if exists)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">'r+'</td><td className="p-3">Read and write</td><td className="p-3">Beginning</td><td className="p-3">No</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">'w+'</td><td className="p-3">Read and write (overwrites)</td><td className="p-3">Beginning</td><td className="p-3">Yes (truncates)</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">'a+'</td><td className="p-3">Read and append</td><td className="p-3">End for writing, beginning for reading</td><td className="p-3">Yes</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">'b'</td><td className="p-3">Binary mode (add to any mode)</td><td className="p-3">-</td><td className="p-3">-</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">'t'</td><td className="p-3">Text mode (default)</td><td className="p-3">-</td><td className="p-3">-</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Open a file (returns a file object)
file = open("data.txt", "r")  # Read mode (default)
file = open("data.txt", "w")  # Write mode (overwrites)
file = open("data.txt", "a")  # Append mode
file = open("data.txt", "rb") # Binary read mode

# Always close when done!
file.close()`}
            </pre>
          </div>
        </section>

        {/* The with Statement */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            The with Statement (Context Manager)
          </h2>
          <p className="text-muted-foreground mb-4">
            The <code className="bg-muted px-1.5 py-0.5 rounded">with</code> statement automatically closes the file, even if an exception occurs.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Best practice: use 'with' for automatic cleanup
with open("data.txt", "r") as file:
    content = file.read()
    # File is automatically closed when exiting the 'with' block

# Equivalent to:
file = open("data.txt", "r")
try:
    content = file.read()
finally:
    file.close()  # Always runs, even on exception`}
            </pre>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <strong>Always prefer the 'with' statement</strong> — it's cleaner, safer, and the Pythonic way to handle files.
              </p>
            </div>
          </div>
        </section>

        {/* Reading Files */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Reading Files
          </h2>
          
          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">read() - Read entire file</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`with open("data.txt", "r") as f:
    content = f.read()          # Reads entire file as string
    print(content)
    
    # Optionally, read only N characters
    f.seek(0)                   # Go back to beginning
    first_10 = f.read(10)       # Read first 10 characters`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">readline() - Read line by line</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`with open("data.txt", "r") as f:
    line = f.readline()         # Reads one line (including newline)
    while line:
        print(line.strip())
        line = f.readline()`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">readlines() - Read all lines into a list</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`with open("data.txt", "r") as f:
    lines = f.readlines()       # Returns list of strings (each line)
    for line in lines:
        print(line.strip())`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">Iterate directly (most memory-efficient)</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Best for large files - doesn't load entire file into memory
with open("data.txt", "r") as f:
    for line in f:
        print(line.strip())     # Processes one line at a time`}
            </pre>
          </div>
        </section>

        {/* Writing Files */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Writing Files
          </h2>
          
          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">write() - Write string to file</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Write mode (overwrites existing content)
with open("output.txt", "w") as f:
    f.write("Hello, World!\\n")
    f.write("Second line\\n")

# Append mode (adds to end of file)
with open("output.txt", "a") as f:
    f.write("This line is appended\\n")`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">writelines() - Write multiple lines</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`lines = ["First line\\n", "Second line\\n", "Third line\\n"]
with open("output.txt", "w") as f:
    f.writelines(lines)  # Writes all strings in the list

# Note: writelines() does NOT add newlines automatically
# You must include \\n in each string if you want line breaks`}
            </pre>
          </div>
        </section>

        {/* Working with Binary Files */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Working with Binary Files
          </h2>
          <p className="text-muted-foreground mb-4">
            Use <code className="bg-muted px-1.5 py-0.5 rounded">'rb'</code> or <code className="bg-muted px-1.5 py-0.5 rounded">'wb'</code> modes for binary data (images, audio, video, pickled objects).
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Reading a binary file
with open("image.jpg", "rb") as f:
    data = f.read()     # Returns bytes object
    print(f"Size: {len(data)} bytes")

# Writing a binary file
with open("copy.jpg", "wb") as f:
    f.write(data)

# Reading a file in chunks (memory-efficient for large files)
with open("large_file.bin", "rb") as f:
    chunk_size = 1024 * 1024  # 1MB
    while True:
        chunk = f.read(chunk_size)
        if not chunk:          # EOF
            break
        process(chunk)`}
            </pre>
          </div>
        </section>

        {/* File Position: seek() and tell() */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            File Position: seek() and tell()
          </h2>
          <p className="text-muted-foreground mb-4">
            Track and move the file pointer (position where reading/writing happens).
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`with open("data.txt", "r") as f:
    print(f.tell())        # 0 (beginning)
    
    content = f.read(5)    # Read first 5 characters
    print(f.tell())        # 5
    
    f.seek(0)              # Go back to beginning
    print(f.tell())        # 0
    
    f.seek(10, 0)          # Move to position 10 from start (0 = SEEK_SET)
    f.seek(5, 1)           # Move forward 5 bytes from current (1 = SEEK_CUR)
    f.seek(-10, 2)         # Move to 10 bytes before end (2 = SEEK_END)`}
            </pre>
          </div>
        </section>

        {/* Common File Operations */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Common File Operations
          </h2>
          <div className="space-y-4">
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">Check if file exists (without opening)</p>
              <pre className="text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                {`import os
if os.path.exists("data.txt"):
    print("File exists")

from pathlib import Path
if Path("data.txt").exists():
    print("File exists")`}
              </pre>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">Copy a file</p>
              <pre className="text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                {`import shutil
shutil.copy("source.txt", "destination.txt")`}
              </pre>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">Delete a file</p>
              <pre className="text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                {`import os
os.remove("unwanted.txt")

from pathlib import Path
Path("unwanted.txt").unlink()`}
              </pre>
            </div>

            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <p className="font-semibold text-foreground mb-2">Rename/move a file</p>
              <pre className="text-sm bg-muted/50 p-2 rounded text-primary overflow-x-auto">
                {`import os
os.rename("old.txt", "new.txt")

import shutil
shutil.move("file.txt", "subdir/file.txt")`}
              </pre>
            </div>
          </div>
        </section>

        {/* Working with CSV Files */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Working with CSV Files
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import csv

# Reading CSV
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)  # row is a list of strings

# Reading CSV with headers
with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["name"], row["age"])

# Writing CSV
with open("output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age", "City"])
    writer.writerow(["Alice", 30, "NYC"])
    writer.writerow(["Bob", 25, "LA"])`}
            </pre>
          </div>
        </section>

        {/* Working with JSON Files */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Working with JSON Files
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import json

# Writing JSON
data = {"name": "Alice", "age": 30, "city": "NYC"}
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)  # Pretty print with indent

# Reading JSON
with open("data.json", "r") as f:
    data = json.load(f)
    print(data["name"])  # Alice

# JSON string to dict
json_string = '{"name": "Bob", "age": 25}'
obj = json.loads(json_string)

# Dict to JSON string
json_string = json.dumps(obj, indent=2)`}
            </pre>
          </div>
        </section>

        {/* Error Handling */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Error Handling
          </h2>
          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4 mb-4">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-foreground">Common file-related exceptions</p>
                <ul className="list-disc list-inside text-muted-foreground text-sm mt-2">
                  <li><code className="bg-muted px-1.5 py-0.5 rounded">FileNotFoundError</code> - File doesn't exist</li>
                  <li><code className="bg-muted px-1.5 py-0.5 rounded">PermissionError</code> - Don't have permission to read/write</li>
                  <li><code className="bg-muted px-1.5 py-0.5 rounded">IsADirectoryError</code> - Tried to read a directory as a file</li>
                  <li><code className="bg-muted px-1.5 py-0.5 rounded">UnicodeDecodeError</code> - Wrong encoding when reading text</li>
                </ul>
              </div>
            </div>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`try:
    with open("data.txt", "r") as f:
        content = f.read()
except FileNotFoundError:
    print("File not found!")
except PermissionError:
    print("Permission denied!")
except Exception as e:
    print(f"Unexpected error: {e}")
else:
    print("File read successfully")
finally:
    print("Done attempting to read file")`}
            </pre>
          </div>
        </section>

        {/* File Encoding */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            File Encoding
          </h2>
          <p className="text-muted-foreground mb-4">
            Specify encoding to handle non-ASCII characters correctly.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Specify encoding (UTF-8 is standard)
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Common encodings: 'utf-8', 'latin-1', 'cp1252', 'ascii'

# Handle encoding errors
with open("data.txt", "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()  # Skips invalid characters

with open("data.txt", "r", encoding="utf-8", errors="replace") as f:
    content = f.read()  # Replaces invalid chars with '�'`}
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
                  <p className="font-semibold text-foreground">Forgetting to close files</p>
                  <p className="text-muted-foreground text-sm">Without <code className="bg-muted px-1.5 py-0.5 rounded">with</code> or <code className="bg-muted px-1.5 py-0.5 rounded">.close()</code>, writes may not be saved and file handles may leak. Always use <code className="bg-muted px-1.5 py-0.5 rounded">with</code>.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Mode 'w' truncates (overwrites) files</p>
                  <p className="text-muted-foreground text-sm">Opening with <code className="bg-muted px-1.5 py-0.5 rounded">'w'</code> destroys existing content. Use <code className="bg-muted px-1.5 py-0.5 rounded">'a'</code> to append or <code className="bg-muted px-1.5 py-0.5 rounded">'r+'</code> to read/write without truncating.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Newline handling differs across platforms</p>
                  <p className="text-muted-foreground text-sm">Windows uses <code className="bg-muted px-1.5 py-0.5 rounded">\r\n</code>, Unix/Linux uses <code className="bg-muted px-1.5 py-0.5 rounded">\n</code>. Python handles this automatically in text mode, but be careful with binary files.</p>
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
                  <p className="font-semibold text-foreground">What is the difference between 'r' and 'r+' modes?</p>
                  <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">'r'</code> is read-only; <code className="bg-muted px-1.5 py-0.5 rounded">'r+'</code> allows both reading and writing. Neither creates a missing file (unlike <code className="bg-muted px-1.5 py-0.5 rounded">'w'</code> or <code className="bg-muted px-1.5 py-0.5 rounded">'a'</code>).</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What does the 'with' statement do?</p>
                  <p className="text-muted-foreground">It's a context manager that ensures resources are cleaned up. For files, it automatically calls <code className="bg-muted px-1.5 py-0.5 rounded">.close()</code> even if an exception occurs.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">How do you read a large file line by line efficiently?</p>
                  <p className="text-muted-foreground">Use <code className="bg-muted px-1.5 py-0.5 rounded">for line in file:</code> which reads one line at a time without loading the entire file into memory.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">How do you read and write JSON files in Python?</p>
                  <p className="text-muted-foreground">Use <code className="bg-muted px-1.5 py-0.5 rounded">json.load()</code> to read and <code className="bg-muted px-1.5 py-0.5 rounded">json.dump()</code> to write. For strings, use <code className="bg-muted px-1.5 py-0.5 rounded">json.loads()</code> and <code className="bg-muted px-1.5 py-0.5 rounded">json.dumps()</code>.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-border mt-4">
          <Link
            href="/learn/programming-fundamentals/error-handling"
            className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            ← Previous: Error Handling
          </Link>
          <Link
            href="/learn/programming-fundamentals/modules"
            className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: Modules →
          </Link>
        </div>
      </div>
    </TopicContent>
  );
}