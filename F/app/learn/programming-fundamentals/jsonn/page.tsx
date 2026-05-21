import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function JsonPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "jsonn");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">
            JSON
          </h1>
          <p className="text-muted-foreground text-lg">
            JSON (JavaScript Object Notation) is a lightweight text format for exchanging data. Python's <code className="bg-muted px-1.5 py-0.5 rounded">json</code> module serializes Python objects to JSON strings and deserializes JSON back to Python. It is built into the standard library and requires no installation.
          </p>
        </div>

        {/* Note Box */}
        <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
          <div className="flex gap-3">
            <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
            <p className="text-sm text-foreground">
              JSON shows up in APIs, config files, and data pipelines. Understanding the type mapping between Python and JSON, plus the common pitfalls (NaN, custom objects, encoding), saves time in interviews and real projects.
            </p>
          </div>
        </div>

        {/* What is JSON? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            What is JSON?
          </h2>
          <p className="text-muted-foreground mb-4">
            JSON is a text format that uses a small set of structures: objects (key-value pairs), arrays (ordered lists), strings, numbers, booleans, and null. It is human-readable and widely supported across languages. The format does not support comments, trailing commas, or single quotes.
          </p>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">JSON Type</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3">Object</td><td className="p-3 font-mono">{'{"a": 2, "b": 4}'}</td></tr>
                <tr className="border-b border-border"><td className="p-3">Array</td><td className="p-3 font-mono">[2, 4, 6, 8]</td></tr>
                <tr className="border-b border-border"><td className="p-3">String</td><td className="p-3 font-mono">"hello"</td></tr>
                <tr className="border-b border-border"><td className="p-3">Number</td><td className="p-3 font-mono">42</td></tr>
                <tr className="border-b border-border"><td className="p-3">Boolean</td><td className="p-3 font-mono">true, false</td></tr>
                <tr className="border-b border-border"><td className="p-3">Null</td><td className="p-3 font-mono">null</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Python to JSON Type Mapping */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Python to JSON Type Mapping
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Python Type</th>
                  <th className="text-left p-3 font-semibold text-foreground">JSON Type</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">dict</td><td className="p-3">object</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">list, tuple</td><td className="p-3">array</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">str</td><td className="p-3">string</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">int, float</td><td className="p-3">number</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">bool</td><td className="p-3">boolean</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">None</td><td className="p-3">null</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <strong>No direct mapping:</strong> <code className="bg-muted px-1.5 py-0.5 rounded">set</code>, <code className="bg-muted px-1.5 py-0.5 rounded">frozenset</code>, <code className="bg-muted px-1.5 py-0.5 rounded">bytes</code>, <code className="bg-muted px-1.5 py-0.5 rounded">complex</code>, <code className="bg-muted px-1.5 py-0.5 rounded">datetime</code>, and custom classes are not JSON types. Serializing them requires custom handling.
              </p>
            </div>
          </div>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mt-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import json

data = {"a": 2, "b": 4, "c": [6, 8], "d": None, "e": True}
s = json.dumps(data)
print(s)  # {"a": 2, "b": 4, "c": [6, 8], "d": null, "e": true}`}
            </pre>
          </div>
        </section>

        {/* Serialize: json.dumps() */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Serialize: <code className="bg-muted px-2 py-0.5 rounded">json.dumps()</code>
          </h2>
          <p className="text-muted-foreground mb-4">
            <code className="bg-muted px-1.5 py-0.5 rounded">json.dumps(obj)</code> converts a Python object to a JSON string. The result is a <code className="bg-muted px-1.5 py-0.5 rounded">str</code>, not bytes.
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import json

data = {"x": 2, "y": 4, "z": 6}
s = json.dumps(data)
print(s)           # {"x": 2, "y": 4, "z": 6}
print(type(s))     # <class 'str'>`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">Common Parameters</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg mb-4">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Parameter</th>
                  <th className="text-left p-3 font-semibold text-foreground">Default</th>
                  <th className="text-left p-3 font-semibold text-foreground">Description</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-mono">indent</td><td className="p-3">None</td><td className="p-3">Pretty-print with this many spaces per level</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">sort_keys</td><td className="p-3">False</td><td className="p-3">Sort dictionary keys in output</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">ensure_ascii</td><td className="p-3">True</td><td className="p-3">Escape non-ASCII characters as \uXXXX</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">separators</td><td className="p-3">(', ', ': ')</td><td className="p-3">Custom separators for compact output</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-mono">default</td><td className="p-3">None</td><td className="p-3">Callable for non-serializable types</td></tr>
              </tbody>
            </table>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`# Pretty printing
data = {"a": 2, "b": 4, "c": [6, 8]}
print(json.dumps(data, indent=2))
# {
#   "a": 2,
#   "b": 4,
#   "c": [6, 8]
# }

# Sort keys
data = {"z": 2, "a": 4, "m": 6}
print(json.dumps(data, sort_keys=True))  # {"a": 4, "m": 6, "z": 2}

# Compact output
print(json.dumps(data, separators=(',', ':')))  # {"a":2,"b":4}

# Non-ASCII
data = {"name": "José"}
print(json.dumps(data))                    # {"name": "Jos\\u00e9"}
print(json.dumps(data, ensure_ascii=False))  # {"name": "José"}`}
            </pre>
          </div>
        </section>

        {/* Deserialize: json.loads() */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Deserialize: <code className="bg-muted px-2 py-0.5 rounded">json.loads()</code>
          </h2>
          <p className="text-muted-foreground mb-4">
            <code className="bg-muted px-1.5 py-0.5 rounded">json.loads(s)</code> parses a JSON string and returns a Python object. The input must be a <code className="bg-muted px-1.5 py-0.5 rounded">str</code> or <code className="bg-muted px-1.5 py-0.5 rounded">bytes</code> (decoded as UTF-8).
          </p>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import json

s = '{"a": 2, "b": 4, "c": [6, 8]}'
data = json.loads(s)
print(data)        # {'a': 2, 'b': 4, 'c': [6, 8]}
print(type(data))  # <class 'dict'>

# JSON arrays become lists
s = '[2, 4, 6, 8]'
data = json.loads(s)
print(type(data))  # <class 'list'>`}
            </pre>
          </div>
        </section>

        {/* File Operations */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            File Operations: <code className="bg-muted px-2 py-0.5 rounded">json.dump()</code> and <code className="bg-muted px-2 py-0.5 rounded">json.load()</code>
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import json

# Write to file
data = {"scores": [82, 84, 86, 88]}
with open("scores.json", "w") as f:
    json.dump(data, f)

# With options
with open("out.json", "w") as f:
    json.dump(data, f, indent=2, sort_keys=True)

# Read from file
with open("scores.json") as f:
    data = json.load(f)
print(data)  # {'scores': [82, 84, 86, 88]}`}
            </pre>
          </div>
        </section>

        {/* Handling Non-Serializable Types */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Handling Non-Serializable Types
          </h2>
          
          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">The <code className="bg-muted px-2 py-0.5 rounded">default</code> Callable</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import json
from datetime import datetime

def serialize(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

data = {"created": datetime(2024, 2, 4), "tags": {2, 4, 6}}
s = json.dumps(data, default=serialize)
print(s)  # {"created": "2024-02-04T00:00:00", "tags": [2, 4, 6]}

# Custom class serialization
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def default(obj):
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"{type(obj).__name__} not serializable")

data = {"p": Point(2, 4)}
print(json.dumps(data, default=default))  # {"p": {"x": 2, "y": 4}}`}
            </pre>
          </div>

          <h3 className="text-xl font-semibold text-foreground mt-4 mb-2">Custom JSONEncoder</h3>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import json

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return {"__set__": True, "items": list(obj)}
        return super().default(obj)

data = {"ids": {2, 4, 6}}
s = json.dumps(data, cls=SetEncoder)
print(s)  # {"ids": {"__set__": true, "items": [2, 4, 6]}}`}
            </pre>
          </div>
        </section>

        {/* Custom Decoding with object_hook */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Custom Decoding with <code className="bg-muted px-2 py-0.5 rounded">object_hook</code>
          </h2>
          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`import json

def decode_set(d):
    if d.get("__set__") is True:
        return set(d["items"])
    return d

s = '{"ids": {"__set__": true, "items": [2, 4, 6]}}'
data = json.loads(s, object_hook=decode_set)
print(data)  # {'ids': {2, 4, 6}}
print(type(data["ids"]))  # <class 'set'>`}
            </pre>
          </div>
        </section>

        {/* JSON vs Other Formats */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            JSON vs Other Formats
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Format</th>
                  <th className="text-left p-3 font-semibold text-foreground">Use case</th>
                  <th className="text-left p-3 font-semibold text-foreground">Python support</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3 font-semibold">JSON</td><td className="p-3">APIs, config, human-readable data exchange</td><td className="p-3 font-mono">json</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-semibold">pickle</td><td className="p-3">Python-only, arbitrary objects, not human-readable</td><td className="p-3 font-mono">pickle</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-semibold">YAML</td><td className="p-3">Config files, comments, more readable</td><td className="p-3 font-mono">pyyaml</td></tr>
                <tr className="border-b border-border"><td className="p-3 font-semibold">TOML</td><td className="p-3">Config files, simple syntax</td><td className="p-3 font-mono">toml</td></tr>
              </tbody>
            </table>
          </div>
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <strong>When to use JSON:</strong> Cross-language data exchange, REST APIs, config that must be readable. Do not use JSON for sensitive data without encryption.
              </p>
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
                  <p className="font-semibold text-foreground">Tuples become arrays</p>
                  <p className="text-muted-foreground text-sm">JSON has no tuple type. <code className="bg-muted px-1.5 py-0.5 rounded">json.dumps((2, 4, 6))</code> produces <code className="bg-muted px-1.5 py-0.5 rounded">[2, 4, 6]</code>. Round-trip returns a list, not a tuple.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Keys must be strings</p>
                 <p className="text-muted-foreground text-sm">
  <code className="bg-muted px-1.5 py-0.5 rounded">
    {`json.dumps({2: "a", 4: "b"})`}
  </code>{" "}
  converts keys to strings:{" "}
  <code className="bg-muted px-1.5 py-0.5 rounded">
    {`{"2": "a", "4": "b"}`}
  </code>
  .
</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">NaN and Infinity</p>
                  <p className="text-muted-foreground text-sm"><code className="bg-muted px-1.5 py-0.5 rounded">float('nan')</code> and <code className="bg-muted px-1.5 py-0.5 rounded">float('inf')</code> serialize to <code className="bg-muted px-1.5 py-0.5 rounded">NaN</code> and <code className="bg-muted px-1.5 py-0.5 rounded">Infinity</code>, which are not valid strict JSON (RFC 8259).</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-4">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">Duplicate keys</p>
                  <p className="text-muted-foreground text-sm">
  <code className="bg-muted px-1.5 py-0.5 rounded">
    {`json.loads('{"a": 2, "a": 4}')`}
  </code>{" "}
  returns{" "}
  <code className="bg-muted px-1.5 py-0.5 rounded">
    {`{"a": 4}`}
  </code>
  . The last value wins.
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
                  <p className="font-semibold text-foreground">What is JSON and what types does it support?</p>
                  <p className="text-muted-foreground">JSON supports objects, arrays, strings, numbers, booleans, and null. It does not support comments, trailing commas, or single-quoted strings.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">How does Python map types to JSON?</p>
                  <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">dict</code> → object, <code className="bg-muted px-1.5 py-0.5 rounded">list/tuple</code> → array, <code className="bg-muted px-1.5 py-0.5 rounded">str</code> → string, <code className="bg-muted px-1.5 py-0.5 rounded">int/float</code> → number, <code className="bg-muted px-1.5 py-0.5 rounded">bool</code> → boolean, <code className="bg-muted px-1.5 py-0.5 rounded">None</code> → null.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">What is the difference between <code className="bg-muted px-1.5 py-0.5 rounded">json.dumps()</code> and <code className="bg-muted px-1.5 py-0.5 rounded">json.dump()</code>?</p>
                  <p className="text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">dumps()</code> returns a JSON string. <code className="bg-muted px-1.5 py-0.5 rounded">dump()</code> writes to a file object.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground">How do you serialize a <code className="bg-muted px-1.5 py-0.5 rounded">datetime</code> or custom class?</p>
                  <p className="text-muted-foreground">Use the <code className="bg-muted px-1.5 py-0.5 rounded">default</code> parameter: <code className="bg-muted px-1.5 py-0.5 rounded">json.dumps(data, default=my_serializer)</code>.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

    
      </div>
    </TopicContent>
  );
}