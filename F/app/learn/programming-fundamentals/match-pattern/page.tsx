// app/learn/programming-fundamentals/match-pattern/page.tsx

import Link from "next/link";

export default function MatchPatternPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Navigation Bar */}
      <nav className="bg-white dark:bg-gray-800 shadow-md sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Link href="/learn" className="text-blue-600 dark:text-blue-400 hover:underline">
                ← Back to Learning Path
              </Link>
            </div>
            <div className="text-gray-700 dark:text-gray-300 font-semibold">
              Programming Fundamentals
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        {/* Header */}
        <div className="mb-8 border-b border-gray-200 dark:border-gray-700 pb-4">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Match (Structural Pattern Matching)
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            Structural pattern matching picks one branch by matching a value against patterns. Python&apos;s <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">match</code> (3.10+) often replaces long <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">if</code>–<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">elif</code>–<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">else</code> chains when you&apos;re branching on &quot;one of these exact values&quot; or on the <span className="font-semibold">shape</span> of the value (e.g. how long a list is or which keys a dict has).
          </p>
        </div>

        {/* Why match? */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Why match?
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            When every branch depends on the <span className="font-semibold">same</span> value (a status code, a command name, the shape of some structure), <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">if</code>–<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">elif</code>–<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">else</code> works but gets noisy. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">match</code>–<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">case</code> puts the value in one place and turns each branch into a <span className="font-semibold">pattern</span>: it either matches or it doesn&apos;t. The first <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">case</code> that matches runs; nothing falls through to the next. Match really shines when you care about more than equality (e.g. &quot;a list of two elements&quot; or &quot;a dict with key <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">'x'</code>&quot;).
          </p>
        </section>

        {/* Basic Syntax */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Basic Syntax
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            You write <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">match</code> followed by a <span className="font-semibold">subject</span> (the value you&apos;re inspecting), then one or more <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">case</code> blocks. Each <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">case</code> has a <span className="font-semibold">pattern</span> and, optionally, a <span className="font-semibold">guard</span>. Python tries the patterns in order; the first one that matches runs, and the rest are skipped.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`match subject:
    case pattern_1:
        block_1
    case pattern_2:
        block_2
    case _:
        block_default`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mb-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              Colons after <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">match</code> and after each <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case</code> are required. The block under a <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case</code> is whatever&apos;s indented under it. The <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">_</code> in the last case is a <span className="font-semibold">wildcard</span>: it matches anything and is commonly used as the default.
            </p>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`status = 4
match status:
    case 2:
        label = "pending"
    case 4:
        label = "active"
    case 6:
        label = "done"
    case _:
        label = "unknown"

print(label) # active`}
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 p-3 mt-4">
            <p className="text-sm text-green-800 dark:text-green-200">
              ✅ Exactly one <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case</code> runs. When it&apos;s done, execution continues after the whole <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">match</code>; no fall-through.
            </p>
          </div>
        </section>

        {/* Literal Patterns */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Literal Patterns
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            Put a number, string, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">None</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">True</code>, or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">False</code> right in the <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">case</code> line and you get a <span className="font-semibold">literal pattern</span>. The subject is compared to that value.
          </p>

          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Literal type</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Example in case</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Matches when</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3">Integer</td>
                  <td className="p-3 font-mono">case 4:</td>
                  <td className="p-3">Subject equals 4</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3">Float</td>
                  <td className="p-3 font-mono">case 2.0:</td>
                  <td className="p-3">Subject equals 2.0</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3">String</td>
                  <td className="p-3 font-mono">case "ok":</td>
                  <td className="p-3">Subject equals "ok"</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3">None</td>
                  <td className="p-3 font-mono">case None:</td>
                  <td className="p-3">Subject is None (identity)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3">True</td>
                  <td className="p-3 font-mono">case True:</td>
                  <td className="p-3">Subject is True (identity)</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3">False</td>
                  <td className="p-3 font-mono">case False:</td>
                  <td className="p-3">Subject is False (identity)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mb-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Important:</span> For <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">None</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">True</code>, and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">False</code>, the pattern uses <span className="font-semibold">identity</span> (<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">is</code>), not <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">==</code>. For everything else (numbers, strings), it uses <span className="font-semibold">equality</span> (<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">==</code>). So <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case None:</code> matches only the actual <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">None</code> object; <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case 0:</code> matches when the subject equals 0.
            </p>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`x = 0
match x:
    case 0:
        print("zero")    # runs
    case None:
        print("none")    # not run

match None:
    case None:
        print("none")    # runs

cmd = "start"
match cmd:
    case "start":
        action = "run"
    case "stop":
        action = "halt"
    case _:
        action = "unknown"
# action is "run"`}
            </pre>
          </div>
        </section>

        {/* Capture Patterns */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Capture Patterns
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            A bare <span className="font-semibold">name</span> in a pattern is a <span className="font-semibold">capture</span>: it matches any value and <span className="font-semibold">binds</span> it to that name so you can use it in the block. It does <span className="font-semibold">not</span> compare the subject to some existing variable.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`value = 6
match value:
    case x:
        print(x)   # 6; x is bound to the subject`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mb-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              So <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case x:</code> means &quot;match anything and call it <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">x</code>.&quot; It does <span className="font-semibold">not</span> mean &quot;match when the subject equals the variable <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">x</code>.&quot; To match a specific value, use a literal: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case 6:</code>.
            </p>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`data = [2, 4, 6]
match data:
    case [a, b, c]:
        total = a + b + c   # a=2, b=4, c=6
# total is 12`}
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 p-3 mt-4">
            <p className="text-sm text-green-800 dark:text-green-200">
              ✅ Those names only exist inside that <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case</code> block, not in other cases or after the <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">match</code>.
            </p>
          </div>
        </section>

        {/* Wildcard Pattern */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Wildcard Pattern
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">_</code> matches any value and doesn&apos;t bind it. Use it when you don&apos;t care what the value is (e.g. for a default or &quot;anything else&quot;).
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`code = 8
match code:
    case 2:
        result = "low"
    case 4 | 6:
        result = "mid"
    case _:
        result = "other"   # 8 falls here; no name bound
# result is "other"`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              One <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">_</code> per case is enough. You can repeat it in the same case; it still means &quot;match anything.&quot;
            </p>
          </div>
        </section>

        {/* OR Patterns */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            OR Patterns
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">|</code> lets you combine several patterns in one <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">case</code>. The case runs if <span className="font-semibold">any</span> of them matches. Python checks the subject against each pattern left to right until one matches.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`match n:
    case 2 | 4 | 6:
        kind = "even_small"
    case 8 | 10:
        kind = "even_large"
    case _:
        kind = "other"`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              The alternatives don&apos;t bind anything; if you need the value in the block, use a capture or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">as</code>.
            </p>
          </div>

          <div className="bg-gray-900 rounded-lg overflow-hidden mt-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`key = "quit"
match key:
    case "q" | "quit" | "exit":
        do_exit()
    case "s" | "save":
        do_save()
    case _:
        pass`}
            </pre>
          </div>
        </section>

        {/* AS Patterns */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            AS Patterns
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">as</code> lets you match a pattern <span className="font-semibold">and</span> give the matched value a name: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">pattern as name</code>.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`match item:
    case [2, 4, 6] as triple:
        print(triple)   # [2, 4, 6]
    case (x, y) as pair:
        print(pair)     # the tuple (x, y)`}
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 p-3 mt-4">
            <p className="text-sm text-green-800 dark:text-green-200">
              ✅ Handy when you need the full subject (or a subpattern) in the block, not just the pieces you captured. The name after <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">as</code> is the value that matched the whole pattern.
            </p>
          </div>
        </section>

        {/* Sequence Patterns */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Sequence Patterns
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <span className="font-semibold">Sequence patterns</span> match lists, tuples, or anything else that has length and indexing. You spell out the length you expect and, optionally, subpatterns for each element.
          </p>

          <div className="mb-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Fixed length:</span> A list or tuple of patterns matches a sequence of that length whose elements match.
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`point = [2, 4]
match point:
    case [0, 0]:
        label = "origin"
    case [x, y]:
        label = f"({x}, {y})"   # x=2, y=4
# label is "(2, 4)"`}
              </pre>
            </div>
          </div>

          <div className="mb-4">
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Variable length with *rest:</span> A <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">*</code> in the pattern soaks up the rest into a name. You get exactly one starred part per pattern.
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`items = [2, 4, 6, 8]
match items:
    case [first, *rest]:
        print(first, rest)   # 2 [4, 6, 8]`}
              </pre>
            </div>
          </div>

          <div>
            <p className="text-gray-700 dark:text-gray-300 mb-2">
              <span className="font-semibold">Example – different lengths:</span>
            </p>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
                {`def describe(seq):
    match seq:
        case []:
            return "empty"
        case [a]:
            return f"one: {a}"
        case [a, b]:
            return f"two: {a}, {b}"
        case [a, b, *rest]:
            return f"many: {a}, {b}, rest={rest}"

describe([])           # "empty"
describe([4])           # "one: 4"
describe([2, 6])        # "two: 2, 6"
describe([2, 4, 6, 8]) # "many: 2, 4, rest=[6, 8]"`}
              </pre>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              <span className="font-semibold">Note:</span> Sequence patterns work on <strong>lists</strong>, <strong>tuples</strong>, and the like. They don&apos;t match <strong>strings</strong> or <strong>bytes</strong>: in match, a string isn&apos;t treated as a sequence of characters. For strings, use a literal or a capture.
            </p>
          </div>
        </section>

        {/* Mapping Patterns */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Mapping Patterns
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <span className="font-semibold">Mapping patterns</span> look like dict literals: key–value pairs. They match a mapping (e.g. a dict) that has <span className="font-semibold">at least</span> those keys; the values are matched (or captured) by the value patterns. Extra keys in the subject are fine.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden mb-4">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`config = {"mode": "fast", "level": 2}
match config:
    case {'{"mode": "fast", "level": n}'}:
        print(n)   # 2

data = {"x": 2, "y": 4, "z": 6}
match data:
    case {"x": a, "y": b}:
        total = a + b   # a=2, b=4; "z" ignored
# total is 6`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
  Pattern keys must be literals (strings, numbers, etc.). To grab the whole dict, use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">as</code>: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case {'{"mode": m} as d:'}</code>.
</p>
          </div>
        </section>

        {/* Class Patterns */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Class Patterns
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            <span className="font-semibold">Class patterns</span> match by type and optionally by attributes. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">ClassName(arg1, arg2)</code> matches an instance of that class whose attributes line up with the patterns.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

p = Point(2, 4)
match p:
    case Point(0, 0):
        label = "origin"
    case Point(x, y):
        label = f"({x}, {y})"
# label is "(2, 4)"`}
            </pre>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3 mt-4">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              To match any instance of a type without caring about fields, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case Point():</code> does it; the parentheses are required.
            </p>
          </div>
        </section>

        {/* Guards */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Guards
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
            A <span className="font-semibold">guard</span> is an <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">if</code> on the end of a <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">case</code>. Python tries the pattern first; if it matches, it evaluates the guard. The block runs only when the pattern matches <span className="font-semibold">and</span> the guard is truthy.
          </p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`match value:
    case x if x > 0:
        result = "positive"
    case x if x < 0:
        result = "negative"
    case _:
        result = "zero"

pair = (2, 4)
match pair:
    case (a, b) if a == b:
        kind = "equal"
    case (a, b) if a < b:
        kind = "ascending"
    case (a, b):
        kind = "descending"
# kind is "ascending"`}
            </pre>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 p-3 mt-4">
            <p className="text-sm text-green-800 dark:text-green-200">
              ✅ The guard can use names bound in the same pattern (e.g. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">x</code> in <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case x if x &gt; 0:</code>). If the guard is false, that case is treated as a non-match and the next one is tried.
            </p>
          </div>
        </section>

        {/* Match vs if-elif-else */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Match vs if–elif–else
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">Aspect</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">match–case</th>
                  <th className="text-left p-3 font-semibold text-gray-900 dark:text-white">if–elif–else</th>
                </tr>
              </thead>
              <tbody className="text-gray-700 dark:text-gray-300 text-sm">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">Subject</td>
                  <td className="p-3">One subject; each case is a pattern</td>
                  <td className="p-3">Each branch has its own condition</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">Comparison</td>
                  <td className="p-3">Pattern match (literal, structure)</td>
                  <td className="p-3">Arbitrary Boolean expressions</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">Default</td>
                  <td className="p-3"><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">case _:</code></td>
                  <td className="p-3"><code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">else:</code></td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">Binding</td>
                  <td className="p-3">Capture and <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">as</code> bind names</td>
                  <td className="p-3">No automatic binding</td>
                </tr>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <td className="p-3 font-semibold">Best for</td>
                  <td className="p-3">Many discrete values or structure</td>
                  <td className="p-3">Few branches; complex conditions</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 p-3 mt-4">
            <p className="text-sm text-green-800 dark:text-green-200">
              Reach for <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">match</code> when you&apos;re branching on one value against many options (literals or shapes). Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if</code>–<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">elif</code>–<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">else</code> when you&apos;re mixing variables, ranges, or logic that isn&apos;t structural.
            </p>
          </div>
        </section>

        {/* Combining Patterns */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Combining Patterns
          </h2>
          <p className="text-gray-700 dark:text-gray-300 mb-4">
  You can mix and match: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">case [2, 4] as pair:</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">case {'{"x": 0, "y": 0}'}:</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono">case (a, b) if a &lt; b:</code>. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">_</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">as</code> work inside sequences and mappings too. Only one <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case</code> runs, and order matters: first match wins.
</p>

          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <pre className="p-4 text-green-400 font-mono text-sm overflow-x-auto">
              {`def handle(event):
    match event:
        case {"type": "click", "x": x, "y": y} if x >= 0 and y >= 0:
            return f"click at ({x}, {y})"
        case {"type": "key", "key": k}:
            return f"key: {k}"
        case _:
            return "unknown"`}
            </pre>
          </div>
        </section>

        {/* Tricky Points */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Tricky Points
          </h2>
          <div className="space-y-3">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Basic syntax: one case runs, no fall-through</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Exactly one <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case</code> block runs. No <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">break</code> is needed.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Basic syntax: match is a statement</p>
              <p className="text-sm text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">match</code> doesn&apos;t produce a value. Assign in each <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">return</code> from each branch inside a function.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">When no case matches</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Nothing runs. Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case _:</code> at the end if every subject should be handled.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Literal match: identity vs equality</p>
              <p className="text-sm text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">None</code>, <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">True</code>, and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">False</code> use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">is</code>; other literals use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">==</code>.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Single name is always a capture</p>
              <p className="text-sm text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case x:</code> matches any value and binds it. Use a literal for a specific value.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Wildcard does not bind</p>
              <p className="text-sm text-gray-700 dark:text-gray-300"><code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">_</code> matches anything without binding. Use for default branch.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Sequence pattern and type</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Sequence patterns match <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">list</code> and <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">tuple</code>, not <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">str</code> or <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">bytes</code>.</p>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border-l-4 border-yellow-500 p-3">
              <p className="text-sm font-semibold text-gray-900 dark:text-white mb-1">Order of cases: first match wins</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">Put specific patterns above broad ones. <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case _:</code> last.</p>
            </div>
          </div>
        </section>

        {/* Interview Questions */}
        <section className="mb-10">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4 border-b border-gray-200 dark:border-gray-700 pb-2">
            Interview Questions
          </h2>
          <div className="space-y-5">
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">When should you use match instead of if–elif–else?</p>
              <p className="text-gray-700 dark:text-gray-300">Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">match</code> when you&apos;re branching on <span className="font-semibold">one</span> value against many options (literals or shapes like &quot;list of two elements&quot;). Use <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if</code>–<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">elif</code>–<code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">else</code> when conditions involve different variables, ranges, or non-structural logic.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Does more than one case block run? Is there fall-through?</p>
              <p className="text-gray-700 dark:text-gray-300">No. Exactly one <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case</code> runs. No fall-through (unlike C&apos;s <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">switch</code>); no <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">break</code> needed.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">What does <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case x:</code> mean?</p>
              <p className="text-gray-700 dark:text-gray-300">It matches <span className="font-semibold">any</span> value and binds it to <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">x</code>. It does <span className="font-semibold">not</span> compare to an existing variable.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">How do you match a dict by some keys?</p>
              <p className="text-gray-700 dark:text-gray-300">Use a mapping pattern: <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">case {'{"key1": v1, "key2": v2}'}:</code>. The subject must have at least those keys; extra keys are fine.</p>
            
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">What is a guard and when is it evaluated?</p>
              <p className="text-gray-700 dark:text-gray-300">A guard is <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">if condition</code> after a pattern. It runs only <span className="font-semibold">after</span> the pattern matches. The block runs only when both match and guard are truthy.</p>
            </div>
          </div>
        </section>

        {/* Navigation */}
        <div className="flex justify-between items-center pt-6 border-t border-gray-200 dark:border-gray-700 mt-4">
          <Link
            href="/learn/programming-fundamentals/loops"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            ← Previous: Loops
          </Link>
          <Link
            href="/learn/programming-fundamentals/collections"
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition duration-200"
          >
            Next: Collections →
          </Link>
        </div>
      </div>
    </div>
  );
}