import Link from "next/link";
import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { AlertCircle, CheckCircle2, Lightbulb } from "lucide-react";

export default function MatchPatternPage() {
  const result = getSubtopicBySlug("programming-fundamentals", "match-pattern");
  if (!result) return null;

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        {/* Header */}
        <div className="mb-8 border-b border-border pb-4">
          <h1 className="text-4xl font-bold text-foreground mb-2">
            Match (Structural Pattern Matching)
          </h1>
          <p className="text-muted-foreground text-lg">
            Structural pattern matching picks one branch by matching a value against patterns. Python&apos;s <code className="bg-muted px-1.5 py-0.5 rounded font-mono">match</code> (3.10+) often replaces long <code className="bg-muted px-1.5 py-0.5 rounded font-mono">if</code>–<code className="bg-muted px-1.5 py-0.5 rounded font-mono">elif</code>–<code className="bg-muted px-1.5 py-0.5 rounded font-mono">else</code> chains when you&apos;re branching on &quot;one of these exact values&quot; or on the <span className="font-semibold text-foreground">shape</span> of the value (e.g. how long a list is or which keys a dict has).
          </p>
        </div>

        {/* Why match? */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Why match?
          </h2>
          <p className="text-muted-foreground mb-4">
            When every branch depends on the <span className="font-semibold text-foreground">same</span> value (a status code, a command name, the shape of some structure), <code className="bg-muted px-1.5 py-0.5 rounded font-mono">if</code>–<code className="bg-muted px-1.5 py-0.5 rounded font-mono">elif</code>–<code className="bg-muted px-1.5 py-0.5 rounded font-mono">else</code> works but gets noisy. <code className="bg-muted px-1.5 py-0.5 rounded font-mono">match</code>–<code className="bg-muted px-1.5 py-0.5 rounded font-mono">case</code> puts the value in one place and turns each branch into a <span className="font-semibold text-foreground">pattern</span>: it either matches or it doesn&apos;t. The first <code className="bg-muted px-1.5 py-0.5 rounded font-mono">case</code> that matches runs; nothing falls through to the next. Match really shines when you care about more than equality (e.g. &quot;a list of two elements&quot; or &quot;a dict with key <code className="bg-muted px-1.5 py-0.5 rounded font-mono">'x'</code>&quot;).
          </p>
        </section>

        {/* Basic Syntax */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Basic Syntax
          </h2>
          <p className="text-muted-foreground mb-4">
            You write <code className="bg-muted px-1.5 py-0.5 rounded font-mono">match</code> followed by a <span className="font-semibold text-foreground">subject</span> (the value you&apos;re inspecting), then one or more <code className="bg-muted px-1.5 py-0.5 rounded font-mono">case</code> blocks. Each <code className="bg-muted px-1.5 py-0.5 rounded font-mono">case</code> has a <span className="font-semibold text-foreground">pattern</span> and, optionally, a <span className="font-semibold text-foreground">guard</span>. Python tries the patterns in order; the first one that matches runs, and the rest are skipped.
          </p>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`match subject:
    case pattern_1:
        block_1
    case pattern_2:
        block_2
    case _:
        block_default`}
            </pre>
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mb-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                Colons after <code className="bg-muted px-1.5 py-0.5 rounded">match</code> and after each <code className="bg-muted px-1.5 py-0.5 rounded">case</code> are required. The block under a <code className="bg-muted px-1.5 py-0.5 rounded">case</code> is whatever&apos;s indented under it. The <code className="bg-muted px-1.5 py-0.5 rounded">_</code> in the last case is a <span className="font-semibold">wildcard</span>: it matches anything and is commonly used as the default.
              </p>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
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

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                ✅ Exactly one <code className="bg-muted px-1.5 py-0.5 rounded">case</code> runs. When it&apos;s done, execution continues after the whole <code className="bg-muted px-1.5 py-0.5 rounded">match</code>; no fall-through.
              </p>
            </div>
          </div>
        </section>

        {/* Literal Patterns */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Literal Patterns
          </h2>
          <p className="text-muted-foreground mb-4">
            Put a number, string, <code className="bg-muted px-1.5 py-0.5 rounded font-mono">None</code>, <code className="bg-muted px-1.5 py-0.5 rounded font-mono">True</code>, or <code className="bg-muted px-1.5 py-0.5 rounded font-mono">False</code> right in the <code className="bg-muted px-1.5 py-0.5 rounded font-mono">case</code> line and you get a <span className="font-semibold text-foreground">literal pattern</span>. The subject is compared to that value.
          </p>

          <div className="overflow-x-auto mb-4">
            <table className="min-w-full bg-card border border-border rounded-lg">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-semibold text-foreground">Literal type</th>
                  <th className="text-left p-3 font-semibold text-foreground">Example in case</th>
                  <th className="text-left p-3 font-semibold text-foreground">Matches when</th>
                </tr>
              </thead>
              <tbody className="text-muted-foreground text-sm">
                <tr className="border-b border-border"><td className="p-3">Integer</td><td className="p-3 font-mono">case 4:</td><td className="p-3">Subject equals 4</td></tr>
                <tr className="border-b border-border"><td className="p-3">Float</td><td className="p-3 font-mono">case 2.0:</td><td className="p-3">Subject equals 2.0</td></tr>
                <tr className="border-b border-border"><td className="p-3">String</td><td className="p-3 font-mono">case "ok":</td><td className="p-3">Subject equals "ok"</td></tr>
                <tr className="border-b border-border"><td className="p-3">None</td><td className="p-3 font-mono">case None:</td><td className="p-3">Subject is None (identity)</td></tr>
                <tr className="border-b border-border"><td className="p-3">True</td><td className="p-3 font-mono">case True:</td><td className="p-3">Subject is True (identity)</td></tr>
                <tr className="border-b border-border"><td className="p-3">False</td><td className="p-3 font-mono">case False:</td><td className="p-3">Subject is False (identity)</td></tr>
              </tbody>
            </table>
          </div>

          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3 mb-4">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                <span className="font-semibold">Important:</span> For <code className="bg-muted px-1.5 py-0.5 rounded">None</code>, <code className="bg-muted px-1.5 py-0.5 rounded">True</code>, and <code className="bg-muted px-1.5 py-0.5 rounded">False</code>, the pattern uses <span className="font-semibold">identity</span> (<code className="bg-muted px-1.5 py-0.5 rounded">is</code>), not <code className="bg-muted px-1.5 py-0.5 rounded">==</code>. For everything else (numbers, strings), it uses <span className="font-semibold">equality</span> (<code className="bg-muted px-1.5 py-0.5 rounded">==</code>).
              </p>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
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
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Capture Patterns
          </h2>
          <p className="text-muted-foreground mb-4">
            A bare <span className="font-semibold text-foreground">name</span> in a pattern is a <span className="font-semibold text-foreground">capture</span>: it matches any value and <span className="font-semibold text-foreground">binds</span> it to that name so you can use it in the block. It does <span className="font-semibold text-foreground">not</span> compare the subject to some existing variable.
          </p>

          <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3 mb-4">
            <div className="flex gap-3">
              <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                So <code className="bg-muted px-1.5 py-0.5 rounded">case x:</code> means &quot;match anything and call it <code className="bg-muted px-1.5 py-0.5 rounded">x</code>.&quot; It does <span className="font-semibold">not</span> mean &quot;match when the subject equals the variable <code className="bg-muted px-1.5 py-0.5 rounded">x</code>.&quot; To match a specific value, use a literal: <code className="bg-muted px-1.5 py-0.5 rounded">case 6:</code>.
              </p>
            </div>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`data = [2, 4, 6]
match data:
    case [a, b, c]:
        total = a + b + c   # a=2, b=4, c=6
# total is 12`}
            </pre>
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                ✅ Those names only exist inside that <code className="bg-muted px-1.5 py-0.5 rounded">case</code> block, not in other cases or after the <code className="bg-muted px-1.5 py-0.5 rounded">match</code>.
              </p>
            </div>
          </div>
        </section>

        {/* Wildcard Pattern */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Wildcard Pattern
          </h2>
          <p className="text-muted-foreground mb-4">
            <code className="bg-muted px-1.5 py-0.5 rounded font-mono">_</code> matches any value and doesn&apos;t bind it. Use it when you don&apos;t care what the value is (e.g. for a default or &quot;anything else&quot;).
          </p>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
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
        </section>

        {/* OR Patterns */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            OR Patterns
          </h2>
          <p className="text-muted-foreground mb-4">
            <code className="bg-muted px-1.5 py-0.5 rounded font-mono">|</code> lets you combine several patterns in one <code className="bg-muted px-1.5 py-0.5 rounded font-mono">case</code>. The case runs if <span className="font-semibold text-foreground">any</span> of them matches.
          </p>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border mb-4">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
              {`match n:
    case 2 | 4 | 6:
        kind = "even_small"
    case 8 | 10:
        kind = "even_large"
    case _:
        kind = "other"`}
            </pre>
          </div>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
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

        {/* Guards */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Guards
          </h2>
          <p className="text-muted-foreground mb-4">
            A <span className="font-semibold text-foreground">guard</span> is an <code className="bg-muted px-1.5 py-0.5 rounded font-mono">if</code> on the end of a <code className="bg-muted px-1.5 py-0.5 rounded font-mono">case</code>. Python tries the pattern first; if it matches, it evaluates the guard. The block runs only when the pattern matches <span className="font-semibold text-foreground">and</span> the guard is truthy.
          </p>

          <div className="bg-muted/50 rounded-lg overflow-hidden border border-border">
            <pre className="p-4 text-primary font-mono text-sm overflow-x-auto">
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

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                ✅ The guard can use names bound in the same pattern. If the guard is false, that case is treated as a non-match and the next one is tried.
              </p>
            </div>
          </div>
        </section>

        {/* Match vs if-elif-else */}
        <section>
          <h2 className="text-2xl font-bold text-foreground mb-4 border-b border-border pb-2">
            Match vs if–elif–else
          </h2>
         <div className="overflow-x-auto">
  <table className="min-w-full bg-card border border-border rounded-lg">
    <thead>
      <tr className="border-b border-border">
        <th className="text-left p-3 font-semibold text-foreground">
          Aspect
        </th>
        <th className="text-left p-3 font-semibold text-foreground">
          match–case
        </th>
        <th className="text-left p-3 font-semibold text-foreground">
          if–elif–else
        </th>
      </tr>
    </thead>

    <tbody className="text-muted-foreground text-sm">
      <tr className="border-b border-border">
        <td className="p-3 font-semibold text-foreground">Subject</td>
        <td className="p-3">
          One subject; each case is a pattern
        </td>
        <td className="p-3">
          Each branch has its own condition
        </td>
      </tr>

      <tr className="border-b border-border">
        <td className="p-3 font-semibold text-foreground">Comparison</td>
        <td className="p-3">
          Pattern match (literal, structure)
        </td>
        <td className="p-3">
          Arbitrary Boolean expressions
        </td>
      </tr>

      <tr className="border-b border-border">
        <td className="p-3 font-semibold text-foreground">Default</td>
        <td className="p-3">
          <code className="bg-muted px-1.5 py-0.5 rounded">
            case _:
          </code>
        </td>
        <td className="p-3">
          <code className="bg-muted px-1.5 py-0.5 rounded">
            else:
          </code>
        </td>
      </tr>

      <tr className="border-b border-border">
        <td className="p-3 font-semibold text-foreground">Binding</td>
        <td className="p-3">
          Capture and{" "}
          <code className="bg-muted px-1.5 py-0.5 rounded">
            as
          </code>{" "}
          bind names
        </td>
        <td className="p-3">No automatic binding</td>
      </tr>

      <tr className="border-b border-border">
        <td className="p-3 font-semibold text-foreground">Best for</td>
        <td className="p-3">
          Many discrete values or structure
        </td>
        <td className="p-3">
          Few branches; complex conditions
        </td>
      </tr>
    </tbody>
  </table>
</div>

          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 mt-4">
            <div className="flex gap-3">
              <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <p className="text-sm text-foreground">
                Reach for <code className="bg-muted px-1.5 py-0.5 rounded">match</code> when you&apos;re branching on one value against many options (literals or shapes). Use <code className="bg-muted px-1.5 py-0.5 rounded">if</code>–<code className="bg-muted px-1.5 py-0.5 rounded">elif</code>–<code className="bg-muted px-1.5 py-0.5 rounded">else</code> when you&apos;re mixing variables, ranges, or logic that isn&apos;t structural.
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
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">Basic syntax: one case runs, no fall-through</p>
                  <p className="text-sm text-muted-foreground">Exactly one <code className="bg-muted px-1.5 py-0.5 rounded">case</code> block runs. No <code className="bg-muted px-1.5 py-0.5 rounded">break</code> is needed.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">Single name is always a capture</p>
                  <p className="text-sm text-muted-foreground"><code className="bg-muted px-1.5 py-0.5 rounded">case x:</code> matches any value and binds it. Use a literal for a specific value.</p>
                </div>
              </div>
            </div>
            <div className="bg-destructive/5 border border-destructive/20 rounded-lg p-3">
              <div className="flex gap-3">
                <AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-foreground mb-1">Order of cases: first match wins</p>
                  <p className="text-sm text-muted-foreground">Put specific patterns above broad ones. <code className="bg-muted px-1.5 py-0.5 rounded">case _:</code> last.</p>
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
                  <p className="font-semibold text-foreground mb-1">When should you use match instead of if–elif–else?</p>
                  <p className="text-muted-foreground">Use <code className="bg-muted px-1.5 py-0.5 rounded">match</code> when you&apos;re branching on <span className="font-semibold">one</span> value against many options (literals or shapes like &quot;list of two elements&quot;).</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">What does <code className="bg-muted px-1.5 py-0.5 rounded">case x:</code> mean?</p>
                  <p className="text-muted-foreground">It matches <span className="font-semibold">any</span> value and binds it to <code className="bg-muted px-1.5 py-0.5 rounded">x</code>. It does <span className="font-semibold">not</span> compare to an existing variable.</p>
                </div>
              </div>
            </div>
            <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
              <div className="flex gap-3">
                <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-foreground mb-1">What is a guard and when is it evaluated?</p>
                  <p className="text-muted-foreground">A guard is <code className="bg-muted px-1.5 py-0.5 rounded">if condition</code> after a pattern. It runs only <span className="font-semibold">after</span> the pattern matches. The block runs only when both match and guard are truthy.</p>
                </div>
              </div>
            </div>
          </div>
        </section>

       
      </div>
    </TopicContent>
  );
}