import Link from "next/link";
import {
  Layers,
  Cpu,
  Code2,
  Box,
  Database,
  Globe,
  Layout,
  Settings,
  Shield,
  Brain,
  ArrowRight,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Header } from "@/components/layout/header";
import { topics } from "@/lib/topics-data";

const iconMap: Record<string, React.ReactNode> = {
  layers: <Layers className="h-5 w-5" />,
  cpu: <Cpu className="h-5 w-5" />,
  code: <Code2 className="h-5 w-5" />,
  "code-2": <Code2 className="h-5 w-5" />,
  box: <Box className="h-5 w-5" />,
  database: <Database className="h-5 w-5" />,
  globe: <Globe className="h-5 w-5" />,
  layout: <Layout className="h-5 w-5" />,
  settings: <Settings className="h-5 w-5" />,
  shield: <Shield className="h-5 w-5" />,
  brain: <Brain className="h-5 w-5" />,
};

const totalLessons = topics.reduce((acc, t) => acc + t.subtopics.length, 0);

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main>
        {/* ============================================================
            HERO — left-aligned copy, curriculum graph as the visual
        ============================================================ */}
        <section className="relative overflow-hidden border-b border-border">
          <div className="grid-fade pointer-events-none absolute inset-0" />

          <div className="relative mx-auto grid max-w-6xl grid-cols-1 gap-12 px-4 py-16 sm:px-6 md:py-24 lg:grid-cols-[1fr_1fr] lg:items-center lg:px-8">
            <div>
              <p className="meta-label mb-5">amni / curriculum</p>

              <h1 className="max-w-xl font-[family-name:var(--font-space-grotesk)] text-4xl font-semibold leading-[1.1] text-foreground md:text-5xl">
                Learn computer science the way engineers actually think about it.
              </h1>

              <p className="mt-6 max-w-md text-base leading-relaxed text-muted-foreground">
                Structured lessons, working code in four languages, and visual
                explanations for the ideas that are usually just hand-waved —
                from arrays to distributed systems.
              </p>

              <div className="mt-8 flex flex-col gap-3 sm:flex-row">
                <Link href="/learn/data-structures/arrays">
                  <Button size="lg" className="w-full sm:w-auto">
                    Start with arrays
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
                <Link href="/learn">
                  <Button size="lg" variant="outline" className="w-full sm:w-auto">
                    Browse the curriculum
                  </Button>
                </Link>
              </div>

              <div className="meta-label mt-10 flex flex-wrap gap-x-6 gap-y-2 border-t border-border pt-6">
                <span>{topics.length} modules</span>
                <span>{totalLessons}+ lessons</span>
                <span>4 languages per example</span>
              </div>
            </div>

            <CurriculumGraph />
          </div>
        </section>

        {/* ============================================================
            APPROACH — three principles, not generic feature cards
        ============================================================ */}
        <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 gap-10 md:grid-cols-3">
            <Principle
              index="01"
              title="Explained, then shown"
              body="Every concept gets a plain-language explanation first, then a visualization you can step through — not just a wall of text."
            />
            <Principle
              index="02"
              title="Code in context"
              body="Examples run in Python, JavaScript, Java, and C++ side by side, so you see the idea, not just one language's syntax for it."
            />
            <Principle
              index="03"
              title="Built for recall"
              body="Short quizzes and a Blind 75 track turn lessons into something you can actually retrieve under interview pressure."
            />
          </div>
        </section>

        {/* ============================================================
            CURRICULUM — genuinely sequential, so numbering is honest
        ============================================================ */}
        <section className="border-t border-border bg-muted/30">
          <div className="mx-auto max-w-6xl px-4 py-16 sm:px-6 lg:px-8">
            <div className="mb-10 flex items-end justify-between gap-4">
              <div>
                <p className="meta-label mb-2">the roadmap</p>
                <h2 className="font-[family-name:var(--font-space-grotesk)] text-2xl font-semibold text-foreground md:text-3xl">
                  Twelve modules, built to stack
                </h2>
              </div>
            </div>

            <div className="divide-y divide-border border-y border-border">
              {topics.map((topic, i) => (
                <Link
                  key={topic.id}
                  href={`/learn/${topic.id}/${topic.subtopics[0]?.slug}`}
                  className="group flex items-center gap-5 py-5 transition-colors hover:bg-card"
                >
                  <span className="meta-label w-8 shrink-0 text-right">
                    {String(i + 1).padStart(2, "0")}
                  </span>

                  <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md border border-border bg-card text-primary">
                    {iconMap[topic.icon] || <Layers className="h-5 w-5" />}
                  </span>

                  <span className="min-w-0 flex-1">
                    <span className="block font-medium text-foreground transition-colors group-hover:text-primary">
                      {topic.title}
                    </span>
                    <span className="block truncate text-sm text-muted-foreground">
                      {topic.description}
                    </span>
                  </span>

                  <span className="meta-label hidden shrink-0 sm:block">
                    {topic.subtopics.length} lessons
                  </span>

                  <ArrowRight className="h-4 w-4 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-primary" />
                </Link>
              ))}
            </div>
          </div>
        </section>

        {/* ============================================================
            CTA
        ============================================================ */}
        <section className="mx-auto max-w-6xl px-4 py-20 sm:px-6 lg:px-8">
          <div className="flex flex-col items-start justify-between gap-6 rounded-lg border border-border bg-card p-8 md:flex-row md:items-center md:p-10">
            <div>
              <h2 className="font-[family-name:var(--font-space-grotesk)] text-xl font-semibold text-foreground md:text-2xl">
                Create a free account to keep your progress
              </h2>
              <p className="mt-2 max-w-md text-sm text-muted-foreground">
                Completed lessons, quiz scores, and bookmarks sync across
                every device you sign in on.
              </p>
            </div>
            <Link href="/sign-up" className="shrink-0">
              <Button size="lg">
                Sign up free
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </div>
        </section>
      </main>
    </div>
  );
}

function Principle({ index, title, body }: { index: string; title: string; body: string }) {
  return (
    <div className="border-t border-border pt-5">
      <span className="meta-label">{index}</span>
      <h3 className="mt-3 font-medium text-foreground">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{body}</p>
    </div>
  );
}

/**
 * Hand-laid-out schematic of how the curriculum's core data structures
 * connect — a genuine dependency graph, not decoration. Doubles as a nod
 * to the graph/tree visualizers used throughout the lessons.
 */
function CurriculumGraph() {
  const rawNodes = [
    { id: "arrays", label: "Arrays", x: 32, y: 40 },
    { id: "lists", label: "Linked Lists", x: 32, y: 150 },
    { id: "stacks", label: "Stacks & Queues", x: 32, y: 260 },
    { id: "trees", label: "Trees", x: 262, y: 95 },
    { id: "graphs", label: "Graphs", x: 262, y: 220 },
    { id: "interview", label: "Interview Ready", x: 420, y: 155 },
  ];

  // Size each box to its label so nothing clips — mono-space glyphs run
  // ~6.4px wide at this font size, plus room for the leading dot and padding.
  const nodes = rawNodes.map((n) => ({
    ...n,
    width: Math.round(n.label.length * 6.4 + 40),
  }));

  const edges: [string, string][] = [
    ["arrays", "trees"],
    ["lists", "trees"],
    ["lists", "graphs"],
    ["stacks", "graphs"],
    ["trees", "interview"],
    ["graphs", "interview"],
  ];

  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const viewWidth = Math.max(...nodes.map((n) => n.x + n.width)) + 20;

  return (
    <div className="relative hidden lg:block">
      <svg
        viewBox={`0 0 ${viewWidth} 320`}
        className="h-auto w-full"
        role="img"
        aria-label="Diagram of how data structure topics build toward interview readiness"
      >
        {edges.map(([from, to], i) => {
          const a = byId[from];
          const b = byId[to];
          return (
            <line
              key={i}
              x1={a.x + a.width}
              y1={a.y + 16}
              x2={b.x}
              y2={b.y + 16}
              stroke="var(--border)"
              strokeWidth="1.5"
            />
          );
        })}

        {nodes.map((n) => (
          <g key={n.id} transform={`translate(${n.x}, ${n.y})`}>
            <rect
              width={n.width}
              height="32"
              rx="4"
              fill="var(--card)"
              stroke={n.id === "interview" ? "var(--primary)" : "var(--border)"}
              strokeWidth={n.id === "interview" ? 1.5 : 1}
            />
            <circle cx="12" cy="16" r="3" fill={n.id === "interview" ? "var(--primary)" : "var(--muted-foreground)"} />
            <text
              x="24"
              y="20"
              fontFamily="var(--font-mono)"
              fontSize="10.5"
              fill="var(--foreground)"
            >
              {n.label}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
