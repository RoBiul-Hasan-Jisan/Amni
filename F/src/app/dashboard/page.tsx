import { redirect } from "next/navigation";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import type { QuizAttempt } from "@prisma/client";

import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { Header } from "@/components/layout/header";
import { ProgressOverview } from "@/components/learn/progress-tracker";
import { SignOutButton } from "@/components/layout/sign-out-button";
import { topics } from "@/lib/topics-data";

export const metadata = { title: "Your dashboard" };

export default async function DashboardPage() {
  const session = await auth();
  if (!session?.user) {
    redirect("/sign-in?callbackUrl=/dashboard");
  }

  const [completedCount, recentAttempts] = await Promise.all([
    prisma.progress.count({ where: { userId: session.user.id, completed: true } }),
    prisma.quizAttempt.findMany({
      where: { userId: session.user.id },
      orderBy: { createdAt: "desc" },
      take: 5,
    }),
  ]);

  const totalLessons = topics.reduce((acc, t) => acc + t.subtopics.length, 0);

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="mx-auto max-w-4xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="meta-label mb-1">signed in as {session.user.email}</p>
            <h1 className="font-[family-name:var(--font-space-grotesk)] text-3xl font-semibold text-foreground">
              Welcome back{session.user.name ? `, ${session.user.name.split(" ")[0]}` : ""}
            </h1>
          </div>
          <SignOutButton />
        </div>

        <div className="mb-10 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Lessons done" value={`${completedCount}/${totalLessons}`} />
          <Stat label="Modules" value={String(topics.length)} />
          <Stat label="Quiz attempts" value={String(recentAttempts.length > 0 ? recentAttempts.length : 0)} />
          <Stat
            label="Best recent score"
            value={
              recentAttempts.length
                ? `${Math.round(
                    Math.max(...recentAttempts.map((a: QuizAttempt) => (a.score / a.total) * 100))
                  )}%`
                : "—"
            }
          />
        </div>

        <ProgressOverview />

        {recentAttempts.length > 0 && (
          <div className="mt-8 rounded-lg border border-border bg-card p-6">
            <h2 className="mb-4 text-lg font-semibold text-foreground">Recent quiz attempts</h2>
            <ul className="divide-y divide-border">
              {recentAttempts.map((attempt: QuizAttempt) => (
                <li key={attempt.id} className="flex items-center justify-between py-3 text-sm">
                  <span className="text-foreground">
                    {attempt.topicId} / {attempt.subtopicId}
                  </span>
                  <span className="meta-label">
                    {Math.round((attempt.score / attempt.total) * 100)}%
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <Link
          href="/learn"
          className="mt-8 inline-flex items-center gap-2 text-sm font-medium text-primary hover:underline"
        >
          Continue browsing the curriculum
          <ArrowRight className="h-4 w-4" />
        </Link>
      </main>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="font-[family-name:var(--font-space-grotesk)] text-2xl font-semibold text-foreground">
        {value}
      </div>
      <div className="mt-1 text-xs text-muted-foreground">{label}</div>
    </div>
  );
}
