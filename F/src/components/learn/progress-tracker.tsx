"use client";

import * as React from "react";
import { useSession } from "next-auth/react";
import { CheckCircle2, Circle, Trophy, LogIn } from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { topics } from "@/lib/topics-data";

interface Progress {
  completed: string[];
  quizScores: Record<string, number>;
}

const STORAGE_KEY = "amni-progress";

function readLocalProgress(): Progress {
  if (typeof window === "undefined") return { completed: [], quizScores: {} };
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : { completed: [], quizScores: {} };
  } catch {
    return { completed: [], quizScores: {} };
  }
}

/**
 * Tracks lesson completion + quiz scores.
 *
 * Signed-in users: reads/writes through /api/progress and /api/quiz so
 * progress follows them across devices. Guests: falls back to
 * localStorage, so the site is still fully usable without an account.
 */
export function useProgress() {
  const { data: session, status } = useSession();
  const isAuthed = status === "authenticated" && !!session?.user;

  const [progress, setProgress] = React.useState<Progress>({ completed: [], quizScores: {} });
  const [loaded, setLoaded] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;

    async function hydrate() {
      if (isAuthed) {
        try {
          const [progressRes, quizRes] = await Promise.all([
            fetch("/api/progress"),
            fetch("/api/quiz"),
          ]);
          const progressData = await progressRes.json();
          const quizData = await quizRes.json();
          if (cancelled) return;

          const completed: string[] = (progressData.progress ?? []).map(
            (p: { topicId: string; subtopicId: string }) => `${p.topicId}/${p.subtopicId}`
          );

          const quizScores: Record<string, number> = {};
          for (const attempt of quizData.attempts ?? []) {
            const key = `${attempt.topicId}/${attempt.subtopicId}`;
            // attempts are ordered most-recent-first; keep the first (latest) per lesson
            if (!(key in quizScores)) {
              quizScores[key] = Math.round((attempt.score / attempt.total) * 100);
            }
          }

          setProgress({ completed, quizScores });
        } catch {
          // network hiccup — fall through with empty state rather than crash
        }
      } else if (status !== "loading") {
        setProgress(readLocalProgress());
      }
      if (!cancelled) setLoaded(true);
    }

    hydrate();
    return () => {
      cancelled = true;
    };
  }, [isAuthed, status]);

  const persistLocal = (next: Progress) => {
    setProgress(next);
    if (typeof window !== "undefined") {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    }
  };

  const markCompleted = (topicId: string, subtopicSlug: string) => {
    const key = `${topicId}/${subtopicSlug}`;
    if (progress.completed.includes(key)) return;

    const next = { ...progress, completed: [...progress.completed, key] };
    setProgress(next);

    if (isAuthed) {
      fetch("/api/progress", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topicId, subtopicId: subtopicSlug, completed: true }),
      }).catch(() => {});
    } else {
      persistLocal(next);
    }
  };

  const saveQuizScore = (topicId: string, subtopicSlug: string, scorePercent: number) => {
    const key = `${topicId}/${subtopicSlug}`;
    const next = { ...progress, quizScores: { ...progress.quizScores, [key]: scorePercent } };
    setProgress(next);

    if (isAuthed) {
      // scorePercent is 0-100; store it as score/total = percent/100 for a stable ratio
      fetch("/api/quiz", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topicId,
          subtopicId: subtopicSlug,
          score: scorePercent,
          total: 100,
        }),
      }).catch(() => {});
    } else {
      persistLocal(next);
    }
  };

  const isCompleted = (topicId: string, subtopicSlug: string) =>
    progress.completed.includes(`${topicId}/${subtopicSlug}`);

  const getQuizScore = (topicId: string, subtopicSlug: string) =>
    progress.quizScores[`${topicId}/${subtopicSlug}`];

  const getTopicProgress = (topicId: string) => {
    const topic = topics.find((t) => t.id === topicId);
    if (!topic) return 0;
    const completed = topic.subtopics.filter((s) =>
      progress.completed.includes(`${topicId}/${s.slug}`)
    ).length;
    return Math.round((completed / topic.subtopics.length) * 100);
  };

  const getTotalProgress = () => {
    const totalSubtopics = topics.reduce((acc, t) => acc + t.subtopics.length, 0);
    return Math.round((progress.completed.length / totalSubtopics) * 100);
  };

  return {
    progress,
    loaded,
    isAuthed,
    markCompleted,
    saveQuizScore,
    isCompleted,
    getQuizScore,
    getTopicProgress,
    getTotalProgress,
  };
}

export function ProgressOverview() {
  const { getTotalProgress, getTopicProgress, isAuthed } = useProgress();
  const totalProgress = getTotalProgress();

  return (
    <div className="rounded-lg border border-border bg-card p-6">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-foreground">Your progress</h3>
        <div className="flex items-center gap-2">
          <Trophy
            className={cn("h-5 w-5", totalProgress === 100 ? "text-warning" : "text-muted-foreground")}
          />
          <span className="font-bold text-foreground">{totalProgress}%</span>
        </div>
      </div>

      {!isAuthed && (
        <Link
          href="/sign-in"
          className="mb-6 flex items-center gap-2 rounded-md border border-dashed border-border px-3 py-2 text-sm text-muted-foreground transition-colors hover:border-primary hover:text-primary"
        >
          <LogIn className="h-4 w-4" />
          Sign in to save progress across devices — currently stored on this device only.
        </Link>
      )}

      <div className="mb-6 h-3 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full bg-primary transition-all duration-500"
          style={{ width: `${totalProgress}%` }}
        />
      </div>

      <div className="space-y-4">
        {topics.map((topic) => {
          const topicProgress = getTopicProgress(topic.id);
          return (
            <div key={topic.id} className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-foreground">{topic.title}</span>
                <span className="text-muted-foreground">{topicProgress}%</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div
                  className={cn(
                    "h-full transition-all duration-500",
                    topicProgress === 100 ? "bg-success" : "bg-primary/70"
                  )}
                  style={{ width: `${topicProgress}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function CompletionBadge({
  topicId,
  subtopicSlug,
}: {
  topicId: string;
  subtopicSlug: string;
}) {
  const { isCompleted, markCompleted } = useProgress();
  const completed = isCompleted(topicId, subtopicSlug);

  return (
    <button
      onClick={() => !completed && markCompleted(topicId, subtopicSlug)}
      className={cn(
        "flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition-colors",
        completed
          ? "bg-success/15 text-success"
          : "bg-muted text-muted-foreground hover:bg-muted/80"
      )}
    >
      {completed ? (
        <>
          <CheckCircle2 className="h-4 w-4" />
          Completed
        </>
      ) : (
        <>
          <Circle className="h-4 w-4" />
          Mark as complete
        </>
      )}
    </button>
  );
}
