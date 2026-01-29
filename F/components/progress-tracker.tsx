"use client";

import * as React from "react";
import { CheckCircle2, Circle, Trophy } from "lucide-react";
import { cn } from "@/lib/utils";
import { topics } from "@/lib/topics-data";

interface Progress {
  completed: string[];
  quizScores: Record<string, number>;
}

export function useProgress() {
  const [progress, setProgress] = React.useState<Progress>({
    completed: [],
    quizScores: {},
  });

  React.useEffect(() => {
    const stored = localStorage.getItem("cse-progress");
    if (stored) {
      setProgress(JSON.parse(stored));
    }
  }, []);

  const markCompleted = (topicId: string, subtopicSlug: string) => {
    const key = `${topicId}/${subtopicSlug}`;
    if (!progress.completed.includes(key)) {
      const newProgress = {
        ...progress,
        completed: [...progress.completed, key],
      };
      setProgress(newProgress);
      localStorage.setItem("cse-progress", JSON.stringify(newProgress));
    }
  };

  const saveQuizScore = (topicId: string, subtopicSlug: string, score: number) => {
    const key = `${topicId}/${subtopicSlug}`;
    const newProgress = {
      ...progress,
      quizScores: { ...progress.quizScores, [key]: score },
    };
    setProgress(newProgress);
    localStorage.setItem("cse-progress", JSON.stringify(newProgress));
  };

  const isCompleted = (topicId: string, subtopicSlug: string) => {
    return progress.completed.includes(`${topicId}/${subtopicSlug}`);
  };

  const getQuizScore = (topicId: string, subtopicSlug: string) => {
    return progress.quizScores[`${topicId}/${subtopicSlug}`];
  };

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
    markCompleted,
    saveQuizScore,
    isCompleted,
    getQuizScore,
    getTopicProgress,
    getTotalProgress,
  };
}

export function ProgressOverview() {
  const { getTotalProgress, getTopicProgress } = useProgress();
  const totalProgress = getTotalProgress();

  return (
    <div className="p-6 bg-card rounded-lg border border-border">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-lg text-foreground">Your Progress</h3>
        <div className="flex items-center gap-2">
          <Trophy className={cn(
            "h-5 w-5",
            totalProgress === 100 ? "text-warning" : "text-muted-foreground"
          )} />
          <span className="font-bold text-foreground">{totalProgress}%</span>
        </div>
      </div>

      <div className="h-3 bg-muted rounded-full overflow-hidden mb-6">
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
              <div className="h-2 bg-muted rounded-full overflow-hidden">
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
        "flex items-center gap-2 px-3 py-1.5 rounded-full text-sm transition-colors",
        completed
          ? "bg-success/20 text-success"
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
          Mark as Complete
        </>
      )}
    </button>
  );
}
