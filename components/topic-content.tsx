"use client";

import * as React from "react";
import Link from "next/link";
import { ChevronLeft, ChevronRight, Bookmark, BookmarkCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Topic, Subtopic, topics } from "@/lib/topics-data";
import { CompletionBadge } from "@/components/progress-tracker";

// --- Custom Hook for Bookmark Logic ---
function useBookmark(topicId: string, subtopicSlug: string) {
  const [isBookmarked, setIsBookmarked] = React.useState(false);
  const key = `${topicId}/${subtopicSlug}`;

  React.useEffect(() => {
    // Check initial state after mount
    try {
      const bookmarks = JSON.parse(localStorage.getItem("cse-bookmarks") || "[]");
      setIsBookmarked(bookmarks.includes(key));
    } catch (e) {
      console.error("Failed to parse bookmarks", e);
    }
  }, [key]);

  const toggleBookmark = () => {
    try {
      const bookmarks = JSON.parse(localStorage.getItem("cse-bookmarks") || "[]");
      let newBookmarks;

      if (isBookmarked) {
        newBookmarks = bookmarks.filter((b: string) => b !== key);
      } else {
        newBookmarks = [...bookmarks, key];
      }

      localStorage.setItem("cse-bookmarks", JSON.stringify(newBookmarks));
      setIsBookmarked(!isBookmarked);
      
      // Optional: Dispatch a custom event if other components need to know immediately
      // window.dispatchEvent(new Event("bookmark-updated"));
    } catch (e) {
      console.error("Failed to update bookmarks", e);
    }
  };

  return { isBookmarked, toggleBookmark };
}

interface TopicContentProps {
  topic: Topic;
  subtopic: Subtopic;
  children: React.ReactNode;
}

export function TopicContent({ topic, subtopic, children }: TopicContentProps) {
  // Use the custom hook
  const { isBookmarked, toggleBookmark } = useBookmark(topic.id, subtopic.slug);

  // Get previous and next subtopics
  const currentIndex = topic.subtopics.findIndex((s) => s.id === subtopic.id);
  const prevSubtopic = topic.subtopics[currentIndex - 1];
  const nextSubtopic = topic.subtopics[currentIndex + 1];

  // Get next topic if we're at the end of current topic
  const currentTopicIndex = topics.findIndex((t) => t.id === topic.id);
  const nextTopic = topics[currentTopicIndex + 1];

  return (
    <div className="p-6 lg:p-8 max-w-4xl mx-auto"> {/* Added mx-auto for centering */}
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-sm text-muted-foreground mb-6 overflow-x-auto whitespace-nowrap">
        <Link href="/learn" className="hover:text-foreground transition-colors">
          Topics
        </Link>
        <span>/</span>
        <Link
          href={`/learn/${topic.id}/${topic.subtopics[0]?.slug}`}
          className="hover:text-foreground transition-colors"
        >
          {topic.title}
        </Link>
        <span>/</span>
        <span className="text-foreground font-medium">{subtopic.title}</span>
      </nav>

      {/* Header */}
      <div className="flex items-start justify-between gap-4 mb-8">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight text-foreground">{subtopic.title}</h1>
          <div className="flex items-center gap-4 text-sm">
            <p className="text-muted-foreground">
              Part of <span className="font-medium text-foreground">{topic.title}</span>
            </p>
            <CompletionBadge topicId={topic.id} subtopicSlug={subtopic.slug} />
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleBookmark}
          aria-label={isBookmarked ? "Remove bookmark" : "Add bookmark"} // Added Accessibility Label
          className={cn(
            "shrink-0 transition-all duration-200", // Added transition
            isBookmarked && "text-primary hover:text-primary/80"
          )}
        >
          {isBookmarked ? (
            <BookmarkCheck className="h-6 w-6 fill-current" /> // Slightly larger, optional fill
          ) : (
            <Bookmark className="h-6 w-6" />
          )}
        </Button>
      </div>

      {/* Content */}
      <div className="prose prose-slate dark:prose-invert max-w-none mb-12">
        {children}
      </div>

      {/* Navigation Footer */}
      <div className="grid grid-cols-2 gap-4 pt-8 border-t border-border">
        {/* Previous Link */}
        <div>
          {prevSubtopic && (
            <Link
              href={`/learn/${topic.id}/${prevSubtopic.slug}`}
              className="group flex flex-col items-start gap-1 p-4 rounded-lg border border-transparent hover:border-border hover:bg-muted/50 transition-all"
            >
              <div className="flex items-center gap-2 text-muted-foreground text-sm group-hover:text-primary">
                <ChevronLeft className="h-4 w-4 transition-transform group-hover:-translate-x-1" />
                <span>Previous</span>
              </div>
              <div className="font-medium text-foreground">{prevSubtopic.title}</div>
            </Link>
          )}
        </div>

        {/* Next Link */}
        <div className="flex justify-end">
          {nextSubtopic ? (
            <Link
              href={`/learn/${topic.id}/${nextSubtopic.slug}`}
              className="group flex flex-col items-end gap-1 p-4 rounded-lg border border-transparent hover:border-border hover:bg-muted/50 transition-all text-right"
            >
              <div className="flex items-center gap-2 text-muted-foreground text-sm group-hover:text-primary">
                <span>Next</span>
                <ChevronRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
              </div>
              <div className="font-medium text-foreground">{nextSubtopic.title}</div>
            </Link>
          ) : nextTopic ? (
            <Link
              href={`/learn/${nextTopic.id}/${nextTopic.subtopics[0]?.slug}`}
              className="group flex flex-col items-end gap-1 p-4 rounded-lg border border-primary/20 bg-primary/5 hover:bg-primary/10 transition-all text-right"
            >
              <div className="flex items-center gap-2 text-primary text-sm font-semibold">
                <span>Next Topic</span>
                <ChevronRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
              </div>
              <div className="font-medium text-foreground">{nextTopic.title}</div>
            </Link>
          ) : null}
        </div>
      </div>
    </div>
  );
}