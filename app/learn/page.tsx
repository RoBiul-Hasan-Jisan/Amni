"use client";

import React from "react";
import Link from "next/link";
import {
  Layers,
  Cpu,
  Code,
  Box,
  Database,
  Globe,
  Layout,
  Settings,
  Shield,
  Brain,
  ArrowRight,
} from "lucide-react";
import { topics } from "@/lib/topics-data";
import { ProgressOverview } from "@/components/progress-tracker";

const iconMap: Record<string, React.ReactNode> = {
  layers: <Layers className="h-6 w-6" />,
  cpu: <Cpu className="h-6 w-6" />,
  code: <Code className="h-6 w-6" />,
  box: <Box className="h-6 w-6" />,
  database: <Database className="h-6 w-6" />,
  globe: <Globe className="h-6 w-6" />,
  layout: <Layout className="h-6 w-6" />,
  settings: <Settings className="h-6 w-6" />,
  shield: <Shield className="h-6 w-6" />,
  brain: <Brain className="h-6 w-6" />,
};

export default function LearnPage() {
  return (
    <div className="p-6 lg:p-8 max-w-5xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2 text-foreground">All Topics</h1>
        <p className="text-muted-foreground">
          Choose a topic to start learning. Each topic includes explanations,
          visualizations, code examples, and practice questions.
        </p>
      </div>

      <div className="mb-8">
        <ProgressOverview />
      </div>

      <h2 className="text-xl font-semibold mb-4 text-foreground">Browse Topics</h2>
      <div className="grid gap-4">
        {topics.map((topic) => (
          <Link
            key={topic.id}
            href={`/learn/${topic.id}/${topic.subtopics[0]?.slug}`}
            className="group flex items-center gap-4 p-4 bg-card rounded-lg border border-border hover:border-primary hover:shadow-md transition-all"
          >
            <div className="p-3 rounded-lg bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors shrink-0">
              {iconMap[topic.icon] || <Layers className="h-6 w-6" />}
            </div>
            <div className="flex-1 min-w-0">
              <h2 className="font-semibold text-lg text-foreground group-hover:text-primary transition-colors">
                {topic.title}
              </h2>
              <p className="text-sm text-muted-foreground line-clamp-1">
                {topic.description}
              </p>
              <div className="flex flex-wrap gap-2 mt-2">
                {topic.subtopics.slice(0, 4).map((subtopic) => (
                  <span
                    key={subtopic.id}
                    className="text-xs px-2 py-1 bg-muted rounded text-muted-foreground"
                  >
                    {subtopic.title}
                  </span>
                ))}
                {topic.subtopics.length > 4 && (
                  <span className="text-xs px-2 py-1 bg-muted rounded text-muted-foreground">
                    +{topic.subtopics.length - 4} more
                  </span>
                )}
              </div>
            </div>
            <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-primary transition-colors shrink-0" />
          </Link>
        ))}
      </div>
    </div>
  );
}
