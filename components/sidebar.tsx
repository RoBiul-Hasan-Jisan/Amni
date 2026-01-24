"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ChevronDown,
  ChevronRight,
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
} from "lucide-react";
import { cn } from "@/lib/utils";
import { topics, Topic } from "@/lib/topics-data";
import { Button } from "@/components/ui/button";

const iconMap: Record<string, React.ReactNode> = {
  layers: <Layers className="h-4 w-4" />,
  cpu: <Cpu className="h-4 w-4" />,
  code: <Code className="h-4 w-4" />,
  box: <Box className="h-4 w-4" />,
  database: <Database className="h-4 w-4" />,
  globe: <Globe className="h-4 w-4" />,
  layout: <Layout className="h-4 w-4" />,
  settings: <Settings className="h-4 w-4" />,
  shield: <Shield className="h-4 w-4" />,
  brain: <Brain className="h-4 w-4" />,
};

interface SidebarProps {
  className?: string;
  onNavigate?: () => void;
}

export function Sidebar({ className, onNavigate }: SidebarProps) {
  const pathname = usePathname();
  const [expandedTopics, setExpandedTopics] = React.useState<string[]>([]);

  // Auto-expand the current topic
  React.useEffect(() => {
    const currentTopic = topics.find((topic) =>
      pathname.includes(`/learn/${topic.id}`)
    );
    if (currentTopic && !expandedTopics.includes(currentTopic.id)) {
      setExpandedTopics((prev) => [...prev, currentTopic.id]);
    }
  }, [pathname, expandedTopics]);

  const toggleTopic = (topicId: string) => {
    setExpandedTopics((prev) =>
      prev.includes(topicId)
        ? prev.filter((id) => id !== topicId)
        : [...prev, topicId]
    );
  };

  return (
    <aside
      className={cn(
        "w-64 border-r border-border bg-sidebar overflow-y-auto",
        className
      )}
    >
      <nav className="p-4 space-y-1">
        <div className="mb-4">
          <h2 className="px-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            Topics
          </h2>
        </div>

        {topics.map((topic) => (
          <TopicItem
            key={topic.id}
            topic={topic}
            isExpanded={expandedTopics.includes(topic.id)}
            onToggle={() => toggleTopic(topic.id)}
            pathname={pathname}
            onNavigate={onNavigate}
          />
        ))}
      </nav>
    </aside>
  );
}

interface TopicItemProps {
  topic: Topic;
  isExpanded: boolean;
  onToggle: () => void;
  pathname: string;
  onNavigate?: () => void;
}

function TopicItem({
  topic,
  isExpanded,
  onToggle,
  pathname,
  onNavigate,
}: TopicItemProps) {
  const isActive = pathname.includes(`/learn/${topic.id}`);

  return (
    <div className="space-y-1">
      <Button
        variant="ghost"
        className={cn(
          "w-full justify-start gap-2 px-2 h-9 font-medium",
          isActive && "bg-sidebar-accent text-sidebar-accent-foreground"
        )}
        onClick={onToggle}
      >
        {iconMap[topic.icon] || <Layers className="h-4 w-4" />}
        <span className="flex-1 text-left text-sm truncate">{topic.title}</span>
        {isExpanded ? (
          <ChevronDown className="h-4 w-4 shrink-0" />
        ) : (
          <ChevronRight className="h-4 w-4 shrink-0" />
        )}
      </Button>

      {isExpanded && (
        <div className="ml-4 pl-2 border-l border-border space-y-1">
          {topic.subtopics.map((subtopic) => {
            const href = `/learn/${topic.id}/${subtopic.slug}`;
            const isSubtopicActive = pathname === href;

            return (
              <Link
                key={subtopic.id}
                href={href}
                onClick={onNavigate}
                className={cn(
                  "block px-2 py-1.5 text-sm rounded-md transition-colors",
                  isSubtopicActive
                    ? "bg-primary text-primary-foreground font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted"
                )}
              >
                {subtopic.title}
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
