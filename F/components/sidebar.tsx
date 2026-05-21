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
  X,
  Home,
  BookOpen,
  Terminal,
  Award,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { topics, Topic } from "@/lib/topics-data";
import { Button } from "@/components/ui/button";
import { useEffect } from "react";

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
  isMobileOpen?: boolean;
  onMobileClose?: () => void;
}

export function Sidebar({
  className,
  onNavigate,
  isMobileOpen = false,
  onMobileClose,
}: SidebarProps) {
  const pathname = usePathname();

  const [expandedTopics, setExpandedTopics] = React.useState<string[]>(() => {
    const currentTopic = topics.find((topic) =>
      pathname.includes(`/learn/${topic.id}`)
    );

    return currentTopic ? [currentTopic.id] : [];
  });

  // AUTO OPEN CURRENT TOPIC
  React.useEffect(() => {
    const currentTopic = topics.find((topic) =>
      pathname.includes(`/learn/${topic.id}`)
    );

    if (
      currentTopic &&
      !expandedTopics.includes(currentTopic.id)
    ) {
      setExpandedTopics((prev) => [...prev, currentTopic.id]);
    }
  }, [pathname]);

  // LOCK BODY SCROLL
  React.useEffect(() => {
    if (isMobileOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }

    return () => {
      document.body.style.overflow = "";
    };
  }, [isMobileOpen]);

  const toggleTopic = (
    topicId: string,
    e?: React.MouseEvent
  ) => {
    e?.preventDefault();
    e?.stopPropagation();

    setExpandedTopics((prev) =>
      prev.includes(topicId)
        ? prev.filter((id) => id !== topicId)
        : [...prev, topicId]
    );
  };

  const handleNavigate = () => {
    onNavigate?.();
    onMobileClose?.();
  };

  return (
    <>
      {/* SIDEBAR */}
      <aside
        className={cn(
          "fixed top-14 left-0 z-50 h-[calc(100vh-56px)] w-72",
          "bg-background border-r border-border",
          "overflow-y-auto",
          "transition-transform duration-300 ease-in-out",
          "lg:sticky lg:top-14 lg:translate-x-0",
          isMobileOpen
            ? "translate-x-0"
            : "-translate-x-full",
          className
        )}
      >
        <div className="p-4">

          {/* MOBILE CLOSE BUTTON */}
          <div className="flex justify-end mb-4 lg:hidden">
            <Button
              variant="ghost"
              size="icon"
              onClick={onMobileClose}
              className="h-9 w-9"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>

          <div className="space-y-6">

            {/* MAIN */}
            <div>
              <h3 className="px-3 mb-2 text-xs font-semibold uppercase text-muted-foreground">
                Main
              </h3>

              <div className="space-y-1">

                <NavItem
                  href="/"
                  icon={<Home className="h-4 w-4" />}
                  label="Home"
                  pathname={pathname}
                  onClick={handleNavigate}
                />

                <NavItem
                  href="/learn"
                  icon={<BookOpen className="h-4 w-4" />}
                  label="All Topics"
                  pathname={pathname}
                  onClick={handleNavigate}
                />

                <NavItem
                  href="/practice"
                  icon={<Terminal className="h-4 w-4" />}
                  label="Practice"
                  pathname={pathname}
                  onClick={handleNavigate}
                />

                <NavItem
                  href="/interview"
                  icon={<Award className="h-4 w-4" />}
                  label="Interview Prep"
                  pathname={pathname}
                  onClick={handleNavigate}
                />

              </div>
            </div>

            {/* TOPICS */}
            <div>
              <h3 className="px-3 mb-2 text-xs font-semibold uppercase text-muted-foreground">
                Topics
              </h3>

              <div className="space-y-1">

                {topics.map((topic) => (
                  <TopicItem
                    key={topic.id}
                    topic={topic}
                    pathname={pathname}
                    isExpanded={expandedTopics.includes(topic.id)}
                    onToggle={(e) =>
                      toggleTopic(topic.id, e)
                    }
                    onNavigate={handleNavigate}
                  />
                ))}

              </div>
            </div>

          </div>
        </div>
      </aside>
    </>
  );
}

/* ========================================
   NAV ITEM
======================================== */

function NavItem({
  href,
  icon,
  label,
  pathname,
  onClick,
}: {
  href: string;
  icon: React.ReactNode;
  label: string;
  pathname: string;
  onClick?: () => void;
}) {
  const isActive =
    pathname === href ||
    (href !== "/" && pathname.startsWith(href));

  return (
    <Link
      href={href}
      onClick={onClick}
      className={cn(
        "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all",
        isActive
          ? "bg-primary text-primary-foreground"
          : "text-muted-foreground hover:bg-muted hover:text-foreground"
      )}
    >
      {icon}
      <span>{label}</span>
    </Link>
  );
}

/* ========================================
   TOPIC ITEM
======================================== */

function TopicItem({
  topic,
  pathname,
  isExpanded,
  onToggle,
  onNavigate,
}: {
  topic: Topic;
  pathname: string;
  isExpanded: boolean;
  onToggle: (e: React.MouseEvent) => void;
  onNavigate?: () => void;
}) {
  const isActive = pathname.includes(`/learn/${topic.id}`);

  return (
    <div className="space-y-1">

      {/* TOPIC BUTTON */}
      <button
        onClick={onToggle}
        className={cn(
          "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all",
          isActive
            ? "bg-muted text-foreground font-medium"
            : "text-muted-foreground hover:bg-muted hover:text-foreground"
        )}
      >
        {iconMap[topic.icon] || (
          <Layers className="h-4 w-4" />
        )}

        <span className="flex-1 text-left">
          {topic.title}
        </span>

        {isExpanded ? (
          <ChevronDown className="h-4 w-4 shrink-0" />
        ) : (
          <ChevronRight className="h-4 w-4 shrink-0" />
        )}
      </button>

      {/* SUBTOPICS */}
      <div
        className={cn(
          "overflow-hidden transition-all duration-300",
          isExpanded
            ? "max-h-[1000px] opacity-100"
            : "max-h-0 opacity-0"
        )}
      >
        <div className="ml-6 border-l pl-3 space-y-1 py-1">

          {topic.subtopics.map((subtopic) => {
            const href = `/learn/${topic.id}/${subtopic.slug}`;

            const active = pathname === href;

            return (
              <Link
                key={subtopic.id}
                href={href}
                onClick={onNavigate}
                className={cn(
                  "block rounded-md px-3 py-1.5 text-sm transition-all",
                  active
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                )}
              >
                {subtopic.title}
              </Link>
            );
          })}

        </div>
      </div>
    </div>
  );
}
