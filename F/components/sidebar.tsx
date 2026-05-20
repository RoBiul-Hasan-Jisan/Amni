"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
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
  isMobileOpen,
  onMobileClose,
}: SidebarProps) {
  const pathname = usePathname();

  const [expandedTopics, setExpandedTopics] = React.useState<string[]>(
    () => {
      const currentTopic = topics.find((topic) =>
        pathname.includes(`/learn/${topic.id}`)
      );
      return currentTopic ? [currentTopic.id] : [];
    }
  );

  // =========================
  // AUTO EXPAND CURRENT TOPIC
  // =========================
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

  // =========================
  // TOGGLE TOPIC
  // =========================
  const toggleTopic = (topicId: string, e?: React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }

    setExpandedTopics((prev) =>
      prev.includes(topicId)
        ? prev.filter((id) => id !== topicId)
        : [...prev, topicId]
    );
  };

  // =========================
  // FIXED NAVIGATION HANDLER
  // =========================
  const handleNavigate = () => {
    onNavigate?.();
    onMobileClose?.();
  };

  // =========================
  // BODY SCROLL LOCK (mobile)
  // =========================
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

  return (
    <>
      {/* MOBILE BACKDROP (FIXED) */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 top-14 bg-black/50 z-30 lg:hidden"
          onClick={onMobileClose}
        />
      )}

      {/* SIDEBAR */}
      <aside
        className={cn(
          "fixed lg:sticky top-14 lg:top-auto left-0 h-[calc(100vh-3.5rem)] bg-sidebar border-r border-border overflow-y-auto transition-transform duration-300 z-40",
          "lg:translate-x-0 lg:relative lg:h-[calc(100vh-3.5rem)]",
          isMobileOpen ? "translate-x-0" : "-translate-x-full",
          className
        )}
      >
        <div className="p-4">

          {/* CLOSE BUTTON (FIXED) */}
          <div className="lg:hidden flex justify-end mb-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={onMobileClose}
              className="h-8 w-8 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          <div className="space-y-6">

            {/* MAIN */}
            <div>
              <h3 className="px-3 text-xs font-semibold text-muted-foreground uppercase mb-2">
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
              <h3 className="px-3 text-xs font-semibold text-muted-foreground uppercase mb-2">
                Topics
              </h3>

              <div className="space-y-1">
                {topics.map((topic) => (
                  <TopicItem
                    key={topic.id}
                    topic={topic}
                    isExpanded={expandedTopics.includes(topic.id)}
                    onToggle={(e) => toggleTopic(topic.id, e)}
                    pathname={pathname}
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

/* =========================
   NAV ITEM
========================= */
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
        "flex items-center gap-3 px-3 py-2 text-sm rounded-md transition-all",
        isActive
          ? "bg-primary text-primary-foreground"
          : "text-muted-foreground hover:text-foreground hover:bg-muted"
      )}
    >
      {icon}
      <span>{label}</span>
    </Link>
  );
}

/* =========================
   TOPIC ITEM
========================= */
function TopicItem({
  topic,
  isExpanded,
  onToggle,
  pathname,
  onNavigate,
}: {
  topic: Topic;
  isExpanded: boolean;
  onToggle: (e: React.MouseEvent) => void;
  pathname: string;
  onNavigate?: () => void;
}) {
  const isActive = pathname.includes(`/learn/${topic.id}`);

  return (
    <div className="space-y-1">
      <button
        onClick={onToggle}
        className={cn(
          "w-full flex items-center gap-3 px-3 py-2 text-sm rounded-md",
          isActive
            ? "bg-sidebar-accent font-medium"
            : "text-muted-foreground hover:bg-muted"
        )}
      >
        {iconMap[topic.icon] || <Layers className="h-4 w-4" />}
        <span className="flex-1 text-left">{topic.title}</span>

        {isExpanded ? (
          <ChevronDown className="h-3 w-3" />
        ) : (
          <ChevronRight className="h-3 w-3" />
        )}
      </button>

      {isExpanded && (
        <div className="ml-6 pl-2 border-l space-y-1">
          {topic.subtopics.map((sub) => {
            const href = `/learn/${topic.id}/${sub.slug}`;
            const active = pathname === href;

            return (
              <Link
                key={sub.id}
                href={href}
                onClick={onNavigate}
                className={cn(
                  "block px-3 py-1.5 text-sm rounded-md",
                  active
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted"
                )}
              >
                {sub.title}
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}