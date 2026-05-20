// D:\cseLernW\F\components/sidebar.tsx
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

export function Sidebar({ className, onNavigate, isMobileOpen, onMobileClose }: SidebarProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [expandedTopics, setExpandedTopics] = React.useState<string[]>(() => {
    // Initialize expanded topics based on current path
    const currentTopic = topics.find((topic) =>
      pathname.includes(`/learn/${topic.id}`)
    );
    return currentTopic ? [currentTopic.id] : [];
  });

  // Auto-expand the current topic when path changes
  React.useEffect(() => {
    const currentTopic = topics.find((topic) =>
      pathname.includes(`/learn/${topic.id}`)
    );
    if (currentTopic && !expandedTopics.includes(currentTopic.id)) {
      setExpandedTopics((prev) => [...prev, currentTopic.id]);
    }
  }, [pathname]);

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

  const handleNavigate = (href?: string) => {
    if (onNavigate) {
      onNavigate();
    }
    if (onMobileClose) {
      onMobileClose();
    }
  };

  // Handle body scroll lock
  React.useEffect(() => {
    if (isMobileOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isMobileOpen]);

  return (
    <>
      {/* Mobile Overlay */}
{isMobileOpen && (
  <div
    className="fixed inset-0 top-14 bg-black/50 z-30 lg:hidden"
    onClick={() => handleNavigate()}
  />
)}

<aside
  className={cn(
    "fixed lg:sticky top-14 lg:top-auto left-0 h-[calc(100vh-3.5rem)] bg-sidebar border-r border-border overflow-y-auto transition-transform duration-300 z-40",
    "lg:translate-x-0 lg:relative lg:h-[calc(100vh-3.5rem)]",
    isMobileOpen ? "translate-x-0" : "-translate-x-full",
    className
  )}
>
   
        <div className="p-4">
          {/* Mobile Close Button */}
          <div className="lg:hidden flex justify-end mb-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleNavigate()}
              className="h-8 w-8 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Navigation Sections */}
          <div className="space-y-6">
            {/* Main Section */}
            <div>
              <h3 className="px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                Main
              </h3>
              <div className="space-y-1">
                <NavItem
                  href="/"
                  icon={<Home className="h-4 w-4" />}
                  label="Home"
                  pathname={pathname}
                  onClick={() => handleNavigate()}
                />
                <NavItem
                  href="/learn"
                  icon={<BookOpen className="h-4 w-4" />}
                  label="All Topics"
                  pathname={pathname}
                  onClick={() => handleNavigate()}
                />
                <NavItem
                  href="/practice"
                  icon={<Terminal className="h-4 w-4" />}
                  label="Practice"
                  pathname={pathname}
                  onClick={() => handleNavigate()}
                />
                <NavItem
                  href="/interview"
                  icon={<Award className="h-4 w-4" />}
                  label="Interview Prep"
                  pathname={pathname}
                  onClick={() => handleNavigate()}
                />
              </div>
            </div>

            {/* Topics Section */}
            <div>
              <h3 className="px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
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
                    onNavigate={() => handleNavigate()}
                  />
                ))}
              </div>
            </div>

            {/* Footer Section in Sidebar */}
            <div className="pt-4 mt-4 border-t border-border">
              <div className="px-3 space-y-2">
                <p className="text-xs text-muted-foreground">
                  Need help?
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full text-xs"
                  asChild
                >
                  <Link href="/contact" onClick={() => handleNavigate()}>
                    Contact Support
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}

interface NavItemProps {
  href: string;
  icon: React.ReactNode;
  label: string;
  pathname: string;
  onClick?: () => void;
}

function NavItem({ href, icon, label, pathname, onClick }: NavItemProps) {
  const isActive = pathname === href || (href !== "/" && pathname.startsWith(href));

  return (
    <Link
      href={href}
      onClick={onClick}
      className={cn(
        "flex items-center gap-3 px-3 py-2 text-sm rounded-md transition-all duration-200",
        isActive
          ? "bg-primary text-primary-foreground shadow-sm"
          : "text-muted-foreground hover:text-foreground hover:bg-muted"
      )}
    >
      {icon}
      <span>{label}</span>
    </Link>
  );
}

interface TopicItemProps {
  topic: Topic;
  isExpanded: boolean;
  onToggle: (e: React.MouseEvent) => void;
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
  const isTopicActive = pathname.includes(`/learn/${topic.id}`);
  const subtopicRefs = React.useRef<{ [key: string]: HTMLAnchorElement | null }>({});

  // Scroll active subtopic into view
  React.useEffect(() => {
    if (isExpanded) {
      const activeSubtopic = topic.subtopics.find(
        (subtopic) => pathname === `/learn/${topic.id}/${subtopic.slug}`
      );
      if (activeSubtopic && subtopicRefs.current[activeSubtopic.id]) {
        setTimeout(() => {
          subtopicRefs.current[activeSubtopic.id]?.scrollIntoView({
            behavior: "smooth",
            block: "nearest",
          });
        }, 100);
      }
    }
  }, [isExpanded, pathname, topic.id, topic.subtopics]);

  return (
    <div className="space-y-1">
      <button
        onClick={onToggle}
        className={cn(
          "w-full flex items-center gap-3 px-3 py-2 text-sm rounded-md transition-all duration-200 group cursor-pointer",
          isTopicActive
            ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
            : "text-muted-foreground hover:text-foreground hover:bg-muted"
        )}
      >
        <span className="text-primary group-hover:scale-110 transition-transform">
          {iconMap[topic.icon] || <Layers className="h-4 w-4" />}
        </span>
        <span className="flex-1 text-left truncate">{topic.title}</span>
        <span className="text-xs text-muted-foreground">
          {isExpanded ? (
            <ChevronDown className="h-3 w-3" />
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
        </span>
      </button>

      {isExpanded && (
        <div className="ml-6 pl-2 border-l-2 border-border space-y-1">
          {topic.subtopics.map((subtopic) => {
            const href = `/learn/${topic.id}/${subtopic.slug}`;
            const isSubtopicActive = pathname === href;

            return (
              <Link
                key={subtopic.id}
                ref={(el) => {
                  if (el) subtopicRefs.current[subtopic.id] = el;
                }}
                href={href}
                onClick={onNavigate}
                className={cn(
                  "block px-3 py-1.5 text-sm rounded-md transition-all duration-200",
                  isSubtopicActive
                    ? "bg-primary text-primary-foreground font-medium shadow-sm"
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