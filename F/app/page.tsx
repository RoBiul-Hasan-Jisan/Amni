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
  BookOpen,
  Target,
  Zap,
  PanelLeftClose,
  PanelLeft,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { ThemeProvider } from "@/components/theme-provider";
import { Header } from "@/components/header";
import { Sidebar } from "@/components/sidebar";
import { topics } from "@/lib/topics-data";

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

export default function HomePage() {
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);
  const [isDesktopSidebarOpen, setIsDesktopSidebarOpen] = React.useState(true);

  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      <div className="min-h-screen bg-background">
        <Header
          isMenuOpen={isMenuOpen}
          onMenuToggle={() => setIsMenuOpen((prev) => !prev)}
        />

       
   {/* Desktop Sidebar Toggle Icon */}

<div className="hidden lg:block fixed left-2 top-3 z-50">
  <Button
    variant="secondary"
    size="icon"
    onClick={() => setIsDesktopSidebarOpen(!isDesktopSidebarOpen)}
    className="h-8 w-8 shadow-md bg-secondary hover:bg-secondary/80"
    title={isDesktopSidebarOpen ? "Hide Sidebar" : "Show Sidebar"}
  >
    {isDesktopSidebarOpen ? (
      <PanelLeftClose className="h-4 w-4" />
    ) : (
      <PanelLeft className="h-4 w-4" />
    )}
  </Button>
</div>

        <div className="flex">
          {/* Mobile Overlay - Only show on mobile when sidebar is open */}
          {isMenuOpen && (
            <div
              className="fixed inset-0 z-[49] lg:hidden bg-black/30 backdrop-blur-sm cursor-pointer"
              onClick={() => setIsMenuOpen(false)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Escape' && setIsMenuOpen(false)}
            />
          )}

          {/* Sidebar */}
          <div className={`${!isDesktopSidebarOpen ? 'lg:hidden' : ''}`}>
            <Sidebar
              isMobileOpen={isMenuOpen}
              onMobileClose={() => setIsMenuOpen(false)}
              onNavigate={() => setIsMenuOpen(false)}
            />
          </div>

          {/* Main Content */}
          <main className="flex-1 min-w-0">
            {/* HERO */}
            <section className="py-20 px-4 lg:px-6">
              <div className="max-w-4xl mx-auto text-center">

                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm mb-6">
                  <Zap className="h-4 w-4" />
                  Interactive Learning Platform
                </div>

                <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6">
                  Master Computer Science with Interactive Learning
                </h1>

                <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
                  Learn Data Structures, Algorithms, Operating Systems,
                  and more with visual explanations and interview prep.
                </p>

                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <Link href="/learn/data-structures/arrays">
                    <Button size="lg">
                      Start Learning
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </Link>

                  <Link href="/learn">
                    <Button variant="outline" size="lg">
                      Browse Topics
                    </Button>
                  </Link>
                </div>

              </div>
            </section>

            {/* FEATURES */}
            <section className="py-16 px-4 lg:px-6 bg-muted/30">
              <div className="max-w-6xl mx-auto">

                <h2 className="text-3xl font-bold text-center mb-12">
                  Why Learn With Us?
                </h2>

                <div className="grid md:grid-cols-3 gap-8">

                  <FeatureCard
                    icon={<BookOpen className="h-8 w-8" />}
                    title="Simple Explanations"
                    description="Complex concepts simplified."
                  />

                  <FeatureCard
                    icon={<Layers className="h-8 w-8" />}
                    title="Interactive Visualizations"
                    description="Understand algorithms visually."
                  />

                  <FeatureCard
                    icon={<Target className="h-8 w-8" />}
                    title="Interview Ready"
                    description="Practice real interview questions."
                  />

                </div>

              </div>
            </section>

            {/* TOPICS */}
            <section className="py-16 px-4 lg:px-6">

              <div className="max-w-6xl mx-auto text-center mb-12">
                <h2 className="text-3xl font-bold">
                  Core CSE Topics
                </h2>

                <p className="text-muted-foreground mt-2">
                  Structured roadmap for CS learning
                </p>
              </div>

              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">

                {topics.map((topic) => (
                  <Link
                    key={topic.id}
                    href={`/learn/${topic.id}/${topic.subtopics[0]?.slug}`}
                    className="group p-6 bg-card rounded-xl border hover:border-primary hover:shadow-xl transition-all"
                  >
                    <div className="flex gap-4">

                      <div className="p-3 rounded-lg bg-primary/10 text-primary">
                        {iconMap[topic.icon] || <Layers />}
                      </div>

                      <div className="text-left">
                        <h3 className="font-semibold group-hover:text-primary transition-colors">
                          {topic.title}
                        </h3>

                        <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                          {topic.description}
                        </p>

                        <span className="text-xs text-muted-foreground mt-2 block">
                          {topic.subtopics.length} lessons
                        </span>
                      </div>

                    </div>
                  </Link>
                ))}

              </div>
            </section>
          </main>
        </div>
      </div>
    </ThemeProvider>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="p-6 bg-card rounded-xl border">
      <div className="text-primary mb-4">
        {icon}
      </div>

      <h3 className="font-semibold mb-2">
        {title}
      </h3>

      <p className="text-muted-foreground text-sm">
        {description}
      </p>
    </div>
  );
}