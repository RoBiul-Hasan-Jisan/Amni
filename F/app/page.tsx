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
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { ThemeProvider } from "@/components/theme-provider";
import { Header } from "@/components/header";
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

  const toggleMenu = () => setIsMenuOpen((p) => !p);
  const closeMenu = () => setIsMenuOpen(false);

  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      <div className="min-h-screen bg-background">

        {/* HEADER */}
        <Header
          isMenuOpen={isMenuOpen}
          onMenuToggle={toggleMenu}
        />

        {/* ✅ MOBILE SIDEBAR (FULL FIXED) */}
        <div
          className={`fixed inset-0 z-50 md:hidden ${
            isMenuOpen ? "block" : "hidden"
          }`}
        >
          {/* BACKDROP */}
          <div
            className="absolute inset-0 bg-black/50"
            onClick={closeMenu}
          />

          {/* PANEL */}
          <div className="absolute left-0 top-0 h-full w-72 bg-background border-r border-border shadow-lg overflow-y-auto">

            <div className="p-4 space-y-6">

              {/* MAIN */}
              <div>
                <h3 className="text-xs font-semibold text-muted-foreground uppercase mb-2">
                  Main
                </h3>

                <nav className="flex flex-col gap-2 text-sm">
                  <Link href="/" className="p-2 rounded hover:bg-muted" onClick={closeMenu}>
                    Home
                  </Link>

                  <Link href="/learn" className="p-2 rounded hover:bg-muted" onClick={closeMenu}>
                    All Topics
                  </Link>

                  <Link href="/practice" className="p-2 rounded hover:bg-muted" onClick={closeMenu}>
                    Practice
                  </Link>

                  <Link href="/interview" className="p-2 rounded hover:bg-muted" onClick={closeMenu}>
                    Interview Prep
                  </Link>
                </nav>
              </div>

              {/* TOPICS */}
              <div>
                <h3 className="text-xs font-semibold text-muted-foreground uppercase mb-2">
                  Topics
                </h3>

                <nav className="flex flex-col gap-2 text-sm">
                  {topics.map((topic) => (
                    <Link
                      key={topic.id}
                      href={`/learn/${topic.id}/${topic.subtopics[0]?.slug}`}
                      className="p-2 rounded hover:bg-muted"
                      onClick={closeMenu}
                    >
                      {topic.title}
                    </Link>
                  ))}
                </nav>
              </div>

              {/* HELP */}
              <div className="border-t pt-4">
                <p className="text-xs text-muted-foreground mb-2">
                  Need help?
                </p>

                <Link
                  href="/contact"
                  className="block p-2 rounded border text-center hover:bg-muted"
                  onClick={closeMenu}
                >
                  Contact Support
                </Link>
              </div>

            </div>
          </div>
        </div>

        {/* MAIN CONTENT */}
        <main>
          {/* HERO */}
          <section className="py-20 px-4 lg:px-6">
            <div className="max-w-4xl mx-auto text-center">

              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm mb-6">
                <Zap className="h-4 w-4" />
                Interactive Learning Platform
              </div>

              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-foreground mb-6 text-balance">
                Master Computer Science with Interactive Learning
              </h1>

              <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
                Learn Data Structures, Algorithms, Operating Systems, and more with
                visual explanations, code examples, and interview preparation materials.
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
            <div className="max-w-6xl mx-auto text-center">
              <h2 className="text-2xl md:text-3xl font-bold mb-12">
                Why Learn With Us?
              </h2>

              <div className="grid md:grid-cols-3 gap-8">

                <FeatureCard
                  icon={<BookOpen className="h-8 w-8" />}
                  title="Simple Explanations"
                  description="Complex concepts broken down into easy-to-understand lessons."
                />

                <FeatureCard
                  icon={<Layers className="h-8 w-8" />}
                  title="Interactive Visualizations"
                  description="See algorithms in action with animations."
                />

                <FeatureCard
                  icon={<Target className="h-8 w-8" />}
                  title="Interview Ready"
                  description="Practice top interview questions."
                />

              </div>
            </div>
          </section>

          {/* TOPICS */}
          <section className="py-16 px-4 lg:px-6">
            <div className="max-w-6xl mx-auto text-center mb-12">
              <h2 className="text-2xl md:text-3xl font-bold">
                Core CSE Topics
              </h2>
              <p className="text-muted-foreground mt-2">
                Structured learning path for CS fundamentals
              </p>
            </div>

            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 max-w-6xl mx-auto">

              {topics.map((topic) => (
                <Link
                  key={topic.id}
                  href={`/learn/${topic.id}/${topic.subtopics[0]?.slug}`}
                  className="p-6 bg-card rounded-lg border hover:border-primary hover:shadow-lg transition"
                >
                  <div className="flex gap-4 items-start">
                    <div className="p-3 rounded bg-primary/10 text-primary">
                      {iconMap[topic.icon] || <Layers />}
                    </div>

                    <div>
                      <h3 className="font-semibold">{topic.title}</h3>
                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {topic.description}
                      </p>
                      <span className="text-xs text-muted-foreground">
                        {topic.subtopics.length} lessons
                      </span>
                    </div>
                  </div>
                </Link>
              ))}

            </div>
          </section>

          {/* CTA */}
          <section className="py-20 px-4 lg:px-6 bg-primary text-primary-foreground text-center">
            <h2 className="text-2xl md:text-3xl font-bold mb-4">
              Ready to Start Learning?
            </h2>

            <p className="mb-8 max-w-xl mx-auto">
              Start with Arrays and build your foundation step by step.
            </p>

            <Link href="/learn/data-structures/arrays">
              <Button variant="secondary" size="lg">
                Start with Arrays
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </section>

        </main>
      </div>
    </ThemeProvider>
  );
}

/* FEATURE CARD */
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
    <div className="p-6 bg-card rounded-lg border">
      <div className="text-primary mb-3">{icon}</div>
      <h3 className="font-semibold mb-2">{title}</h3>
      <p className="text-muted-foreground">{description}</p>
    </div>
  );
}