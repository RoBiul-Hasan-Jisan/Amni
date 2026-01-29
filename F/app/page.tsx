"use client";

import React from "react"

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
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      <div className="min-h-screen bg-background">
        <Header />

        <main>
          {/* Hero Section */}
          <section className="py-20 px-4 lg:px-6">
            <div className="max-w-4xl mx-auto text-center">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm mb-6">
                <Zap className="h-4 w-4" />
                Interactive Learning Platform
              </div>
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-foreground mb-6 text-balance">
                Master Computer Science with Interactive Learning
              </h1>
              <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto text-pretty">
                Learn Data Structures, Algorithms, Operating Systems, and more with 
                visual explanations, code examples, and interview preparation materials.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link href="/learn/data-structures/arrays">
                  <Button size="lg" className="w-full sm:w-auto">
                    Start Learning
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
                <Link href="/learn">
                  <Button variant="outline" size="lg" className="w-full sm:w-auto bg-transparent">
                    Browse Topics
                  </Button>
                </Link>
              </div>
            </div>
          </section>

          {/* Features Section */}
          <section className="py-16 px-4 lg:px-6 bg-muted/30">
            <div className="max-w-6xl mx-auto">
              <h2 className="text-2xl md:text-3xl font-bold text-center mb-12 text-foreground">
                Why Learn With Us?
              </h2>
              <div className="grid md:grid-cols-3 gap-8">
                <FeatureCard
                  icon={<BookOpen className="h-8 w-8" />}
                  title="Simple Explanations"
                  description="Complex concepts broken down into easy-to-understand lessons with real-world examples."
                />
                <FeatureCard
                  icon={<Layers className="h-8 w-8" />}
                  title="Interactive Visualizations"
                  description="See algorithms in action with step-by-step animations and interactive controls."
                />
                <FeatureCard
                  icon={<Target className="h-8 w-8" />}
                  title="Interview Ready"
                  description="Practice with common interview questions and patterns from top tech companies."
                />
              </div>
            </div>
          </section>

          {/* Topics Section */}
          <section className="py-16 px-4 lg:px-6">
            <div className="max-w-6xl mx-auto">
              <div className="text-center mb-12">
                <h2 className="text-2xl md:text-3xl font-bold mb-4 text-foreground">
                  Core CSE Topics
                </h2>
                <p className="text-muted-foreground max-w-2xl mx-auto">
                  Comprehensive coverage of fundamental Computer Science and Engineering topics,
                  structured for students and interview preparation.
                </p>
              </div>

              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {topics.map((topic) => (
                  <Link
                    key={topic.id}
                    href={`/learn/${topic.id}/${topic.subtopics[0]?.slug}`}
                    className="group p-6 bg-card rounded-lg border border-border hover:border-primary hover:shadow-lg transition-all"
                  >
                    <div className="flex items-start gap-4">
                      <div className="p-3 rounded-lg bg-primary/10 text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                        {iconMap[topic.icon] || <Layers className="h-6 w-6" />}
                      </div>
                      <div className="flex-1">
                        <h3 className="font-semibold mb-1 text-foreground group-hover:text-primary transition-colors">
                          {topic.title}
                        </h3>
                        <p className="text-sm text-muted-foreground line-clamp-2">
                          {topic.description}
                        </p>
                        <div className="mt-3 text-xs text-muted-foreground">
                          {topic.subtopics.length} lessons
                        </div>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          </section>

          {/* CTA Section */}
          <section className="py-20 px-4 lg:px-6 bg-primary text-primary-foreground">
            <div className="max-w-4xl mx-auto text-center">
              <h2 className="text-2xl md:text-3xl font-bold mb-4">
                Ready to Start Learning?
              </h2>
              <p className="text-primary-foreground/80 mb-8 max-w-xl mx-auto">
                Begin with Arrays - the fundamental building block of all data structures,
                and progress through increasingly complex topics.
              </p>
              <Link href="/learn/data-structures/arrays">
                <Button
                  size="lg"
                  variant="secondary"
                  className="bg-primary-foreground text-primary hover:bg-primary-foreground/90"
                >
                  Start with Arrays
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>
          </section>
        </main>

        <footer className="py-8 px-4 lg:px-6 border-t border-border">
          <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <div className="flex h-6 w-6 items-center justify-center rounded bg-primary">
                <BookOpen className="h-3 w-3 text-primary-foreground" />
              </div>
              <span className="font-semibold text-foreground">CSE Learn</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Built for students, by educators. Learn Computer Science the right way.
            </p>
          </div>
        </footer>
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
    <div className="p-6 bg-card rounded-lg border border-border">
      <div className="p-3 rounded-lg bg-primary/10 text-primary w-fit mb-4">
        {icon}
      </div>
      <h3 className="font-semibold text-lg mb-2 text-foreground">{title}</h3>
      <p className="text-muted-foreground">{description}</p>
    </div>
  );
}
