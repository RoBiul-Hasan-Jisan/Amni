"use client";

import * as React from "react";
import Link from "next/link";

import {
  Search,
  Menu,
  Moon,
  Sun,
  X,
  BookOpen,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

import { useTheme } from "next-themes";

import { topics } from "@/lib/topics-data";
import { cn } from "@/lib/utils";

interface HeaderProps {
  onMenuToggle?: () => void;
  isMenuOpen?: boolean;
}

export function Header({
  onMenuToggle,
  isMenuOpen = false,
}: HeaderProps) {
  const { theme, setTheme } = useTheme();

  const [mounted, setMounted] = React.useState(false);

  const [searchOpen, setSearchOpen] =
    React.useState(false);

  const [searchQuery, setSearchQuery] =
    React.useState("");

  const [searchResults, setSearchResults] =
    React.useState<
      Array<{
        topic: string;
        subtopic: string;
        slug: string;
      }>
    >([]);

  // =========================
  // FIX HYDRATION
  // =========================
  React.useEffect(() => {
    setMounted(true);
  }, []);

  // =========================
  // SEARCH FUNCTION
  // =========================
  const handleSearch = (query: string) => {
    setSearchQuery(query);

    if (query.trim().length < 2) {
      setSearchResults([]);
      return;
    }

    const results: Array<{
      topic: string;
      subtopic: string;
      slug: string;
    }> = [];

    topics.forEach((topic) => {
      topic.subtopics.forEach((subtopic) => {
        if (
          subtopic.title
            .toLowerCase()
            .includes(query.toLowerCase()) ||
          topic.title
            .toLowerCase()
            .includes(query.toLowerCase())
        ) {
          results.push({
            topic: topic.title,
            subtopic: subtopic.title,
            slug: `/learn/${topic.id}/${subtopic.slug}`,
          });
        }
      });
    });

    setSearchResults(results.slice(0, 8));
  };

  // =========================
  // CLOSE SEARCH
  // =========================
  const closeSearch = () => {
    setSearchOpen(false);
    setSearchQuery("");
    setSearchResults([]);
  };

  return (
    <>
      {/* =========================
          MOBILE SEARCH OVERLAY
      ========================= */}
      <div
        onClick={closeSearch}
        className={cn(
          "fixed inset-0 z-40 bg-black/30 backdrop-blur-sm transition-all duration-300 md:hidden",
          searchOpen
            ? "visible opacity-100"
            : "invisible opacity-0"
        )}
      />

      {/* =========================
          HEADER
      ========================= */}
      <header className="sticky top-0 z-[60] w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="flex h-14 items-center px-4 lg:px-6">

          {/* =========================
              MOBILE MENU BUTTON
          ========================= */}
          <Button
            variant="ghost"
            size="icon"
            className="mr-2 lg:hidden"
            onClick={onMenuToggle}
            aria-label={
              isMenuOpen
                ? "Close menu"
                : "Open menu"
            }
          >
            {isMenuOpen ? (
              <X className="h-5 w-5" />
            ) : (
              <Menu className="h-5 w-5" />
            )}
          </Button>

          {/* =========================
              LOGO
          ========================= */}
          <Link
            href="/"
            className="mr-6 flex items-center gap-2 shrink-0"
          >
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
              <BookOpen className="h-4 w-4 text-primary-foreground" />
            </div>

            <span className="hidden text-sm font-bold text-foreground sm:inline-block">
              CSE Learn
            </span>
          </Link>

          {/* =========================
              DESKTOP NAV
          ========================= */}
          <nav className="hidden items-center gap-6 text-sm md:flex">

            <Link
              href="/learn"
              className="text-muted-foreground transition-colors hover:text-foreground"
            >
              Topics
            </Link>

            <Link
              href="/learn/data-structures/arrays"
              className="text-muted-foreground transition-colors hover:text-foreground"
            >
              Get Started
            </Link>

          </nav>

          <div className="flex-1" />

          {/* =========================
              RIGHT SIDE
          ========================= */}
          <div className="flex items-center gap-1 sm:gap-2">

            {/* =========================
                SEARCH
            ========================= */}
            <div className="relative z-[70]">

              <div
                className={cn(
                  "flex items-center transition-all duration-300 ease-in-out",
                  searchOpen
                    ? "w-[220px] sm:w-64"
                    : "w-auto"
                )}
              >

                {searchOpen ? (
                  <div className="relative w-full">

                    {/* SEARCH ICON */}
                    <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />

                    {/* INPUT */}
                    <Input
                      type="search"
                      placeholder="Search topics..."
                      className="h-9 pl-8 pr-9"
                      value={searchQuery}
                      onChange={(e) =>
                        handleSearch(
                          e.target.value
                        )
                      }
                      autoFocus
                    />

                    {/* CLOSE BUTTON */}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="absolute right-0 top-0 h-9 w-9"
                      onClick={closeSearch}
                    >
                      <X className="h-4 w-4" />
                    </Button>

                    {/* =========================
                        SEARCH RESULTS
                    ========================= */}
                    {searchResults.length > 0 && (
                      <div className="absolute top-full left-0 right-0 mt-2 overflow-hidden rounded-xl border border-border bg-popover shadow-2xl">

                        {searchResults.map(
                          (result, index) => (
                            <Link
                              key={index}
                              href={result.slug}
                              onClick={closeSearch}
                              className="block border-b border-border/50 px-3 py-3 transition-colors last:border-b-0 hover:bg-muted"
                            >
                              <div className="text-sm font-medium text-foreground">
                                {result.subtopic}
                              </div>

                              <div className="mt-1 text-xs text-muted-foreground">
                                {result.topic}
                              </div>
                            </Link>
                          )
                        )}

                      </div>
                    )}
                  </div>
                ) : (
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() =>
                      setSearchOpen(true)
                    }
                    aria-label="Search"
                  >
                    <Search className="h-5 w-5" />
                  </Button>
                )}
              </div>
            </div>

            {/* =========================
                THEME TOGGLE
            ========================= */}
            {mounted && (
              <Button
                variant="ghost"
                size="icon"
                aria-label="Toggle theme"
                className="relative"
                onClick={() =>
                  setTheme(
                    theme === "dark"
                      ? "light"
                      : "dark"
                  )
                }
              >
                <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />

                <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
              </Button>
            )}

          </div>
        </div>
      </header>
    </>
  );
}
