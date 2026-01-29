"use client";

import * as React from "react";
import Link from "next/link";
import { Search, Menu, Moon, Sun, X, BookOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useTheme } from "next-themes";
import { topics } from "@/lib/topics-data";
import { cn } from "@/lib/utils";

interface HeaderProps {
  onMenuToggle?: () => void;
  isMenuOpen?: boolean;
}

export function Header({ onMenuToggle, isMenuOpen }: HeaderProps) {
  const { theme, setTheme } = useTheme();
  const [searchOpen, setSearchOpen] = React.useState(false);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [searchResults, setSearchResults] = React.useState<
    Array<{ topic: string; subtopic: string; slug: string }>
  >([]);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }

    const results: Array<{ topic: string; subtopic: string; slug: string }> = [];
    topics.forEach((topic) => {
      topic.subtopics.forEach((subtopic) => {
        if (
          subtopic.title.toLowerCase().includes(query.toLowerCase()) ||
          topic.title.toLowerCase().includes(query.toLowerCase())
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

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-14 items-center px-4 lg:px-6">
        <Button
          variant="ghost"
          size="icon"
          className="mr-2 lg:hidden"
          onClick={onMenuToggle}
          aria-label={isMenuOpen ? "Close menu" : "Open menu"}
        >
          {isMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </Button>

        <Link href="/" className="flex items-center gap-2 mr-6">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
            <BookOpen className="h-4 w-4 text-primary-foreground" />
          </div>
          <span className="hidden font-semibold sm:inline-block text-foreground">
            CSE Learn
          </span>
        </Link>

        <nav className="hidden md:flex items-center gap-6 text-sm">
          <Link
            href="/learn"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            Topics
          </Link>
          <Link
            href="/learn/data-structures/arrays"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            Get Started
          </Link>
        </nav>

        <div className="flex-1" />

        <div className="flex items-center gap-2">
          <div className="relative">
            <div
              className={cn(
                "flex items-center transition-all duration-200",
                searchOpen ? "w-64" : "w-auto"
              )}
            >
              {searchOpen ? (
                <div className="relative w-full">
                  <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input
                    type="search"
                    placeholder="Search topics..."
                    className="pl-8 pr-8 h-9"
                    value={searchQuery}
                    onChange={(e) => handleSearch(e.target.value)}
                    autoFocus
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute right-0 top-0 h-9 w-9"
                    onClick={() => {
                      setSearchOpen(false);
                      setSearchQuery("");
                      setSearchResults([]);
                    }}
                  >
                    <X className="h-4 w-4" />
                  </Button>

                  {searchResults.length > 0 && (
                    <div className="absolute top-full left-0 right-0 mt-1 bg-popover border border-border rounded-lg shadow-lg overflow-hidden z-50">
                      {searchResults.map((result, index) => (
                        <Link
                          key={index}
                          href={result.slug}
                          className="block px-3 py-2 hover:bg-muted transition-colors"
                          onClick={() => {
                            setSearchOpen(false);
                            setSearchQuery("");
                            setSearchResults([]);
                          }}
                        >
                          <div className="text-sm font-medium text-foreground">
                            {result.subtopic}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {result.topic}
                          </div>
                        </Link>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setSearchOpen(true)}
                  aria-label="Search"
                >
                  <Search className="h-5 w-5" />
                </Button>
              )}
            </div>
          </div>

          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            aria-label="Toggle theme"
          >
            <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          </Button>
        </div>
      </div>
    </header>
  );
}
