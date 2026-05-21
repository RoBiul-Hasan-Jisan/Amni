"use client";

import * as React from "react";
import { ThemeProvider } from "@/components/theme-provider";
import { Header } from "@/components/header";
import { Sidebar } from "@/components/sidebar";
import { Button } from "@/components/ui/button";
import { PanelLeftClose, PanelLeft } from "lucide-react";
import { cn } from "@/lib/utils";

export default function LearnLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = React.useState(false);
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
          onMenuToggle={() => setSidebarOpen(!sidebarOpen)}
          isMenuOpen={sidebarOpen}
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
          {/* Mobile Overlay - Click to close sidebar */}
          {sidebarOpen && (
            <div
              className="fixed inset-0 z-[49] lg:hidden bg-black/30 backdrop-blur-sm cursor-pointer"
              onClick={() => setSidebarOpen(false)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Escape' && setSidebarOpen(false)}
            />
          )}

          {/* Sidebar - Desktop toggleable, Mobile conditional */}
          <div className={`${!isDesktopSidebarOpen ? 'lg:hidden' : ''}`}>
            <Sidebar
              isMobileOpen={sidebarOpen}
              onMobileClose={() => setSidebarOpen(false)}
              onNavigate={() => setSidebarOpen(false)}
            />
          </div>

          {/* Main Content */}
          <main className="flex-1 min-w-0">{children}</main>
        </div>
      </div>
    </ThemeProvider>
  );
}