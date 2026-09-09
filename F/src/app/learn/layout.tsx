"use client";

import * as React from "react";
import { Header } from "@/components/layout/header";
import { Sidebar } from "@/components/layout/sidebar";

export default function LearnLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = React.useState(false);

  return (
    <div className="min-h-screen bg-background">
      <Header onMenuToggle={() => setSidebarOpen(!sidebarOpen)} isMenuOpen={sidebarOpen} />

      <div className="flex">
        {/* Mobile overlay — click to close sidebar */}
        {sidebarOpen && (
          <div
            className="fixed inset-0 z-[49] cursor-pointer bg-black/30 backdrop-blur-sm lg:hidden"
            onClick={() => setSidebarOpen(false)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Escape" && setSidebarOpen(false)}
          />
        )}

        <Sidebar
          isMobileOpen={sidebarOpen}
          onMobileClose={() => setSidebarOpen(false)}
          onNavigate={() => setSidebarOpen(false)}
        />

        <main className="min-w-0 flex-1">{children}</main>
      </div>
    </div>
  );
}
