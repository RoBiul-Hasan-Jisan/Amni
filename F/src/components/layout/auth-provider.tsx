"use client";

import * as React from "react";
import { SessionProvider } from "next-auth/react";

import { ThemeProvider } from "@/components/layout/theme-provider";

// Single top-level provider stack: NextAuth session + theme. Kept as one
// component so the root layout doesn't accumulate nested providers as more
// are added later.
export function AuthProvider({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
        {children}
      </ThemeProvider>
    </SessionProvider>
  );
}
