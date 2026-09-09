import React from "react";
import type { Metadata, Viewport } from "next";

import { Inter, JetBrains_Mono, Space_Grotesk } from "next/font/google";

import { Analytics } from "@vercel/analytics/next";

import Link from "next/link";

import { Github, Linkedin, Mail, BookOpen } from "lucide-react";

import { AuthProvider } from "@/components/layout/auth-provider";

import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

// ========================================
// BASE URL
// ========================================
const BASE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://amni.dev";

// ========================================
// METADATA
// ========================================
export const metadata: Metadata = {
  metadataBase: new URL(BASE_URL),

  title: {
    default: "Amni — Learn computer science like an engineer",
    template: "%s | Amni",
  },

  description:
    "A structured, visual path through data structures, algorithms, operating systems, and machine learning — with working code in four languages and interview prep built in.",

  keywords: [
    "computer science",
    "data structures",
    "algorithms",
    "dsa",
    "operating systems",
    "dbms",
    "computer networks",
    "system design",
    "machine learning",
    "interview preparation",
    "software engineering",
    "coding interviews",
    "visual learning",
  ],

  authors: [{ name: "Amni", url: BASE_URL }],
  creator: "Amni",
  publisher: "Amni",

  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },

  openGraph: {
    title: "Amni — Learn computer science like an engineer",
    description:
      "A structured, visual path through data structures, algorithms, systems, and machine learning.",
    url: BASE_URL,
    siteName: "Amni",
    locale: "en_US",
    type: "website",
    images: [{ url: "/logo.png", width: 1200, height: 630, alt: "Amni" }],
  },

  twitter: {
    card: "summary_large_image",
    title: "Amni — Learn computer science like an engineer",
    description: "Learn DSA, OS, DBMS, Networks, and ML — interactively.",
    images: ["/logo.png"],
  },

  icons: {
    icon: [
      { url: "/logo.png", media: "(prefers-color-scheme: light)" },
      { url: "/logo.png", media: "(prefers-color-scheme: dark)" },
      { url: "/logo.png", type: "image/svg+xml" },
    ],
  },
};

// ========================================
// VIEWPORT
// ========================================
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f7f6f2" },
    { media: "(prefers-color-scheme: dark)", color: "#101216" },
  ],
};

// ========================================
// ROOT LAYOUT
// ========================================
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        suppressHydrationWarning
        className={`${inter.variable} ${spaceGrotesk.variable} ${jetbrainsMono.variable} min-h-screen overflow-x-hidden bg-background font-sans antialiased`}
      >
        <AuthProvider>
          <div className="flex min-h-screen flex-col">
            <main className="flex-1">{children}</main>
            <SiteFooter />
          </div>
        </AuthProvider>

        <Analytics />
      </body>
    </html>
  );
}

// ========================================
// FOOTER — hairline-rule structure, no card chrome
// ========================================
function SiteFooter() {
  return (
    <footer className="mt-auto border-t border-border bg-background">
      <div className="mx-auto max-w-6xl px-4 py-14 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-10 md:grid-cols-[1.3fr_1fr_1fr_1fr]">
          {/* BRAND */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary">
                <BookOpen className="h-4 w-4 text-primary-foreground" />
              </div>
              <span className="font-[family-name:var(--font-space-grotesk)] text-lg font-semibold text-foreground">
                Amni
              </span>
            </div>

            <p className="max-w-xs text-sm leading-relaxed text-muted-foreground">
              A structured, visual path through computer science —
              built for the way engineers actually learn.
            </p>

            <div className="flex gap-3 pt-1">
              <SocialLink href="https://github.com/RoBiul-Hasan-Jisan" icon={<Github className="h-4 w-4" />} label="GitHub" />
              <SocialLink href="https://www.linkedin.com/in/robiul-hasan-jisan-45766228b/" icon={<Linkedin className="h-4 w-4" />} label="LinkedIn" />
              <SocialLink href="mailto:contact@amni.dev" icon={<Mail className="h-4 w-4" />} label="Email" />
            </div>
          </div>

          <FooterSection
            title="Curriculum"
            links={[
              { href: "/learn", label: "Browse all topics" },
              { href: "/learn/data-structures/arrays", label: "Data structures" },
              { href: "/learn/operating-systems/process-thread", label: "Operating systems" },
              { href: "/learn/machine-learning/ml-intro", label: "Machine learning" },
              { href: "/learn/blind-75/two-sum", label: "Interview prep" },
            ]}
          />

          <FooterSection
            title="Account"
            links={[
              { href: "/sign-in", label: "Sign in" },
              { href: "/sign-up", label: "Create account" },
              { href: "/dashboard", label: "Your progress" },
            ]}
          />

          <FooterSection
            title="Project"
            links={[
              { href: "/", label: "Home" },
              { href: "/about", label: "About" },
            ]}
          />
        </div>

        <div className="mt-12 flex flex-col gap-3 border-t border-border pt-6 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
          <span>© {new Date().getFullYear()} Amni. All rights reserved.</span>
          <span className="meta-label">built with next.js · prisma · auth.js</span>
        </div>
      </div>
    </footer>
  );
}

function SocialLink({
  href,
  icon,
  label,
}: {
  href: string;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      aria-label={label}
      className="flex h-9 w-9 items-center justify-center rounded-md border border-border text-muted-foreground transition-colors hover:border-primary hover:text-primary"
    >
      {icon}
    </a>
  );
}

function FooterSection({
  title,
  links,
}: {
  title: string;
  links: { href: string; label: string }[];
}) {
  return (
    <div>
      <h3 className="mb-4 text-sm font-medium text-foreground">{title}</h3>
      <ul className="space-y-3">
        {links.map((link) => (
          <li key={link.href}>
            <Link
              href={link.href}
              className="text-sm text-muted-foreground transition-colors hover:text-primary"
            >
              {link.label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
