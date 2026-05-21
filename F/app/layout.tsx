import React from "react";
import type { Metadata, Viewport } from "next";

import { Inter, JetBrains_Mono } from "next/font/google";

import { Analytics } from "@vercel/analytics/next";

import Link from "next/link";

import {
  Github,
  Linkedin,
  Twitter,
  Mail,
  Heart,
  BookOpen,
} from "lucide-react";

import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
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
const BASE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ||
  "https://cselearn.vercel.app";

// ========================================
// METADATA
// ========================================
export const metadata: Metadata = {
  metadataBase: new URL(BASE_URL),

  title: {
    default: "CSE Learn - Master Computer Science",
    template: "%s | CSE Learn",
  },

  description:
    "Interactive learning platform for Computer Science & Engineering. Learn Data Structures, Algorithms, Operating Systems, DBMS, Networks, and more with visual explanations.",

  keywords: [
    "computer science",
    "cse",
    "data structures",
    "algorithms",
    "dsa",
    "operating systems",
    "dbms",
    "computer networks",
    "system design",
    "competitive programming",
    "machine learning",
    "interview preparation",
    "software engineering",
    "web development",
    "coding interviews",
    "visual learning",
  ],

  authors: [
    {
      name: "CSE Learn Team",
      url: "https://robiulhasanjisan.vercel.app",
    },
  ],

  creator: "CSE Learn",

  publisher: "CSE Learn",

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
    title: "CSE Learn - Master Computer Science",

    description:
      "Master Computer Science concepts with interactive visualizations and structured learning.",

    url: BASE_URL,

    siteName: "CSE Learn",

    locale: "en_US",

    type: "website",

    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "CSE Learn Preview",
      },
    ],
  },

  twitter: {
    card: "summary_large_image",

    title: "CSE Learn - Master Computer Science",

    description:
      "Learn DSA, OS, DBMS, Networks, and more interactively.",

    creator: "@cselearn_official",

    images: ["/og-image.png"],
  },

  icons: {
    icon: [
      {
        url: "/icon-light-32x32.png",
        media: "(prefers-color-scheme: light)",
      },

      {
        url: "/icon-dark-32x32.png",
        media: "(prefers-color-scheme: dark)",
      },

      {
        url: "/icon.svg",
        type: "image/svg+xml",
      },
    ],

    apple: "/apple-icon.png",
  },
};

// ========================================
// VIEWPORT
// ========================================
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,

  themeColor: [
    {
      media: "(prefers-color-scheme: light)",
      color: "#ffffff",
    },

    {
      media: "(prefers-color-scheme: dark)",
      color: "#0a0a0a",
    },
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
    <html
      lang="en"
      suppressHydrationWarning
    >
      <body
        suppressHydrationWarning
        className={`${inter.variable} ${jetbrainsMono.variable} min-h-screen overflow-x-hidden bg-background font-sans antialiased`}
      >
        {/* PAGE WRAPPER */}
        <div className="flex min-h-screen flex-col">

          {/* MAIN CONTENT */}
          <main className="flex-1">
            {children}
          </main>

          {/* ========================================
              FOOTER
          ======================================== */}
          <footer className="mt-auto border-t border-border bg-card">

            <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">

              {/* ========================================
                  MAIN GRID
              ======================================== */}
              <div className="mb-10 grid grid-cols-1 gap-10 md:grid-cols-2 lg:grid-cols-4">

                {/* BRAND */}
                <div className="space-y-4">

                  <div className="flex items-center gap-2">

                    <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary">
                      <BookOpen className="h-4 w-4 text-primary-foreground" />
                    </div>

                    <span className="text-lg font-bold text-foreground">
                      CSE Learn
                    </span>
                  </div>

                  <p className="max-w-sm text-sm leading-relaxed text-muted-foreground">
                    Interactive platform for mastering
                    Computer Science through visual
                    learning and interview preparation.
                  </p>

                  {/* SOCIALS */}
                  <div className="flex flex-wrap gap-2">

                    <SocialLink
                      href="https://github.com/RoBiul-Hasan-Jisan"
                      icon={
                        <Github className="h-4 w-4" />
                      }
                    />

                    <SocialLink
                      href="https://www.linkedin.com/in/robiul-hasan-jisan-45766228b/"
                      icon={
                        <Linkedin className="h-4 w-4" />
                      }
                    />

                    <SocialLink
                      href="https://twitter.com/"
                      icon={
                        <Twitter className="h-4 w-4" />
                      }
                    />

                    <SocialLink
                      href="mailto:contact@cselearn.com"
                      icon={
                        <Mail className="h-4 w-4" />
                      }
                    />

                  </div>
                </div>

                {/* QUICK LINKS */}
                <FooterSection
                  title="Quick Links"
                  links={[
                    {
                      href: "/",
                      label: "Home",
                    },
                    {
                      href: "/learn",
                      label: "Browse Topics",
                    },
                    {
                      href:
                        "/learn/data-structures/arrays",
                      label: "Start Learning",
                    },
                    {
                      href: "/about",
                      label: "About",
                    },
                    {
                      href: "/contact",
                      label: "Contact",
                    },
                  ]}
                />

                {/* POPULAR TOPICS */}
                <FooterSection
                  title="Popular Topics"
                  links={[
                    {
                      href: "/learn/data-structures",
                      label: "Data Structures",
                    },
                    {
                      href: "/learn/algorithms",
                      label: "Algorithms",
                    },
                    {
                      href:
                        "/learn/operating-systems",
                      label:
                        "Operating Systems",
                    },
                    {
                      href: "/learn/databases",
                      label: "Databases",
                    },
                    {
                      href:
                        "/learn/computer-networks",
                      label:
                        "Computer Networks",
                    },
                  ]}
                />

                {/* RESOURCES */}
                <FooterSection
                  title="Resources"
                  links={[
                    {
                      href: "/interview",
                      label:
                        "Interview Preparation",
                    },
                    {
                      href: "/practice",
                      label:
                        "Practice Problems",
                    },
                    {
                      href: "/guides",
                      label: "Study Guides",
                    },
                    {
                      href: "/cheatsheets",
                      label: "Cheat Sheets",
                    },
                    {
                      href: "/blog",
                      label: "Blog",
                    },
                  ]}
                />

              </div>

              {/* ========================================
                  NEWSLETTER
              ======================================== */}
              <div className="mb-10 border-t border-border pt-8">

                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 lg:items-center">

                  <div>
                    <h3 className="mb-2 text-lg font-semibold text-foreground">
                      Stay Updated
                    </h3>

                    <p className="text-sm text-muted-foreground">
                      Get tutorials, interview tips,
                      and resources directly in your
                      inbox.
                    </p>
                  </div>

                  <div className="flex flex-col gap-3 sm:flex-row">

                    <input
                      type="email"
                      placeholder="Enter your email"
                      className="h-11 flex-1 rounded-xl border border-border bg-background px-4 text-sm outline-none transition-all focus:ring-2 focus:ring-primary"
                    />

                    <button className="h-11 rounded-xl bg-primary px-5 text-sm font-medium text-primary-foreground transition-all hover:bg-primary/90">
                      Subscribe
                    </button>

                  </div>
                </div>
              </div>

              {/* ========================================
                  BOTTOM BAR
              ======================================== */}
              <div className="flex flex-col gap-4 border-t border-border pt-8 md:flex-row md:items-center md:justify-between">

                {/* COPYRIGHT */}
                <div className="text-center text-sm text-muted-foreground md:text-left">
                  © {new Date().getFullYear()} CSE
                  Learn. All rights reserved.
                </div>

                {/* MADE WITH */}
                <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">

                  <span>Made with</span>

                  <Heart className="h-4 w-4 fill-red-500 text-red-500" />

                  <span>
                    for students worldwide
                  </span>

                </div>

                {/* LEGAL */}
                <div className="flex flex-wrap items-center justify-center gap-4 text-xs text-muted-foreground md:justify-end">

                  <Link
                    href="/privacy"
                    className="transition-colors hover:text-primary"
                  >
                    Privacy Policy
                  </Link>

                  <Link
                    href="/terms"
                    className="transition-colors hover:text-primary"
                  >
                    Terms
                  </Link>

                  <Link
                    href="/cookies"
                    className="transition-colors hover:text-primary"
                  >
                    Cookies
                  </Link>

                </div>
              </div>
            </div>
          </footer>
        </div>

        <Analytics />
      </body>
    </html>
  );
}

// ========================================
// SOCIAL LINK
// ========================================
function SocialLink({
  href,
  icon,
}: {
  href: string;
  icon: React.ReactNode;
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      aria-label="Social Link"
      className="flex h-10 w-10 items-center justify-center rounded-xl bg-muted text-muted-foreground transition-all duration-200 hover:scale-110 hover:bg-primary/10 hover:text-primary"
    >
      {icon}
    </a>
  );
}

// ========================================
// FOOTER SECTION
// ========================================
function FooterSection({
  title,
  links,
}: {
  title: string;

  links: {
    href: string;
    label: string;
  }[];
}) {
  return (
    <div>

      <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-foreground">
        {title}
      </h3>

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