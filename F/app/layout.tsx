import React from "react"
import type { Metadata, Viewport } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import Link from 'next/link'
import {
  Github,
  Linkedin,
  Twitter,
  Mail,
  Heart,
  BookOpen,
} from 'lucide-react'
import './globals.css'

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains" });

// Define your base URL for consistent absolute links
const BASE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'https://cselearn.vercel.app/'; 

export const metadata: Metadata = {
  metadataBase: new URL(BASE_URL),
  title: {
    default: 'CSE Learn - Master Computer Science',
    template: '%s | CSE Learn'
  },
  description: 'Interactive learning platform for Computer Science & Engineering. Master Data Structures, Algorithms, and OS with visualizations.',
  keywords: [
    'computer science', 'cse', 'computer engineering',
    'data structures', 'algorithms', 'dsa', 'big o notation', 'time complexity',
    'operating systems', 'os concepts', 'computer architecture', 
    'computer networks', 'distributed systems', 'linux', 'concurrency',
    'software engineering', 'system design', 'design patterns', 
    'object oriented programming', 'oop', 'solid principles', 'uml',
    'database management systems', 'dbms', 'sql', 'normalization',
    'discrete mathematics', 'automata theory', 'compiler design', 'graph theory',
    'web development', 'full stack', 'cloud computing', 
    'artificial intelligence', 'machine learning',
    'interview preparation', 'coding interviews', 'technical interviews',
    'faang prep', 'competitive programming', 'leetcode style',
    'visual learning', 'interactive coding', 'visualization', 'roadmap'
  ],
  authors: [{ name: 'CSE Learn Team', url: 'https://robiulhasanjisan.vercel.app/' }],
  creator: 'CSE Learn',
  publisher: 'CSE Learn',
  openGraph: {
    title: 'CSE Learn - Master Computer Science',
    description: 'The interactive way to master Computer Science concepts. visualizations, quizzes, and interview prep.',
    url: BASE_URL,
    siteName: 'CSE Learn',
    locale: 'en_US',
    type: 'website',
    images: [
      {
        url: '/og-image.png', 
        width: 1200,
        height: 630,
        alt: 'CSE Learn Dashboard Preview',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'CSE Learn - Master Computer Science',
    description: 'Master Data Structures & Algorithms with interactive visualizations.',
    creator: '@cselearn_official', 
    images: ['/og-image.png'],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  icons: {
    icon: [
      { url: '/icon-light-32x32.png', media: '(prefers-color-scheme: light)' },
      { url: '/icon-dark-32x32.png', media: '(prefers-color-scheme: dark)' },
      { url: '/icon.svg', type: 'image/svg+xml' },
    ],
    apple: '/apple-icon.png',
  },
}

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0a0a0a' },
  ],
  width: 'device-width',
  initialScale: 1,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased min-h-screen flex flex-col`}>
        {children}
        
        {/* Global Footer - Appears on all pages */}
        <footer className="bg-card border-t border-border mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            {/* Main Footer Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-8">
              {/* Brand Section */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
                    <BookOpen className="h-4 w-4 text-primary-foreground" />
                  </div>
                  <span className="font-bold text-lg text-foreground">CSE Learn</span>
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Master Computer Science through interactive learning, visual explanations, and comprehensive interview preparation.
                </p>
                <div className="flex gap-2">
                  <SocialLink href="https://github.com/RoBiul-Hasan-Jisan" icon={<Github className="h-4 w-4" />} />
                  <SocialLink href="https://www.linkedin.com/in/robiul-hasan-jisan-45766228b/" icon={<Linkedin className="h-4 w-4" />} />
                  <SocialLink href="https://twitter.com/" icon={<Twitter className="h-4 w-4" />} />
                  <SocialLink href="mailto:contact@cselearn.com" icon={<Mail className="h-4 w-4" />} />
                </div>
              </div>

              {/* Quick Links */}
              <div>
                <h3 className="font-semibold text-foreground mb-4">Quick Links</h3>
                <ul className="space-y-2">
                  <FooterLink href="/">Home</FooterLink>
                  <FooterLink href="/learn">Browse Topics</FooterLink>
                  <FooterLink href="/learn/data-structures/arrays">Start Learning</FooterLink>
                  <FooterLink href="/about">About Us</FooterLink>
                  <FooterLink href="/contact">Contact</FooterLink>
                </ul>
              </div>

              {/* Popular Topics */}
              <div>
                <h3 className="font-semibold text-foreground mb-4">Popular Topics</h3>
                <ul className="space-y-2">
                  <FooterLink href="/learn/data-structures">Data Structures</FooterLink>
                  <FooterLink href="/learn/algorithms">Algorithms</FooterLink>
                  <FooterLink href="/learn/operating-systems">Operating Systems</FooterLink>
                  <FooterLink href="/learn/databases">Databases</FooterLink>
                  <FooterLink href="/learn/computer-networks">Computer Networks</FooterLink>
                </ul>
              </div>

              {/* Resources */}
              <div>
                <h3 className="font-semibold text-foreground mb-4">Resources</h3>
                <ul className="space-y-2">
                  <FooterLink href="/interview">Interview Preparation</FooterLink>
                  <FooterLink href="/practice">Practice Problems</FooterLink>
                  <FooterLink href="/guides">Study Guides</FooterLink>
                  <FooterLink href="/cheatsheets">Cheat Sheets</FooterLink>
                  <FooterLink href="/blog">Blog</FooterLink>
                </ul>
              </div>
            </div>

            {/* Newsletter Section */}
            <div className="border-t border-border pt-8 mb-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-center">
                <div>
                  <h3 className="font-semibold text-foreground mb-2">Stay Updated</h3>
                  <p className="text-sm text-muted-foreground">
                    Get the latest tutorials, interview tips, and resources delivered to your inbox.
                  </p>
                </div>
                <div className="flex gap-2">
                  <input
                    type="email"
                    placeholder="Enter your email"
                    className="flex-1 px-4 py-2 text-sm rounded-lg border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                  />
                  <button className="px-4 py-2 text-sm font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors">
                    Subscribe
                  </button>
                </div>
              </div>
            </div>

            {/* Bottom Bar */}
            <div className="pt-8 border-t border-border">
              <div className="flex flex-col md:flex-row justify-between items-center gap-4">
                <div className="text-sm text-muted-foreground">
                  © {new Date().getFullYear()} CSE Learn. All rights reserved.
                </div>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <span>Made with</span>
                  <Heart className="h-4 w-4 text-red-500 fill-red-500" />
                  <span>for students worldwide</span>
                </div>
                <div className="flex gap-4 text-xs text-muted-foreground">
                  <Link href="/privacy" className="hover:text-primary transition-colors">
                    Privacy Policy
                  </Link>
                  <Link href="/terms" className="hover:text-primary transition-colors">
                    Terms of Service
                  </Link>
                  <Link href="/cookies" className="hover:text-primary transition-colors">
                    Cookie Policy
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </footer>
        
        <Analytics />
      </body>
    </html>
  )
}

// Helper Components
function SocialLink({ href, icon }: { href: string; icon: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="p-2 rounded-lg bg-muted hover:bg-primary/10 text-muted-foreground hover:text-primary transition-all duration-200 hover:scale-110"
      aria-label="Social link"
    >
      {icon}
    </a>
  );
}

function FooterLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <li>
      <Link href={href} className="text-sm text-muted-foreground hover:text-primary transition-colors">
        {children}
      </Link>
    </li>
  );
}