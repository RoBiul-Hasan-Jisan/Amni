import React from "react"
import type { Metadata, Viewport } from 'next' // Added Viewport type
import { Inter, JetBrains_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'

const _inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const _jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains" });

//  Define your base URL for consistent absolute links
const BASE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'https://cselearn.vercel.app/'; 

export const metadata: Metadata = {
  // MetadataBase makes all relative URLs (like '/og.png') absolute automatically
  metadataBase: new URL(BASE_URL),

  title: {
    default: 'CSE Learn - Master Computer Science',
    template: '%s | CSE Learn' // Allows child pages to have "Topic | CSE Learn"
    
  },
  description: 'Interactive learning platform for Computer Science & Engineering. Master Data Structures, Algorithms, and OS with visualizations.',
  
  // Expanded Keywords for better reach
 keywords: [
    // Core CS
    'computer science', 'cse', 'computer engineering',
    'data structures', 'algorithms', 'dsa', 'big o notation', 'time complexity',
    
    // Systems & Architecture
    'operating systems', 'os concepts', 'computer architecture', 
    'computer networks', 'distributed systems', 'linux', 'concurrency',
    
    // Software Engineering & Design
    'software engineering', 'system design', 'design patterns', 
    'object oriented programming', 'oop', 'solid principles', 'uml',
    
    // Data & Databases
    'database management systems', 'dbms', 'sql', 'normalization',
    
    // Theory & Math
    'discrete mathematics', 'automata theory', 'compiler design', 'graph theory',
    
    // Modern Technologies
    'web development', 'full stack', 'cloud computing', 
    'artificial intelligence', 'machine learning',
    
    // Career & Prep
    'interview preparation', 'coding interviews', 'technical interviews',
    'faang prep', 'competitive programming', 'leetcode style',
    
    // Learning Method
    'visual learning', 'interactive coding', 'visualization', 'roadmap'
  ],

  //Authors and Creator info
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
      <body className={`${_inter.variable} ${_jetbrainsMono.variable} font-sans antialiased`}>
        {children}
        <Analytics />
      </body>
    </html>
  )
}