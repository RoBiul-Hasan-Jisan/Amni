"use client";

import * as React from "react";
import { Check, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface CodeBlockProps {
  code: string;
  language: string;
  filename?: string;
  className?: string;
}

export function CodeBlock({ code, language, filename, className }: CodeBlockProps) {
  const [copied, setCopied] = React.useState(false);

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={cn("rounded-lg border border-border overflow-hidden", className)}>
      {filename && (
        <div className="flex items-center justify-between px-4 py-2 bg-muted/50 border-b border-border">
          <span className="text-sm text-muted-foreground font-mono">{filename}</span>
          <span className="text-xs text-muted-foreground uppercase">{language}</span>
        </div>
      )}
      <div className="relative">
        <pre className="p-4 overflow-x-auto bg-[var(--code-bg)] text-sm">
          <code className="font-mono text-foreground">{code}</code>
        </pre>
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-2 right-2 h-8 w-8 opacity-70 hover:opacity-100"
          onClick={copyToClipboard}
        >
          {copied ? (
            <Check className="h-4 w-4 text-success" />
          ) : (
            <Copy className="h-4 w-4" />
          )}
        </Button>
      </div>
    </div>
  );
}

interface MultiLanguageCodeProps {
  codes: {
    language: string;
    code: string;
    label: string;
  }[];
  className?: string;
}

export function MultiLanguageCode({ codes, className }: MultiLanguageCodeProps) {
  const [activeTab, setActiveTab] = React.useState(0);

  return (
    <div className={cn("rounded-lg border border-border overflow-hidden", className)}>
      <div className="flex border-b border-border bg-muted/50">
        {codes.map((item, index) => (
          <button
            key={item.language}
            onClick={() => setActiveTab(index)}
            className={cn(
              "px-4 py-2 text-sm font-medium transition-colors",
              activeTab === index
                ? "text-primary border-b-2 border-primary bg-background"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {item.label}
          </button>
        ))}
      </div>
      <CodeBlock
        code={codes[activeTab].code}
        language={codes[activeTab].language}
        className="border-0 rounded-none"
      />
    </div>
  );
}
