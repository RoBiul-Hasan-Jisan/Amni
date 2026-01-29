import { TopicContent } from "@/components/topic-content";
import { getSubtopicBySlug } from "@/lib/topics-data";
import { notFound } from "next/navigation";
import { Construction, ArrowRight } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";

interface PageProps {
  params: Promise<{
    topic: string;
    subtopic: string;
  }>;
}

export default async function TopicSubtopicPage({ params }: PageProps) {
  const { topic: topicSlug, subtopic: subtopicSlug } = await params;
  
  const result = getSubtopicBySlug(topicSlug, subtopicSlug);
  
  if (!result) {
    notFound();
  }

  const { topic, subtopic } = result;

  return (
    <TopicContent topic={topic} subtopic={subtopic}>
      <div className="space-y-8">
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="p-4 rounded-full bg-primary/10 text-primary mb-6">
            <Construction className="h-12 w-12" />
          </div>
          <h2 className="text-2xl font-bold mb-4 text-foreground">
            Content Coming Soon
          </h2>
          <p className="text-muted-foreground max-w-md mb-8">
            We are working hard to create comprehensive content for <strong className="text-foreground">{subtopic.title}</strong>. 
            Check back soon or explore other available topics.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4">
            <Link href="/learn/data-structures/arrays">
              <Button>
                Try Arrays Tutorial
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
            <Link href="/learn/operating-systems/process-thread">
              <Button variant="outline">
                Try Process vs Thread
              </Button>
            </Link>
          </div>
        </div>

        <div className="p-6 bg-muted/30 rounded-lg">
          <h3 className="font-semibold mb-4 text-foreground">What to Expect</h3>
          <ul className="space-y-3 text-muted-foreground">
            <li className="flex items-start gap-3">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">1</span>
              <span>Clear, beginner-friendly explanations of core concepts</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">2</span>
              <span>Interactive visualizations to help you understand</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">3</span>
              <span>Code examples in multiple programming languages</span>
            </li>
            <li className="flex items-start gap-3">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs shrink-0">4</span>
              <span>Practice quizzes and interview preparation tips</span>
            </li>
          </ul>
        </div>
      </div>
    </TopicContent>
  );
}
