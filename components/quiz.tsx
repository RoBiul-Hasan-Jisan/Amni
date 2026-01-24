"use client";

import * as React from "react";
import { CheckCircle2, XCircle, RotateCcw, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface QuizQuestion {
  id: number;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

interface QuizProps {
  questions: QuizQuestion[];
  title?: string;
}

export function Quiz({ questions, title = "Quick Quiz" }: QuizProps) {
  const [currentQuestion, setCurrentQuestion] = React.useState(0);
  const [selectedAnswer, setSelectedAnswer] = React.useState<number | null>(null);
  const [showResult, setShowResult] = React.useState(false);
  const [score, setScore] = React.useState(0);
  const [answers, setAnswers] = React.useState<number[]>([]);

  const question = questions[currentQuestion];
  const isCorrect = selectedAnswer === question.correctAnswer;

  const handleAnswer = (index: number) => {
    if (selectedAnswer !== null) return;
    setSelectedAnswer(index);
    setAnswers([...answers, index]);
    if (index === question.correctAnswer) {
      setScore(score + 1);
    }
  };

  const nextQuestion = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
      setSelectedAnswer(null);
    } else {
      setShowResult(true);
    }
  };

  const reset = () => {
    setCurrentQuestion(0);
    setSelectedAnswer(null);
    setShowResult(false);
    setScore(0);
    setAnswers([]);
  };

  if (showResult) {
    const percentage = Math.round((score / questions.length) * 100);
    return (
      <div className="p-6 bg-card rounded-lg border border-border">
        <h3 className="text-xl font-semibold mb-4 text-foreground">{title} - Results</h3>
        <div className="text-center py-8">
          <div
            className={cn(
              "inline-flex items-center justify-center w-24 h-24 rounded-full text-3xl font-bold mb-4",
              percentage >= 70
                ? "bg-success/20 text-success"
                : percentage >= 50
                ? "bg-warning/20 text-warning"
                : "bg-destructive/20 text-destructive"
            )}
          >
            {percentage}%
          </div>
          <p className="text-lg text-foreground mb-2">
            You scored {score} out of {questions.length}
          </p>
          <p className="text-muted-foreground mb-6">
            {percentage >= 70
              ? "Excellent work! You have a solid understanding."
              : percentage >= 50
              ? "Good effort! Review the topics you missed."
              : "Keep practicing! Review the material and try again."}
          </p>
          <Button onClick={reset}>
            <RotateCcw className="h-4 w-4 mr-2" />
            Try Again
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 bg-card rounded-lg border border-border">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-semibold text-foreground">{title}</h3>
        <span className="text-sm text-muted-foreground">
          Question {currentQuestion + 1} of {questions.length}
        </span>
      </div>

      <div className="mb-2 h-2 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{
            width: `${((currentQuestion + 1) / questions.length) * 100}%`,
          }}
        />
      </div>

      <div className="py-6">
        <p className="text-lg font-medium mb-6 text-foreground">{question.question}</p>

        <div className="space-y-3">
          {question.options.map((option, index) => (
            <button
              key={index}
              onClick={() => handleAnswer(index)}
              disabled={selectedAnswer !== null}
              className={cn(
                "w-full p-4 text-left rounded-lg border-2 transition-all",
                selectedAnswer === null
                  ? "border-border hover:border-primary hover:bg-muted/50"
                  : index === question.correctAnswer
                  ? "border-success bg-success/10"
                  : selectedAnswer === index
                  ? "border-destructive bg-destructive/10"
                  : "border-border opacity-50"
              )}
            >
              <div className="flex items-center justify-between">
                <span className="text-foreground">{option}</span>
                {selectedAnswer !== null && index === question.correctAnswer && (
                  <CheckCircle2 className="h-5 w-5 text-success" />
                )}
                {selectedAnswer === index && index !== question.correctAnswer && (
                  <XCircle className="h-5 w-5 text-destructive" />
                )}
              </div>
            </button>
          ))}
        </div>

        {selectedAnswer !== null && (
          <div
            className={cn(
              "mt-6 p-4 rounded-lg",
              isCorrect ? "bg-success/10" : "bg-destructive/10"
            )}
          >
            <p className="font-medium text-foreground mb-1">
              {isCorrect ? "Correct!" : "Incorrect"}
            </p>
            <p className="text-sm text-muted-foreground">{question.explanation}</p>
          </div>
        )}
      </div>

      {selectedAnswer !== null && (
        <div className="flex justify-end">
          <Button onClick={nextQuestion}>
            {currentQuestion < questions.length - 1 ? (
              <>
                Next Question
                <ArrowRight className="h-4 w-4 ml-2" />
              </>
            ) : (
              "See Results"
            )}
          </Button>
        </div>
      )}
    </div>
  );
}
