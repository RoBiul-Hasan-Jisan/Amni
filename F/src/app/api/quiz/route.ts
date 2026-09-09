import { NextResponse } from "next/server";
import { z } from "zod";

import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";

const quizSchema = z.object({
  topicId: z.string().min(1),
  subtopicId: z.string().min(1),
  score: z.number().int().min(0),
  total: z.number().int().min(1),
});

// GET /api/quiz — recent quiz attempts for the signed-in user
export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ attempts: [] });
  }

  const attempts = await prisma.quizAttempt.findMany({
    where: { userId: session.user.id },
    orderBy: { createdAt: "desc" },
    take: 50,
  });

  return NextResponse.json({ attempts });
}

// POST /api/quiz — record a quiz submission
export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Sign in to save your score." }, { status: 401 });
  }

  const body = await request.json().catch(() => null);
  const parsed = quizSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid quiz payload." }, { status: 400 });
  }

  const attempt = await prisma.quizAttempt.create({
    data: { userId: session.user.id, ...parsed.data },
  });

  return NextResponse.json({ attempt }, { status: 201 });
}
