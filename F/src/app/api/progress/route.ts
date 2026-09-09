import { NextResponse } from "next/server";
import { z } from "zod";

import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";

const progressSchema = z.object({
  topicId: z.string().min(1),
  subtopicId: z.string().min(1),
  completed: z.boolean().default(true),
});

// GET /api/progress — all completed lessons for the signed-in user
export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ progress: [] });
  }

  const progress = await prisma.progress.findMany({
    where: { userId: session.user.id, completed: true },
    select: { topicId: true, subtopicId: true, updatedAt: true },
  });

  return NextResponse.json({ progress });
}

// POST /api/progress — mark a lesson complete/incomplete
export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Sign in to save progress." }, { status: 401 });
  }

  const body = await request.json().catch(() => null);
  const parsed = progressSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid progress payload." }, { status: 400 });
  }

  const { topicId, subtopicId, completed } = parsed.data;

  const entry = await prisma.progress.upsert({
    where: {
      userId_topicId_subtopicId: { userId: session.user.id, topicId, subtopicId },
    },
    update: { completed },
    create: { userId: session.user.id, topicId, subtopicId, completed },
  });

  return NextResponse.json({ progress: entry });
}
