import { NextResponse } from "next/server";
import { z } from "zod";

import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";

const bookmarkSchema = z.object({
  topicId: z.string().min(1),
  subtopicId: z.string().min(1),
});

// GET /api/bookmarks
export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ bookmarks: [] });
  }

  const bookmarks = await prisma.bookmark.findMany({
    where: { userId: session.user.id },
    select: { topicId: true, subtopicId: true },
  });

  return NextResponse.json({ bookmarks });
}

// POST /api/bookmarks — toggle a bookmark on/off
export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Sign in to save bookmarks." }, { status: 401 });
  }

  const body = await request.json().catch(() => null);
  const parsed = bookmarkSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid bookmark payload." }, { status: 400 });
  }

  const { topicId, subtopicId } = parsed.data;
  const userId = session.user.id;

  const existing = await prisma.bookmark.findUnique({
    where: { userId_topicId_subtopicId: { userId, topicId, subtopicId } },
  });

  if (existing) {
    await prisma.bookmark.delete({ where: { id: existing.id } });
    return NextResponse.json({ bookmarked: false });
  }

  await prisma.bookmark.create({ data: { userId, topicId, subtopicId } });
  return NextResponse.json({ bookmarked: true });
}
