import { prisma } from "./prisma";

interface Post {
  id: string;

  content: string;

  createdAt: Date;

  userId: string;

  user: {
    id: string;

    username: string;

    name: string;

    _count: {
      followers: number;
    };

    followers: Array<{ id: string }>;
  };

  likes: Array<{ id: string }>;

  _count: {
    likes: number;

    comments: number;
  };
}

function calculateScore(
  post: Post,
  now: number,
  likedAuthorIds: Set<string>,
  followingIds: Set<string>
): { score: number; breakdown: string } {
  // Signal 1
  const isFollowing = followingIds.has(post.userId);
  const followingScore = isFollowing ? 100 : 0;

  // Signal 2
  const hasLikedFromAuthor = likedAuthorIds.has(post.userId);
  const interactionScore = hasLikedFromAuthor ? 50 : 0;

  // Signal 3
  const ageInHours = (now - post.createdAt.getTime()) / (1000 * 60 * 60);
  const recencyScore = 10 * Math.exp(-ageInHours / 24);

  // Signal 4
  const totalEngagement = post._count.likes + post._count.comments * 2;
  const engagementScore = Math.log(totalEngagement + 1) * 3;

  // Signal 5
  const velocity = totalEngagement / Math.max(ageInHours, 0.5);
  const velocityScore = Math.log(velocity + 1) * 2;

  const WEIGHTS = {
    following: 3,
    interaction: 2,
    recency: 1,
    engagement: 1,
    velocity: 1,
  };

  const totalScore =
    followingScore * WEIGHTS.following + // 300 points max
    interactionScore * WEIGHTS.interaction + // 100 points max
    recencyScore * WEIGHTS.recency + // ~10 points max
    engagementScore * WEIGHTS.engagement + // ~10 points max
    velocityScore * WEIGHTS.velocity; // ~5 points max

  // Build breakdown string

  const reasons = [];

  if (followingScore > 0)
    reasons.push(
      `Following (+${(followingScore * WEIGHTS.following).toFixed(0)})`
    );

  if (interactionScore > 0)
    reasons.push(
      `Liked before (+${(interactionScore * WEIGHTS.interaction).toFixed(0)})`
    );

  reasons.push(`Recent (+${(recencyScore * WEIGHTS.recency).toFixed(1)})`);

  if (engagementScore > 2)
    reasons.push(
      `Popular (+${(engagementScore * WEIGHTS.engagement).toFixed(1)})`
    );

  if (velocityScore > 1)
    reasons.push(`Viral (+${(velocityScore * WEIGHTS.velocity).toFixed(1)})`);

  const breakdown = reasons.length > 0 ? reasons.join(" • ") : "Recommended";

  return { score: totalScore, breakdown };
}

export async function generateForYouFeed(userId: string) {
  const now = Date.now();
  const threeDaysAgo = new Date(now - 3 * 24 * 60 * 60 * 1000);
  const oneDayAgo = new Date(now - 24 * 60 * 60 * 1000);
  const oneHourAgo = new Date(now - 60 * 60 * 1000);

  const userFollows = await prisma.follow.findMany({
    where: { followerId: userId },
    select: { followingId: true },
  });

  const followingIds = new Set(userFollows.map((f) => f.followingId));

  const userLikes = await prisma.like.findMany({
    where: { userId },
    select: {
      postId: true,
      post: { select: { userId: true } },
    },
  });

  const likedPostIds = new Set(userLikes.map((like) => like.postId));
  const likedAuthorIds = new Set(userLikes.map((like) => like.post.userId));

  const [followingPosts, popularPosts, recentPosts] = await Promise.all([
    prisma.post.findMany({
      where: {
        user: {
          followers: {
            some: { followerId: userId },
          },
        },

        createdAt: { gte: threeDaysAgo },
        NOT: { id: { in: Array.from(likedPostIds) } },
      },
      take: 200,
      orderBy: { createdAt: "desc" },
      include: {
        user: {
          include: {
            _count: { select: { followers: true } },
            followers: {
              where: { followerId: userId },
            },
          },
        },

        likes: {
          where: { userId },
          select: { id: true },
        },
        _count: {
          select: { likes: true, comments: true },
        },
      },
    }),

    prisma.post.findMany({
      where: {
        createdAt: { gte: oneDayAgo },
        NOT: { id: { in: Array.from(likedPostIds) } },
      },
      take: 100,
      orderBy: [
        { likes: { _count: "desc" } },
        { comments: { _count: "desc" } },
      ],

      include: {
        user: {
          include: {
            _count: { select: { followers: true } },
            followers: {
              where: { followerId: userId },
            },
          },
        },
        likes: {
          where: { userId },
          select: { id: true },
        },
        _count: {
          select: { likes: true, comments: true },
        },
      },
    }),

    prisma.post.findMany({
      where: {
        createdAt: { gte: oneHourAgo },
        NOT: { id: { in: Array.from(likedPostIds) } },
      },
      take: 100,
      orderBy: { createdAt: "desc" },
      include: {
        user: {
          include: {
            _count: { select: { followers: true } },
            followers: {
              where: { followerId: userId },
            },
          },
        },
        likes: {
          where: { userId },
          select: { id: true },
        },
        _count: {
          select: { likes: true, comments: true },
        },
      },
    }),
  ]);

  const allPosts = [...followingPosts, ...popularPosts, ...recentPosts];
  const uniquePostMap = new Map<string, Post>();

  allPosts.forEach((post) => {
    if (!uniquePostMap.has(post.id)) {
      uniquePostMap.set(post.id, post as Post);
    }
  });

  const candidates = Array.from(uniquePostMap.values());

  // Score and rank
  const scoredPosts = candidates.map(post => {
    const {score, breakdown} = calculateScore(post, now, likedAuthorIds, followingIds)
    return{
      ...post,
      score,
      scoreBreakdown: breakdown
    }
  })

  return scoredPosts.sort((a, b) => b.score - a.score).slice(0, 50);
}
