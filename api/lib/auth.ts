// Shared authentication utilities for Vercel serverless functions
import { VercelRequest } from '@vercel/node';

export interface AuthUser {
  id: string;
  email: string;
  name: string | null;
  role: string;
  hospitalCode: string;
}

export function getAuthToken(req: VercelRequest): string | null {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return null;
  }
  return authHeader.substring(7);
}

export function getUserIdFromToken(token: string): string | null {
  // Simplified token parsing - in production, verify JWT properly
  try {
    // For demo tokens like "demo-token-{userId}-{timestamp}"
    const parts = token.split('-');
    if (parts.length >= 3 && parts[0] === 'demo' && parts[1] === 'token') {
      return parts[2];
    }
    return null;
  } catch {
    return null;
  }
}

export async function getCurrentUser(req: VercelRequest): Promise<AuthUser | null> {
  const token = getAuthToken(req);
  if (!token) return null;
  
  const userId = getUserIdFromToken(token);
  if (!userId) return null;
  
  const { prisma } = await import('./prisma');
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: {
      id: true,
      email: true,
      name: true,
      role: true,
      hospitalCode: true,
    },
  });
  
  if (!user) return null;
  
  return {
    id: user.id,
    email: user.email,
    name: user.name,
    role: user.role,
    hospitalCode: user.hospitalCode,
  };
}







