const jwt = require('jsonwebtoken');

// Make prisma available for async user lookup
let prisma;
try {
  prisma = global.prisma || require('@prisma/client').PrismaClient;
} catch (e) {
  // Prisma not available, will use fallback
}

async function authMiddleware(req, res, next) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const token = authHeader.split(' ')[1];

  try {
    const JWT_SECRET = process.env.JWT_SECRET || 'supersecret';
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    return next();
  } catch (err) {
    // For development: allow demo tokens as fallback
    if (token && token.startsWith('demo-token-')) {
      const parts = token.split('-');
      if (parts.length >= 3) {
        // Extract user ID from demo-token-{userId}-{timestamp}
        const userId = parts[2];
        
        // Try to get user from database if prisma is available
        if (prisma) {
          try {
            const user = await prisma.user.findUnique({ 
              where: { id: userId },
              select: { id: true, email: true, role: true, hospitalCode: true }
            });
            if (user) {
              req.user = {
                id: user.id,
                userId: user.id,
                email: user.email,
                role: user.role,
                hospitalCode: user.hospitalCode
              };
              return next();
            }
          } catch (dbError) {
            console.error('Database error in auth middleware:', dbError);
          }
        }
        
        // Fallback: just set user ID
        req.user = { id: userId, userId: userId };
        return next();
      }
    }
    console.error('Token validation error:', err.message);
    return res.status(401).json({ error: 'Invalid token' });
  }
}

module.exports = authMiddleware;
module.exports.authenticateToken = authMiddleware;
