const jwt = require('jsonwebtoken');

// Middleware to check if user has required role
const requireRole = (allowedRoles) => {
  return (req, res, next) => {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const token = authHeader.split(' ')[1];

    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET || 'supersecret');
      
      // Check if user has required role
      if (!decoded.role || !allowedRoles.includes(decoded.role)) {
        return res.status(403).json({ error: 'Insufficient permissions' });
      }
      
      req.user = decoded;
      next();
    } catch (err) {
      return res.status(401).json({ error: 'Invalid token' });
    }
  };
};

// Specific role middlewares
const requirePatient = requireRole(['patient']);
const requireClinician = requireRole(['clinician']);
const requireAnyRole = requireRole(['patient', 'clinician']);

module.exports = {
  requireRole,
  requirePatient,
  requireClinician,
  requireAnyRole
};


