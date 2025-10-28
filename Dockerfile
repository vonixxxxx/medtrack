# MedTrack Backend Production Dockerfile for AWS Elastic Beanstalk
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Install dependencies for building
RUN apk add --no-cache python3 make g++

# Copy package files
COPY backend/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application files
COPY backend/ .

# Generate Prisma Client
RUN npx prisma generate

# Create non-root user
RUN addgroup -g 1001 -S nodejs && adduser -S medtrack -u 1001
RUN chown -R medtrack:nodejs /app

# Switch to non-root user
USER medtrack

# Expose port (Elastic Beanstalk will map this)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/api/test-public || exit 1

# Start the application
CMD ["node", "simple-server.js"]
