FROM node:20-bookworm

WORKDIR /app

# Install system dependencies for Prisma + PostgreSQL
RUN apt-get update && \
    apt-get install -y openssl libssl-dev build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Copy package files + lockfile + npmrc from backend directory
COPY backend/package*.json ./
COPY backend/package-lock.json backend/.npmrc* ./

# Production install using lockfile (replaces --only=production)
RUN npm ci --omit=dev

# Copy backend source code
COPY backend/ .

# Generate Prisma client (critical for production)
RUN npx prisma generate

# Expose port
EXPOSE 8080
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1

# Start server
CMD ["npm", "start"]
