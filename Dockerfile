FROM node:20-bookworm

WORKDIR /app

RUN apt-get update && \
    apt-get install -y openssl libssl-dev build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Copy only package files from backend folder
COPY backend/package*.json ./
COPY backend/package-lock.json* ./
COPY backend/.npmrc* ./

RUN npm ci --omit=dev

# Copy the actual backend code
COPY backend/ .

RUN npx prisma generate

EXPOSE 8080
ENV PORT=8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1

CMD ["npm", "start"]
