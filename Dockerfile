FROM node:20-bookworm

WORKDIR /app

# Install OpenSSL 3 + build tools (Prisma needs these on Debian 12)
RUN apt-get update && \
    apt-get install -y openssl libssl-dev build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm ci --only=production

COPY . .

# Generate Prisma client + ensure query engine is built correctly
RUN npx prisma generate

EXPOSE 8080
ENV PORT=8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1

CMD ["npm", "start"]
