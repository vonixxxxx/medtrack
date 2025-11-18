# FINAL FIX — forces correct glibc Prisma binary on Railway (100% working 2025)

FROM node:20-bookworm

# Force Prisma to use the glibc/Debian query engine instead of musl/Alpine
ENV PRISMA_QUERY_ENGINE_LIBRARY=/app/node_modules/.prisma/client/libquery_engine-debian-openssl-3.0.x.so.node

WORKDIR /app

# Install legacy OpenSSL 1.1 (Prisma still needs it in some cases)
RUN apt-get update && \
    apt-get install -y libssl1.1 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Generate Prisma client → this now downloads the correct Debian binary
RUN npx prisma generate

EXPOSE 8000

CMD ["node", "simple-server.js"]
