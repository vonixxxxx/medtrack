# Use official Node.js runtime as the base image
FROM node:18-alpine

# Set working directory in the container
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy the rest of the application code
COPY . .

# Generate Prisma client
RUN npx prisma generate

# Create a non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S medtrack -u 1001

# Change ownership of the app directory to the nodejs user
RUN chown -R medtrack:nodejs /app

# Switch to the non-root user
USER medtrack

# Expose the port the app runs on
EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:4000/health || exit 1

# Command to run the application
CMD ["npm", "start"]
