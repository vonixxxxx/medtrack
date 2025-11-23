import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  resolve: {
    extensions: ['.js', '.jsx', '.ts', '.tsx', '.json'],
  },
  plugins: [
    react(),
    // Custom plugin to remove X-Frame-Options header
    {
      name: 'remove-x-frame-options',
      configureServer(server) {
        // Intercept all responses to remove X-Frame-Options
        server.middlewares.use((req, res, next) => {
          // Override setHeader to block X-Frame-Options
          const originalSetHeader = res.setHeader;
          res.setHeader = function(name, ...args) {
            if (name && name.toLowerCase() === 'x-frame-options') {
              // Don't set X-Frame-Options
              return this;
            }
            return originalSetHeader.apply(this, [name, ...args]);
          };
          
          // Remove X-Frame-Options if already set
          if (res.hasHeader('x-frame-options')) {
            res.removeHeader('x-frame-options');
          }
          if (res.hasHeader('X-Frame-Options')) {
            res.removeHeader('X-Frame-Options');
          }
          
          // Set CORS headers
          res.setHeader('Access-Control-Allow-Origin', '*');
          res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
          res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
          
          // Intercept writeHead to remove X-Frame-Options from headers
          const originalWriteHead = res.writeHead;
          res.writeHead = function(statusCode, statusMessage, headers) {
            if (typeof statusMessage === 'object') {
              headers = statusMessage;
              statusMessage = undefined;
            }
            if (headers) {
              delete headers['x-frame-options'];
              delete headers['X-Frame-Options'];
            }
            if (statusMessage) {
              return originalWriteHead.call(this, statusCode, statusMessage, headers);
            }
            return originalWriteHead.call(this, statusCode, headers);
          };
          
          next();
        });
      },
    },
  ],
  // Don't set base path - Next.js proxy strips /app/ prefix
  // React Router basename handles the routing
  server: {
    port: 3000,
    cors: true,
    hmr: {
      protocol: 'ws',
      host: 'localhost',
      port: 3000,
    },
  },
});
