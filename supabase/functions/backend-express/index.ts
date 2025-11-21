// Supabase Edge Function wrapper for Express.js backend
// This bridges your existing Express app to Supabase Edge Functions

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

// Enable Node.js compatibility for require() and CommonJS
// @deno-types="https://deno.land/x/node@18.15.0/types/index.d.ts"
import { createRequire } from "https://deno.land/std@0.168.0/node/module.ts";
const require = createRequire(import.meta.url);

// Polyfill process.env from Deno.env and Supabase secrets
if (typeof globalThis.process === "undefined") {
  // @ts-ignore
  globalThis.process = {
    env: {
      ...Deno.env.toObject(),
      // Supabase automatically injects secrets as env vars
      // Access them via Deno.env.get("SECRET_NAME")
    },
    version: "v20.0.0",
    versions: {},
    platform: Deno.build.os,
    nextTick: (fn: () => void) => setTimeout(fn, 0),
    cwd: () => Deno.cwd(),
    exit: (code?: number) => Deno.exit(code || 0),
  };
}

// Set NODE_ENV
if (!process.env.NODE_ENV) {
  process.env.NODE_ENV = Deno.env.get("NODE_ENV") || "production";
}

// Load the Express app from simple-server.js
// IMPORTANT: The backend folder must be accessible from this function
// During deployment, ensure backend files are included (see README)
serve(async (req: Request) => {
  try {
    const path = require("node:path");
    const fs = require("node:fs");
    
    // Resolve backend path - try multiple possible locations
    let backendPath: string;
    const currentDir = Deno.cwd();
    
    // In Supabase Edge Functions, files are bundled in the function directory
    // Backend files are copied to the function root during deployment
    // Try multiple locations
    let simpleServerPath: string;
    
    // Option 1: Backend files in function root (deployed)
    const rootPath = path.resolve(currentDir, "simple-server.js");
    // Option 2: Backend subdirectory (local dev)
    const backendPath = path.resolve(currentDir, "backend/simple-server.js");
    // Option 3: Parent backend directory
    const parentBackend = path.resolve(currentDir, "../../backend/simple-server.js");
    
    if (fs.existsSync(rootPath)) {
      simpleServerPath = rootPath;
    } else if (fs.existsSync(backendPath)) {
      simpleServerPath = backendPath;
    } else if (fs.existsSync(parentBackend)) {
      simpleServerPath = parentBackend;
    } else {
      throw new Error(`Express app not found. Tried: ${rootPath}, ${backendPath}, ${parentBackend}`);
    }
    
    // Verify the file exists
    if (!fs.existsSync(simpleServerPath)) {
      throw new Error(`Express app not found at: ${simpleServerPath}. Run deploy.sh to copy backend files.`);
    }
    
    console.log(`Loading Express app from: ${simpleServerPath}`);
    
    // Load the Express app
    const app = require(simpleServerPath);
    
    // Convert Deno Request to Express req/res and handle
    return await handleExpressRequest(app, req);
    
  } catch (error: any) {
    console.error("Edge function error:", error);
    console.error("Error stack:", error.stack);
    return new Response(
      JSON.stringify({ 
        error: error.message, 
        stack: process.env.NODE_ENV === "development" ? error.stack : undefined,
        details: "Failed to load Express app. Ensure backend files are accessible."
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
});

// Helper function to convert Deno Request/Response to Express req/res
async function handleExpressRequest(app: any, denoReq: Request): Promise<Response> {
  return new Promise((resolve, reject) => {
    const url = new URL(denoReq.url);
    
    // Read request body
    denoReq.text().then((bodyText) => {
      // Create Express-compatible request object
      const req: any = {
        method: denoReq.method,
        url: url.pathname + url.search,
        path: url.pathname,
        query: Object.fromEntries(url.searchParams),
        headers: {} as Record<string, string>,
        body: null,
        params: {},
        get: function(name: string) {
          return this.headers[name.toLowerCase()];
        },
        header: function(name: string) {
          return this.headers[name.toLowerCase()];
        },
        ip: denoReq.headers.get("x-forwarded-for") || "unknown",
        protocol: url.protocol.slice(0, -1), // Remove trailing ':'
        secure: url.protocol === "https:",
        hostname: url.hostname,
        originalUrl: url.pathname + url.search,
      };
      
      // Copy headers from Deno Request
      denoReq.headers.forEach((value, key) => {
        req.headers[key.toLowerCase()] = value;
      });
      
      // Parse JSON body if present
      if (bodyText) {
        const contentType = denoReq.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
          try {
            req.body = JSON.parse(bodyText);
          } catch {
            req.body = bodyText;
          }
        } else if (contentType.includes("application/x-www-form-urlencoded")) {
          // Parse form data
          const params = new URLSearchParams(bodyText);
          req.body = Object.fromEntries(params);
        } else {
          req.body = bodyText;
        }
      }
      
      // Create Express-compatible response object
      let responseSent = false;
      const res: any = {
        statusCode: 200,
        headers: {} as Record<string, string>,
        body: null as string | null,
        status: function(code: number) {
          this.statusCode = code;
          return this;
        },
        json: function(data: any) {
          if (responseSent) return this;
          responseSent = true;
          this.body = JSON.stringify(data);
          this.headers["Content-Type"] = "application/json";
          resolve(
            new Response(this.body, {
              status: this.statusCode,
              headers: this.headers,
            })
          );
          return this;
        },
        send: function(data: any) {
          if (responseSent) return this;
          responseSent = true;
          if (typeof data === "object") {
            this.body = JSON.stringify(data);
            if (!this.headers["Content-Type"]) {
              this.headers["Content-Type"] = "application/json";
            }
          } else {
            this.body = String(data);
            if (!this.headers["Content-Type"]) {
              this.headers["Content-Type"] = "text/plain";
            }
          }
          resolve(
            new Response(this.body, {
              status: this.statusCode,
              headers: this.headers,
            })
          );
          return this;
        },
        setHeader: function(name: string, value: string) {
          this.headers[name] = value;
          return this;
        },
        getHeader: function(name: string) {
          return this.headers[name];
        },
        end: function(data?: any) {
          if (responseSent) return this;
          responseSent = true;
          if (data !== undefined) {
            this.body = typeof data === "string" ? data : JSON.stringify(data);
          }
          resolve(
            new Response(this.body || "", {
              status: this.statusCode,
              headers: this.headers,
            })
          );
          return this;
        },
      };
      
      // Call Express app with req, res, and next callback
      try {
        app(req, res, (err?: any) => {
          if (err) {
            console.error("Express error:", err);
            if (!responseSent) {
              responseSent = true;
              resolve(
                new Response(
                  JSON.stringify({ error: err.message, stack: err.stack }),
                  {
                    status: 500,
                    headers: { "Content-Type": "application/json" },
                  }
                )
              );
            }
          } else if (!responseSent && res.statusCode === 200) {
            // No response was sent, return 404
            responseSent = true;
            resolve(new Response("Not Found", { status: 404 }));
          }
        });
      } catch (err: any) {
        if (!responseSent) {
          responseSent = true;
          resolve(
            new Response(
              JSON.stringify({ error: err.message, stack: err.stack }),
              {
                status: 500,
                headers: { "Content-Type": "application/json" },
              }
            )
          );
        }
      }
    }).catch((err) => {
      console.error("Error reading request body:", err);
      if (!responseSent) {
        resolve(
          new Response(
            JSON.stringify({ error: "Failed to read request body" }),
            {
              status: 500,
              headers: { "Content-Type": "application/json" },
            }
          )
        );
      }
    });
  });
}
