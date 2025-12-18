/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['images.unsplash.com'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
        pathname: '/**',
      },
    ],
  },
  async rewrites() {
    return [
      // Proxy Vite dev server assets (must come first to catch @vite, src, etc.)
      {
        source: '/@vite/:path*',
        destination: 'http://localhost:5173/@vite/:path*',
      },
      {
        source: '/@react-refresh',
        destination: 'http://localhost:5173/@react-refresh',
      },
      {
        source: '/src/:path*',
        destination: 'http://localhost:5173/src/:path*',
      },
      {
        source: '/node_modules/:path*',
        destination: 'http://localhost:5173/node_modules/:path*',
      },
      {
        source: '/@fs/:path*',
        destination: 'http://localhost:5173/@fs/:path*',
      },
      // Proxy all Vite app routes - this must be last
      {
        source: '/app/:path*',
        destination: 'http://localhost:5173/:path*',
      },
    ];
  },
}

module.exports = nextConfig


