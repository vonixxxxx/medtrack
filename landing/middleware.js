import { NextResponse } from 'next/server';

export function middleware(request) {
  // Only process requests to /app/* paths
  if (request.nextUrl.pathname.startsWith('/app/')) {
    const response = NextResponse.next();
    
    // Remove X-Frame-Options header if present
    response.headers.delete('x-frame-options');
    response.headers.delete('X-Frame-Options');
    
    return response;
  }
  
  return NextResponse.next();
}

export const config = {
  matcher: '/app/:path*',
};



