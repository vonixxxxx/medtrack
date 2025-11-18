const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000/api';

export interface LoginRequest {
  email: string;
  password: string;
}

export interface SignupRequest {
  email: string;
  password: string;
  role: 'patient' | 'clinician';
  hospitalCode: string;
  name?: string;
  patientData?: {
    name?: string;
    [key: string]: any;
  };
}

export interface AuthResponse {
  token: string;
  user: {
    id: string;
    email: string;
    role: 'patient' | 'clinician';
    hospitalCode: string;
  };
}

export async function login(data: LoginRequest): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Login failed' }));
    throw new Error(error.error || 'Invalid credentials');
  }

  const result = await response.json();
  
  // Handle different response formats from backend
  if (result.success && result.user && result.token) {
    // Format from simple-server.js
    return {
      token: result.token,
      user: {
        id: result.user.id,
        email: result.user.email,
        role: result.user.role,
        hospitalCode: result.user.hospitalCode,
      },
    };
  }
  
  // Format from authController.js (already correct)
  return result;
}

export async function signup(data: SignupRequest): Promise<AuthResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/signup`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    const result = await response.json().catch(() => {
      throw new Error('Failed to parse server response');
    });

    if (!response.ok) {
      // Log the full error for debugging
      console.error('Signup error response:', result);
      throw new Error(result.error || result.details || `Signup failed: ${response.status} ${response.statusText}`);
    }
    
    // Handle different response formats from backend
    if (result.success && result.user && result.token) {
      // Format from simple-server.js
      return {
        token: result.token,
        user: {
          id: result.user.id,
          email: result.user.email,
          role: result.user.role,
          hospitalCode: result.user.hospitalCode,
        },
      };
    }
    
    // Format from authController.js (already correct)
    return result;
  } catch (error) {
    // Re-throw with more context if it's not already an Error
    if (error instanceof Error) {
      throw error;
    }
    throw new Error(`Signup failed: ${String(error)}`);
  }
}

export function getDashboardUrl(role: 'patient' | 'clinician'): string {
  // Use Next.js routes - same domain
  return `/dashboard/${role}`;
}

export function redirectToDashboard(role: 'patient' | 'clinician'): void {
  const url = getDashboardUrl(role);
  window.location.href = url;
}

