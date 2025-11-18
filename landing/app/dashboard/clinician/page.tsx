"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function ClinicianDashboard() {
  const router = useRouter();
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    // Check if user is authenticated
    const token = localStorage.getItem("token");
    const user = localStorage.getItem("user");

    if (!token || !user) {
      // Redirect to landing page if not authenticated
      router.push("/");
      return;
    }

    setIsAuthenticated(true);
  }, [router]);

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <iframe
        src="/app/dashboard/clinician"
        className="w-full h-screen border-0"
        title="Clinician Dashboard"
        allow="clipboard-read; clipboard-write"
      />
    </div>
  );
}

