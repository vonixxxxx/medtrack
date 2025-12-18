import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ 
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MedTrack — Enterprise-grade medication & clinical intelligence",
  description: "Privacy-first AI to improve adherence, power research, and streamline clinical workflows.",
  keywords: ["healthcare", "medication tracking", "patient management", "health data", "medical records", "enterprise healthcare", "HIPAA compliant", "clinical intelligence"],
  authors: [{ name: "MedTrack" }],
  openGraph: {
    title: "MedTrack — Enterprise-grade medication & clinical intelligence",
    description: "Privacy-first AI to improve adherence, power research, and streamline clinical workflows.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "MedTrack — Enterprise-grade medication & clinical intelligence",
    description: "Privacy-first AI to improve adherence, power research, and streamline clinical workflows.",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="antialiased bg-white text-gray-900">
        {children}
      </body>
    </html>
  );
}

