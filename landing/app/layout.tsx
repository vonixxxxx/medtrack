import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ 
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MedTrack - Redefining Connected Healthcare",
  description: "MedTrack empowers patients, doctors, and institutions to collaborate seamlessly through intelligent health data management.",
  keywords: ["healthcare", "medication tracking", "patient management", "health data", "medical records"],
  authors: [{ name: "MedTrack" }],
  openGraph: {
    title: "MedTrack - Redefining Connected Healthcare",
    description: "Empowering patients, doctors, and institutions through intelligent health data management.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "MedTrack - Redefining Connected Healthcare",
    description: "Empowering patients, doctors, and institutions through intelligent health data management.",
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

