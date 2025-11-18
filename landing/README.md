# MedTrack Landing Page

A modern, professional, Apple-level landing page for MedTrack - a healthcare management platform serving both patients and institutions.

## 🎨 Design Features

- **Minimalistic & Clean**: Inspired by Apple's design philosophy with ample white space and elegant typography
- **Smooth Animations**: Framer Motion powered micro-interactions and transitions
- **Fully Responsive**: Pixel-perfect across mobile, tablet, and desktop devices
- **Accessible**: Built with accessibility standards in mind
- **SEO Optimized**: Next.js metadata for optimal search engine performance

## 🚀 Tech Stack

- **Next.js 14** with App Router
- **React 18** with TypeScript
- **TailwindCSS** for styling
- **Framer Motion** for animations
- **Lucide React** for icons

## 📦 Installation

```bash
cd landing
npm install
```

## ⚙️ Configuration

Create a `.env.local` file in the `landing` directory:

```bash
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:4000/api

# Main App URL (where your Vite React app runs - for dashboard redirects)
NEXT_PUBLIC_APP_URL=http://localhost:5173
```

**Important**: Make sure your Vite app (main MedTrack application) runs on port 5173, and the Next.js landing page runs on port 3000.

To change the Vite app port, update `frontend/vite.config.js`:
```js
server: {
  port: 5173, // or any other port
}
```

## 🏃 Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the landing page.

**Note**: Make sure your backend API is running on port 4000 and your main Vite app is running on port 5173 for full functionality.

## 🏗️ Build

```bash
npm run build
npm start
```

## 📁 Project Structure

```
landing/
├── app/
│   ├── layout.tsx      # Root layout with metadata
│   ├── page.tsx        # Main landing page
│   └── globals.css     # Global styles
├── components/
│   ├── Header.tsx      # Navigation header
│   ├── Hero.tsx        # Hero section
│   ├── Features.tsx    # Feature highlights
│   ├── Collaboration.tsx # Partnership section
│   ├── AISecurity.tsx  # AI & Security section
│   ├── CTA.tsx         # Call to action
│   ├── Footer.tsx      # Footer
│   └── AuthModal.tsx   # Login/Signup modal
└── lib/
    └── utils.ts        # Utility functions
```

## 🎯 Features

### Sections

1. **Hero Section**
   - Compelling headline and subtext
   - Animated background elements
   - Call-to-action buttons
   - Dashboard mockup visualization

2. **Feature Highlights**
   - For Patients: medication tracking, appointments, health metrics
   - For Clinicians: real-time data, secure communication
   - For Institutions: AI-processed data aggregation

3. **Collaboration Section**
   - Imperial College London partnership
   - Institution statistics
   - Trust indicators

4. **AI & Security Section**
   - AI-powered data processing
   - HIPAA & GDPR compliance
   - Security features showcase

5. **Call to Action**
   - Patient and Clinician signup options
   - Clear value proposition

6. **Footer**
   - Navigation links
   - Legal information
   - Compliance badges

### Authentication Modal

- **Fully functional** login and signup connected to backend API
- **Patient and Clinician** user type selection for signup
- **Hospital code field** required for signup (validates with backend)
- **Form validation** with real-time error messages
- **Password visibility toggle** for better UX
- **Loading states** with spinner during authentication
- **Automatic redirect** to appropriate dashboard after successful auth
- **Token storage** in localStorage for seamless integration with main app
- **Smooth animations** and transitions

## 🎨 Color Palette

- **Primary**: Blue tones (#0ea5e9, #0284c7)
- **Medical**: Green accents for health-related elements
- **Neutral**: Grays for text and backgrounds
- **Gradients**: Subtle gradients for depth and visual interest

## 📱 Responsive Breakpoints

- Mobile: < 640px
- Tablet: 640px - 1024px
- Desktop: > 1024px

## ♿ Accessibility

- Semantic HTML
- ARIA labels where needed
- Keyboard navigation support
- Focus states
- Screen reader friendly

## 🔒 Security & Compliance

- HIPAA compliant messaging
- GDPR compliance indicators
- Secure data handling references

## 📝 License

MIT License - see LICENSE file for details

---

**MedTrack** - Redefining Connected Healthcare 🏥✨

