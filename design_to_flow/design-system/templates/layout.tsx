// =============================================================
// TEMPLATE: app/layout.tsx
// Root layout — fonts + theme provider
// =============================================================
import { RootProvider } from 'fumadocs-ui/provider/next';
import './global.css';
import { Libre_Franklin, Fraunces } from 'next/font/google';

// ---------------- Google Fonts --------------------------------
// Body: Libre Franklin (clean, editorial sans-serif)
const libreFranklin = Libre_Franklin({
  subsets: ['latin'],
  variable: '--font-body',
});

// Heading: Fraunces (expressive optical serif)
const fraunces = Fraunces({
  subsets: ['latin'],
  variable: '--font-heading',
});
// --------------------------------------------------------------

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${libreFranklin.variable} ${fraunces.variable} font-sans`}
      suppressHydrationWarning
    >
      <body className="flex flex-col min-h-screen">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
