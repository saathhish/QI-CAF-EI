// =============================================================
// TEMPLATE: app/(home)/page.tsx
// Landing page — 2-col hero + 3-col feature grid
// EDIT: Update href, text, and image paths for your project
// =============================================================
import Link from 'next/link';
import Image from 'next/image';
import { ArrowRight, ExternalLink } from 'lucide-react';

const FEATURES = [
  {
    title: 'Feature One',
    desc: 'Short description of what this feature does and why it matters.',
    iconAsset: '/images/icon_one.png',
  },
  {
    title: 'Feature Two',
    desc: 'Short description of what this feature does and why it matters.',
    iconAsset: '/images/icon_two.png',
  },
  {
    title: 'Feature Three',
    desc: 'Short description of what this feature does and why it matters.',
    iconAsset: '/images/icon_three.png',
  },
];

export default function HomePage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background text-foreground overflow-hidden">
      <main className="flex w-full max-w-5xl flex-col items-start px-6 py-12 md:py-24">

        {/* ── Hero: 2-Column Grid ───────────────────────────── */}
        <div className="grid w-full grid-cols-1 md:grid-cols-2 gap-12 items-center">

          {/* Left Column: Text Content */}
          <div className="flex flex-col items-start z-10">

            {/* Main Heading */}
            <h1 className="font-serif text-[36px] font-semibold tracking-[-0.02em] md:text-[48px] max-w-[500px] animate-fade-in-up">
              Your Project Title Here
            </h1>

            {/* Subtitle */}
            <p className="mt-4 max-w-[500px] text-[16px] leading-[26px] text-muted-foreground animate-fade-in-up delay-100">
              One or two sentences describing what this project teaches or does.
              Be specific and compelling.
            </p>

            {/* CTA Buttons */}
            <div className="mt-8 flex flex-col gap-4 sm:flex-row sm:items-center w-full animate-fade-in-up delay-200">
              {/* Primary CTA */}
              <Link
                href="/docs"
                className="group inline-flex items-center justify-center gap-2 rounded-md bg-foreground px-6 py-3 text-[14px] font-medium text-background transition-colors hover:bg-foreground/90 w-full sm:w-auto"
              >
                Get Started
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              </Link>

              {/* Secondary CTA */}
              <Link
                href="https://example.com"
                target="_blank"
                className="inline-flex items-center justify-center gap-2 rounded-md border border-border bg-background px-6 py-3 text-[14px] font-medium text-foreground transition-colors hover:bg-muted w-full sm:w-auto"
              >
                External Resource
                <ExternalLink className="h-4 w-4" />
              </Link>
            </div>
          </div>

          {/* Right Column: Hero Focal Image */}
          <div className="relative flex items-center justify-center h-[350px] w-full hidden md:flex">
            <div className="z-10 transition-transform duration-500 hover:scale-105 cursor-default animate-fade-in delay-300">
              <Image
                src="/images/hero.png"
                alt="Hero illustration"
                width={260}
                height={260}
                className="drop-shadow-2xl"
              />
            </div>
          </div>
        </div>

        <hr className="my-16 w-full border-border" />

        {/* ── Feature Grid: 3-Column ───────────────────────── */}
        <div className="grid w-full grid-cols-1 gap-6 md:grid-cols-3 animate-fade-in-up delay-300">
          {FEATURES.map((feature, i) => (
            <div
              key={i}
              className="group flex flex-col items-start overflow-hidden rounded-xl border border-border bg-card p-6 text-card-foreground transition-all hover:bg-muted/30 hover:border-foreground/20 hover:shadow-sm"
            >
              {/* Icon Box */}
              <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-lg bg-muted/40 border border-border/50">
                <Image
                  src={feature.iconAsset}
                  alt={feature.title}
                  width={40}
                  height={40}
                  className="drop-shadow-sm transition-transform duration-300 ease-out group-hover:scale-125 group-hover:rotate-6"
                />
              </div>

              <h3 className="text-[16px] font-semibold tracking-[-0.01em]">
                {feature.title}
              </h3>

              <p className="mt-2 text-[14px] leading-[22px] text-muted-foreground">
                {feature.desc}
              </p>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
