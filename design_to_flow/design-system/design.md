# Design System — Alabaster × Midnight Editor

> **Portable design system** for Next.js + Fumadocs projects.  
> Drop the `design-system/` folder into any project and follow the setup guide to replicate the exact look and feel.

---

## 1. Overview

This design system defines the complete visual language for documentation-style web applications built with:

| Layer | Technology |
|---|---|
| Framework | **Next.js 16+** |
| UI Library | **Fumadocs UI 16+** |
| Styling | **Tailwind CSS v4** |
| Language | **TypeScript** |
| Icons | **Lucide React** |
| MDX Engine | **Fumadocs MDX** |

The aesthetic is **Editorial Minimalism**: print-quality typography, restrained color, purposeful whitespace, and subtle motion.

---

## 2. Color Palette

### Light Theme — "Alabaster Print"

| Token | HSL Value | Hex Approx | Usage |
|---|---|---|---|
| `--background` | `44 26% 96%` | `#F9F7F3` | Page background (warm off-white) |
| `--foreground` | `0 0% 11%` | `#1C1C1C` | Primary text |
| `--card` | `44 26% 96%` | `#F9F7F3` | Card backgrounds |
| `--secondary` | `40 13% 90%` | `#E6E2D8` | Subtle fills, dividers |
| `--muted` | `40 13% 90%` | `#E6E2D8` | Muted backgrounds |
| `--muted-foreground` | `0 0% 40%` | `#666666` | Secondary text |
| `--border` | `40 13% 90%` | `#E6E2D8` | Borders |
| `--primary` | `0 0% 11%` | `#1C1C1C` | Primary buttons (text-on-bg) |
| `--primary-foreground` | `44 26% 96%` | `#F9F7F3` | Text on primary buttons |

### Dark Theme — "Midnight Editor"

| Token | HSL Value | Hex Approx | Usage |
|---|---|---|---|
| `--background` | `240 4% 9%` | `#161618` | Page background (deep near-black) |
| `--foreground` | `0 0% 95%` | `#F2F2F2` | Primary text |
| `--card` | `240 4% 9%` | `#161618` | Card backgrounds |
| `--secondary` | `240 2% 17%` | `#2C2C2E` | Subtle fills |
| `--muted` | `240 2% 17%` | `#2C2C2E` | Muted backgrounds |
| `--muted-foreground` | `0 0% 70%` | `#B3B3B3` | Secondary text |
| `--border` | `240 2% 17%` | `#2C2C2E` | Borders |
| `--primary` | `0 0% 95%` | `#F2F2F2` | Primary buttons |
| `--primary-foreground` | `240 4% 9%` | `#161618` | Text on primary buttons |

### Design Philosophy
- **No vivid accent colors** — the palette is intentionally chromatic-neutral (warm light, cool dark).
- Contrast is created through **value** (lightness), not hue.
- This makes the content — headings, code, diagrams — the visual hero.

---

## 3. Typography

### Font Stack

| Role | Font Family | Weight(s) | Variable |
|---|---|---|---|
| Body / Sans | **Libre Franklin** | 400, 500, 600 | `--font-body` -> `--font-sans` |
| Heading / Serif | **Fraunces** | 400, 600, 700 | `--font-heading` -> `--font-serif` |

Both are loaded via `next/font/google` with `subsets: ['latin']`.

### Typographic Rules

| Element | Size | Line-height | Letter-spacing |
|---|---|---|---|
| h1 | 36px (mobile) / 48px (md+) | 1.15 | -0.02em |
| h2, h3 | Scaled by Fumadocs defaults | 1.15 | -0.02em |
| Body | 1.05rem | 1.45 | Default |
| Muted text | 14px | 22px | Default |
| Prose paragraphs | Inherit | 1.45 | Default |

### Prose Constraints
- Max prose width: **65ch** (optimal reading measure)
- Prose is centered inline: margin-inline: auto
- Paragraph bottom margin: 1rem

---

## 4. Spacing & Layout

### Breakpoints (Tailwind defaults)
| Name | Min-width |
|---|---|
| sm | 640px |
| md | 768px |
| lg | 1024px |
| xl | 1280px |

### Layout Patterns

**Hero Section (2-column grid)**
```
[Left: Text Content]  [Right: Focal Image]
   grid-cols-2 gap-12 max-w-5xl
```

**Feature Grid (3-column)**
```
[Card] [Card] [Card]
grid-cols-1 -> md:grid-cols-3 gap-6
```

**Docs Layout**
- Managed by Fumadocs DocsLayout
- Sidebar navigation on left
- TOC on right (desktop)
- Content in center column

---

## 5. Animation & Motion

### Entrance Animations
Two utility classes for progressive reveal:

```
.animate-fade-in-up   /* fade + rise 15px, duration 500ms */
.animate-fade-in      /* fade only, duration 500ms */
```

Easing: cubic-bezier(0.16, 1, 0.3, 1) — fast start, spring-like settle.

### Delay Hierarchy
- .delay-100 => 100ms
- .delay-200 => 200ms
- .delay-300 => 300ms
- .delay-500 => 500ms
- .delay-700 => 700ms

### Interactive Micro-animations
| Interaction | Effect |
|---|---|
| Primary button hover | group-hover:translate-x-0.5 on arrow icon |
| Card hover | hover:bg-muted/30 hover:border-foreground/20 hover:shadow-sm |
| Feature icon hover | group-hover:scale-125 group-hover:rotate-6 |
| Hero image hover | hover:scale-105 with duration-500 |

**Principle**: Motion is forwards-fill, not looping. It signals arrival, not distraction.

---

## 6. Component Patterns

### Primary Button
```
className="group inline-flex items-center justify-center gap-2 rounded-md
           bg-foreground px-6 py-3 text-[14px] font-medium text-background
           transition-colors hover:bg-foreground/90"
```

### Secondary / Ghost Button
```
className="inline-flex items-center justify-center gap-2 rounded-md
           border border-border bg-background px-6 py-3 text-[14px]
           font-medium text-foreground transition-colors hover:bg-muted"
```

### Feature Card
```
className="group flex flex-col items-start overflow-hidden rounded-xl
           border border-border bg-card p-6 text-card-foreground
           transition-all hover:bg-muted/30 hover:border-foreground/20 hover:shadow-sm"
```

### Icon Box (inside card)
```
className="mb-6 flex h-16 w-16 items-center justify-center
           rounded-lg bg-muted/40 border border-border/50"
```

### Nav Logo
```
// Logo image + serif wordmark side by side
className="flex items-center gap-3"
// Image: width=72 height=72, className="dark:invert drop-shadow-sm -my-2"
// Text: className="font-serif font-bold text-2xl tracking-tight mt-1"
```

---

## 7. Fumadocs Configuration

### Preset / CSS Imports (in global.css)
```
@import 'tailwindcss';
@import 'fumadocs-ui/css/neutral.css';   /* neutral color preset */
@import 'fumadocs-ui/css/preset.css';    /* Fumadocs base reset */
```

### Layout Provider
```
// app/layout.tsx
import { RootProvider } from 'fumadocs-ui/provider/next';
// Wraps the entire app — enables dark mode, search, sidebar state
```

### Home Layout
```
// app/(home)/layout.tsx
import { HomeLayout } from 'fumadocs-ui/layouts/home';
// Marketing/landing page shell with nav
```

### Docs Layout
```
// app/docs/layout.tsx
import { DocsLayout } from 'fumadocs-ui/layouts/docs';
// Sidebar + TOC shell for documentation pages
```

---

## 8. File Structure

```
project-root/
├── app/
│   ├── global.css              # Design tokens + animation utilities
│   ├── layout.tsx              # Root layout (fonts + RootProvider)
│   ├── (home)/
│   │   ├── layout.tsx          # HomeLayout wrapper
│   │   └── page.tsx            # Landing page
│   └── docs/
│       ├── layout.tsx          # DocsLayout wrapper
│       └── [[...slug]]/
│           └── page.tsx        # Dynamic docs page
├── components/
│   ├── mdx.tsx                 # MDX component overrides
│   └── mermaid.tsx             # Mermaid diagram renderer
├── lib/
│   ├── cn.ts                   # tailwind-merge utility
│   ├── shared.ts               # App-wide constants (name, git config)
│   ├── layout.shared.tsx       # Shared nav/layout options
│   └── source.ts               # Fumadocs source adapter
├── content/
│   └── docs/                   # MDX content files
├── public/
│   └── images/                 # Static assets (logo, icons)
├── next.config.mjs             # MDX + image domains config
├── source.config.ts            # Fumadocs collections config
├── tsconfig.json               # TypeScript paths (@/* alias)
├── postcss.config.mjs          # PostCSS + Tailwind v4
└── package.json
```

---

## 9. Design Principles

1. **Content First** — UI chrome is invisible; the writing is the product.
2. **No Accent Color** — Neutral palette keeps focus on diagrams, code blocks, and headings.
3. **Editorial Rhythm** — Consistent vertical spacing and type scale create a calm, trustworthy reading experience.
4. **Progressive Enhancement** — Animations use opacity: 0 initial state so they degrade gracefully.
5. **Accessible by Default** — Fumadocs neutral preset meets WCAG AA contrast in both themes.
6. **Portable** — All tokens are CSS custom properties; swapping a palette requires editing only global.css.

---

## 10. Quick Customization Guide

| What to change | Where |
|---|---|
| App name | lib/shared.ts -> appName |
| GitHub link | lib/shared.ts -> gitConfig |
| Nav logo | lib/layout.shared.tsx |
| Color palette | app/global.css -> :root and .dark blocks |
| Body font | app/layout.tsx -> different Google Font |
| Heading font | app/layout.tsx -> replace Fraunces |
| Docs content | content/docs/ -> add/edit .mdx files |
| Remote image hosts | next.config.mjs -> images.remotePatterns |
