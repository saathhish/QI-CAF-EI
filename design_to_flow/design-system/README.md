# Alabaster × Midnight Editor Design System

This directory is a **complete, standalone design system bundle** tailored for Next.js 16+, Fumadocs UI 16+, and Tailwind CSS v4.

---

## 📁 Directory Overview

```
design-system/
├── design.md                  # Master Design Reference & System Guidelines
├── README.md                  # Quickstart & Integration Guide
├── tokens/
│   └── global.css             # Theme CSS custom properties & entrance animations
├── components/
│   ├── cn.ts                  # Tailwind class merge helper
│   ├── shared.ts              # Global app constants configuration
│   └── layout.shared.tsx      # Shared Fumadocs nav options & branding
├── templates/
│   ├── layout.tsx             # Root layout template with Google Fonts
│   ├── home-layout.tsx        # Home layout wrapper template
│   └── home-page.tsx          # 2-column hero & 3-column feature grid template
└── setup/
    ├── package.json           # Required dependencies & scripts
    ├── next.config.mjs        # Next.js & Fumadocs plugin configuration
    ├── source.config.ts       # Fumadocs MDX source collection configuration
    ├── tsconfig.json          # TypeScript compiler options & `@/*` alias
    └── postcss.config.mjs     # PostCSS setup for Tailwind v4
```

---

## 🚀 How to Use in Any New Project

### Step 1: Copy Setup Configuration
Copy all files from `setup/` to your target project root:
- `package.json`
- `next.config.mjs`
- `source.config.ts`
- `tsconfig.json`
- `postcss.config.mjs`

Install dependencies:
```bash
npm install
```

### Step 2: Add Design Tokens
Copy `tokens/global.css` into your project's `app/global.css`.

### Step 3: Add Shared Utilities & Components
Copy the files in `components/` to your project:
- Place `cn.ts`, `shared.ts`, and `layout.shared.tsx` inside your project's `lib/` directory.

### Step 4: Apply Templates
Copy templates from `templates/`:
- `layout.tsx` → `app/layout.tsx`
- `home-layout.tsx` → `app/(home)/layout.tsx`
- `home-page.tsx` → `app/(home)/page.tsx`

### Step 5: Customize Application Branding
Edit `lib/shared.ts` to update your application name and GitHub link:
```typescript
export const appName = 'My New App';
export const gitConfig = {
  user: 'my-github-user',
  repo: 'my-repo',
  branch: 'main',
};
```

---

## 🎨 Design System Highlights

- **Palette**: Chromatic-neutral warm light theme ("Alabaster Print") and deep dark theme ("Midnight Editor").
- **Typography**: Libre Franklin for crisp body text and Fraunces for editorial serif headings.
- **Micro-Animations**: Eased entrance animations (`animate-fade-in-up`, `animate-fade-in`) with progressive delay utilities (`delay-100` through `delay-700`).
- **Responsive Layouts**: Editorial hero section, feature cards with icon hover states, and Fumadocs doc layout integration.

Refer to [`design.md`](./design.md) for full architectural guidelines and design principles.
