import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  images: {
    remotePatterns: [
      // Add remote image hostnames here
      // Example:
      // {
      //   protocol: 'https',
      //   hostname: 'example.com',
      // },
    ],
  },
};

export default withMDX(config);
