import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Facial Recognition System',
  description: 'A facial recognition system built with Next.js',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
