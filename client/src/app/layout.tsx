import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "treehacks26 client",
  description: "Spotify connect and playlist browser",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
