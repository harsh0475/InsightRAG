import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'InsightRAG — Advanced Knowledge Intelligence System',
  description: 'Enterprise RAG with Hybrid Retrieval, Cross-Encoder Reranking, and Strict Grounding',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 min-h-screen font-sans">{children}</body>
    </html>
  )
}

