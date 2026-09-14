'use client'

import React, { useState } from 'react'

export default function Home() {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Welcome to InsightRAG! Ask any question regarding your enterprise documents (e.g. Redis caching architecture, OAuth 2.0 PKCE, or Kubernetes deployment scaling). All responses are strictly grounded in retrieved evidence.',
      citations: [],
    },
  ])
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState('hybrid')
  const [rerank, setRerank] = useState(true)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || loading) return

    const userText = query.trim()
    setQuery('')
    setMessages(prev => [...prev, { role: 'user', content: userText, citations: [] }])
    setLoading(true)

    try {
      const res = await fetch('http://localhost:8000/api/v1/rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userText,
          top_k: 3,
          retrieval_mode: mode,
          enable_reranking: rerank,
          candidate_pool_size: 15,
        }),
      })
      const data = await res.json()
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          citations: data.citations || [],
          rewritten: data.rewritten_query,
          latency: data.execution_time_ms,
        },
      ])
    } catch (err: any) {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message}`, citations: [] },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="max-w-6xl mx-auto p-6 flex flex-col min-h-screen">
      <header className="flex justify-between items-center pb-6 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-600/30">
            IR
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">InsightRAG Dashboard</h1>
            <p className="text-xs text-slate-400">Two-Stage Hybrid Retrieval & Grounded Generation</p>
          </div>
        </div>
        <div className="flex gap-2">
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300 hover:text-white"
          >
            API Docs
          </a>
          <a
            href="http://localhost:8000/app"
            className="px-3 py-1.5 rounded-lg bg-indigo-600 text-xs font-semibold text-white hover:bg-indigo-500"
          >
            Open Standalone UI
          </a>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mt-6 flex-1">
        {/* Chat window (3 cols) */}
        <div className="lg:col-span-3 flex flex-col bg-slate-900/60 rounded-2xl border border-slate-800 p-5 min-h-[500px]">
          <div className="flex-1 overflow-y-auto space-y-4 pr-2">
            {messages.map((m, idx) => (
              <div
                key={idx}
                className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`rounded-2xl p-4 max-w-2xl text-sm leading-relaxed ${
                    m.role === 'user'
                      ? 'bg-indigo-600 text-white rounded-tr-none'
                      : 'bg-slate-800/80 border border-slate-700/60 text-slate-200 rounded-tl-none'
                  }`}
                >
                  {(m as any).rewritten && (
                    <div className="text-xs text-indigo-400 font-mono mb-1">
                      Contextualized Query: &quot;{(m as any).rewritten}&quot;
                    </div>
                  )}
                  <div>{m.content}</div>
                  {m.citations && m.citations.length > 0 && (
                    <div className="mt-3 pt-2 border-t border-slate-700/60 flex flex-wrap gap-1 text-xs">
                      <span className="text-slate-400 font-medium">Citations:</span>
                      {m.citations.map((c: any, ci: number) => (
                        <span
                          key={ci}
                          className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-mono text-[11px]"
                        >
                          {c.document_name}
                          {c.page_number ? ` p.${c.page_number}` : ''}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="text-xs text-indigo-400 animate-pulse">
                Retrieving candidate chunks & generating grounded completion...
              </div>
            )}
          </div>

          <form onSubmit={handleSubmit} className="mt-4 flex gap-2">
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Ask a technical question..."
              className="flex-1 bg-slate-950 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-indigo-500"
            />
            <button
              type="submit"
              disabled={loading}
              className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded-xl text-xs transition"
            >
              Send
            </button>
          </form>
        </div>

        {/* Sidebar Controls */}
        <div className="flex flex-col gap-4">
          <div className="bg-slate-900/60 rounded-2xl border border-slate-800 p-5 text-xs space-y-4">
            <h2 className="font-bold text-white text-sm">Configuration</h2>
            <div>
              <label className="block text-slate-400 mb-1">Retrieval Mode</label>
              <select
                value={mode}
                onChange={e => setMode(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200"
              >
                <option value="hybrid">Hybrid (Dense + BM25 via RRF)</option>
                <option value="vector">Vector Only</option>
                <option value="bm25">BM25 Only</option>
              </select>
            </div>
            <div className="flex items-center justify-between p-2 rounded bg-slate-950 border border-slate-800">
              <span className="text-slate-300">Cross-Encoder Rerank</span>
              <input
                type="checkbox"
                checked={rerank}
                onChange={e => setRerank(e.target.checked)}
                className="w-4 h-4 accent-indigo-600"
              />
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}

