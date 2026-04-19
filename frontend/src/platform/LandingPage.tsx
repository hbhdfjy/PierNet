import { ArrowRight, Database, FlaskConical, GitBranch, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(56,189,248,0.14),transparent_34%),radial-gradient(circle_at_bottom_right,rgba(16,185,129,0.14),transparent_28%),linear-gradient(180deg,var(--surface-0),var(--surface-1))] text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-6 py-10 lg:px-10">
        <header className="flex items-center justify-between gap-4 border-b border-slate-700/40 pb-6">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.24em] text-sky-300/80">PiERN</div>
            <h1 className="mt-3 text-4xl font-semibold tracking-tight text-white">Platform Gateway</h1>
            <p className="mt-3 max-w-2xl text-base leading-7 text-slate-300">
              Choose a workspace. Data synthesis and model training are now separate platform surfaces with independent routing and shared infrastructure.
            </p>
          </div>
          <div className="hidden rounded-3xl border border-slate-700/40 bg-slate-900/40 px-5 py-4 text-sm text-slate-300 lg:block">
            Root: <span className="font-mono text-slate-100">/</span>
          </div>
        </header>

        <main className="grid flex-1 gap-6 py-10 lg:grid-cols-2">
          <Link
            to="/synth"
            className="group flex flex-col rounded-[28px] border border-slate-700/40 bg-slate-900/40 p-7 transition hover:border-sky-400/40 hover:bg-slate-900/60"
          >
            <div className="flex items-center justify-between">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-sky-400/20 bg-sky-400/10 text-sky-300">
                <Database size={22} />
              </div>
              <ArrowRight size={18} className="text-slate-500 transition group-hover:translate-x-1 group-hover:text-sky-300" />
            </div>
            <h2 className="mt-8 text-2xl font-semibold text-white">Data Synthesis Platform</h2>
            <p className="mt-3 text-sm leading-7 text-slate-300">
              Stage 1 to Stage 4 workflow for physical simulation, registration, template generation, sample filling, and router-data construction.
            </p>
            <div className="mt-8 grid gap-3 text-sm text-slate-200">
              <div className="flex items-center gap-3 rounded-2xl border border-slate-700/30 bg-slate-950/30 px-4 py-3">
                <Sparkles size={16} className="text-amber-300" />
                <span>Simulation and template pipeline</span>
              </div>
              <div className="flex items-center gap-3 rounded-2xl border border-slate-700/30 bg-slate-950/30 px-4 py-3">
                <FlaskConical size={16} className="text-emerald-300" />
                <span>Sample filling and dataset browsing</span>
              </div>
            </div>
          </Link>

          <Link
            to="/training"
            className="group flex flex-col rounded-[28px] border border-slate-700/40 bg-slate-900/40 p-7 transition hover:border-emerald-400/40 hover:bg-slate-900/60"
          >
            <div className="flex items-center justify-between">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-emerald-400/20 bg-emerald-400/10 text-emerald-300">
                <GitBranch size={22} />
              </div>
              <ArrowRight size={18} className="text-slate-500 transition group-hover:translate-x-1 group-hover:text-emerald-300" />
            </div>
            <h2 className="mt-8 text-2xl font-semibold text-white">Training Platform</h2>
            <p className="mt-3 text-sm leading-7 text-slate-300">
              Single-GPU Token Router training workspace with job management, curve inspection, logs, checkpoints, and GPU assignment.
            </p>
            <div className="mt-8 grid gap-3 text-sm text-slate-200">
              <div className="flex items-center gap-3 rounded-2xl border border-slate-700/30 bg-slate-950/30 px-4 py-3">
                <GitBranch size={16} className="text-emerald-300" />
                <span>Managed training jobs and checkpoints</span>
              </div>
              <div className="flex items-center gap-3 rounded-2xl border border-slate-700/30 bg-slate-950/30 px-4 py-3">
                <Sparkles size={16} className="text-sky-300" />
                <span>Training and test curves by run</span>
              </div>
            </div>
          </Link>
        </main>
      </div>
    </div>
  )
}
