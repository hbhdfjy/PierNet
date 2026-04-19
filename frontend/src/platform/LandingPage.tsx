import { ArrowRight, Database, GitBranch, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'

function EntryCard({
  to,
  title,
  copy,
  tone,
  icon,
  points,
}: {
  to: string
  title: string
  copy: string
  tone: string
  icon: React.ReactNode
  points: string[]
}) {
  return (
    <Link
      to={to}
      className="group relative overflow-hidden rounded-[28px] border border-slate-700/45 bg-slate-950/30 p-7 transition hover:-translate-y-0.5 hover:border-slate-600/70 hover:bg-slate-950/40"
    >
      <div className={tone} />
      <div className="relative">
        <div className="flex items-start justify-between gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-white">
            {icon}
          </div>
          <ArrowRight size={18} className="mt-1 text-slate-500 transition group-hover:translate-x-1 group-hover:text-slate-200" />
        </div>

        <div className="mt-7">
          <h2 className="text-[1.9rem] font-semibold tracking-tight text-white">{title}</h2>
          <p className="mt-3 max-w-xl text-[15px] leading-7 text-slate-300">{copy}</p>
        </div>

        <div className="mt-7 space-y-2.5">
          {points.map(point => (
            <div key={point} className="rounded-2xl border border-slate-700/35 bg-slate-950/30 px-4 py-3 text-[15px] text-slate-200">
              {point}
            </div>
          ))}
        </div>
      </div>
    </Link>
  )
}

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.16),transparent_28%),radial-gradient(circle_at_100%_0%,rgba(16,185,129,0.14),transparent_26%),linear-gradient(180deg,#060b17,#0b1120)] text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-6 py-8 lg:px-10 lg:py-10">
        <header className="relative overflow-hidden rounded-[32px] border border-slate-700/45 bg-slate-950/28 px-7 py-8 backdrop-blur xl:px-8 xl:py-9">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.18),transparent_28%),radial-gradient(circle_at_85%_18%,rgba(52,211,153,0.14),transparent_24%)]" />
          <div className="relative flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
            <div className="max-w-3xl">
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <span className="badge border border-sky-400/20 bg-sky-500/10 text-sky-300">PiERN</span>
                <span className="badge border border-slate-700/50 bg-slate-900/40 text-slate-400">Gateway</span>
              </div>
              <h1 className="text-[2.4rem] font-semibold tracking-tight text-white xl:text-[3rem]">{"\u9009\u62e9\u5e73\u53f0"}</h1>
              <p className="mt-3 max-w-2xl text-[16px] leading-8 text-slate-300">
                {"\u6570\u636e\u5408\u6210\u548c\u6a21\u578b\u8bad\u7ec3\u5df2\u62c6\u5206\u6210\u4e24\u4e2a\u5165\u53e3\u3002\u4ece\u8fd9\u91cc\u76f4\u63a5\u8fdb\u5165\u5bf9\u5e94\u5de5\u4f5c\u9762\u3002"}
              </p>
            </div>
            <div className="grid grid-cols-2 gap-3 xl:w-[360px]">
              <div className="rounded-2xl border border-slate-700/40 bg-slate-950/30 px-4 py-4">
                <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">synth</div>
                <div className="mt-2 text-[22px] font-semibold text-white">/synth</div>
              </div>
              <div className="rounded-2xl border border-slate-700/40 bg-slate-950/30 px-4 py-4">
                <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">training</div>
                <div className="mt-2 text-[22px] font-semibold text-white">/training</div>
              </div>
            </div>
          </div>
        </header>

        <main className="grid flex-1 gap-6 py-8 lg:grid-cols-2">
          <EntryCard
            to="/synth"
            title={"\u6570\u636e\u5408\u6210"}
            copy={"\u7528\u4e8e\u7269\u7406\u4eff\u771f\u3001\u6ce8\u518c\u3001\u6a21\u677f\u751f\u6210\u3001\u6837\u672c\u586b\u5145\u548c Router \u6570\u636e\u6784\u5efa\u3002"}
            tone="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.12),transparent_34%)]"
            icon={<Database size={22} className="text-sky-300" />}
            points={[
              '\u6837\u672c\u7edf\u8ba1\u4e0e\u5185\u5bb9\u603b\u89c8',
              '\u6a21\u677f\u3001\u6837\u672c\u3001Router \u6570\u636e\u6d41\u8f6c',
              '\u6ce8\u518c\u4e0e\u914d\u7f6e\u7ba1\u7406',
            ]}
          />

          <EntryCard
            to="/training"
            title={"\u6a21\u578b\u8bad\u7ec3"}
            copy={"\u7528\u4e8e Token Router \u8bad\u7ec3\u3001GPU \u5206\u914d\u3001\u4efb\u52a1\u7ba1\u7406\u3001\u66f2\u7ebf\u67e5\u770b\u548c checkpoint \u8ddf\u8e2a\u3002"}
            tone="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(52,211,153,0.12),transparent_34%)]"
            icon={<GitBranch size={22} className="text-emerald-300" />}
            points={[
              '\u5355 GPU \u8bad\u7ec3\u8c03\u5ea6',
              '\u8fd0\u884c\u65e5\u5fd7\u3001\u6307\u6807\u66f2\u7ebf\u4e0e\u6d4b\u8bd5\u7ed3\u679c',
              '\u4efb\u52a1\u5217\u8868\u4e0e checkpoint \u7ba1\u7406',
            ]}
          />
        </main>
      </div>
    </div>
  )
}
