import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { useTheme } from '../shared/theme'

const LandingPage = lazy(() => import('./LandingPage'))
const SynthApp = lazy(() => import('../synth/SynthApp'))
const TrainingApp = lazy(() => import('../training/TrainingApp'))
const TrainingSimpleApp = lazy(() => import('../training/TrainingSimpleApp'))

function PlatformFallback() {
  return <div className="min-h-screen bg-slate-950 text-sm text-slate-400" />
}

const SYNTH_LEGACY_ROUTES = [
  '/simulate',
  '/upload',
  '/register',
  '/templates',
  '/fill',
  '/router',
  '/template-viewer',
  '/samples',
  '/router-viewer',
  '/stats',
  '/registry',
  '/llm-config',
]

export default function PlatformRouter() {
  const [theme, toggleTheme] = useTheme()

  return (
    <Suspense fallback={<PlatformFallback />}>
      <Routes>
        <Route path="/" element={<LandingPage theme={theme} toggleTheme={toggleTheme} />} />
        <Route path="/training/simple/*" element={<TrainingSimpleApp theme={theme} toggleTheme={toggleTheme} />} />
        <Route path="/training/*" element={<TrainingApp theme={theme} toggleTheme={toggleTheme} />} />
        <Route path="/files" element={<Navigate to="/synth/files" replace />} />
        <Route path="/synth/*" element={<SynthApp theme={theme} toggleTheme={toggleTheme} />} />
        {SYNTH_LEGACY_ROUTES.map(path => (
          <Route key={path} path={path} element={<Navigate to={`/synth${path}`} replace />} />
        ))}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  )
}
