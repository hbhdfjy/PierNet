import { Navigate, Route, Routes } from 'react-router-dom'
import LandingPage from './LandingPage'
import SynthApp from '../synth/SynthApp'
import TrainingApp from '../training/TrainingApp'
import { useTheme } from '../shared/theme'

const SYNTH_LEGACY_ROUTES = [
  '/simulate',
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
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/training/*" element={<TrainingApp theme={theme} toggleTheme={toggleTheme} />} />
      <Route path="/synth/*" element={<SynthApp theme={theme} toggleTheme={toggleTheme} />} />
      {SYNTH_LEGACY_ROUTES.map((path) => (
        <Route key={path} path={path} element={<Navigate to={`/synth${path}`} replace />} />
      ))}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
