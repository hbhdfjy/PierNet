import { Navigate, Route, Routes } from 'react-router-dom'
import LandingPage from './LandingPage'
import SynthApp from '../synth/SynthApp'
import TrainingApp from '../training/TrainingApp'
import FileManagerPage from '../files/FileManagerPage'
import { useTheme } from '../shared/theme'

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
    <Routes>
      <Route path="/" element={<LandingPage theme={theme} toggleTheme={toggleTheme} />} />
      <Route path="/training/*" element={<TrainingApp theme={theme} toggleTheme={toggleTheme} />} />
      <Route path="/files" element={<FileManagerPage theme={theme} toggleTheme={toggleTheme} />} />
      <Route path="/synth/*" element={<SynthApp theme={theme} toggleTheme={toggleTheme} />} />
      {SYNTH_LEGACY_ROUTES.map((path) => (
        <Route key={path} path={path} element={<Navigate to={`/synth${path}`} replace />} />
      ))}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
