import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const frontendPort = Number(process.env.PierNet_FRONTEND_PORT ?? 3000)
const backendPort = Number(process.env.PierNet_BACKEND_PORT ?? 8000)
const newSynthPort = Number(process.env.PierNet_NEW_SYNTH_PORT ?? 3002)
const allowedHostsSetting = (process.env.PierNet_FRONTEND_ALLOWED_HOSTS ?? '').trim()
const configuredAllowedHosts = allowedHostsSetting
  .split(',')
  .map(host => host.trim())
  .filter(Boolean)
const allowedHosts =
  allowedHostsSetting.toLowerCase() === 'true'
    ? true
    : configuredAllowedHosts.length > 0
      ? configuredAllowedHosts
      : undefined

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: frontendPort,
    strictPort: true,
    allowedHosts,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${backendPort}`,
        changeOrigin: true,
      },
      '/new-synth': {
        target: `http://127.0.0.1:${newSynthPort}`,
        changeOrigin: true,
        ws: true,
      },
    },
  },
  preview: {
    host: '0.0.0.0',
    port: 4173,
    strictPort: true,
  },
})
