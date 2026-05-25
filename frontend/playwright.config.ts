import { defineConfig, devices } from '@playwright/test'

const externalBaseURL = process.env.PierNet_E2E_BASE_URL
const e2ePort = process.env.PierNet_E2E_PORT ?? '5174'
const baseURL = externalBaseURL ?? `http://127.0.0.1:${e2ePort}`
const shouldStartServer =
  process.env.PierNet_E2E_START_SERVER === '1' || (!externalBaseURL && process.env.PierNet_E2E_START_SERVER !== '0')

export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  expect: { timeout: 8_000 },
  fullyParallel: true,
  reporter: [['list']],
  webServer: shouldStartServer
    ? {
        command: `npm run dev -- --host 127.0.0.1 --port ${e2ePort}`,
        url: baseURL,
        reuseExistingServer: false,
        timeout: 60_000,
      }
    : undefined,
  use: {
    baseURL,
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
})
