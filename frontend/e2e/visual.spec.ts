import { expect, test, type Page } from '@playwright/test'

async function mockApi(page: Page) {
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    if (route.request().method() !== 'GET') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true }) })
      return
    }
    if (url.pathname === '/api/training/overview') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ completed_job_count: 0, running_job_count: 0, datasets: [], gpus: [], jobs: [] }),
      })
      return
    }
    if (url.pathname === '/api/training/jobs') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([]) })
      return
    }
    if (url.pathname === '/api/config/text2comp-scenarios') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          modflow: [
            {
              name: 'coastal_seawater',
              simulator: 'MODFLOW',
              h5_file: 'data/modflow/modflow_coastal_seawater.h5',
              sample_count: 1000,
              output_shape: [5, 365],
              existing_jsonl_count: 1000,
              has_jsonl: true,
              has_h5: true,
              registered: true,
            },
          ],
        }),
      })
      return
    }
    if (url.pathname === '/api/config') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          seed: 42,
          generation: {
            n_samples_per_scenario: 100,
            max_workers: 2,
            language_mix: 0.5,
            styles: ['technical', 'popular', 'concise'],
            style_weights: [0.5, 0.3, 0.2],
            transform_prob: 0.3,
          },
        }),
      })
      return
    }
    if (url.pathname === '/api/generate/jobs') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([]) })
      return
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) })
  })
}

test('key workbench pages produce non-empty screenshots without overflow', async ({ page }) => {
  await mockApi(page)
  await page.setViewportSize({ width: 1440, height: 900 })
  const routes = [
    { path: '/', text: 'PiERN 控制台' },
    { path: '/training', text: 'PiERN 训练' },
    { path: '/training/jobs', text: '训练任务' },
    { path: '/synth/fill', text: '样本填充' },
  ]
  for (const route of routes) {
    await page.goto(route.path)
    await expect(page.getByText(route.text).first()).toBeVisible()
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
    expect(overflow).toBeLessThanOrEqual(3)
    const screenshot = await page.screenshot({ fullPage: true })
    expect(screenshot.readUInt32BE(16)).toBeGreaterThanOrEqual(1200)
    expect(screenshot.readUInt32BE(20)).toBeGreaterThanOrEqual(700)
  }
})
