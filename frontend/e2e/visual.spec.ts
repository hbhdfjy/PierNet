import { expect, test, type Page } from '@playwright/test'

async function mockApi(page: Page) {
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    if (route.request().method() !== 'GET') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true }) })
      return
    }
    if (url.pathname === '/api/jobs') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            job_id: 'router-ci-001',
            platform: 'synth',
            job_type: 'router',
            status: 'running',
            name: 'router-ci-001',
            created_at: 1,
            started_at: 1,
            finished_at: null,
            progress: {},
            stats: {},
            error_message: null,
            source: 'synth_jobs',
          },
        ]),
      })
      return
    }
    if (url.pathname === '/api/jobs/audit/events') {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ items: [] }) })
      return
    }
    if (url.pathname === '/api/storage/integrity') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ok: true,
          manifest_exists: true,
          manifest_path: 'data/.manifests/source_integrity.json',
          checked_entries: 44,
          scanned_entries: 44,
          errors: [],
          generated_at: 1,
        }),
      })
      return
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) })
  })
}

test('key workbench pages produce non-empty screenshots without overflow', async ({ page }) => {
  await mockApi(page)
  await page.setViewportSize({ width: 1440, height: 900 })
  for (const route of ['/tasks', '/training/tasks', '/synth/tasks']) {
    await page.goto(route)
    await expect(page.locator('body')).toBeVisible()
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
    expect(overflow).toBeLessThanOrEqual(3)
    const visibleTextLength = await page.evaluate(() => document.body.innerText.trim().length)
    expect(visibleTextLength).toBeGreaterThan(20)
    const screenshot = await page.screenshot({ fullPage: true })
    expect(screenshot.readUInt32BE(16)).toBeGreaterThanOrEqual(1200)
    expect(screenshot.readUInt32BE(20)).toBeGreaterThanOrEqual(700)
  }
})
