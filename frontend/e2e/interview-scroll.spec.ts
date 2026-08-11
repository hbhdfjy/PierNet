import { expect, test } from '@playwright/test'

test.use({ viewport: { width: 1592, height: 896 } })

test('long interview confirmation editor scrolls without hiding actions', async ({ page }) => {
  await page.route('**/api/registry', route => route.fulfill({ json: {} }))
  await page.route('**/api/text2comp/scenarios', route => route.fulfill({ json: {} }))
  await page.route('**/api/llm/config', route =>
    route.fulfill({
      json: {
        provider: 'test',
        model: 'test-model',
        has_api_key: true,
      },
    }),
  )
  await page.route('**/api/interview/start', route =>
    route.fulfill({
      json: {
        session_id: 'scroll-test',
        step: 3,
        step_label: '输出通道结构',
        total_steps: 6,
        question: '请确认输出通道结构',
        extracted: {
          output_info: [
            { name: 'critical_load_kN', name_zh: '临界载荷', description: '临界载荷', unit: 'kN', slice: [0, 1] },
            { name: 'safety_factor', name_zh: '安全系数', description: '稳定裕度', unit: '无量纲', slice: [1, 2] },
            { name: 'stable_flag', name_zh: '稳定标志', description: '稳定状态', unit: 'bool', slice: [2, 3] },
          ],
        },
        extraction_uncertain: false,
        needs_confirmation: true,
        done: false,
        saved: false,
        registry_key: null,
        error: null,
        hdf5_loaded: true,
        github_prefilled: null,
      },
    }),
  )

  await page.goto('/synth/register')
  await page.getByPlaceholder('如 modflow、simpeg、fenics').fill('mechanics')
  const startResponse = page.waitForResponse(response => response.url().endsWith('/api/interview/start'))
  await page.getByRole('button', { name: '开始注册仿真器' }).click()
  await startResponse

  const scrollRegion = page.getByTestId('interview-extraction-scroll')
  await expect(scrollRegion).toBeVisible()
  const before = await scrollRegion.evaluate(element => ({
    clientHeight: element.clientHeight,
    scrollHeight: element.scrollHeight,
    scrollTop: element.scrollTop,
  }))
  expect(before.scrollHeight).toBeGreaterThan(before.clientHeight)

  await scrollRegion.hover()
  await page.mouse.wheel(0, 500)
  await expect.poll(() => scrollRegion.evaluate(element => element.scrollTop)).toBeGreaterThan(0)
  await expect(page.getByRole('button', { name: '确认，下一步' })).toBeVisible()
})
