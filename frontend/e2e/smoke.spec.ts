import { expect, test, type Page } from '@playwright/test'

const now = Math.floor(Date.now() / 1000)

const trainingDatasets = [
  {
    simulator: 'MODFLOW',
    total_count: 4_400_000,
    scenarios: ['coastal_seawater', 'heterogeneous_unconfined', 'lake_interaction', 'monsoon_seasonal'].map(
      scenario => ({
        simulator: 'MODFLOW',
        scenario,
        router_count: 1_100_000,
        file_size_bytes: 480_000_000,
        mtime: now,
        path: `data/router/modflow/${scenario}.parquet`,
      }),
    ),
  },
]

const trainingGpus = [
  {
    index: 0,
    name: 'CI GPU Stub',
    memory_used_mib: 128,
    memory_total_mib: 24_576,
    utilization_gpu: 0,
    available: true,
    locked_by_job_id: null,
    reason: null,
  },
]

const text2CompScenarios = {
  modflow: [
    {
      name: 'coastal_seawater',
      simulator: 'MODFLOW',
      h5_file: 'data/raw/modflow/coastal_seawater.h5',
      sample_count: 1000,
      output_shape: [5, 16],
      existing_jsonl_count: 1000,
      has_jsonl: true,
      has_h5: true,
      registered: true,
    },
  ],
}

const templates = [
  {
    scenario: 'coastal_seawater',
    template_count: 1000,
    file_size_bytes: 128_000,
    mtime: now,
    path: 'data/templates/coastal_seawater_templates.jsonl',
  },
]

function apiPayloadFor(pathname: string): unknown {
  if (pathname === '/api/training/overview') {
    return {
      completed_job_count: 0,
      running_job_count: 0,
      datasets: trainingDatasets,
      gpus: trainingGpus,
      jobs: [],
    }
  }
  if (pathname === '/api/training/datasets') return trainingDatasets
  if (pathname === '/api/training/gpus') return trainingGpus
  if (pathname === '/api/training/jobs') return []
  if (pathname === '/api/config/text2comp-scenarios') return text2CompScenarios
  if (pathname === '/api/config') {
    return {
      seed: 42,
      generation: {
        n_samples_per_scenario: 100,
        max_workers: 2,
        language_mix: 0.5,
        styles: ['technical', 'popular', 'concise'],
        style_weights: [0.5, 0.3, 0.2],
        transform_prob: 0.3,
      },
    }
  }
  if (pathname === '/api/templates') return templates
  if (pathname === '/api/generate/jobs') return []
  return {}
}

async function mockApi(page: Page) {
  await page.route('**/api/**', async route => {
    if (route.request().method() !== 'GET') {
      await route.continue()
      return
    }

    const url = new URL(route.request().url())
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(apiPayloadFor(url.pathname)),
    })
  })
}

async function expectNoHorizontalOverflow(page: Page) {
  const overflow = await page.evaluate(() => {
    const root = document.documentElement
    return root.scrollWidth - window.innerWidth
  })
  expect(overflow).toBeLessThanOrEqual(3)
}

async function setTheme(page: Page, theme: 'dark' | 'light') {
  await page.addInitScript(value => {
    localStorage.setItem('piern-theme', value)
  }, theme)
}

test.beforeEach(async ({ page }) => {
  await mockApi(page)
})

test('platform entry and core workbenches render without layout overflow', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })

  await page.goto('/')
  await expect(page.getByText('PiERN 控制台')).toBeVisible()
  await expect(page.getByRole('link', { name: /打开训练平台/ })).toBeVisible()
  await expectNoHorizontalOverflow(page)

  await page.goto('/training')
  await expect(page.getByText('PiERN 训练')).toBeVisible()
  await expect(page.getByText('训练平台')).toBeVisible()
  await expect(page.getByRole('link', { name: /任务管理/ })).toBeVisible()
  await expectNoHorizontalOverflow(page)

  await page.goto('/training/jobs')
  await expect(page.getByRole('heading', { name: '训练任务' })).toBeVisible()
  await expect(page.getByText('任务列表')).toBeVisible()
  await expectNoHorizontalOverflow(page)

  await page.goto('/synth/fill')
  await expect(page.getByText('PiERN 数据')).toBeVisible()
  await expect(page.getByRole('heading', { name: '样本填充' })).toBeVisible()
  await expectNoHorizontalOverflow(page)
})

test('mobile shell keeps primary navigation usable', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })

  await page.goto('/')
  await expect(page.getByText('PiERN 控制台')).toBeVisible()
  await expect(page.getByRole('link', { name: /打开数据平台/ })).toBeVisible()
  await expectNoHorizontalOverflow(page)
})

test('dark and light themes keep core pages visually stable', async ({ page }) => {
  const routes = ['/', '/training', '/training/new', '/training/jobs', '/synth/fill', '/synth/router']

  for (const theme of ['dark', 'light'] as const) {
    await setTheme(page, theme)
    await page.setViewportSize({ width: 1366, height: 820 })

    for (const route of routes) {
      await page.goto(route)
      await expect(page.locator('body')).toBeVisible()
      await expect
        .poll(async () => page.locator('body').evaluate(node => node.textContent?.trim().length ?? 0), {
          message: `${theme} ${route} rendered text`,
        })
        .toBeGreaterThan(20)
      await expectNoHorizontalOverflow(page)
    }
  }
})

test('mobile training and synthesis shells avoid horizontal overflow', async ({ page }) => {
  await setTheme(page, 'light')
  await page.setViewportSize({ width: 390, height: 844 })

  for (const route of ['/training', '/training/new', '/synth/fill', '/synth/router']) {
    await page.goto(route)
    await expect(page.locator('body')).toBeVisible()
    await expect
      .poll(async () => page.locator('body').evaluate(node => node.textContent?.trim().length ?? 0), {
        message: `${route} rendered text`,
      })
      .toBeGreaterThan(20)
    await expectNoHorizontalOverflow(page)
  }
})
