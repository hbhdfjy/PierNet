import { expect, test, type Page } from '@playwright/test'

const now = Math.floor(Date.now() / 1000)

test.describe.configure({ mode: 'serial' })

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

const text2CompOverview = {
  expert_models: [{ name: 'MODFLOW', domain: 'groundwater', output_dim: 128, description: 'MODFLOW Text2Comp' }],
  datasets: [
    {
      path: 'data/text2comp/modflow/coastal_seawater.jsonl',
      simulator: 'MODFLOW',
      scenario: 'coastal_seawater',
      n_samples: 1000,
      file_size_bytes: 128_000,
      mtime: now,
    },
  ],
  gpus: trainingGpus,
  jobs: [],
  running_job_count: 0,
  completed_job_count: 0,
}

const uploadedExpertModels = {
  interface: 'def predict(inputs: list[float]) -> float | list[float]',
  interface_version: 3,
  constraints: {},
  example_source: '',
  max_model_bytes: 1_073_741_824,
  max_input_dim: 4096,
  max_input_points: 1_000_000,
  models: [
    {
      model_id: 'uploaded-expert-ci',
      name: 'uploaded_expert_ci',
      status: 'active',
      runtime: 'python',
      file_name: 'expert.py',
      path: '/models/uploaded/expert.py',
      created_at: now,
      file_size_bytes: 1024,
      input_dim: 128,
      output_dim: 64,
      assembly_enabled: true,
      data_generation_enabled: true,
      interface: 'predict',
      interface_version: 3,
      exists: true,
    },
  ],
}

const assemblyStatus = {
  llms: [{ name: 'Qwen2.5-0.5B-Instruct', path: '/models/qwen', size: '0.5B', downloaded: true }],
  routers: [
    {
      name: 'router-final',
      path: '/models/router_final.pt',
      num_classes: 3,
      class_names: ['normal', 'modflow', 'gcam'],
      description: 'CI Router',
      trained: true,
    },
  ],
  text2comps: [
    {
      name: 'text2comp-final',
      simulator: 'MODFLOW',
      output_dim: 128,
      path: '/models/text2comp/final_model.pt',
      trained: true,
      description: 'CI Text2Comp',
    },
  ],
  fno_experts: [
    {
      name: 'fno-modflow',
      simulator: 'MODFLOW',
      input_dim: 128,
      output_shape: [5, 16],
      path: '/models/fno/modflow.pt',
      trained: true,
      description: 'CI FNO',
    },
  ],
  gpus: [
    {
      index: 0,
      name: 'CI GPU Stub',
      memory_used_mb: 128,
      memory_free_mb: 24_448,
      memory_total_mb: 24_576,
      available: true,
    },
  ],
  loaded_models: {
    llm: { loaded: false, path: null },
    router: { loaded: false, path: null },
    text2comp: { loaded: false, paths: [] },
    fno: { loaded: false, paths: [] },
    uploaded_expert: { loaded: false, model_id: null, path: null, executor: null },
  },
  custom_experts: uploadedExpertModels.models,
  gpu_available: true,
  architecture_note: 'CI assembly stub',
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

const fileCatalog = {
  generated_at: now,
  assets: [
    {
      id: 'synth-template-modflow',
      platform: 'synth',
      platform_label: '数据合成',
      stage: 'stage2',
      stage_label: '阶段 2 模板',
      kind: 'template',
      kind_label: '模板 JSONL',
      title: 'coastal_seawater_templates.jsonl',
      simulator: 'MODFLOW',
      scenario: 'coastal_seawater',
      job_id: null,
      path: 'data/templates/coastal_seawater_templates.jsonl',
      count: 1000,
      count_label: 'templates',
      file_size_bytes: 128_000,
      mtime: now,
      valid: true,
      status: 'ok',
      protected: true,
      deletable: false,
      warnings: [],
      errors: [],
      details: {},
    },
    {
      id: 'training-job-001',
      platform: 'training',
      platform_label: '训练平台',
      stage: 'training',
      stage_label: '训练产物',
      kind: 'training_job',
      kind_label: '训练任务',
      title: 'train-job-001',
      simulator: 'MODFLOW',
      scenario: 'coastal_seawater',
      job_id: 'train-job-001',
      path: 'data/training/jobs/train-job-001',
      count: 1,
      count_label: 'job',
      file_size_bytes: 64_000,
      mtime: now,
      valid: true,
      status: 'ok',
      protected: true,
      deletable: false,
      warnings: [],
      errors: [],
      details: {},
    },
  ],
  summary: {
    total_assets: 2,
    total_size_bytes: 192_000,
    invalid_count: 0,
    deletable_count: 0,
    protected_count: 2,
    by_platform: { synth: 1, training: 1 },
    by_stage: { stage2: 1, training: 1 },
    by_kind: { template: 1, training_job: 1 },
  },
}

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
  if (pathname === '/api/training/simple-datasets') return trainingDatasets
  if (pathname === '/api/training/gpus') return trainingGpus
  if (pathname === '/api/training/jobs') return []
  if (pathname === '/api/expert-models') return uploadedExpertModels
  if (pathname === '/api/text2comp/overview' || pathname === '/api/text2comp/status') return text2CompOverview
  if (pathname === '/api/text2comp/datasets') return text2CompOverview.datasets
  if (pathname === '/api/text2comp/gpus') return text2CompOverview.gpus
  if (pathname === '/api/text2comp/jobs') return []
  if (pathname === '/api/assembly/status') return assemblyStatus
  if (pathname === '/api/assembly/gpus') return assemblyStatus.gpus
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
  if (pathname === '/api/files/catalog') return fileCatalog
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

async function gotoApp(page: Page, path: string) {
  await page.goto(path, { waitUntil: 'domcontentloaded' })
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
    localStorage.setItem('PierNet-theme', value)
  }, theme)
}

test.beforeEach(async ({ page }) => {
  await mockApi(page)
})

test('platform entry and core workbenches render without layout overflow', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })

  await gotoApp(page, '/')
  await expect(page.getByRole('heading', { name: '工业与科学时序语言多模态大模型自动化训练平台' })).toBeVisible()
  await expect(page.getByRole('link', { name: /开始创建/ })).toBeVisible()
  await expect(page.getByRole('link', { name: /查看已有任务/ })).toBeVisible()
  await expectNoHorizontalOverflow(page)

  await gotoApp(page, '/training')
  await expect(page.getByText('Piern 训练')).toBeVisible()
  await expect(page.getByText('训练平台')).toBeVisible()
  await expect(page.getByRole('link', { name: /任务管理/ })).toBeVisible()
  await expectNoHorizontalOverflow(page)

  await gotoApp(page, '/training/jobs')
  await expect(page.getByRole('heading', { name: '训练任务' })).toBeVisible()
  await expect(page.getByText('任务列表')).toBeVisible()
  await expectNoHorizontalOverflow(page)

  await gotoApp(page, '/training/files')
  await expect(page.getByRole('heading', { name: '训练文件管理' })).toBeVisible()
  await expect(page.getByRole('table').getByText('train-job-001', { exact: true })).toBeVisible()
  await expect(page.getByText('coastal_seawater_templates.jsonl')).not.toBeVisible()
  await expectNoHorizontalOverflow(page)

  await gotoApp(page, '/synth/fill')
  await expect(page.getByText('Piern 数据')).toBeVisible()
  await expect(page.getByRole('heading', { name: '样本填充' })).toBeVisible()
  await expectNoHorizontalOverflow(page)
})

test('mobile shell keeps primary navigation usable', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })

  await gotoApp(page, '/')
  await expect(page.getByRole('heading', { name: '工业与科学时序语言多模态大模型自动化训练平台' })).toBeVisible()
  await expect(page.getByRole('link', { name: /开始创建/ })).toBeVisible()
  await expectNoHorizontalOverflow(page)
})

test('dark and light themes keep core pages visually stable', async ({ page }) => {
  test.setTimeout(60_000)

  const routes = [
    '/',
    '/training',
    '/training/simple',
    '/training/simple/tasks',
    '/training/simple/assembly',
    '/training/simple/chat',
    '/training/new',
    '/training/jobs',
    '/training/chat',
    '/training/files',
    '/synth/fill',
    '/synth/router',
    '/synth/files',
  ]

  for (const theme of ['dark', 'light'] as const) {
    await setTheme(page, theme)
    await page.setViewportSize({ width: 1366, height: 820 })

    for (const route of routes) {
      await gotoApp(page, route)
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

  for (const route of [
    '/training',
    '/training/simple',
    '/training/simple/tasks',
    '/training/simple/assembly',
    '/training/simple/chat',
    '/training/new',
    '/training/chat',
    '/training/files',
    '/synth/fill',
    '/synth/router',
    '/synth/files',
  ]) {
    await gotoApp(page, route)
    await expect(page.locator('body')).toBeVisible()
    await expect
      .poll(async () => page.locator('body').evaluate(node => node.textContent?.trim().length ?? 0), {
        message: `${route} rendered text`,
      })
      .toBeGreaterThan(20)
    await expectNoHorizontalOverflow(page)
  }
})

test('desktop shell keeps brand, navigation and utility controls visible on deep pages', async ({ page }) => {
  await setTheme(page, 'dark')
  await page.setViewportSize({ width: 1440, height: 900 })

  for (const route of ['/synth/files', '/training/assembly', '/training/chat']) {
    await gotoApp(page, route)
    await expect(page.locator('.app-brand')).toBeVisible()
    await expect(page.locator('.app-sidebar__footer')).toBeVisible()
    await expect(page.locator('.app-nav .nav-item--active')).toBeVisible()

    const geometry = await page.evaluate(() => {
      const rect = (selector: string) => document.querySelector(selector)?.getBoundingClientRect()
      const brand = rect('.app-brand')
      const switcher = rect('.platform-switcher')
      const nav = rect('.app-nav')
      const active = rect('.app-nav .nav-item--active')
      const footer = rect('.app-sidebar__footer')
      return {
        viewportHeight: window.innerHeight,
        brandTop: brand?.top ?? -1,
        brandBottom: brand?.bottom ?? -1,
        switcherTop: switcher?.top ?? -1,
        navTop: nav?.top ?? -1,
        navBottom: nav?.bottom ?? -1,
        activeTop: active?.top ?? -1,
        activeBottom: active?.bottom ?? -1,
        footerTop: footer?.top ?? -1,
        footerBottom: footer?.bottom ?? -1,
      }
    })

    expect(geometry.brandTop).toBeGreaterThanOrEqual(0)
    expect(geometry.switcherTop).toBeGreaterThanOrEqual(geometry.brandBottom)
    expect(geometry.navTop).toBeGreaterThanOrEqual(geometry.switcherTop)
    expect(geometry.activeTop).toBeGreaterThanOrEqual(geometry.navTop - 1)
    expect(geometry.activeBottom).toBeLessThanOrEqual(geometry.navBottom + 1)
    expect(geometry.footerTop).toBeGreaterThanOrEqual(geometry.navTop)
    expect(geometry.footerBottom).toBeLessThanOrEqual(geometry.viewportHeight + 1)
    await expectNoHorizontalOverflow(page)
  }
})

test('conversation training workflow renders as a standalone entry', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await gotoApp(page, '/conversation-training-demo.html')

  await expect(page.getByRole('heading', { name: '创建完整模型' })).toBeVisible()
  await expect(page.getByRole('button', { name: /新建对话/ })).toBeVisible()
  await expect(page.getByText('完整模型工作流')).toBeVisible()
  await expectNoHorizontalOverflow(page)
})
