import { expect, test } from '@playwright/test'

async function expectNoHorizontalOverflow(page) {
  const overflow = await page.evaluate(() => {
    const root = document.documentElement
    return root.scrollWidth - window.innerWidth
  })
  expect(overflow).toBeLessThanOrEqual(3)
}

test('platform entry and core workbenches render without layout overflow', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })

  await page.goto('/')
  await expect(page.getByText('PiERN 工作台')).toBeVisible()
  await expect(page.getByRole('link', { name: /进入训练平台/ })).toBeVisible()
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
  await expect(page.getByText('PiERN 工作台')).toBeVisible()
  await expect(page.getByRole('link', { name: /打开数据平台/ })).toBeVisible()
  await expectNoHorizontalOverflow(page)
})
