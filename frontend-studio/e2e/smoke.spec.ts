import { expect, test } from "@playwright/test";

test("opens the independent Studio workspace", async ({ page }) => {
  await page.goto("./");
  await expect(page).toHaveTitle(/PiERN Studio/);
  await expect(page.getByRole("heading", { name: "科学计算项目" })).toBeVisible();
  await expect(page.getByRole("link", { name: "创建项目" })).toBeVisible();
});

test("creates a project from the user goal", async ({ page }) => {
  await page.goto("./new");
  await page.getByLabel("项目名称").fill(`界面验证 ${Date.now()}`);
  await page.getByLabel("计算目标").fill("根据用户给定的三个参数，预测下一时刻的八个状态值。");
  await page.getByRole("button", { name: "准备资源" }).click();
  await expect(page).toHaveURL(/\/studio\/projects\/studio-[a-f0-9]+\/resources/);
  await expect(page.getByRole("heading", { name: "上传你的数据与计算模型" })).toBeVisible();

  await page.goto(page.url().replace(/\/resources$/, ""));
  await expect(page.getByRole("heading", { name: /界面验证/ })).toBeVisible();
  page.once("dialog", (dialog) => void dialog.accept());
  await page.getByRole("button", { name: "删除项目" }).click();
  await expect(page).toHaveURL(/\/studio\/?$/);
  await expect(page.getByRole("heading", { name: "科学计算项目" })).toBeVisible();
});

test("keeps core controls labeled and keyboard reachable", async ({ page }) => {
  await page.goto("./new");
  await expect(page.locator("main")).toHaveCount(1);
  await expect(page.getByLabel("项目名称")).toBeVisible();
  await expect(page.getByLabel("计算目标")).toBeVisible();
  await page.getByLabel("项目名称").focus();
  await expect(page.getByLabel("项目名称")).toBeFocused();
  const unnamedInteractiveCount = await page
    .locator("button, a[href], input:not([type='hidden']), textarea, select")
    .evaluateAll(
      (elements) =>
        elements.filter((element) => {
          const text = element.textContent?.trim();
          const label = element.getAttribute("aria-label");
          const labelledBy = element.getAttribute("aria-labelledby");
          const title = element.getAttribute("title");
          if (text || label || labelledBy || title) return false;
          if (
            element instanceof HTMLInputElement ||
            element instanceof HTMLTextAreaElement ||
            element instanceof HTMLSelectElement
          ) {
            return !element.labels?.length;
          }
          return true;
        }).length,
    );
  expect(unnamedInteractiveCount).toBe(0);
});

test("hides the field editor after the mapping is saved", async ({ page }) => {
  await page.route("**/api/studio/projects/studio-mapped", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        project_id: "studio-mapped",
        name: "字段确认测试",
        goal: "根据输入计算输出",
        status: "draft",
        current_stage: "inspection",
        created_at: 0,
        updated_at: 0,
        stages: [],
        data: {
          filename: "mapped.csv",
          format: "csv",
          size_bytes: 1024,
          needs_mapping: false,
          samples: 8,
          input_shape: [2],
          input_dim: 2,
          output_shape: [1],
          output_dim: 1,
          input_names: ["a", "b"],
          output_names: ["result"],
        },
        expert: null,
        inspection: {
          data: {
            kind: "table",
            needs_mapping: true,
            columns: [
              { name: "a", dtype: "float64", numeric: true, sample: [1, 2] },
              { name: "b", dtype: "float64", numeric: true, sample: [3, 4] },
              { name: "result", dtype: "float64", numeric: true, sample: [4, 6] },
            ],
            suggested_input_fields: ["a", "b"],
            suggested_output_fields: ["result"],
          },
        },
        compatibility: null,
        artifacts: null,
        result: null,
        error: null,
        recommended_prompt: null,
        can_run: false,
        can_chat: false,
      }),
    });
  });

  await page.goto("./projects/studio-mapped/resources");
  await expect(page.getByRole("heading", { name: "结构清晰，可以用于构建" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "哪些列是输入，哪些列是输出？" })).toHaveCount(0);
});

test("downloads the visible calculation result as JSON", async ({ page }) => {
  await page.route("**/api/studio/projects/studio-download", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        project_id: "studio-download",
        name: "结果下载测试",
        goal: "根据输入计算输出",
        status: "ready",
        current_stage: "validation",
        created_at: 0,
        updated_at: 0,
        stages: [],
        data: {
          filename: "data.npz",
          format: "npz",
          size_bytes: 1024,
          needs_mapping: false,
          samples: 8,
          input_shape: [2],
          input_dim: 2,
          output_shape: [2],
          output_dim: 2,
          input_names: ["a", "b"],
          output_names: ["x", "y"],
        },
        expert: null,
        inspection: null,
        compatibility: null,
        artifacts: null,
        result: {
          message: "a=1, b=2",
          answer: "计算完成。",
          routed: true,
          confidence: 0.99,
          inputs: [1, 2],
          output: [3, 4],
          chart: {
            kind: "line",
            x: [1, 2],
            series: [{ name: "结果", values: [3, 4] }],
          },
          latency_ms: 12,
        },
        error: null,
        recommended_prompt: "a=1, b=2",
        can_run: false,
        can_chat: true,
      }),
    });
  });

  await page.goto("./projects/studio-download/demo");
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "下载 JSON" }).click();
  const download = await downloadPromise;

  expect(download.suggestedFilename()).toBe("piern-studio-result.json");
});
