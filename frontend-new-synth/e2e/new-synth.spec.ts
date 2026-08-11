import { expect, test, type Page } from "@playwright/test";

const hdf5Path = process.env.NEW_SYNTH_E2E_HDF5;
const captureScreenshots = process.env.NEW_SYNTH_CAPTURE_SCREENSHOTS === "1";
const testLiveLlm = process.env.NEW_SYNTH_E2E_LLM === "1";

async function capture(page: Page, projectName: string, name: string) {
  if (!captureScreenshots) return;
  await page.screenshot({
    path: `artifacts/screenshots/${projectName}-${name}.png`,
    fullPage: projectName !== "mobile",
  });
}

async function captureMobileSection(
  page: Page,
  projectName: string,
  selector: string,
  name: string,
) {
  if (!captureScreenshots || projectName !== "mobile") return;
  await page
    .locator(selector)
    .evaluate((element) => element.scrollIntoView({ block: "start" }));
  await page.waitForTimeout(150);
  await capture(page, projectName, name);
}

test("a real HDF5 file reaches both training entry points", async ({
  page,
}, testInfo) => {
  test.skip(!hdf5Path, "NEW_SYNTH_E2E_HDF5 is required for the real-data flow");
  await page.goto("./");
  await expect(
    page.getByRole("heading", { name: "从你的数据开始" }),
  ).toBeVisible();

  await page
    .locator('input[type="file"][accept=".h5,.hdf5"]')
    .setInputFiles(hdf5Path!);
  await expect(page.getByText("数据可用")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("4 维", { exact: true })).toBeVisible();
  await capture(page, testInfo.project.name, "01-source");
  await captureMobileSection(
    page,
    testInfo.project.name,
    ".inspection-panel",
    "01b-source-details",
  );

  await page.getByRole("button", { name: /确认并继续/ }).click();
  await expect(
    page.getByRole("heading", { name: "确认数据定义" }),
  ).toBeVisible();
  await page.getByLabel("领域标识").fill(`e2e_${testInfo.project.name}`);
  await page.getByLabel("场景标识").fill("real_hdf5");
  await capture(page, testInfo.project.name, "02-definition");
  await captureMobileSection(
    page,
    testInfo.project.name,
    ".definition-section:nth-of-type(2)",
    "02a-definition-parameters",
  );
  await captureMobileSection(
    page,
    testInfo.project.name,
    ".definition-section:nth-of-type(3)",
    "02b-definition-output",
  );
  await page.getByRole("button", { name: /保存并继续/ }).click();

  await expect(
    page.getByRole("heading", { name: "开始自动生成" }),
  ).toBeVisible();
  await page.getByLabel("使用源样本").fill("4");
  await page.getByLabel("每条语言变体").fill("1");
  await page.getByLabel("负样本比例").fill("1");
  await capture(page, testInfo.project.name, "03-generate");
  await page.getByRole("button", { name: /开始生成/ }).click();

  await expect(
    page.getByRole("heading", { name: "训练数据已就绪" }),
  ).toBeVisible({ timeout: 30_000 });
  const simpleLink = page.getByRole("link", { name: /进入简洁训练/ });
  const complexLink = page.getByRole("link", { name: "复杂训练" });
  await expect(simpleLink).toHaveAttribute(
    "href",
    /\/training\/simple\?datasetId=router-/,
  );
  await expect(complexLink).toHaveAttribute(
    "href",
    /\/training\/new\?datasetId=router-/,
  );
  await expect(page.getByText("专家输入标签")).toBeVisible();
  await capture(page, testInfo.project.name, "04-result");
  await captureMobileSection(
    page,
    testInfo.project.name,
    ".result-section",
    "04b-result-datasets",
  );

  const routerId = (await simpleLink.getAttribute("href"))?.match(
    /datasetId=(router-[^&]+)/,
  )?.[1];
  expect(routerId).toBeTruthy();
  const platformBaseUrl =
    process.env.PIERN_PLATFORM_BASE_URL ?? "http://127.0.0.1:3000";

  await page.goto(`${platformBaseUrl}/training/simple?datasetId=${routerId}`);
  await expect(page.getByRole("heading", { name: "模型训练" })).toBeVisible();
  await expect(page.locator(".training-simple-dataset--active")).toContainText(
    "new-synth-real · Router",
  );

  await page.goto(`${platformBaseUrl}/training/new?datasetId=${routerId}`);
  await expect(
    page.getByRole("heading", { name: "Token Router 训练" }),
  ).toBeVisible();
  await expect(page.getByRole("combobox").first()).toHaveValue(routerId!);
});

test("an existing simulation reaches registered training data", async ({
  page,
}) => {
  await page.goto("./");
  await page.getByRole("tab", { name: "运行内置仿真" }).click();
  await page.getByLabel("仿真场景").selectOption("modflow/coastal_seawater");
  await page.getByLabel("样本数").fill("4");
  await page.getByRole("button", { name: "接入已有数据" }).click();

  await expect(page.getByText("数据可用")).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: /确认并继续/ }).click();
  await expect(
    page.getByRole("heading", { name: "确认数据定义" }),
  ).toBeVisible();
  await page.getByRole("button", { name: /保存并继续/ }).click();

  await expect(
    page.getByRole("heading", { name: "开始自动生成" }),
  ).toBeVisible();
  await page.getByLabel("使用源样本").fill("4");
  await page.getByLabel("每条语言变体").fill("1");
  await page.getByLabel("负样本比例").fill("1");
  await page.getByRole("button", { name: /开始生成/ }).click();

  await expect(
    page.getByRole("heading", { name: "训练数据已就绪" }),
  ).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("专家输入标签")).toBeVisible();
  await expect(
    page.getByRole("link", { name: /进入简洁训练/ }),
  ).toHaveAttribute("href", /datasetId=router-/);
});

test("API settings are available inside the isolated frontend", async ({
  page,
}, testInfo) => {
  await page.goto("./");
  await page.getByRole("button", { name: /API 设置/ }).click();
  const dialog = page.getByRole("dialog", { name: "智能识别 API" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByLabel("服务商")).toBeVisible();
  await expect(dialog.getByLabel("模型")).toBeVisible();
  await expect(dialog.getByLabel("API Key")).toBeVisible();
  await capture(page, testInfo.project.name, "05-api-settings");
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
  await page.getByRole("button", { name: "关闭 API 设置" }).click();
  await expect(dialog).not.toBeVisible();
});

test("the saved API enriches a real definition without changing its contract", async ({
  page,
}, testInfo) => {
  test.skip(!testLiveLlm, "NEW_SYNTH_E2E_LLM=1 is required");
  test.skip(
    testInfo.project.name !== "desktop",
    "live LLM verification runs once",
  );
  test.setTimeout(180_000);

  await page.goto("./");
  await page.getByRole("button", { name: /API 设置/ }).click();
  const dialog = page.getByRole("dialog", { name: "智能识别 API" });
  await expect(dialog.getByLabel("模型")).toHaveValue(
    "deepseek-ai/DeepSeek-V3.2",
  );
  await dialog.getByRole("button", { name: "保存并测试" }).click();
  await expect(dialog.getByText("连接成功")).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "关闭 API 设置" }).click();

  await page.getByRole("tab", { name: "运行内置仿真" }).click();
  await page.getByLabel("仿真场景").selectOption("modflow/coastal_seawater");
  await page.getByLabel("样本数").fill("4");
  await page.getByRole("button", { name: "接入已有数据" }).click();
  await expect(page.getByText("数据可用")).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: /确认并继续/ }).click();
  await expect(
    page.getByRole("heading", { name: "确认数据定义" }),
  ).toBeVisible();

  const taskDescription = page.getByLabel("任务描述");
  const taskBefore = await taskDescription.inputValue();
  const contractBefore = await page
    .locator(".definition-table tbody code")
    .allTextContents();
  const strideBefore = await page.getByLabel("时间步长").inputValue();
  await page.getByRole("button", { name: "智能补全说明" }).click();
  await expect(page.getByText("说明已智能补全，请检查后再保存。")).toBeVisible({
    timeout: 150_000,
  });
  await expect(taskDescription).not.toHaveValue(taskBefore);
  expect(
    await page.locator(".definition-table tbody code").allTextContents(),
  ).toEqual(contractBefore);
  await expect(page.getByLabel("时间步长")).toHaveValue(strideBefore);
  await capture(page, testInfo.project.name, "06-definition-suggested");
});
