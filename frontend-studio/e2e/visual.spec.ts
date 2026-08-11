import { expect, test } from "@playwright/test";
import type { ProjectSnapshot, ProjectStatus, StageSnapshot } from "../src/types";

const projectId = process.env.STUDIO_VISUAL_PROJECT_ID;
const fieldProjectId = process.env.STUDIO_VISUAL_FIELD_PROJECT_ID;
const sessionId = process.env.STUDIO_VISUAL_SESSION;

const baseStages: StageSnapshot[] = [
  "resources",
  "inspection",
  "compatibility",
  "preparation",
  "training",
  "assembly",
  "validation",
].map((id, index) => ({
  id,
  title: [
    "准备自己的资源",
    "理解数据与模型",
    "检查资源匹配",
    "准备训练内容",
    "训练 Demo",
    "创建可对话 Demo",
    "验证计算结果",
  ][index],
  status: index < 3 ? "succeeded" : "waiting",
  progress: index < 3 ? 1 : null,
  message: index < 3 ? "已完成" : "等待开始",
  retryable: false,
  started_at: null,
  finished_at: null,
}));

function buildState(status: ProjectStatus): ProjectSnapshot {
  const stages = baseStages.map((stage) => ({ ...stage }));
  const training = stages.find((stage) => stage.id === "training")!;
  const preparation = stages.find((stage) => stage.id === "preparation")!;
  preparation.status = "succeeded";
  preparation.progress = 1;
  preparation.message = "已准备 192 条训练语句";
  if (status === "running") {
    training.status = "running";
    training.progress = 0.46;
    training.message = "正在学习生成计算参数";
  } else {
    training.status = status === "cancelled" ? "cancelled" : "failed";
    training.progress = 0.46;
    training.message =
      status === "cancelled" ? "已安全停止，可以继续" : "构建没有完成，可以安全重试";
    training.retryable = true;
  }
  return {
    project_id: "visual-state",
    name: "热响应计算",
    goal: "根据材料参数预测温度响应",
    status,
    current_stage: "training",
    created_at: 0,
    updated_at: 0,
    stages,
    data: null,
    expert: null,
    inspection: null,
    compatibility: {
      compatible: true,
      sample_count: 3,
      input_shape: [3],
      expected_output_shape: [8],
      actual_output_shape: [8],
      finite: true,
    },
    artifacts: null,
    result: null,
    error:
      status === "running"
        ? null
        : {
            code: status,
            message: training.message,
          },
    recommended_prompt: null,
    can_run: status !== "running",
    can_chat: false,
  };
}

test("captures the workspace and creation flow", async ({ page }, testInfo) => {
  await page.goto("./");
  await expect(page.getByRole("heading", { name: "科学计算项目" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "把一组科学数据变成可对话 Demo" })).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("projects.png"),
    fullPage: true,
  });

  await page.goto("./new");
  await expect(page.getByRole("heading", { name: "你想完成什么计算？" })).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("create-project.png"),
    fullPage: true,
  });

  await page.getByLabel("项目名称").fill(`视觉验证 ${Date.now()}`);
  await page.getByLabel("计算目标").fill("根据用户上传的数据和计算模型预测系统响应。");
  await page.getByRole("button", { name: "准备资源" }).click();
  await expect(page.getByRole("heading", { name: "上传你的数据与计算模型" })).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("project-resources-empty.png"),
    fullPage: true,
  });
  await page
    .locator("input[type='file']")
    .first()
    .setInputFiles({
      name: "unsupported.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("not scientific data"),
    });
  await expect(page.getByText(/不支持此数据格式/)).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("project-resources-error.png"),
    fullPage: true,
  });
});

test("captures a prepared project when supplied", async ({ page }, testInfo) => {
  test.skip(!projectId || !sessionId, "Studio visual project credentials are not set");
  await page.context().addCookies([
    {
      name: "piern_studio_session",
      value: sessionId!,
      url: "http://127.0.0.1:3001",
      httpOnly: true,
      sameSite: "Lax",
    },
  ]);
  for (const route of ["", "resources", "mapping", "build", "demo"]) {
    await page.goto(`./projects/${projectId!}${route ? `/${route}` : ""}`);
    await expect(page.locator(".feedback--loading")).toHaveCount(0);
    if (route === "resources" || route === "mapping") {
      await expect(page.getByRole("heading", { name: "上传你的数据与计算模型" })).toBeVisible();
      await expect(page.getByRole("heading", { name: "数据与计算模型匹配" })).toBeVisible();
    } else if (route === "build") {
      await expect(page.getByRole("heading", { name: "你的计算 Demo 已就绪" })).toBeVisible();
    } else if (route === "demo") {
      await expect(page.getByRole("heading", { name: "计算模型已返回结果" })).toBeVisible();
    } else {
      await expect(page.getByText("端到端验证已完成，可以输入新参数进行计算。")).toBeVisible();
    }
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= document.documentElement.clientWidth,
      ),
    ).toBe(true);
    await page.screenshot({
      path: testInfo.outputPath(`project-${route || "overview"}.png`),
      fullPage: true,
    });
  }
});

test("captures bounded build failure and cancellation states", async ({ page }, testInfo) => {
  let snapshot = buildState("running");
  await page.route("**/api/studio/projects/visual-state", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(snapshot),
    });
  });
  for (const status of ["running", "failed", "cancelled"] as const) {
    snapshot = buildState(status);
    await page.goto("./projects/visual-state/build");
    await expect(page.locator(".feedback--loading")).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "把资源连接成可对话 Demo" })).toBeVisible();
    await page.screenshot({
      path: testInfo.outputPath(`build-${status}.png`),
      fullPage: true,
    });
  }
});

test("captures a real two-dimensional result", async ({ page }, testInfo) => {
  test.skip(!fieldProjectId || !sessionId, "Studio field project credentials are not set");
  await page.context().addCookies([
    {
      name: "piern_studio_session",
      value: sessionId!,
      url: "http://127.0.0.1:3001",
      httpOnly: true,
      sameSite: "Lax",
    },
  ]);
  await page.goto(`./projects/${fieldProjectId!}/demo`);
  await expect(page.getByLabel("4 行 6 列结果热力图")).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("project-field-demo.png"),
    fullPage: true,
  });
});
