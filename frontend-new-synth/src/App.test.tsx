import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("new synthesis shell", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("starts with the isolated three-step data workflow", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/session"))
        return jsonResponse({ session_id: "a".repeat(32) });
      if (url.endsWith("/api/llm-config")) {
        return jsonResponse({
          provider: "siliconflow",
          model: "deepseek-ai/DeepSeek-V3.2",
          base_url: "",
          api_key_masked: "sk-t****test",
          has_api_key: true,
          temperature: 0.5,
          max_tokens: 4096,
          thinking: "disabled",
        });
      }
      if (url.endsWith("/presets")) {
        return jsonResponse({
          accepted_uploads: [".h5", ".hdf5"],
          max_upload_bytes: 1024 ** 3,
          max_generation_samples: 5_000_000,
          router_mode: "binary",
          simulations: [],
          experts: [],
        });
      }
      if (url.endsWith("/workflows")) return jsonResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App />);

    expect(
      await screen.findByRole("heading", { name: "从你的数据开始" }),
    ).toBeInTheDocument();
    const navigation = screen.getByRole("navigation", { name: "数据合成步骤" });
    expect(navigation).toHaveTextContent("接入数据");
    expect(navigation).toHaveTextContent("确认定义");
    expect(navigation).toHaveTextContent("生成数据");
    expect(screen.getByRole("tab", { name: "上传 HDF5" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByText("新数据合成")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /API 设置/ })).toHaveTextContent(
      "已连接",
    );
  });
});
