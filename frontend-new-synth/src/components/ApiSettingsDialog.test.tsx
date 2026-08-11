import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiSettingsDialog } from "./ApiSettingsDialog";

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("API settings dialog", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("saves the shared LLM config and tests the persisted key", async () => {
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        const method = init?.method ?? "GET";
        if (url.endsWith("/api/llm-config") && method === "GET") {
          return jsonResponse({
            provider: "siliconflow",
            model: "deepseek-ai/DeepSeek-V3.2",
            base_url: "",
            api_key_masked: "sk-a****test",
            has_api_key: true,
            temperature: 0.5,
            max_tokens: 4096,
            thinking: "disabled",
          });
        }
        if (url.endsWith("/api/llm-config") && method === "POST")
          return jsonResponse({ ok: true });
        if (url.endsWith("/api/llm-config/test") && method === "POST")
          return jsonResponse({
            ok: true,
            message: "连接成功",
            response_preview: "ok",
          });
        throw new Error(`Unexpected request: ${method} ${url}`);
      },
    );
    vi.stubGlobal("fetch", fetchMock);
    const onConfigured = vi.fn();

    render(<ApiSettingsDialog onClose={vi.fn()} onConfigured={onConfigured} />);

    expect(
      await screen.findByDisplayValue("deepseek-ai/DeepSeek-V3.2"),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "保存并测试" }));

    expect(await screen.findByText("连接成功")).toBeInTheDocument();
    expect(onConfigured).toHaveBeenCalledWith(
      expect.objectContaining({ has_api_key: true }),
    );
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([input, init]) =>
            String(input).endsWith("/api/llm-config/test") &&
            init?.method === "POST" &&
            JSON.parse(String(init.body)).api_key === "",
        ),
      ).toBe(true);
    });
  });
});
