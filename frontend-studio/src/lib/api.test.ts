import { describe, expect, it } from "vitest";

import { parseApiError } from "./api";

describe("parseApiError", () => {
  it("reads the platform error envelope", () => {
    expect(
      parseApiError(
        {
          code: "invalid_data",
          message: "暂不支持此数据格式",
          request_id: "request-1",
        },
        "fallback",
      ),
    ).toEqual({
      code: "invalid_data",
      message: "暂不支持此数据格式",
    });
  });

  it("keeps compatibility with FastAPI detail errors", () => {
    expect(
      parseApiError(
        {
          detail: {
            code: "mapping_required",
            message: "请先确认数据字段",
          },
        },
        "fallback",
      ),
    ).toEqual({
      code: "mapping_required",
      message: "请先确认数据字段",
    });
  });
});
