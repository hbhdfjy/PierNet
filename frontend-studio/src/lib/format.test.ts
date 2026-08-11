import { describe, expect, it } from "vitest";

import type { ProjectSnapshot } from "../types";
import { flattenNumbers, nextDestination, shapeLabel } from "./format";

function project(overrides: Partial<ProjectSnapshot>): ProjectSnapshot {
  return {
    project_id: "studio-test",
    name: "Test",
    goal: "Test",
    status: "draft",
    current_stage: "resources",
    created_at: 1,
    updated_at: 1,
    stages: [],
    data: null,
    expert: null,
    inspection: null,
    compatibility: null,
    artifacts: null,
    result: null,
    error: null,
    recommended_prompt: null,
    can_run: false,
    can_chat: false,
    ...overrides,
  };
}

describe("project navigation", () => {
  it("keeps incomplete projects in resources", () => {
    expect(nextDestination(project({}))).toBe("resources");
  });

  it("opens build after compatibility succeeds", () => {
    expect(
      nextDestination(
        project({
          compatibility: {
            compatible: true,
            sample_count: 3,
            input_shape: [3],
            expected_output_shape: [8],
            actual_output_shape: [8],
            finite: true,
          },
          can_run: true,
        }),
      ),
    ).toBe("build");
  });

  it("opens a ready demo", () => {
    expect(nextDestination(project({ status: "ready", can_chat: true }))).toBe("demo");
  });
});

describe("numeric helpers", () => {
  it("flattens nested model output", () => {
    expect(
      flattenNumbers([
        [1, 2],
        [3, 4],
      ]),
    ).toEqual([1, 2, 3, 4]);
  });

  it("formats multidimensional shapes", () => {
    expect(shapeLabel([4, 6])).toBe("4 × 6");
  });
});
