import type {
  DataDefinition,
  ExpertPreset,
  GenerateOptions,
  LLMConfig,
  LLMConfigRequest,
  LLMTestResult,
  Presets,
  WorkflowSnapshot,
  WorkflowSummary,
} from "./types";

const API_ROOT = "/api/new-synth";

function detailMessage(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") return fallback;
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object") {
    const message = (detail as { message?: unknown }).message;
    if (typeof message === "string") return message;
  }
  return fallback;
}

async function requestUrl<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    credentials: "same-origin",
    ...init,
    headers: {
      ...(init?.body && typeof init.body === "string"
        ? { "Content-Type": "application/json" }
        : {}),
      ...init?.headers,
    },
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(detailMessage(payload, `请求失败 (${response.status})`));
  }
  return payload as T;
}

function request<T>(path: string, init?: RequestInit): Promise<T> {
  return requestUrl<T>(`${API_ROOT}${path}`, init);
}

export const api = {
  createSession: () =>
    request<{ session_id: string }>("/session", { method: "POST" }),
  presets: () => request<Presets>("/presets"),
  workflows: () => request<WorkflowSummary[]>("/workflows"),
  createWorkflow: (name: string) =>
    request<WorkflowSnapshot>("/workflows", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  workflow: (workflowId: string) =>
    request<WorkflowSnapshot>(`/workflows/${workflowId}`),
  uploadSource: (workflowId: string, file: File) =>
    request<WorkflowSnapshot>(`/workflows/${workflowId}/source/upload`, {
      method: "POST",
      headers: { "X-File-Name": encodeURIComponent(file.name) },
      body: file,
    }),
  useSimulation: (
    workflowId: string,
    payload: {
      simulator: string;
      scenario: string;
      n_samples: number;
      seed: number;
      reuse_existing: boolean;
    },
  ) =>
    request<WorkflowSnapshot>(`/workflows/${workflowId}/source/simulation`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  uploadExpert: (name: string, file: File) =>
    request<{ ok: boolean; model: ExpertPreset }>(
      `/experts/upload?name=${encodeURIComponent(name)}`,
      {
        method: "POST",
        headers: { "X-File-Name": encodeURIComponent(file.name) },
        body: file,
      },
    ),
  useExpert: (
    workflowId: string,
    payload: {
      model_id: string;
      scenario: string;
      prompt: string;
      input_dim: number | null;
    },
  ) =>
    request<WorkflowSnapshot>(`/workflows/${workflowId}/source/expert`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  saveDefinition: (workflowId: string, definition: DataDefinition) =>
    request<WorkflowSnapshot>(`/workflows/${workflowId}/definition`, {
      method: "PUT",
      body: JSON.stringify({
        simulator: definition.simulator,
        scenario: definition.scenario,
        task_description: definition.task_description,
        parameters: definition.parameters,
        outputs: definition.outputs,
        sampling: definition.sampling,
      }),
    }),
  suggestDefinition: (workflowId: string, definition: DataDefinition) =>
    request<DataDefinition>(`/workflows/${workflowId}/definition/suggest`, {
      method: "POST",
      body: JSON.stringify({
        simulator: definition.simulator,
        scenario: definition.scenario,
        task_description: definition.task_description,
        parameters: definition.parameters,
        outputs: definition.outputs,
        sampling: definition.sampling,
      }),
    }),
  getLLMConfig: () => requestUrl<LLMConfig>("/api/llm-config"),
  saveLLMConfig: (config: LLMConfigRequest) =>
    requestUrl<{ ok: boolean }>("/api/llm-config", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  testLLMConfig: (config: LLMConfigRequest) =>
    requestUrl<LLMTestResult>("/api/llm-config/test", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  generate: (workflowId: string, options: GenerateOptions, retry = false) =>
    request<{ workflow_id: string; status: string; message: string }>(
      `/workflows/${workflowId}/${retry ? "retry" : "generate"}`,
      { method: "POST", body: JSON.stringify(options) },
    ),
  cancel: (workflowId: string) =>
    request<WorkflowSnapshot>(`/workflows/${workflowId}/cancel`, {
      method: "POST",
    }),
};
