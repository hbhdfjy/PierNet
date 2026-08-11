import type { ChatResponse, ProjectSnapshot, ProjectSummary, RunResponse } from "../types";

const API_BASE = "/api/studio";
let sessionPromise: Promise<void> | null = null;

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

export function parseApiError(
  payload: unknown,
  fallback: string,
): { message: string; code?: string } {
  if (typeof payload !== "object" || payload === null) {
    return { message: fallback };
  }
  const record = payload as Record<string, unknown>;
  if (typeof record.message === "string") {
    return {
      message: record.message,
      code: typeof record.code === "string" ? record.code : undefined,
    };
  }
  const detail = record.detail;
  if (typeof detail === "string") {
    return { message: detail };
  }
  if (typeof detail === "object" && detail !== null) {
    const nested = detail as Record<string, unknown>;
    return {
      message: typeof nested.message === "string" ? nested.message : fallback,
      code: typeof nested.code === "string" ? nested.code : undefined,
    };
  }
  return { message: fallback };
}

async function ensureSession(): Promise<void> {
  if (!sessionPromise) {
    sessionPromise = fetch(`${API_BASE}/session`, {
      method: "POST",
      credentials: "include",
    }).then(async (response) => {
      if (!response.ok) {
        sessionPromise = null;
        throw new ApiError("无法建立工作会话", response.status);
      }
    });
  }
  return sessionPromise;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  await ensureSession();
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "include",
    headers: {
      ...(init?.body instanceof Blob ? {} : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const parsed = parseApiError(payload, `请求失败 (${response.status})`);
    throw new ApiError(parsed.message, response.status, parsed.code);
  }
  return (await response.json()) as T;
}

export const api = {
  listProjects: () => request<ProjectSummary[]>("/projects"),

  createProject: (name: string, goal: string) =>
    request<ProjectSnapshot>("/projects", {
      method: "POST",
      body: JSON.stringify({ name, goal }),
    }),

  getProject: (projectId: string) => request<ProjectSnapshot>(`/projects/${projectId}`),

  deleteProject: (projectId: string) =>
    request<{ project_id: string; deleted: boolean; message: string }>(`/projects/${projectId}`, {
      method: "DELETE",
    }),

  uploadData: (projectId: string, file: File) =>
    request<ProjectSnapshot>(`/projects/${projectId}/data`, {
      method: "POST",
      body: file,
      headers: {
        "Content-Type": "application/octet-stream",
        "X-File-Name": encodeURIComponent(file.name),
      },
    }),

  uploadExpert: (projectId: string, file: File) =>
    request<ProjectSnapshot>(`/projects/${projectId}/expert`, {
      method: "POST",
      body: file,
      headers: {
        "Content-Type": "application/octet-stream",
        "X-File-Name": encodeURIComponent(file.name),
      },
    }),

  applyMapping: (projectId: string, inputFields: string[], outputFields: string[]) =>
    request<ProjectSnapshot>(`/projects/${projectId}/mapping`, {
      method: "POST",
      body: JSON.stringify({
        input_fields: inputFields,
        output_fields: outputFields,
      }),
    }),

  inspectProject: (projectId: string) =>
    request<ProjectSnapshot>(`/projects/${projectId}/inspect`, {
      method: "POST",
    }),

  checkCompatibility: (projectId: string) =>
    request<ProjectSnapshot>(`/projects/${projectId}/compatibility-check`, {
      method: "POST",
    }),

  run: (projectId: string) =>
    request<RunResponse>(`/projects/${projectId}/run`, { method: "POST" }),

  retry: (projectId: string) =>
    request<RunResponse>(`/projects/${projectId}/retry`, { method: "POST" }),

  cancel: (projectId: string) =>
    request<RunResponse>(`/projects/${projectId}/cancel`, { method: "POST" }),

  chat: (projectId: string, message: string) =>
    request<ChatResponse>(`/projects/${projectId}/chat`, {
      method: "POST",
      body: JSON.stringify({ message }),
    }),
};
