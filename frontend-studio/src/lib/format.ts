import type { ProjectSnapshot, StageStatus } from "../types";

export function formatDate(timestamp: number): string {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(timestamp * 1000));
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const absolute = Math.abs(value);
  if ((absolute > 0 && absolute < 0.001) || absolute >= 1_000_000) {
    return value.toExponential(3);
  }
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 5 }).format(value);
}

export function formatDuration(value: number): string {
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: value >= 100 ? 0 : 1,
  }).format(value);
}

export function shapeLabel(shape?: number[]): string {
  return shape?.length ? shape.join(" × ") : "—";
}

export const statusLabel: Record<StageStatus | ProjectSnapshot["status"], string> = {
  waiting: "待开始",
  running: "进行中",
  succeeded: "已完成",
  failed: "需处理",
  cancelled: "已停止",
  draft: "准备中",
  ready: "可使用",
};

export type NextDestination = "resources" | "build" | "demo";

export function nextDestination(project: ProjectSnapshot): NextDestination {
  if (project.can_chat) return "demo";
  if (project.compatibility?.compatible) return "build";
  return "resources";
}

export function flattenNumbers(value: unknown): number[] {
  if (typeof value === "number") return [value];
  if (!Array.isArray(value)) return [];
  return value.flat(Infinity).filter((item): item is number => typeof item === "number");
}
