import { useCallback, useEffect, useState } from "react";

import { api } from "../lib/api";
import type { ProjectSnapshot } from "../types";

export function useProject(projectId: string | undefined) {
  const [project, setProject] = useState<ProjectSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!projectId) return;
    try {
      const value = await api.getProject(projectId);
      setProject(value);
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "项目加载失败");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (project?.status !== "running") return;
    const timer = window.setInterval(() => void refresh(), 1200);
    return () => window.clearInterval(timer);
  }, [project?.status, refresh]);

  return { project, setProject, loading, error, refresh };
}
