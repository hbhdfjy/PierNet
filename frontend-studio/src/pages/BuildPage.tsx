import {
  ArrowRight,
  Check,
  Circle,
  LoaderCircle,
  OctagonX,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { AppShell } from "../components/AppShell";
import { Button } from "../components/Button";
import { ErrorState, LoadingState } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { useProject } from "../hooks/useProject";
import { api } from "../lib/api";
import type { StageSnapshot } from "../types";

const buildStageIds = new Set(["preparation", "training", "assembly", "validation"]);

function BuildStageIcon({ stage }: { stage: StageSnapshot }) {
  if (stage.status === "succeeded") return <Check size={17} />;
  if (stage.status === "running") return <LoaderCircle className="spin" size={18} />;
  if (stage.status === "failed") return <OctagonX size={18} />;
  if (stage.status === "cancelled") return <Pause size={18} />;
  return <Circle size={13} />;
}

export function BuildPage() {
  const { projectId } = useParams();
  const { project, loading, error, refresh } = useProject(projectId);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const stages = useMemo(
    () => project?.stages.filter((stage) => buildStageIds.has(stage.id)) ?? [],
    [project],
  );
  const overallProgress = useMemo(() => {
    if (!stages.length) return 0;
    const total = stages.reduce((sum, stage) => {
      if (stage.status === "succeeded") return sum + 1;
      if (stage.status === "running") return sum + (stage.progress ?? 0.08);
      return sum;
    }, 0);
    return total / stages.length;
  }, [stages]);

  if (loading) {
    return (
      <AppShell>
        <LoadingState label="正在读取构建状态" />
      </AppShell>
    );
  }
  if (!project || !projectId || error) {
    return (
      <AppShell>
        <ErrorState message={error ?? "项目不存在"} onRetry={() => void refresh()} />
      </AppShell>
    );
  }

  const run = async (retry = false) => {
    setBusy(true);
    setActionError(null);
    try {
      await (retry ? api.retry(projectId) : api.run(projectId));
      await refresh();
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : "构建没有开始");
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    setActionError(null);
    try {
      await api.cancel(projectId);
      await refresh();
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : "停止请求没有完成");
    } finally {
      setBusy(false);
    }
  };

  const hasStarted = stages.some((stage) => stage.status !== "waiting");
  const failed = project.status === "failed" || project.status === "cancelled";

  return (
    <AppShell project={project}>
      <PageHeader
        eyebrow="构建 Demo"
        title={project.status === "ready" ? "你的计算 Demo 已就绪" : "把资源连接成可对话 Demo"}
        description={
          project.status === "running"
            ? "页面可以安全关闭，稍后回来继续查看。"
            : "构建过程会保留训练结果和验证记录。"
        }
      />

      <section className="build-console">
        <div className="build-overview">
          <div className="build-overview__value">
            <span>总体进度</span>
            <strong>{Math.round(overallProgress * 100)}%</strong>
          </div>
          <div
            className="progress-track"
            aria-label={`总体进度 ${Math.round(overallProgress * 100)}%`}
          >
            <span style={{ width: `${overallProgress * 100}%` }} />
          </div>
        </div>

        <ol className="build-stages">
          {stages.map((stage) => (
            <li className={`build-stage build-stage--${stage.status}`} key={stage.id}>
              <span className="build-stage__icon">
                <BuildStageIcon stage={stage} />
              </span>
              <div>
                <strong>{stage.title}</strong>
                <span>{stage.message}</span>
              </div>
              {stage.status === "running" && stage.progress !== null ? (
                <strong className="build-stage__progress">
                  {Math.round(stage.progress * 100)}%
                </strong>
              ) : null}
            </li>
          ))}
        </ol>
      </section>

      {project.error ? <ErrorState message={project.error.message} /> : null}
      {actionError ? <ErrorState message={actionError} /> : null}

      <div className="page-action-bar">
        <div>
          {project.status === "running" ? (
            <LoaderCircle className="spin" size={20} />
          ) : (
            <Play size={20} />
          )}
          <span>
            {project.status === "ready"
              ? "端到端计算验证通过"
              : project.status === "running"
                ? "正在使用你的资源构建"
                : failed
                  ? "已有进度已保留"
                  : "准备就绪"}
          </span>
        </div>
        {project.status === "ready" ? (
          <Link to={`/projects/${projectId}/demo`}>
            <Button icon={<ArrowRight size={17} />}>打开 Demo</Button>
          </Link>
        ) : project.status === "running" ? (
          <Button variant="secondary" busy={busy} onClick={() => void stop()}>
            安全停止
          </Button>
        ) : failed ? (
          <Button busy={busy} icon={<RotateCcw size={17} />} onClick={() => void run(true)}>
            从现有进度重试
          </Button>
        ) : (
          <Button
            busy={busy}
            disabled={!project.can_run || hasStarted}
            icon={<Play size={17} />}
            onClick={() => void run()}
          >
            开始构建
          </Button>
        )}
      </div>
    </AppShell>
  );
}
