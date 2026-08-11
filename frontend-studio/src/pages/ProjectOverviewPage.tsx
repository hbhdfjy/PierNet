import { ArrowRight, Box, Database, MessageSquareText, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { AppShell } from "../components/AppShell";
import { Button } from "../components/Button";
import { ErrorState, LoadingState } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useProject } from "../hooks/useProject";
import { api } from "../lib/api";
import { nextDestination } from "../lib/format";

const destinationCopy = {
  resources: { label: "准备资源", path: "resources" },
  build: { label: "开始构建", path: "build" },
  demo: { label: "打开 Demo", path: "demo" },
};

export function ProjectOverviewPage() {
  const { projectId } = useParams();
  const navigate = useNavigate();
  const { project, loading, error, refresh } = useProject(projectId);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  if (loading) {
    return (
      <AppShell>
        <LoadingState label="正在读取项目" />
      </AppShell>
    );
  }
  if (!project || error) {
    return (
      <AppShell>
        <ErrorState message={error ?? "项目不存在"} onRetry={() => void refresh()} />
      </AppShell>
    );
  }

  const destination = destinationCopy[nextDestination(project)];
  const handleDelete = async () => {
    if (
      !window.confirm(
        `确认删除“${project.name}”？项目数据、计算模型、训练结果和对话记录都会一并删除。`,
      )
    ) {
      return;
    }
    setDeleting(true);
    setDeleteError(null);
    try {
      await api.deleteProject(project.project_id);
      navigate("/");
    } catch (caught) {
      setDeleteError(caught instanceof Error ? caught.message : "项目暂时无法删除");
      setDeleting(false);
    }
  };

  return (
    <AppShell project={project}>
      <PageHeader
        eyebrow="项目总览"
        title={project.name}
        description={project.goal}
        action={<StatusBadge status={project.status} />}
      />
      <section className="project-overview">
        <div className="overview-primary">
          <span className="eyebrow">当前状态</span>
          <h2>{project.stages.find((stage) => stage.id === project.current_stage)?.message}</h2>
          <p>
            {project.can_chat
              ? "端到端验证已完成，可以输入新参数进行计算。"
              : "系统会保留每一步结果，关闭页面后仍可继续。"}
          </p>
          <Link to={`/projects/${project.project_id}/${destination.path}`}>
            <Button icon={<ArrowRight size={17} />}>{destination.label}</Button>
          </Link>
        </div>
        <div className="overview-facts">
          <div>
            <Database size={19} />
            <span>数据</span>
            <strong>{project.data ? project.data.filename : "尚未上传"}</strong>
          </div>
          <div>
            <Box size={19} />
            <span>计算模型</span>
            <strong>{project.expert ? project.expert.filename : "尚未上传"}</strong>
          </div>
          <div>
            <MessageSquareText size={19} />
            <span>对话 Demo</span>
            <strong>{project.can_chat ? "可使用" : "尚未完成"}</strong>
          </div>
        </div>
      </section>
      <section className="project-maintenance" aria-labelledby="project-maintenance-title">
        <div>
          <h2 id="project-maintenance-title">项目管理</h2>
          <p>删除后，该项目上传的资源、训练结果和对话记录将无法恢复。</p>
        </div>
        <Button
          variant="quiet"
          className="project-delete-button"
          icon={<Trash2 size={16} />}
          busy={deleting}
          disabled={project.status === "running"}
          onClick={() => void handleDelete()}
        >
          删除项目
        </Button>
      </section>
      {deleteError ? (
        <p className="project-maintenance__error" role="alert">
          {deleteError}
        </p>
      ) : null}
    </AppShell>
  );
}
