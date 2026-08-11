import { ArrowRight, Plus, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { AppShell } from "../components/AppShell";
import { Button } from "../components/Button";
import { ErrorState, LoadingState } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { api } from "../lib/api";
import { formatDate } from "../lib/format";
import type { ProjectSummary } from "../types";

export function ProjectsPage() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      setProjects(await api.listProjects());
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "项目加载失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  return (
    <AppShell>
      <PageHeader
        eyebrow="工作区"
        title="科学计算项目"
        description="从你自己的数据和计算模型开始。"
        action={
          projects.length ? (
            <Link to="/new">
              <Button icon={<Plus size={17} />}>新建项目</Button>
            </Link>
          ) : null
        }
      />

      {loading ? <LoadingState label="正在读取项目" /> : null}
      {error ? <ErrorState message={error} onRetry={() => void load()} /> : null}

      {!loading && !error && projects.length === 0 ? (
        <section className="empty-workspace">
          <div className="empty-workspace__visual" aria-hidden="true">
            <span className="signal-line signal-line--one" />
            <span className="signal-line signal-line--two" />
            <span className="signal-line signal-line--three" />
            <Sparkles size={23} />
          </div>
          <div>
            <span className="eyebrow">第一个项目</span>
            <h2>把一组科学数据变成可对话 Demo</h2>
            <p>准备一份输入输出数据，以及能够处理输入的计算模型。</p>
            <Link to="/new">
              <Button icon={<Plus size={17} />}>创建项目</Button>
            </Link>
          </div>
        </section>
      ) : null}

      {projects.length ? (
        <section className="project-grid" aria-label="项目列表">
          {projects.map((project) => (
            <Link
              className="project-card"
              to={`/projects/${project.project_id}`}
              key={project.project_id}
            >
              <div className="project-card__top">
                <StatusBadge status={project.status} />
                <ArrowRight size={18} aria-hidden="true" />
              </div>
              <h2>{project.name}</h2>
              <p>{project.goal}</p>
              <div className="project-card__meta">
                <span>更新于 {formatDate(project.updated_at)}</span>
              </div>
            </Link>
          ))}
        </section>
      ) : null}
    </AppShell>
  );
}
