import { ArrowRight, FlaskConical, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";

import { AppShell } from "../components/AppShell";
import { Button } from "../components/Button";
import { CompatibilityPanel } from "../components/CompatibilityPanel";
import { DataProfile } from "../components/DataProfile";
import { ErrorState, LoadingState } from "../components/Feedback";
import { FieldMapping } from "../components/FieldMapping";
import { PageHeader } from "../components/PageHeader";
import { UploadSurface } from "../components/UploadSurface";
import { useProject } from "../hooks/useProject";
import { api } from "../lib/api";

export function ResourcesPage() {
  const { projectId } = useParams();
  const { project, setProject, loading, error, refresh } = useProject(projectId);
  const [busy, setBusy] = useState<"data" | "expert" | "mapping" | "check" | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  if (loading) {
    return (
      <AppShell>
        <LoadingState label="正在读取资源" />
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

  const act = async (kind: typeof busy, operation: () => Promise<typeof project>) => {
    setBusy(kind);
    setActionError(null);
    try {
      setProject(await operation());
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : "操作没有完成");
    } finally {
      setBusy(null);
    }
  };

  const inspection = project.inspection?.data;
  const readyToCheck =
    Boolean(project.data) && !project.data?.needs_mapping && Boolean(project.expert);

  return (
    <AppShell project={project}>
      <PageHeader
        eyebrow="准备资源"
        title="上传你的数据与计算模型"
        description="两项资源可以按任意顺序上传。"
      />

      <div className="resource-grid">
        <UploadSurface
          kind="data"
          resource={project.data}
          busy={busy === "data"}
          onUpload={(file) => act("data", () => api.uploadData(projectId, file))}
        />
        <UploadSurface
          kind="expert"
          resource={project.expert}
          busy={busy === "expert"}
          onUpload={(file) => act("expert", () => api.uploadExpert(projectId, file))}
        />
      </div>

      <details className="interface-note">
        <summary>计算模型接口</summary>
        <div>
          <code>predict(inputs) → outputs</code>
          <span>输入首维为样本数，输出形状需要与数据中的输出一致。</span>
        </div>
      </details>

      {project.data?.needs_mapping && inspection?.columns ? (
        <FieldMapping
          columns={inspection.columns}
          suggestedInputs={inspection.suggested_input_fields}
          suggestedOutputs={inspection.suggested_output_fields}
          busy={busy === "mapping"}
          onConfirm={(inputs, outputs) =>
            act("mapping", () => api.applyMapping(projectId, inputs, outputs))
          }
        />
      ) : null}

      {project.data && !project.data.needs_mapping ? <DataProfile data={project.data} /> : null}

      {project.compatibility ? <CompatibilityPanel report={project.compatibility} /> : null}

      {actionError ? <ErrorState message={actionError} /> : null}

      <div className="page-action-bar">
        <div>
          <ShieldCheck size={20} />
          <span>
            {project.compatibility?.compatible
              ? "资源检查已通过"
              : readyToCheck
                ? "已准备好检查真实前向结果"
                : "请先准备完整资源"}
          </span>
        </div>
        {project.compatibility?.compatible ? (
          <Link to={`/projects/${projectId}/build`}>
            <Button icon={<ArrowRight size={17} />}>进入构建</Button>
          </Link>
        ) : (
          <Button
            busy={busy === "check"}
            disabled={!readyToCheck}
            icon={<FlaskConical size={17} />}
            onClick={() =>
              act("check", async () => {
                await api.inspectProject(projectId);
                return api.checkCompatibility(projectId);
              })
            }
          >
            检查资源
          </Button>
        )}
      </div>
    </AppShell>
  );
}
