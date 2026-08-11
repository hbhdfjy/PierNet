import { ArrowRight } from "lucide-react";
import { type FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";

import { AppShell } from "../components/AppShell";
import { Button } from "../components/Button";
import { PageHeader } from "../components/PageHeader";
import { api } from "../lib/api";

export function CreateProjectPage() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [goal, setGoal] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!name.trim() || !goal.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const project = await api.createProject(name.trim(), goal.trim());
      navigate(`/projects/${project.project_id}/resources`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "项目创建失败");
      setBusy(false);
    }
  };

  return (
    <AppShell>
      <div className="narrow-page">
        <PageHeader
          eyebrow="新项目"
          title="你想完成什么计算？"
          description="项目名称用于查找；目标会用于生成与你的数据匹配的训练内容。"
        />
        <form className="project-form" onSubmit={(event) => void submit(event)}>
          <label>
            <span>项目名称</span>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="例如：换热器温度场预测"
              maxLength={100}
            />
          </label>
          <label>
            <span>计算目标</span>
            <textarea
              value={goal}
              onChange={(event) => setGoal(event.target.value)}
              placeholder="例如：根据入口温度、流速和材料参数，预测下一时刻的二维温度场。"
              rows={5}
              maxLength={1000}
            />
            <small>{goal.length} / 1000</small>
          </label>
          {error ? <p className="form-error">{error}</p> : null}
          <div className="form-actions">
            <Button
              type="submit"
              busy={busy}
              disabled={!name.trim() || !goal.trim()}
              icon={<ArrowRight size={17} />}
            >
              准备资源
            </Button>
          </div>
        </form>
      </div>
    </AppShell>
  );
}
