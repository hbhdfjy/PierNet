import {
  ArrowLeft,
  ArrowRight,
  Check,
  Circle,
  DatabaseZap,
  FileText,
  LoaderCircle,
  Network,
  RotateCcw,
  ShieldCheck,
  Square,
} from "lucide-react";
import { useMemo, useState } from "react";

import type { GenerateOptions, WorkflowSnapshot } from "../types";

interface GenerateStepProps {
  workflow: WorkflowSnapshot;
  busy: boolean;
  error: string | null;
  onBack: () => void;
  onGenerate: (options: GenerateOptions, retry: boolean) => Promise<void>;
  onCancel: () => Promise<void>;
}

const phases = [
  { id: "templates", label: "生成语言模板" },
  { id: "text2comp", label: "生成 Text2Comp 数据" },
  { id: "datasets", label: "构建二分类 Router 数据" },
  { id: "validation", label: "校验并登记数据集" },
];

const phaseIndex: Record<string, number> = {
  templates: 0,
  text2comp: 1,
  datasets: 2,
  validation: 3,
  complete: 4,
};

export function GenerateStep({
  workflow,
  busy,
  error,
  onBack,
  onGenerate,
  onCancel,
}: GenerateStepProps) {
  const sourceCount = workflow.source?.sample_count ?? 4;
  const [options, setOptions] = useState<GenerateOptions>({
    max_samples: Math.min(sourceCount, 1000),
    variants_per_sample: 2,
    negative_ratio: 1,
    seed: 42,
  });
  const running = workflow.status === "running";
  const complete =
    workflow.status === "succeeded" && Boolean(workflow.artifacts?.router);
  const failed =
    workflow.status === "failed" || workflow.status === "cancelled";
  const progress = Math.max(
    0,
    Math.min(1, workflow.artifacts?.progress ?? (complete ? 1 : 0)),
  );
  const currentPhase = workflow.artifacts?.phase ?? "templates";
  const activeIndex = phaseIndex[currentPhase] ?? 0;
  const routerId = workflow.artifacts?.router?.dataset_id;
  const estimates = useMemo(() => {
    const text2comp = options.max_samples * options.variants_per_sample;
    return { text2comp, router: text2comp * (1 + options.negative_ratio) };
  }, [options]);

  return (
    <main className="page-shell" data-page="generation">
      <header className="page-heading">
        <div>
          <span className="eyebrow">第三步</span>
          <h1>
            {complete
              ? "训练数据已就绪"
              : running
                ? "正在生成训练数据"
                : "开始自动生成"}
          </h1>
          <p>
            {complete
              ? "数据集已经登记，可直接带入简洁训练或复杂训练。"
              : "模板、Text2Comp 数据和 Router 数据将在一个任务中自动完成。"}
          </p>
        </div>
        {complete ? (
          <div className="ready-pill">
            <ShieldCheck size={17} /> 已通过校验
          </div>
        ) : null}
      </header>

      {error || workflow.error?.message ? (
        <div className="inline-error" role="alert">
          {error ?? workflow.error?.message}
        </div>
      ) : null}

      {!running && !complete ? (
        <section className="generation-setup">
          <div className="section-heading">
            <span>
              <DatabaseZap size={20} />
            </span>
            <div>
              <h2>生成规模</h2>
              <p>先用小批量验证链路，确认后可增加规模。</p>
            </div>
          </div>
          <div className="generation-controls">
            <label className="field">
              <span>使用源样本</span>
              <input
                type="number"
                min={4}
                max={sourceCount}
                value={options.max_samples}
                onChange={(event) =>
                  setOptions({
                    ...options,
                    max_samples: Math.max(
                      4,
                      Math.min(sourceCount, Number(event.target.value)),
                    ),
                  })
                }
              />
              <small>最多 {sourceCount.toLocaleString()} 条</small>
            </label>
            <label className="field">
              <span>每条语言变体</span>
              <input
                type="number"
                min={1}
                max={8}
                value={options.variants_per_sample}
                onChange={(event) =>
                  setOptions({
                    ...options,
                    variants_per_sample: Math.max(
                      1,
                      Math.min(8, Number(event.target.value)),
                    ),
                  })
                }
              />
              <small>增加表达覆盖</small>
            </label>
            <label className="field">
              <span>负样本比例</span>
              <input
                type="number"
                min={1}
                max={10}
                value={options.negative_ratio}
                onChange={(event) =>
                  setOptions({
                    ...options,
                    negative_ratio: Math.max(
                      1,
                      Math.min(10, Number(event.target.value)),
                    ),
                  })
                }
              />
              <small>当前为二分类 Router</small>
            </label>
            <label className="field">
              <span>随机种子</span>
              <input
                type="number"
                min={0}
                value={options.seed}
                onChange={(event) =>
                  setOptions({
                    ...options,
                    seed: Math.max(0, Number(event.target.value)),
                  })
                }
              />
              <small>保证可复现</small>
            </label>
          </div>
          <div className="estimate-line">
            <div>
              <span>预计 Text2Comp</span>
              <strong>{estimates.text2comp.toLocaleString()}</strong>
            </div>
            <div>
              <span>预计 Router</span>
              <strong>{estimates.router.toLocaleString()}</strong>
            </div>
            <div>
              <span>训练标签</span>
              <strong>专家输入参数</strong>
            </div>
          </div>
        </section>
      ) : null}

      {running || complete ? (
        <section className="progress-surface" aria-live="polite">
          <div className="progress-header">
            <div>
              <span>
                {complete ? "完成" : `${Math.round(progress * 100)}%`}
              </span>
              <h2>{workflow.artifacts?.message ?? "正在准备生成任务"}</h2>
            </div>
            {running ? (
              <LoaderCircle className="spin progress-spinner" size={28} />
            ) : (
              <Check className="progress-check" size={28} />
            )}
          </div>
          <div className="progress-track">
            <span style={{ width: `${progress * 100}%` }} />
          </div>
          <ol className="phase-list">
            {phases.map((phase, index) => {
              const done = complete || index < activeIndex;
              const active = !complete && index === activeIndex;
              return (
                <li
                  key={phase.id}
                  className={active ? "is-active" : done ? "is-done" : ""}
                >
                  <span>
                    {done ? (
                      <Check size={15} />
                    ) : active ? (
                      <LoaderCircle className="spin" size={15} />
                    ) : (
                      <Circle size={12} />
                    )}
                  </span>
                  {phase.label}
                </li>
              );
            })}
          </ol>
        </section>
      ) : null}

      {complete ? (
        <section className="result-section">
          <div className="section-heading">
            <span>
              <ShieldCheck size={20} />
            </span>
            <div>
              <h2>已登记的数据集</h2>
              <p>三个产物各司其职，不再混放同一种 schema。</p>
            </div>
          </div>
          <div className="dataset-results">
            <article>
              <span className="dataset-icon teal">
                <FileText size={20} />
              </span>
              <div>
                <small>Text2Comp</small>
                <h3>
                  {workflow.artifacts?.text2comp?.sample_count.toLocaleString()}{" "}
                  条
                </h3>
                <code>{workflow.artifacts?.text2comp?.dataset_id}</code>
              </div>
              <em>专家输入标签</em>
            </article>
            <article>
              <span className="dataset-icon amber">
                <Network size={20} />
              </span>
              <div>
                <small>Router</small>
                <h3>
                  {workflow.artifacts?.router?.sample_count.toLocaleString()} 条
                </h3>
                <code>{routerId}</code>
              </div>
              <em>二分类路由</em>
            </article>
            <article>
              <span className="dataset-icon violet">
                <ShieldCheck size={20} />
              </span>
              <div>
                <small>专家评测</small>
                <h3>
                  {workflow.artifacts?.evaluation?.sample_count.toLocaleString()}{" "}
                  条
                </h3>
                <code>{workflow.artifacts?.evaluation?.dataset_id}</code>
              </div>
              <em>物理输出真值</em>
            </article>
          </div>
        </section>
      ) : null}

      <footer className="page-actions">
        <button
          className="button ghost"
          type="button"
          disabled={running}
          onClick={onBack}
        >
          <ArrowLeft size={18} /> 返回定义
        </button>
        {running ? (
          <button
            className="button danger"
            type="button"
            disabled={busy}
            onClick={() => void onCancel()}
          >
            <Square size={16} /> 取消任务
          </button>
        ) : complete ? (
          <div className="training-actions">
            <a
              className="button secondary"
              href={`/training/new?datasetId=${encodeURIComponent(routerId ?? "")}`}
            >
              复杂训练
            </a>
            <a
              className="button primary"
              href={`/training/simple?datasetId=${encodeURIComponent(routerId ?? "")}`}
            >
              进入简洁训练 <ArrowRight size={18} />
            </a>
          </div>
        ) : (
          <button
            className="button primary"
            type="button"
            disabled={busy}
            onClick={() => void onGenerate(options, failed)}
          >
            {busy ? (
              <LoaderCircle className="spin" size={18} />
            ) : failed ? (
              <RotateCcw size={18} />
            ) : (
              <DatabaseZap size={18} />
            )}
            {failed ? "重新生成" : "开始生成"}
          </button>
        )}
      </footer>
    </main>
  );
}
