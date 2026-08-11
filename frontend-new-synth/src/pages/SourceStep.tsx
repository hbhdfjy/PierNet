import {
  ArrowRight,
  Box,
  CheckCircle2,
  FileUp,
  FlaskConical,
  HardDriveUpload,
  LoaderCircle,
  UploadCloud,
  Activity,
} from "lucide-react";
import { useMemo, useState } from "react";

import type { Presets, SourceMode, WorkflowSnapshot } from "../types";

interface SourceStepProps {
  workflow: WorkflowSnapshot | null;
  presets: Presets | null;
  busy: boolean;
  error: string | null;
  onUpload: (file: File) => Promise<void>;
  onSimulation: (payload: {
    simulator: string;
    scenario: string;
    n_samples: number;
    seed: number;
    reuse_existing: boolean;
  }) => Promise<void>;
  onExpert: (payload: {
    model_id: string;
    scenario: string;
    prompt: string;
    input_dim: number | null;
  }) => Promise<void>;
  onExpertUpload: (name: string, file: File) => Promise<void>;
  onContinue: () => void;
}

const sourceModes: Array<{
  id: SourceMode;
  label: string;
  icon: typeof FileUp;
}> = [
  { id: "upload", label: "上传 HDF5", icon: HardDriveUpload },
  { id: "simulation", label: "运行内置仿真", icon: FlaskConical },
  { id: "expert", label: "使用专家模型", icon: Box },
];

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const power = Math.min(
    Math.floor(Math.log(value) / Math.log(1024)),
    units.length - 1,
  );
  return `${(value / 1024 ** power).toFixed(power === 0 ? 0 : 1)} ${units[power]}`;
}

export function SourceStep({
  workflow,
  presets,
  busy,
  error,
  onUpload,
  onSimulation,
  onExpert,
  onExpertUpload,
  onContinue,
}: SourceStepProps) {
  const [mode, setMode] = useState<SourceMode>("upload");
  const [dragging, setDragging] = useState(false);
  const [simulationKey, setSimulationKey] = useState("");
  const [sampleCount, setSampleCount] = useState(32);
  const [seed, setSeed] = useState(42);
  const [expertId, setExpertId] = useState("");
  const [expertScenario, setExpertScenario] = useState("custom_scenario");
  const [expertPrompt, setExpertPrompt] = useState(
    "请生成覆盖典型工况的专家模型输入与输出样本",
  );
  const [expertInputDim, setExpertInputDim] = useState("");
  const [expertName, setExpertName] = useState("");

  const simulation = useMemo(
    () =>
      presets?.simulations.find(
        (item) => `${item.simulator}/${item.scenario}` === simulationKey,
      ),
    [presets, simulationKey],
  );
  const readyExperts =
    presets?.experts.filter(
      (item) => item.active && item.data_generation_enabled,
    ) ?? [];
  const source = workflow?.source;

  const acceptFile = (file: File | undefined) => {
    if (!file || busy) return;
    void onUpload(file);
  };

  return (
    <main className="page-shell" data-page="source">
      <header className="page-heading">
        <div>
          <span className="eyebrow">第一步</span>
          <h1>从你的数据开始</h1>
          <p>选择一种接入方式。系统只保留训练真正需要的规范化数据和定义。</p>
        </div>
        {source?.ready ? (
          <div className="ready-pill">
            <CheckCircle2 size={17} /> 数据可用
          </div>
        ) : null}
      </header>

      <div className="mode-switch" role="tablist" aria-label="数据接入方式">
        {sourceModes.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={mode === item.id ? "is-active" : ""}
              type="button"
              role="tab"
              aria-selected={mode === item.id}
              onClick={() => setMode(item.id)}
            >
              <Icon size={18} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>

      {error ? (
        <div className="inline-error" role="alert">
          {error}
        </div>
      ) : null}

      <section className="source-layout">
        <div className="source-workspace">
          {mode === "upload" ? (
            <div
              className={`drop-zone${dragging ? " is-dragging" : ""}`}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                acceptFile(event.dataTransfer.files[0]);
              }}
            >
              <span className="drop-icon">
                <UploadCloud size={28} />
              </span>
              <div>
                <h2>
                  {busy ? "正在校验并规范化数据" : "拖入一个 HDF5 数据文件"}
                </h2>
                <p>需要包含 params、param_names 和 timeseries；上限 1 GB。</p>
              </div>
              <label className={`button primary${busy ? " is-disabled" : ""}`}>
                {busy ? (
                  <LoaderCircle className="spin" size={18} />
                ) : (
                  <FileUp size={18} />
                )}
                {busy ? "处理中" : "选择文件"}
                <input
                  type="file"
                  accept=".h5,.hdf5"
                  disabled={busy}
                  onChange={(event) => acceptFile(event.target.files?.[0])}
                />
              </label>
            </div>
          ) : null}

          {mode === "simulation" ? (
            <div className="form-surface">
              <div className="section-title">
                <span>
                  <FlaskConical size={20} />
                </span>
                <div>
                  <h2>运行内置仿真</h2>
                  <p>可复用已有结果，也可生成一小批新样本。</p>
                </div>
              </div>
              <div className="form-grid two-columns">
                <label className="field span-two">
                  <span>仿真场景</span>
                  <select
                    value={simulationKey}
                    onChange={(event) => setSimulationKey(event.target.value)}
                  >
                    <option value="">选择一个场景</option>
                    {presets?.simulations.map((item) => (
                      <option
                        key={`${item.simulator}/${item.scenario}`}
                        value={`${item.simulator}/${item.scenario}`}
                      >
                        {item.simulator} / {item.scenario}
                        {item.has_data ? " · 可复用" : ""}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>样本数</span>
                  <input
                    type="number"
                    min={4}
                    value={sampleCount}
                    onChange={(event) =>
                      setSampleCount(Number(event.target.value))
                    }
                  />
                </label>
                <label className="field">
                  <span>随机种子</span>
                  <input
                    type="number"
                    min={0}
                    value={seed}
                    onChange={(event) => setSeed(Number(event.target.value))}
                  />
                </label>
              </div>
              <button
                className="button primary align-end"
                type="button"
                disabled={!simulation || busy}
                onClick={() =>
                  simulation &&
                  void onSimulation({
                    simulator: simulation.simulator,
                    scenario: simulation.scenario,
                    n_samples: sampleCount,
                    seed,
                    reuse_existing: simulation.has_data,
                  })
                }
              >
                {busy ? (
                  <LoaderCircle className="spin" size={18} />
                ) : (
                  <FlaskConical size={18} />
                )}
                {busy
                  ? "仿真运行中"
                  : simulation?.has_data
                    ? "接入已有数据"
                    : "运行并接入"}
              </button>
            </div>
          ) : null}

          {mode === "expert" ? (
            <div className="form-surface expert-form">
              <div className="section-title">
                <span>
                  <Box size={20} />
                </span>
                <div>
                  <h2>让专家模型生成样本</h2>
                  <p>选择已注册且支持数据生成的专家模型。</p>
                </div>
              </div>
              <div className="form-grid two-columns">
                <label className="field span-two">
                  <span>专家模型</span>
                  <select
                    value={expertId}
                    onChange={(event) => setExpertId(event.target.value)}
                  >
                    <option value="">选择已注册模型</option>
                    {readyExperts.map((item) => (
                      <option key={item.model_id} value={item.model_id}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>场景标识</span>
                  <input
                    value={expertScenario}
                    onChange={(event) => setExpertScenario(event.target.value)}
                  />
                </label>
                <label className="field">
                  <span>输入维度（可选）</span>
                  <input
                    type="number"
                    min={1}
                    value={expertInputDim}
                    onChange={(event) => setExpertInputDim(event.target.value)}
                  />
                </label>
                <label className="field span-two">
                  <span>样本生成要求</span>
                  <textarea
                    rows={3}
                    value={expertPrompt}
                    onChange={(event) => setExpertPrompt(event.target.value)}
                  />
                </label>
              </div>
              <button
                className="button primary align-end"
                type="button"
                disabled={!expertId || !expertScenario || !expertPrompt || busy}
                onClick={() =>
                  void onExpert({
                    model_id: expertId,
                    scenario: expertScenario,
                    prompt: expertPrompt,
                    input_dim: expertInputDim ? Number(expertInputDim) : null,
                  })
                }
              >
                {busy ? (
                  <LoaderCircle className="spin" size={18} />
                ) : (
                  <Activity size={18} />
                )}
                {busy ? "生成中" : "生成并接入"}
              </button>
              <div className="expert-upload-line">
                <input
                  placeholder="新专家模型名称"
                  value={expertName}
                  onChange={(event) => setExpertName(event.target.value)}
                />
                <label className="button secondary">
                  <FileUp size={17} /> 上传模型包
                  <input
                    type="file"
                    disabled={!expertName || busy}
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (file) void onExpertUpload(expertName, file);
                    }}
                  />
                </label>
              </div>
            </div>
          ) : null}
        </div>

        <aside className="inspection-panel" aria-label="数据识别结果">
          <div className="inspection-heading">
            <span>
              <Activity size={19} />
            </span>
            <div>
              <h2>数据识别</h2>
              <p>{source?.ready ? source.filename : "等待接入"}</p>
            </div>
          </div>
          {source?.ready ? (
            <>
              <dl className="metric-list">
                <div>
                  <dt>样本</dt>
                  <dd>{source.sample_count.toLocaleString()}</dd>
                </div>
                <div>
                  <dt>专家输入</dt>
                  <dd>{source.input_dim} 维</dd>
                </div>
                <div>
                  <dt>物理输出</dt>
                  <dd>{source.output_shape.join(" × ")}</dd>
                </div>
                <div>
                  <dt>文件</dt>
                  <dd>{formatBytes(source.file_size_bytes)}</dd>
                </div>
              </dl>
              <div className="parameter-preview">
                <span>已识别参数</span>
                <p>
                  {source.param_names.slice(0, 8).join("、")}
                  {source.param_names.length > 8
                    ? ` 等 ${source.param_names.length} 项`
                    : ""}
                </p>
              </div>
            </>
          ) : (
            <div className="empty-inspection">
              <HardDriveUpload size={28} />
              <p>接入后将在这里显示样本、输入参数与物理输出结构。</p>
            </div>
          )}
        </aside>
      </section>

      <footer className="page-actions">
        <div>
          <strong>{source?.ready ? "数据已准备好" : "尚未接入数据"}</strong>
          <span>
            {source?.ready
              ? "下一步只需确认系统识别的数据定义。"
              : "选择任一接入方式开始。"}
          </span>
        </div>
        <button
          className="button primary"
          type="button"
          disabled={!source?.ready || busy}
          onClick={onContinue}
        >
          确认并继续 <ArrowRight size={18} />
        </button>
      </footer>
    </main>
  );
}
