import {
  ArrowLeft,
  ArrowRight,
  Braces,
  Check,
  Info,
  KeyRound,
  LoaderCircle,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import type { DataDefinition, SourceSnapshot } from "../types";

interface DefinitionStepProps {
  source: SourceSnapshot;
  definition: DataDefinition;
  busy: boolean;
  error: string | null;
  llmConfigured: boolean;
  onBack: () => void;
  onConfigureApi: () => void;
  onSuggest: (definition: DataDefinition) => Promise<DataDefinition | null>;
  onSave: (definition: DataDefinition) => Promise<void>;
}

function copyDefinition(value: DataDefinition): DataDefinition {
  return JSON.parse(JSON.stringify(value)) as DataDefinition;
}

export function DefinitionStep({
  source,
  definition,
  busy,
  error,
  llmConfigured,
  onBack,
  onConfigureApi,
  onSuggest,
  onSave,
}: DefinitionStepProps) {
  const [draft, setDraft] = useState<DataDefinition>(() =>
    copyDefinition(definition),
  );
  const [suggestionNotice, setSuggestionNotice] = useState<string | null>(null);
  const [suggesting, setSuggesting] = useState(false);
  useEffect(() => {
    setDraft(copyDefinition(definition));
    setSuggestionNotice(null);
  }, [definition]);

  const selectedChannels =
    draft.sampling.channels ?? draft.outputs.map((item) => item.index);
  const allChannelsSelected = selectedChannels.length === draft.outputs.length;
  const valid = useMemo(
    () =>
      /^[A-Za-z0-9_-]+$/.test(draft.simulator) &&
      /^[A-Za-z0-9_-]+$/.test(draft.scenario) &&
      Boolean(draft.task_description.trim()) &&
      draft.parameters.every((item) => Boolean(item.name.trim())) &&
      draft.outputs.every((item) => Boolean(item.name.trim())) &&
      selectedChannels.length > 0,
    [draft, selectedChannels.length],
  );

  const setParameter = (
    index: number,
    key: "display_name" | "description" | "unit",
    value: string,
  ) => {
    setDraft((current) => ({
      ...current,
      parameters: current.parameters.map((item) =>
        item.index === index ? { ...item, [key]: value } : item,
      ),
    }));
  };

  const setOutput = (
    index: number,
    key: "display_name" | "description" | "unit",
    value: string,
  ) => {
    setDraft((current) => ({
      ...current,
      outputs: current.outputs.map((item) =>
        item.index === index ? { ...item, [key]: value } : item,
      ),
    }));
  };

  const toggleChannel = (index: number) => {
    setDraft((current) => {
      const channels =
        current.sampling.channels ?? current.outputs.map((item) => item.index);
      const next = channels.includes(index)
        ? channels.filter((item) => item !== index)
        : [...channels, index].sort();
      return { ...current, sampling: { ...current.sampling, channels: next } };
    });
  };

  const handleSuggestion = async () => {
    if (!llmConfigured) {
      onConfigureApi();
      return;
    }
    setSuggestionNotice(null);
    setSuggesting(true);
    try {
      const suggested = await onSuggest(draft);
      if (!suggested) return;
      setDraft(copyDefinition(suggested));
      setSuggestionNotice("说明已智能补全，请检查后再保存。");
    } finally {
      setSuggesting(false);
    }
  };

  return (
    <main className="page-shell" data-page="definition">
      <header className="page-heading">
        <div>
          <span className="eyebrow">第二步</span>
          <h1>确认数据定义</h1>
          <p>系统已经识别结构。补充人能读懂的名称，并确认训练范围。</p>
        </div>
        <div className="definition-tools">
          <div className="definition-count">
            <strong>{source.input_dim}</strong>
            <span>专家输入维度</span>
          </div>
          <button
            className="button secondary"
            type="button"
            disabled={busy}
            onClick={() => void handleSuggestion()}
          >
            {suggesting ? (
              <LoaderCircle className="spin" size={17} />
            ) : llmConfigured ? (
              <Sparkles size={17} />
            ) : (
              <KeyRound size={17} />
            )}
            {suggesting
              ? "识别中"
              : llmConfigured
                ? "智能补全说明"
                : "配置智能识别 API"}
          </button>
        </div>
      </header>

      {error ? (
        <div className="inline-error" role="alert">
          {error}
        </div>
      ) : null}
      {suggestionNotice ? (
        <div className="suggestion-notice" role="status">
          <Sparkles size={17} /> {suggestionNotice}
        </div>
      ) : null}

      <div className="contract-note">
        <Info size={19} />
        <div>
          <strong>训练契约</strong>
          <p>
            Text2Comp
            学习生成下方“专家输入参数”；物理时序只用于专家验证和端到端评测。
          </p>
        </div>
      </div>

      <section className="definition-section identity-section">
        <div className="section-heading">
          <span>
            <Braces size={20} />
          </span>
          <div>
            <h2>任务身份</h2>
            <p>用于隔离数据集、训练产物和场景名称。</p>
          </div>
        </div>
        <div className="form-grid two-columns">
          <label className="field">
            <span>领域标识</span>
            <input
              value={draft.simulator}
              pattern="[A-Za-z0-9_-]+"
              onChange={(event) =>
                setDraft({ ...draft, simulator: event.target.value })
              }
            />
            <small>字母、数字、下划线或短横线</small>
          </label>
          <label className="field">
            <span>场景标识</span>
            <input
              value={draft.scenario}
              pattern="[A-Za-z0-9_-]+"
              onChange={(event) =>
                setDraft({ ...draft, scenario: event.target.value })
              }
            />
            <small>将在训练选择器中显示</small>
          </label>
          <label className="field span-two">
            <span>任务描述</span>
            <textarea
              rows={2}
              value={draft.task_description}
              onChange={(event) =>
                setDraft({ ...draft, task_description: event.target.value })
              }
            />
          </label>
        </div>
      </section>

      <section className="definition-section">
        <div className="section-heading">
          <span>
            <SlidersHorizontal size={20} />
          </span>
          <div>
            <h2>专家输入参数</h2>
            <p>这些参数就是 Text2Comp 的训练标签和输出。</p>
          </div>
        </div>
        <div className="definition-table-wrap">
          <table className="definition-table">
            <thead>
              <tr>
                <th>序号</th>
                <th>字段</th>
                <th>显示名称</th>
                <th>含义</th>
                <th>单位</th>
              </tr>
            </thead>
            <tbody>
              {draft.parameters.map((item) => (
                <tr key={item.index}>
                  <td data-label="序号">
                    <span className="row-index">
                      {String(item.index + 1).padStart(2, "0")}
                    </span>
                  </td>
                  <td data-label="字段">
                    <code>{item.name}</code>
                  </td>
                  <td data-label="显示名称">
                    <input
                      aria-label={`${item.name} 显示名称`}
                      value={item.display_name}
                      onChange={(event) =>
                        setParameter(
                          item.index,
                          "display_name",
                          event.target.value,
                        )
                      }
                    />
                  </td>
                  <td data-label="含义">
                    <input
                      aria-label={`${item.name} 含义`}
                      placeholder="可选"
                      value={item.description}
                      onChange={(event) =>
                        setParameter(
                          item.index,
                          "description",
                          event.target.value,
                        )
                      }
                    />
                  </td>
                  <td data-label="单位">
                    <input
                      aria-label={`${item.name} 单位`}
                      placeholder="可选"
                      value={item.unit}
                      onChange={(event) =>
                        setParameter(item.index, "unit", event.target.value)
                      }
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="definition-section">
        <div className="section-heading">
          <span>
            <SlidersHorizontal size={20} />
          </span>
          <div>
            <h2>物理输出与评测范围</h2>
            <p>只决定专家模型验证时保留哪些输出，不会成为 Text2Comp 标签。</p>
          </div>
        </div>
        <div className="output-grid">
          {draft.outputs.map((item) => {
            const selected = selectedChannels.includes(item.index);
            return (
              <div
                className={`output-row${selected ? " is-selected" : ""}`}
                key={item.index}
              >
                <button
                  className="channel-toggle"
                  type="button"
                  onClick={() => toggleChannel(item.index)}
                  aria-label={`${selected ? "取消" : "选择"}${item.name}`}
                >
                  {selected ? <Check size={16} /> : null}
                </button>
                <code>{item.name}</code>
                <label className="output-field">
                  <span>显示名称</span>
                  <input
                    aria-label={`${item.name} 显示名称`}
                    value={item.display_name}
                    onChange={(event) =>
                      setOutput(item.index, "display_name", event.target.value)
                    }
                  />
                </label>
                <label className="output-field">
                  <span>输出含义</span>
                  <input
                    aria-label={`${item.name} 描述`}
                    placeholder="可选"
                    value={item.description}
                    onChange={(event) =>
                      setOutput(item.index, "description", event.target.value)
                    }
                  />
                </label>
                <label className="output-field">
                  <span>单位</span>
                  <input
                    aria-label={`${item.name} 单位`}
                    placeholder="可选"
                    value={item.unit}
                    onChange={(event) =>
                      setOutput(item.index, "unit", event.target.value)
                    }
                  />
                </label>
              </div>
            );
          })}
        </div>
        <div className="sampling-line">
          <span>
            {allChannelsSelected
              ? "全部输出通道"
              : `已选 ${selectedChannels.length}/${draft.outputs.length} 个通道`}
          </span>
          <label className="compact-field">
            时间步长{" "}
            <input
              type="number"
              min={1}
              value={draft.sampling.time_stride}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  sampling: {
                    ...draft.sampling,
                    time_stride: Math.max(1, Number(event.target.value)),
                  },
                })
              }
            />
          </label>
          <label className="compact-field">
            最多时间点{" "}
            <input
              type="number"
              min={1}
              placeholder="全部"
              value={draft.sampling.max_time_points ?? ""}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  sampling: {
                    ...draft.sampling,
                    max_time_points: event.target.value
                      ? Math.max(1, Number(event.target.value))
                      : null,
                  },
                })
              }
            />
          </label>
        </div>
      </section>

      <footer className="page-actions">
        <button className="button ghost" type="button" onClick={onBack}>
          <ArrowLeft size={18} /> 返回接入
        </button>
        <div className="action-summary">
          <strong>{valid ? "定义完整" : "还有必填项未完成"}</strong>
          <span>
            {source.sample_count.toLocaleString()} 条源样本 ·{" "}
            {selectedChannels.length} 个评测输出通道
          </span>
        </div>
        <button
          className="button primary"
          type="button"
          disabled={!valid || busy}
          onClick={() => void onSave(draft)}
        >
          {busy ? <LoaderCircle className="spin" size={18} /> : null}
          {busy ? "保存中" : "保存并继续"}
          <ArrowRight size={18} />
        </button>
      </footer>
    </main>
  );
}
