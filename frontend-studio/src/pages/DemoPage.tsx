import { Download, Send, Sparkles } from "lucide-react";
import { type FormEvent, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";

import { AppShell } from "../components/AppShell";
import { Button } from "../components/Button";
import { ErrorState, LoadingState } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { ResultVisual } from "../components/ResultVisual";
import { useProject } from "../hooks/useProject";
import { api } from "../lib/api";
import { flattenNumbers, formatDuration, formatNumber } from "../lib/format";
import type { InferenceResult } from "../types";

function downloadResult(result: InferenceResult) {
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "piern-studio-result.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

export function DemoPage() {
  const { projectId } = useParams();
  const { project, loading, error, refresh } = useProject(projectId);
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [promptInjected, setPromptInjected] = useState(false);
  const [resultOrigin, setResultOrigin] = useState<"validation" | "chat" | null>(null);

  useEffect(() => {
    if (!project || promptInjected) return;
    setMessage(project.recommended_prompt ?? "");
    setResult(project.result);
    setResultOrigin(project.result ? "validation" : null);
    setPromptInjected(true);
  }, [project, promptInjected]);

  const values = useMemo(() => flattenNumbers(result?.output), [result]);
  const usingRecommendedPrompt =
    Boolean(project?.recommended_prompt) && message.trim() === project?.recommended_prompt?.trim();

  if (loading) {
    return (
      <AppShell>
        <LoadingState label="正在打开 Demo" />
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
  if (!project.can_chat) {
    return (
      <AppShell project={project}>
        <ErrorState message="这个 Demo 尚未构建完成，请先返回构建页面。" />
      </AppShell>
    );
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!message.trim()) return;
    setBusy(true);
    setActionError(null);
    setResult(null);
    setResultOrigin(null);
    try {
      setResult(await api.chat(projectId, message.trim()));
      setResultOrigin("chat");
    } catch (reason) {
      setActionError(reason instanceof Error ? reason.message : "计算没有完成");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AppShell project={project}>
      <PageHeader
        eyebrow="对话测试"
        title={project.name}
        description="修改参数后发送，结果来自你上传的计算模型。"
      />

      <form className="composer" onSubmit={(event) => void submit(event)}>
        <textarea
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          rows={4}
          aria-label="计算问题"
          placeholder="输入包含参数的计算问题"
        />
        <div className="composer__footer">
          <span>
            <Sparkles size={15} />
            {usingRecommendedPrompt
              ? "已填入与当前数据匹配的问题"
              : "已修改问题，将按当前内容重新计算"}
          </span>
          <Button type="submit" busy={busy} disabled={!message.trim()} icon={<Send size={17} />}>
            开始计算
          </Button>
        </div>
      </form>

      {actionError ? <ErrorState message={actionError} /> : null}

      {result ? (
        <section className="result-workspace">
          <div className="result-heading">
            <div>
              <span className="eyebrow">
                {resultOrigin === "validation" ? "构建验证" : "计算结果"}
              </span>
              <h2>{resultOrigin === "validation" ? "构建时验证结果" : "计算模型已返回结果"}</h2>
            </div>
            <Button
              variant="quiet"
              icon={<Download size={17} />}
              onClick={() => downloadResult(result)}
            >
              下载 JSON
            </Button>
          </div>
          <p className="result-answer">{result.answer}</p>
          <div className="result-layout">
            <div className="result-visual-panel">
              <ResultVisual chart={result.chart} />
            </div>
            <div className="result-meta">
              <div>
                <span>任务匹配度</span>
                <strong>{Math.round(result.confidence * 100)}%</strong>
              </div>
              <div>
                <span>计算耗时</span>
                <strong>{formatDuration(result.latency_ms)} ms</strong>
              </div>
              <div>
                <span>输出数值</span>
                <strong>{values.length}</strong>
              </div>
            </div>
          </div>
          <div className="numeric-section">
            <div className="numeric-section__heading">
              <h3>输入参数</h3>
              <span>{result.inputs.length} 个数值</span>
            </div>
            <div className="numeric-grid">
              {result.inputs.map((value, index) => (
                <div key={`input-${index}`}>
                  <span>{project.data?.input_names?.[index] ?? `输入 ${index + 1}`}</span>
                  <strong>{formatNumber(value)}</strong>
                </div>
              ))}
            </div>
          </div>
          <div className="numeric-section">
            <div className="numeric-section__heading">
              <h3>模型输出</h3>
              <span>
                {values.length > 96 ? `显示前 96 / ${values.length}` : `${values.length} 个数值`}
              </span>
            </div>
            <div className="output-table">
              {values.slice(0, 96).map((value, index) => (
                <div key={`output-${index}`}>
                  <span>{index + 1}</span>
                  <strong>{formatNumber(value)}</strong>
                </div>
              ))}
            </div>
          </div>
        </section>
      ) : null}
    </AppShell>
  );
}
