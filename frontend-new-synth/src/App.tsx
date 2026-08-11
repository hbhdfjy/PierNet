import {
  ArrowLeft,
  Clock3,
  DatabaseZap,
  History,
  KeyRound,
  Plus,
  RefreshCw,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "./api";
import { ApiSettingsDialog } from "./components/ApiSettingsDialog";
import { StepRail } from "./components/StepRail";
import { DefinitionStep } from "./pages/DefinitionStep";
import { GenerateStep } from "./pages/GenerateStep";
import { SourceStep } from "./pages/SourceStep";
import type {
  DataDefinition,
  GenerateOptions,
  LLMConfig,
  Presets,
  WizardStep,
  WorkflowSnapshot,
  WorkflowSummary,
} from "./types";

function stepFromWorkflow(workflow: WorkflowSnapshot | null): WizardStep {
  if (!workflow?.source?.ready) return "source";
  if (
    workflow.status === "succeeded" ||
    workflow.current_step === "generation" ||
    workflow.current_step === "complete"
  ) {
    return "generation";
  }
  return "definition";
}

function dateLabel(timestamp: number): string {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(timestamp * 1000);
}

function statusLabel(status: WorkflowSummary["status"]): string {
  return {
    draft: "进行中",
    running: "处理中",
    succeeded: "已完成",
    failed: "失败",
    cancelled: "已取消",
  }[status];
}

function workflowAllowsStep(
  workflow: WorkflowSnapshot,
  step: WizardStep,
): boolean {
  if (step === "source") return true;
  if (step === "definition")
    return Boolean(workflow.can_define || workflow.definition);
  return Boolean(workflow.can_generate || workflow.artifacts);
}

export default function App() {
  const [booting, setBooting] = useState(true);
  const [fatalError, setFatalError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [presets, setPresets] = useState<Presets | null>(null);
  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [workflow, setWorkflow] = useState<WorkflowSnapshot | null>(null);
  const [step, setStep] = useState<WizardStep>("source");
  const [llmConfig, setLLMConfig] = useState<LLMConfig | null>(null);
  const [apiSettingsOpen, setApiSettingsOpen] = useState(false);
  const closeApiSettings = useCallback(() => setApiSettingsOpen(false), []);

  const updateUrl = useCallback(
    (workflowId: string | null, nextStep: WizardStep) => {
      const params = new URLSearchParams();
      if (workflowId) params.set("workflow", workflowId);
      params.set("step", nextStep);
      window.history.replaceState({}, "", `/new-synth/?${params.toString()}`);
    },
    [],
  );

  const refreshWorkflows = useCallback(async () => {
    const result = await api.workflows();
    setWorkflows(result);
  }, []);

  const openWorkflow = useCallback(
    async (workflowId: string, requestedStep?: WizardStep) => {
      setBusy(true);
      setError(null);
      try {
        const snapshot = await api.workflow(workflowId);
        const nextStep =
          requestedStep && workflowAllowsStep(snapshot, requestedStep)
            ? requestedStep
            : stepFromWorkflow(snapshot);
        setWorkflow(snapshot);
        setStep(nextStep);
        updateUrl(snapshot.workflow_id, nextStep);
        window.scrollTo({ top: 0, behavior: "auto" });
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "无法打开任务");
      } finally {
        setBusy(false);
      }
    },
    [updateUrl],
  );

  useEffect(() => {
    let active = true;
    const boot = async () => {
      try {
        await api.createSession();
        const [presetData, workflowData, llmData] = await Promise.all([
          api.presets(),
          api.workflows(),
          api.getLLMConfig().catch(() => null),
        ]);
        if (!active) return;
        setPresets(presetData);
        setWorkflows(workflowData);
        setLLMConfig(llmData);
        const params = new URLSearchParams(window.location.search);
        const workflowId = params.get("workflow");
        const requested = params.get("step");
        if (workflowId) {
          const snapshot = await api.workflow(workflowId);
          if (!active) return;
          const requestedStep =
            requested === "source" ||
            requested === "definition" ||
            requested === "generation"
              ? requested
              : null;
          const allowedStep =
            requestedStep && workflowAllowsStep(snapshot, requestedStep)
              ? requestedStep
              : stepFromWorkflow(snapshot);
          setWorkflow(snapshot);
          setStep(allowedStep);
        }
      } catch (caught) {
        if (active)
          setFatalError(
            caught instanceof Error ? caught.message : "新数据合成服务不可用",
          );
      } finally {
        if (active) setBooting(false);
      }
    };
    void boot();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!workflow || workflow.status !== "running") return;
    const timer = window.setInterval(() => {
      void api
        .workflow(workflow.workflow_id)
        .then((snapshot) => {
          setWorkflow(snapshot);
          if (snapshot.status !== "running") void refreshWorkflows();
        })
        .catch((caught: unknown) =>
          setError(caught instanceof Error ? caught.message : "刷新任务失败"),
        );
    }, 800);
    return () => window.clearInterval(timer);
  }, [refreshWorkflows, workflow]);

  const run = useCallback(
    async (task: () => Promise<WorkflowSnapshot | void>): Promise<boolean> => {
      setBusy(true);
      setError(null);
      try {
        const snapshot = await task();
        if (snapshot) setWorkflow(snapshot);
        await refreshWorkflows();
        return true;
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "操作失败，请重试");
        return false;
      } finally {
        setBusy(false);
      }
    },
    [refreshWorkflows],
  );

  const ensureWorkflow = useCallback(
    async (name: string): Promise<WorkflowSnapshot> => {
      if (workflow) return workflow;
      const created = await api.createWorkflow(name);
      setWorkflow(created);
      updateUrl(created.workflow_id, "source");
      return created;
    },
    [updateUrl, workflow],
  );

  const enabled = useMemo<Record<WizardStep, boolean>>(
    () => ({
      source: true,
      definition: Boolean(workflow?.can_define || workflow?.definition),
      generation: Boolean(workflow?.can_generate || workflow?.artifacts),
    }),
    [workflow],
  );

  const changeStep = (nextStep: WizardStep) => {
    if (!enabled[nextStep]) return;
    setStep(nextStep);
    updateUrl(workflow?.workflow_id ?? null, nextStep);
    window.scrollTo({ top: 0, behavior: "auto" });
  };

  const reset = () => {
    setWorkflow(null);
    setStep("source");
    setError(null);
    updateUrl(null, "source");
    window.scrollTo({ top: 0, behavior: "auto" });
  };

  if (booting) {
    return (
      <div className="boot-screen">
        <span className="brand-mark">
          <DatabaseZap size={24} />
        </span>
        <p>正在准备新数据合成</p>
      </div>
    );
  }

  if (fatalError) {
    return (
      <div className="fatal-screen">
        <span className="brand-mark">
          <DatabaseZap size={24} />
        </span>
        <h1>暂时无法连接数据合成服务</h1>
        <p>{fatalError}</p>
        <button
          className="button primary"
          type="button"
          onClick={() => window.location.reload()}
        >
          <RefreshCw size={18} /> 重新连接
        </button>
      </div>
    );
  }

  return (
    <div className="app-frame">
      <header className="app-header">
        <a className="back-link" href="/">
          <ArrowLeft size={18} />
          <span>返回 Piern</span>
        </a>
        <div className="product-title">
          <span className="brand-mark small">
            <DatabaseZap size={19} />
          </span>
          <div>
            <strong>Piern</strong>
            <span>新数据合成</span>
          </div>
        </div>
        <div className="header-actions">
          <button
            className={`api-settings-trigger${llmConfig?.has_api_key ? " is-configured" : ""}`}
            type="button"
            aria-label="API 设置"
            onClick={() => setApiSettingsOpen(true)}
          >
            <KeyRound size={17} />
            <span>
              <strong>API 设置</strong>
              <small>{llmConfig?.has_api_key ? "已连接" : "未配置"}</small>
            </span>
          </button>
          <details className="history-menu">
            <summary>
              <History size={18} />
              <span>历史任务</span>
            </summary>
            <div className="history-popover">
              <div className="popover-heading">
                <strong>最近任务</strong>
                <span>{workflows.length} 个</span>
              </div>
              {workflows.length ? (
                workflows.slice(0, 8).map((item) => (
                  <button
                    type="button"
                    key={item.workflow_id}
                    onClick={() => void openWorkflow(item.workflow_id)}
                  >
                    <span>
                      <strong>{item.name}</strong>
                      <small>
                        <Clock3 size={13} /> {dateLabel(item.updated_at)}
                      </small>
                    </span>
                    <em data-status={item.status}>
                      {statusLabel(item.status)}
                    </em>
                  </button>
                ))
              ) : (
                <p className="empty-history">还没有数据合成任务</p>
              )}
            </div>
          </details>
          <button
            className="icon-button"
            type="button"
            title="新建数据合成任务"
            onClick={reset}
          >
            <Plus size={20} />
          </button>
        </div>
      </header>

      {apiSettingsOpen ? (
        <ApiSettingsDialog
          onClose={closeApiSettings}
          onConfigured={setLLMConfig}
        />
      ) : null}

      <div className="workspace">
        <StepRail current={step} enabled={enabled} onChange={changeStep} />
        {step === "source" ? (
          <SourceStep
            workflow={workflow}
            presets={presets}
            busy={busy || workflow?.status === "running"}
            error={error}
            onUpload={async (file) => {
              await run(async () => {
                const current = await ensureWorkflow(
                  file.name.replace(/\.(h5|hdf5)$/i, "") || "新数据任务",
                );
                return api.uploadSource(current.workflow_id, file);
              });
            }}
            onSimulation={async (payload) => {
              await run(async () => {
                const current = await ensureWorkflow(
                  `${payload.simulator} · ${payload.scenario}`,
                );
                return api.useSimulation(current.workflow_id, payload);
              });
            }}
            onExpert={async (payload) => {
              await run(async () => {
                const current = await ensureWorkflow(
                  `专家生成 · ${payload.scenario}`,
                );
                return api.useExpert(current.workflow_id, payload);
              });
            }}
            onExpertUpload={async (name, file) => {
              setBusy(true);
              setError(null);
              try {
                await api.uploadExpert(name, file);
                setPresets(await api.presets());
              } catch (caught) {
                setError(
                  caught instanceof Error ? caught.message : "专家模型上传失败",
                );
              } finally {
                setBusy(false);
              }
            }}
            onContinue={() => changeStep("definition")}
          />
        ) : null}

        {step === "definition" && workflow?.source && workflow.definition ? (
          <DefinitionStep
            source={workflow.source}
            definition={workflow.definition}
            busy={busy}
            error={error}
            llmConfigured={Boolean(llmConfig?.has_api_key)}
            onBack={() => changeStep("source")}
            onConfigureApi={() => setApiSettingsOpen(true)}
            onSuggest={async (definition: DataDefinition) => {
              setBusy(true);
              setError(null);
              try {
                const suggested = await api.suggestDefinition(
                  workflow.workflow_id,
                  definition,
                );
                return suggested;
              } catch (caught) {
                setError(
                  caught instanceof Error
                    ? caught.message
                    : "智能补全数据定义失败",
                );
                return null;
              } finally {
                setBusy(false);
              }
            }}
            onSave={async (definition: DataDefinition) => {
              const saved = await run(async () =>
                api.saveDefinition(workflow.workflow_id, definition),
              );
              if (saved) changeStep("generation");
            }}
          />
        ) : null}

        {step === "generation" && workflow ? (
          <GenerateStep
            workflow={workflow}
            busy={busy}
            error={error}
            onBack={() => changeStep("definition")}
            onGenerate={async (options: GenerateOptions, retry) => {
              setBusy(true);
              setError(null);
              try {
                await api.generate(workflow.workflow_id, options, retry);
                setWorkflow(await api.workflow(workflow.workflow_id));
              } catch (caught) {
                setError(
                  caught instanceof Error ? caught.message : "训练数据生成失败",
                );
              } finally {
                setBusy(false);
              }
            }}
            onCancel={async () => {
              await run(async () => api.cancel(workflow.workflow_id));
            }}
          />
        ) : null}
      </div>
    </div>
  );
}
