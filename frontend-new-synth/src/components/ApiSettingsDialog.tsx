import {
  CheckCircle2,
  KeyRound,
  LoaderCircle,
  Save,
  Wifi,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import type { LLMConfig, LLMConfigRequest, LLMTestResult } from "../types";

interface ApiSettingsDialogProps {
  onClose: () => void;
  onConfigured: (config: LLMConfig) => void;
}

const PROVIDERS = [
  { value: "siliconflow", label: "SiliconFlow" },
  { value: "openai", label: "OpenAI" },
  { value: "deepseek", label: "DeepSeek" },
  { value: "anthropic", label: "Anthropic" },
];

export function ApiSettingsDialog({
  onClose,
  onConfigured,
}: ApiSettingsDialogProps) {
  const [loading, setLoading] = useState(true);
  const [working, setWorking] = useState(false);
  const [provider, setProvider] = useState("siliconflow");
  const [model, setModel] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [hasSavedKey, setHasSavedKey] = useState(false);
  const [maskedKey, setMaskedKey] = useState("");
  const [temperature, setTemperature] = useState(0.5);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [thinking, setThinking] = useState<"enabled" | "disabled">("disabled");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LLMTestResult | null>(null);

  useEffect(() => {
    let active = true;
    void api
      .getLLMConfig()
      .then((config) => {
        if (!active) return;
        setProvider(config.provider || "siliconflow");
        setModel(config.model || "");
        setBaseUrl(config.base_url || "");
        setHasSavedKey(config.has_api_key);
        setMaskedKey(config.api_key_masked || "");
        setTemperature(config.temperature);
        setMaxTokens(config.max_tokens);
        setThinking(config.thinking);
      })
      .catch((caught: unknown) => {
        if (active)
          setError(
            caught instanceof Error ? caught.message : "读取 API 设置失败",
          );
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      active = false;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  const request = useMemo<LLMConfigRequest>(
    () => ({
      provider,
      model: model.trim(),
      api_key: apiKey.trim(),
      base_url: baseUrl.trim(),
      temperature,
      max_tokens: maxTokens,
      thinking,
    }),
    [apiKey, baseUrl, maxTokens, model, provider, temperature, thinking],
  );

  const validate = () => {
    if (!request.model) {
      setError("请填写模型名称。");
      return false;
    }
    if (!request.api_key && !hasSavedKey) {
      setError("请填写 API Key。");
      return false;
    }
    return true;
  };

  const test = async (payload: LLMConfigRequest) => {
    const response = await api.testLLMConfig(payload);
    setResult(response);
    if (!response.ok) setError(response.message || "API 连接测试失败");
    return response;
  };

  const handleTest = async () => {
    if (!validate()) return;
    setWorking(true);
    setError(null);
    setResult(null);
    try {
      await test(request);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "API 连接测试失败");
    } finally {
      setWorking(false);
    }
  };

  const handleSave = async () => {
    if (!validate()) return;
    setWorking(true);
    setError(null);
    setResult(null);
    try {
      await api.saveLLMConfig(request);
      const saved = await api.getLLMConfig();
      setHasSavedKey(saved.has_api_key);
      setMaskedKey(saved.api_key_masked || "");
      setApiKey("");
      onConfigured(saved);
      await test({ ...request, api_key: "" });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "保存 API 设置失败");
    } finally {
      setWorking(false);
    }
  };

  return (
    <div
      className="dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="api-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="api-dialog-title"
      >
        <header className="dialog-heading">
          <span className="dialog-icon">
            <KeyRound size={20} />
          </span>
          <div>
            <h2 id="api-dialog-title">智能识别 API</h2>
            <p>用于补全任务、参数和物理输出说明，不参与数值计算。</p>
          </div>
          <button
            className="icon-button"
            type="button"
            aria-label="关闭 API 设置"
            onClick={onClose}
          >
            <X size={19} />
          </button>
        </header>

        {loading ? (
          <div className="dialog-loading">
            <LoaderCircle className="spin" size={20} /> 正在读取设置
          </div>
        ) : (
          <div className="dialog-body">
            {error ? (
              <div className="inline-error" role="alert">
                {error}
              </div>
            ) : null}
            {result?.ok ? (
              <div className="connection-result is-success" role="status">
                <CheckCircle2 size={18} />
                <span>
                  <strong>{result.message}</strong>
                  {result.response_preview
                    ? ` · 返回 ${result.response_preview}`
                    : ""}
                </span>
              </div>
            ) : null}

            <div className="api-form-grid">
              <label className="field">
                <span>服务商</span>
                <select
                  value={provider}
                  onChange={(event) => setProvider(event.target.value)}
                >
                  {PROVIDERS.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>模型</span>
                <input
                  value={model}
                  placeholder="例如 deepseek-ai/DeepSeek-V3.2"
                  onChange={(event) => setModel(event.target.value)}
                />
              </label>
              <label className="field span-two">
                <span>Base URL</span>
                <input
                  value={baseUrl}
                  placeholder="留空使用服务商默认地址"
                  onChange={(event) => setBaseUrl(event.target.value)}
                />
              </label>
              <label className="field span-two">
                <span>API Key</span>
                <input
                  type="password"
                  autoComplete="new-password"
                  value={apiKey}
                  placeholder={
                    hasSavedKey ? "留空保持现有 Key" : "填写服务商 API Key"
                  }
                  onChange={(event) => setApiKey(event.target.value)}
                />
                <small>
                  {hasSavedKey
                    ? `当前已保存 ${maskedKey || "****"}`
                    : "Key 仅提交给当前 Piern 服务器，不会回显明文"}
                </small>
              </label>
            </div>

            <details className="api-advanced">
              <summary>高级参数</summary>
              <div className="api-form-grid">
                <label className="field">
                  <span>Temperature</span>
                  <input
                    type="number"
                    min={0}
                    max={2}
                    step={0.1}
                    value={temperature}
                    onChange={(event) =>
                      setTemperature(Number(event.target.value))
                    }
                  />
                </label>
                <label className="field">
                  <span>最大输出 Token</span>
                  <input
                    type="number"
                    min={64}
                    max={8192}
                    step={64}
                    value={maxTokens}
                    onChange={(event) =>
                      setMaxTokens(Number(event.target.value))
                    }
                  />
                </label>
                <label className="field span-two">
                  <span>思考模式</span>
                  <select
                    value={thinking}
                    onChange={(event) =>
                      setThinking(
                        event.target.value === "enabled"
                          ? "enabled"
                          : "disabled",
                      )
                    }
                  >
                    <option value="disabled">关闭</option>
                    <option value="enabled">开启</option>
                  </select>
                </label>
              </div>
            </details>
          </div>
        )}

        <footer className="dialog-actions">
          <button
            className="button secondary"
            type="button"
            disabled={loading || working}
            onClick={() => void handleTest()}
          >
            <Wifi size={17} /> 测试连接
          </button>
          <button
            className="button primary"
            type="button"
            disabled={loading || working}
            onClick={() => void handleSave()}
          >
            {working ? (
              <LoaderCircle className="spin" size={17} />
            ) : (
              <Save size={17} />
            )}
            保存并测试
          </button>
        </footer>
      </section>
    </div>
  );
}
