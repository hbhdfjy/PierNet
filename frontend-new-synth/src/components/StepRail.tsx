import { Check, Database, FileCheck2, Sparkles } from "lucide-react";

import type { WizardStep } from "../types";

const steps: Array<{
  id: WizardStep;
  label: string;
  detail: string;
  icon: typeof Database;
}> = [
  {
    id: "source",
    label: "接入数据",
    detail: "上传、仿真或专家生成",
    icon: Database,
  },
  {
    id: "definition",
    label: "确认定义",
    detail: "参数、输出与采样范围",
    icon: FileCheck2,
  },
  {
    id: "generation",
    label: "生成数据",
    detail: "自动完成模板与数据集",
    icon: Sparkles,
  },
];

const order: Record<WizardStep, number> = {
  source: 0,
  definition: 1,
  generation: 2,
};

interface StepRailProps {
  current: WizardStep;
  enabled: Record<WizardStep, boolean>;
  onChange: (step: WizardStep) => void;
}

export function StepRail({ current, enabled, onChange }: StepRailProps) {
  const currentIndex = order[current];
  return (
    <nav className="step-rail" aria-label="数据合成步骤">
      {steps.map((step, index) => {
        const Icon = step.icon;
        const completed = index < currentIndex;
        const active = step.id === current;
        return (
          <button
            className={`step-item${active ? " is-active" : ""}${completed ? " is-complete" : ""}`}
            type="button"
            key={step.id}
            disabled={!enabled[step.id]}
            onClick={() => onChange(step.id)}
            aria-current={active ? "step" : undefined}
          >
            <span className="step-icon">
              {completed ? <Check size={18} /> : <Icon size={18} />}
            </span>
            <span className="step-copy">
              <strong>{step.label}</strong>
              <small>{step.detail}</small>
            </span>
            <span className="step-number">0{index + 1}</span>
          </button>
        );
      })}
    </nav>
  );
}
