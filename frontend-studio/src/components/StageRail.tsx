import { Check, ChevronDown, Circle, LoaderCircle, TriangleAlert } from "lucide-react";

import type { ProjectSnapshot, StageSnapshot } from "../types";

function StageIcon({ stage }: { stage: StageSnapshot }) {
  if (stage.status === "succeeded") return <Check size={14} />;
  if (stage.status === "running") return <LoaderCircle className="spin" size={15} />;
  if (stage.status === "failed") return <TriangleAlert size={15} />;
  return <Circle size={11} />;
}

export function StageRail({ project }: { project: ProjectSnapshot }) {
  const currentIndex = Math.max(
    0,
    project.stages.findIndex((stage) => stage.id === project.current_stage),
  );
  const currentStage = project.stages[currentIndex] ?? project.stages[0];

  return (
    <aside className="stage-rail" aria-label="项目进度">
      <div className="stage-rail__project">
        <span>当前项目</span>
        <strong>{project.name}</strong>
      </div>
      <ol className="stage-list">
        {project.stages.map((stage, index) => (
          <li
            className={`stage-item stage-item--${stage.status}`}
            key={stage.id}
            aria-current={project.current_stage === stage.id ? "step" : undefined}
          >
            <span className="stage-item__track" aria-hidden="true">
              <span className="stage-item__marker">
                <StageIcon stage={stage} />
              </span>
              {index < project.stages.length - 1 ? <span className="stage-item__line" /> : null}
            </span>
            <span className="stage-item__copy">
              <strong>{stage.title}</strong>
              <small>{stage.message}</small>
            </span>
          </li>
        ))}
      </ol>
      {currentStage ? (
        <details className="stage-mobile">
          <summary>
            <span
              className={`stage-mobile__marker stage-mobile__marker--${currentStage.status}`}
              aria-hidden="true"
            >
              <StageIcon stage={currentStage} />
            </span>
            <span className="stage-mobile__current">
              <small>
                当前阶段 {currentIndex + 1}/{project.stages.length}
              </small>
              <strong>{currentStage.title}</strong>
            </span>
            <ChevronDown className="stage-mobile__chevron" size={18} aria-hidden="true" />
          </summary>
          <ol className="stage-mobile__list">
            {project.stages.map((stage, index) => (
              <li
                className={`stage-mobile__item stage-mobile__item--${stage.status}`}
                key={stage.id}
                aria-current={project.current_stage === stage.id ? "step" : undefined}
              >
                <span aria-hidden="true">
                  <StageIcon stage={stage} />
                </span>
                <strong>
                  {index + 1}. {stage.title}
                </strong>
                <small>{stage.message}</small>
              </li>
            ))}
          </ol>
        </details>
      ) : null}
    </aside>
  );
}
