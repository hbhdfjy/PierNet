import { AlertCircle, CheckCircle2, Clock3, LoaderCircle, PauseCircle } from "lucide-react";

import { statusLabel } from "../lib/format";
import type { ProjectStatus, StageStatus } from "../types";

const icons = {
  waiting: Clock3,
  running: LoaderCircle,
  succeeded: CheckCircle2,
  failed: AlertCircle,
  cancelled: PauseCircle,
  draft: Clock3,
  ready: CheckCircle2,
};

export function StatusBadge({ status }: { status: StageStatus | ProjectStatus }) {
  const Icon = icons[status];
  return (
    <span className={`status-badge status-badge--${status}`}>
      <Icon size={14} aria-hidden="true" />
      {statusLabel[status]}
    </span>
  );
}
