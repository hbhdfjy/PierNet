import { AlertTriangle, LoaderCircle } from "lucide-react";

import { Button } from "./Button";

export function LoadingState({ label = "正在加载" }: { label?: string }) {
  return (
    <div className="feedback feedback--loading" role="status">
      <LoaderCircle className="spin" size={22} />
      <span>{label}</span>
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="feedback feedback--error" role="alert">
      <AlertTriangle size={22} />
      <div>
        <strong>暂时无法继续</strong>
        <p>{message}</p>
      </div>
      {onRetry ? (
        <Button variant="secondary" onClick={onRetry}>
          重试
        </Button>
      ) : null}
    </div>
  );
}
