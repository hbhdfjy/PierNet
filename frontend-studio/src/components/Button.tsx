import clsx from "clsx";
import type { ButtonHTMLAttributes, ReactNode } from "react";

type ButtonVariant = "primary" | "secondary" | "quiet" | "danger";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  icon?: ReactNode;
  busy?: boolean;
}

export function Button({
  variant = "primary",
  icon,
  busy = false,
  className,
  children,
  disabled,
  ...props
}: ButtonProps) {
  return (
    <button
      className={clsx("button", `button--${variant}`, className)}
      disabled={disabled || busy}
      {...props}
    >
      {busy ? <span className="button__spinner" aria-hidden="true" /> : icon}
      <span>{busy ? "处理中" : children}</span>
    </button>
  );
}
