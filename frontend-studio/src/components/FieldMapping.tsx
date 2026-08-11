import { useEffect, useMemo, useState } from "react";

import type { DataColumn } from "../types";
import { Button } from "./Button";

interface FieldMappingProps {
  columns: DataColumn[];
  suggestedInputs?: string[];
  suggestedOutputs?: string[];
  busy: boolean;
  onConfirm: (inputs: string[], outputs: string[]) => Promise<void>;
}

export function FieldMapping({
  columns,
  suggestedInputs = [],
  suggestedOutputs = [],
  busy,
  onConfirm,
}: FieldMappingProps) {
  const numericColumns = useMemo(() => columns.filter((column) => column.numeric), [columns]);
  const [roles, setRoles] = useState<Record<string, "input" | "output" | "ignore">>({});

  useEffect(() => {
    setRoles(
      Object.fromEntries(
        numericColumns.map((column) => [
          column.name,
          suggestedInputs.includes(column.name)
            ? "input"
            : suggestedOutputs.includes(column.name)
              ? "output"
              : "ignore",
        ]),
      ),
    );
  }, [numericColumns, suggestedInputs, suggestedOutputs]);

  const inputs = numericColumns
    .filter((column) => roles[column.name] === "input")
    .map((column) => column.name);
  const outputs = numericColumns
    .filter((column) => roles[column.name] === "output")
    .map((column) => column.name);

  return (
    <section className="mapping-section">
      <div className="section-heading section-heading--split">
        <div>
          <span className="eyebrow">确认字段</span>
          <h2>哪些列是输入，哪些列是输出？</h2>
        </div>
        <span className="mapping-count">
          {inputs.length} 输入 · {outputs.length} 输出
        </span>
      </div>
      <div className="mapping-table">
        <div className="mapping-table__head">
          <span>字段</span>
          <span>示例</span>
          <span>用途</span>
        </div>
        {numericColumns.map((column) => (
          <div className="mapping-row" key={column.name}>
            <strong>{column.name}</strong>
            <code>{column.sample.slice(0, 2).join(", ")}</code>
            <div className="segmented" aria-label={`${column.name} 字段用途`}>
              {(["input", "output", "ignore"] as const).map((role) => (
                <button
                  type="button"
                  className={roles[column.name] === role ? "is-active" : undefined}
                  onClick={() => setRoles((current) => ({ ...current, [column.name]: role }))}
                  key={role}
                >
                  {{ input: "输入", output: "输出", ignore: "忽略" }[role]}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="section-footer">
        <Button
          busy={busy}
          disabled={!inputs.length || !outputs.length}
          onClick={() => void onConfirm(inputs, outputs)}
        >
          确认字段
        </Button>
      </div>
    </section>
  );
}
