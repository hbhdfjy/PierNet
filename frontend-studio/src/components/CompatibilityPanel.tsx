import { CheckCircle2, CircleDashed, TriangleAlert } from "lucide-react";

import { formatNumber, shapeLabel } from "../lib/format";
import type { CompatibilityReport } from "../types";

export function CompatibilityPanel({ report }: { report: CompatibilityReport }) {
  const Icon = report.compatible ? CheckCircle2 : TriangleAlert;
  return (
    <section
      className={`compatibility-panel compatibility-panel--${report.compatible ? "success" : "error"}`}
    >
      <div className="compatibility-panel__title">
        <Icon size={24} />
        <div>
          <span className="eyebrow">匹配结果</span>
          <h2>{report.compatible ? "数据与计算模型匹配" : "需要调整资源"}</h2>
          <p>
            {report.compatible
              ? `已用 ${report.sample_count} 条真实样本完成前向验证`
              : report.message}
          </p>
        </div>
      </div>
      <div className="compatibility-metrics">
        <div>
          <span>期望输出</span>
          <strong>{shapeLabel(report.expected_output_shape)}</strong>
        </div>
        <div>
          <span>实际输出</span>
          <strong>{shapeLabel(report.actual_output_shape)}</strong>
        </div>
        <div>
          <span>样本误差</span>
          <strong>{report.sample_mse === undefined ? "—" : formatNumber(report.sample_mse)}</strong>
        </div>
        <div>
          <span>数值有效</span>
          <strong className="inline-status">
            {report.finite ? <CheckCircle2 size={15} /> : <CircleDashed size={15} />}
            {report.finite ? "是" : "否"}
          </strong>
        </div>
      </div>
    </section>
  );
}
