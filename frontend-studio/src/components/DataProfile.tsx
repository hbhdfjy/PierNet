import { ArrowRight, Braces, Database, Sigma } from "lucide-react";

import { formatNumber, shapeLabel } from "../lib/format";
import type { DataResource } from "../types";

export function DataProfile({ data }: { data: DataResource }) {
  if (!data.samples || !data.input_shape || !data.output_shape) return null;
  return (
    <section className="data-profile">
      <div className="section-heading">
        <div>
          <span className="eyebrow">数据概览</span>
          <h2>结构清晰，可以用于构建</h2>
        </div>
      </div>
      <div className="profile-flow">
        <div className="profile-node">
          <span>
            <Braces size={17} />
            输入
          </span>
          <strong>{shapeLabel(data.input_shape)}</strong>
          <small>{data.input_dim} 个数值维度</small>
        </div>
        <ArrowRight className="profile-flow__arrow" size={22} />
        <div className="profile-node profile-node--accent">
          <span>
            <Sigma size={17} />
            输出
          </span>
          <strong>{shapeLabel(data.output_shape)}</strong>
          <small>{data.output_dim} 个数值维度</small>
        </div>
        <div className="profile-samples">
          <Database size={18} />
          <span>样本</span>
          <strong>{data.samples.toLocaleString("zh-CN")}</strong>
        </div>
      </div>
      {data.input_stats && data.output_stats ? (
        <div className="range-table">
          <div className="range-table__head">
            <span />
            <span>最小值</span>
            <span>平均值</span>
            <span>最大值</span>
          </div>
          <div>
            <strong>输入</strong>
            <span>{formatNumber(data.input_stats.min)}</span>
            <span>{formatNumber(data.input_stats.mean)}</span>
            <span>{formatNumber(data.input_stats.max)}</span>
          </div>
          <div>
            <strong>输出</strong>
            <span>{formatNumber(data.output_stats.min)}</span>
            <span>{formatNumber(data.output_stats.mean)}</span>
            <span>{formatNumber(data.output_stats.max)}</span>
          </div>
        </div>
      ) : null}
    </section>
  );
}
