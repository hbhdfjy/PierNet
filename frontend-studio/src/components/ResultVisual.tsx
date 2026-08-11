import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { flattenNumbers, formatNumber } from "../lib/format";
import type { ChartSpec } from "../types";

function Heatmap({ values }: { values: number[][] }) {
  const flat = values.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const spread = Math.max(max - min, Number.EPSILON);
  return (
    <div
      className="heatmap"
      style={{ gridTemplateColumns: `repeat(${values[0]?.length ?? 1}, minmax(10px, 1fr))` }}
      aria-label={`${values.length} 行 ${values[0]?.length ?? 0} 列结果热力图`}
    >
      {values.flatMap((row, rowIndex) =>
        row.map((value, columnIndex) => {
          const ratio = (value - min) / spread;
          return (
            <span
              key={`${rowIndex}-${columnIndex}`}
              title={`[${rowIndex + 1}, ${columnIndex + 1}] ${formatNumber(value)}`}
              style={{
                backgroundColor: `color-mix(in srgb, #0d9488 ${20 + ratio * 75}%, #edf4f3)`,
              }}
            />
          );
        }),
      )}
    </div>
  );
}

export function ResultVisual({ chart }: { chart: ChartSpec }) {
  const lineData = useMemo(() => {
    if (chart.kind !== "line") return [];
    return chart.x.map((x, index) => {
      const item: Record<string, number> = { x };
      chart.series.forEach((series) => {
        item[series.name] = series.values[index];
      });
      return item;
    });
  }, [chart]);

  if (chart.kind === "metric") {
    return (
      <div className="metric-result">
        <span>{chart.label}</span>
        <strong>{formatNumber(chart.value)}</strong>
      </div>
    );
  }

  if (chart.kind === "heatmap") {
    return <Heatmap values={chart.values} />;
  }

  return (
    <div className="line-chart" aria-label="计算结果折线图">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={lineData} margin={{ top: 12, right: 14, left: -12, bottom: 0 }}>
          <CartesianGrid stroke="#dce4e7" strokeDasharray="3 4" vertical={false} />
          <XAxis
            dataKey="x"
            tickLine={false}
            axisLine={false}
            tick={{ fill: "#718087", fontSize: 11 }}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tick={{ fill: "#718087", fontSize: 11 }}
            tickFormatter={(value: number) => formatNumber(value)}
          />
          <Tooltip
            contentStyle={{
              border: "1px solid #d3dddf",
              borderRadius: 6,
              boxShadow: "0 10px 28px rgba(29, 50, 54, 0.1)",
              fontSize: 12,
            }}
            formatter={(value) => formatNumber(Number(value))}
          />
          {chart.series.map((series, index) => (
            <Line
              key={series.name}
              type="monotone"
              dataKey={series.name}
              stroke={index === 0 ? "#0d9488" : "#e76f51"}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              activeDot={{ r: 4, strokeWidth: 0 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <span className="sr-only">
        {flattenNumbers(chart.series.map((series) => series.values)).length} 个数值
      </span>
    </div>
  );
}
