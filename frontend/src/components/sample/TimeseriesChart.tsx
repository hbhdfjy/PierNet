import { useState, useMemo, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import type { SampleRecord } from '../../lib/types'
import { parseTimeseries, toRechartsData } from '../../lib/parseTimeseries'
import { getLineColor } from '../../lib/utils'
import { TrendingUp, AlertCircle } from 'lucide-react'

interface Props {
  sample: SampleRecord
}

const MAX_CHANNELS_DEFAULT = 1

export default function TimeseriesChart({ sample }: Props) {
  const parsed = useMemo(
    () => parseTimeseries(sample.target, sample.metadata),
    [sample],
  )

  const [visibleChannels, setVisibleChannels] = useState<Set<number>>(
    () => new Set(parsed ? parsed.channels.map((_, i) => i) : [])
  )

  // 当解析结果变化（换了样本）时重置通道选择
  useEffect(() => {
    setVisibleChannels(new Set(parsed ? parsed.channels.map((_, i) => i) : []))
  }, [parsed])

  if (!parsed) {
    return (
      <div className="flex items-center gap-2 text-slate-500 text-sm py-3 px-1 bg-slate-900/30 rounded-lg border border-slate-700/30">
        <AlertCircle size={14} className="flex-shrink-0" />
        <span>无法解析时序数据（target 中未找到 [[...]] 矩阵）</span>
      </div>
    )
  }

  const { channels, labels } = parsed
  const n_ch = channels.length

  const visibleLabels = labels.filter((_, i) => visibleChannels.has(i))
  const visibleChannelData = channels.filter((_, i) => visibleChannels.has(i))
  const chartData = toRechartsData(visibleChannelData, visibleLabels)

  const toggleChannel = (i: number) => {
    setVisibleChannels(prev => {
      const next = new Set(prev)
      if (next.has(i)) { if (next.size > 1) next.delete(i) }
      else next.add(i)
      return next
    })
  }

  return (
    <div className="space-y-3">
      {/* 通道选择器 */}
      {n_ch > MAX_CHANNELS_DEFAULT && (
        <div className="flex flex-wrap gap-1.5 p-2.5 bg-slate-900/40 rounded-lg border border-slate-700/30">
          <span className="text-xs text-slate-500 self-center mr-1">通道：</span>
          {labels.map((label, i) => (
            <button
              key={i}
              onClick={() => toggleChannel(i)}
              className={`text-xs px-2 py-0.5 rounded-full border transition-all duration-150 ${
                visibleChannels.has(i)
                  ? 'border-transparent text-white shadow-sm'
                  : 'border-slate-600/60 text-slate-500 hover:border-slate-400 hover:text-slate-300'
              }`}
              style={visibleChannels.has(i) ? { background: getLineColor(i) } : {}}
              title={label}
            >
              {i + 1}
            </button>
          ))}
          <div className="w-px bg-slate-700/60 self-stretch mx-0.5" />
          <button
            onClick={() => setVisibleChannels(new Set(labels.map((_, i) => i)))}
            className="text-xs px-2 py-0.5 rounded-full border border-slate-600/60 text-slate-400 hover:border-slate-400 hover:text-slate-200 transition-all duration-150"
          >
            全选
          </button>
          <button
            onClick={() => setVisibleChannels(new Set([0]))}
            className="text-xs px-2 py-0.5 rounded-full border border-slate-600/60 text-slate-400 hover:border-slate-400 hover:text-slate-200 transition-all duration-150"
          >
            仅第1
          </button>
        </div>
      )}

      {/* 图表 */}
      <div className="bg-slate-900/30 rounded-lg p-2 border border-slate-700/20">
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(51,65,85,0.4)" />
            <XAxis
              dataKey="t"
              tick={{ fill: '#64748b', fontSize: 11 }}
              axisLine={{ stroke: 'rgba(51,65,85,0.4)' }}
              tickLine={false}
              label={{ value: '时间步', position: 'insideBottomRight', fill: '#475569', fontSize: 10, offset: -4 }}
            />
            <YAxis
              tick={{ fill: '#64748b', fontSize: 11 }}
              axisLine={{ stroke: 'rgba(51,65,85,0.4)' }}
              tickLine={false}
              width={52}
            />
            <Tooltip
              contentStyle={{
                background: '#1e293b',
                border: '1px solid rgba(51,65,85,0.8)',
                borderRadius: 10,
                fontSize: 12,
                boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
              }}
              labelStyle={{ color: '#94a3b8', marginBottom: 4 }}
              itemStyle={{ color: '#cbd5e1' }}
              labelFormatter={(v) => `t = ${v}`}
            />
            {n_ch <= 6 && (
              <Legend
                wrapperStyle={{ fontSize: 11, color: '#64748b', paddingTop: 6 }}
              />
            )}
            {visibleLabels.map((label, i) => {
              // use the original channel index for consistent color assignment
              // find by position in the full labels array to handle duplicate labels
              const origIdx = labels.findIndex((l, idx) => l === label && visibleChannels.has(idx))
              const colorIdx = origIdx >= 0 ? origIdx : i
              return (
                <Line
                  key={`${label}-${i}`}
                  type="monotone"
                  dataKey={label}
                  stroke={getLineColor(colorIdx)}
                  strokeWidth={1.5}
                  dot={chartData.length <= 20 ? { r: 2.5, fill: getLineColor(colorIdx) } : false}
                  activeDot={{ r: 4, strokeWidth: 0 }}
                />
              )
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* 形状信息 */}
      <div className="grid grid-cols-2 gap-2 text-xs text-slate-500">
        <div className="bg-slate-900/30 rounded px-2.5 py-1.5 border border-slate-700/20">
          <span className="text-slate-600">原始</span>
          <span className="ml-2 font-mono text-slate-400">
            {sample.metadata.timeseries_shape[0]} × {sample.metadata.timeseries_shape[1]}
          </span>
        </div>
        <div className="bg-slate-900/30 rounded px-2.5 py-1.5 border border-slate-700/20">
          <span className="text-slate-600">观测</span>
          <span className="ml-2 font-mono text-slate-400">
            {sample.metadata.timeseries_shape_obs[0]} × {sample.metadata.timeseries_shape_obs[1]}
          </span>
        </div>
      </div>
    </div>
  )
}
