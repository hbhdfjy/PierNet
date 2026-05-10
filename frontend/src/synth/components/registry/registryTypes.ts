export type OutputInfoItem = {
  name: string
  name_zh?: string
  description: string
  unit: string
  slice: [number, number | null]
}
export type TimeModeItem = { name: string; desc_en: string; desc_zh?: string; indices: string }
export type ObsConfig = {
  fixed_time_mode?: string
  fixed_channels?: Array<number | string> | null // null=全选，列表=指定索引或 output_info.name
  channel_level?: string
  channel_name_template?: string
  channel_name_template_zh?: string
  time_modes?: TimeModeItem[]
}
export type RegistryEntry = {
  domain_context?: string
  output_description?: string
  param_info?: Record<string, [string, string]>
  output_info?: OutputInfoItem[]
  observation_config?: ObsConfig
  [k: string]: unknown
}
