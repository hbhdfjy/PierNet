import { describe, expect, it } from 'vitest'
import { formatAssemblyAnswer } from './assemblyResult'

describe('formatAssemblyAnswer', () => {
  it('turns MODFLOW matrices into readable value rows', () => {
    const answer = formatAssemblyAnswer(
      'MODFLOW地下水专家输出：\n[[10.759705543518066,13.775484085083008],[10.423967361450195,13.48708438873291]]\n中文趋势总结：\n1. 井1：末段高于起始水平。\n2. 井2：末段高于起始水平。',
    )

    expect(answer).toContain('已完成 MODFLOW 地下水预测')
    expect(answer).toContain('井1：10.7597，13.7755')
    expect(answer).toContain('井2：10.4240，13.4871')
    expect(answer).not.toContain('10.759705543518066')
  })

  it('summarizes dense expert vectors without echoing the array', () => {
    const answer = formatAssemblyAnswer(
      '好的，科学计算预测结果为\n\n[[0.78702, 0.71876, 0.65693, 0.60376, 0.58216, 0.57277]]。',
    )

    expect(answer).toContain('好的，科学计算预测结果为')
    expect(answer).toContain('专家模型返回了一组数值预测')
    expect(answer).not.toContain('[[0.78702')
  })

  it('keeps already readable prediction rows', () => {
    const answer = formatAssemblyAnswer('预测数值：\n第 1-4 点：0.78702，0.71876，0.65693，0.60376')

    expect(answer).toContain('第 1-4 点：0.78702，0.71876')
  })
})
