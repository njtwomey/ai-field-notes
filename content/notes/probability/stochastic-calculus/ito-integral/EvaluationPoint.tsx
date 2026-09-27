import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'

const FINE = 2 ** 12
type Point = 'left' | 'mid' | 'right'

/**
 * Running sums Σ W(τ_j) ΔW_j for one Brownian path on [0, 2], with τ_j the left end, the average of the two ends, or
 * the right end of each step. They converge to ½(W² − t), ½W² and ½(W² + t) respectively.
 */
export function EvaluationPoint() {
  const k = useParam(6, { min: 2, max: 12, step: 1 })
  const [point, setPoint] = useState<Point>('left')
  const path = useMemo(() => {
    const { normal } = rng(3)
    const b = new Float64Array(FINE + 1)
    const sd = Math.sqrt(2 / FINE)
    for (let i = 1; i <= FINE; i++) b[i] = b[i - 1] + sd * normal()
    return b
  }, [])

  const { series, final } = useMemo(() => {
    const m = 2 ** k.value
    const stride = FINE / m
    const t: number[] = [0]
    const sum: number[] = [0]
    let s = 0
    for (let j = 1; j <= m; j++) {
      const a = path[(j - 1) * stride]
      const b = path[j * stride]
      const w = point === 'left' ? a : point === 'right' ? b : (a + b) / 2
      s += w * (b - a)
      t.push((2 * j) / m)
      sum.push(s)
    }
    // Reference curves on the fine grid, thinned for drawing.
    const ft: number[] = []
    const ito: number[] = []
    const strat: number[] = []
    const right: number[] = []
    for (let i = 0; i <= FINE; i += 16) {
      const ti = (2 * i) / FINE
      const w2 = path[i] ** 2
      ft.push(ti)
      ito.push((w2 - ti) / 2)
      strat.push(w2 / 2)
      right.push((w2 + ti) / 2)
    }
    const out: XYSeries[] = [
      { name: '½(W² − t): Itô', type: 'line', x: ft, y: ito, dashed: true, slot: 0 },
      { name: '½W²: Stratonovich', type: 'line', x: ft, y: strat, dashed: true, slot: 1 },
      { name: '½(W² + t): right point', type: 'line', x: ft, y: right, dashed: true, slot: 2 },
      { name: `running sum (${point} point)`, type: 'line', x: t, y: sum, emphasis: true },
    ]
    return { series: out, final: s }
  }, [path, k.value, point])

  const wT = path[FINE]
  return (
    <Interactive
      title="Where the integrand is evaluated changes the integral"
      caption="One Brownian path on [0, 2]. The solid line is the running sum Σ W(τⱼ)(W(tⱼ₊₁) − W(tⱼ)) over 2ᵏ steps, with τⱼ the left end of each step (Itô), the average of the two ends (Stratonovich) or the right end. As k grows each sum settles on a different curve: ½(W² − t), ½W² or ½(W² + t). The gaps between them are ½t, half the accumulated (ΔW)²."
      controls={
        <>
          <ParamSlider label="k (2ᵏ steps)" param={k} withArrows />
          <ParamChoice
            label="evaluate at"
            value={point}
            onChange={setPoint}
            options={[
              { value: 'left', label: 'left' },
              { value: 'mid', label: 'average' },
              { value: 'right', label: 'right' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="sum at t = 2" value={formatNumber(final)} />
          <Readout label="½(W₂² − 2)" value={formatNumber((wT * wT - 2) / 2)} />
          <Readout label="½W₂²" value={formatNumber((wT * wT) / 2)} />
          <Readout label="½(W₂² + 2)" value={formatNumber((wT * wT + 2) / 2)} />
        </>
      }
    >
      <XYChart height={300} xLabel="t" yLabel="∫₀ᵗ W dW" series={series} xRange={[0, 2]} />
    </Interactive>
  )
}
