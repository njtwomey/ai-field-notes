import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'

const ALPHA = linspace(0, 1, 201)

/**
 * Estimating a target mean from n_t target and n_s source observations whose mean is off by δ. The combined estimate
 * α x̄_s + (1 − α) x̄_t has MSE α²(δ² + σ²/n_s) + (1 − α)² σ²/n_t; everything is shown relative to target only (α = 0).
 */
export function SourceTargetTradeoff() {
  const [delta, setDelta] = useState(0.3)
  const [nt, setNt] = useState(10)
  const [ns, setNs] = useState(500)
  const alpha = useParam(0.5, { min: 0, max: 1, step: 0.005 })

  const r = useMemo(() => {
    const vt = 1 / nt
    const vs = 1 / ns
    const rel = (a: number) => (a * a * (delta * delta + vs) + (1 - a) * (1 - a) * vt) / vt
    const star = vt / (delta * delta + vs + vt)
    const pooled = ns / (ns + nt)
    const series: XYSeries[] = [
      { name: 'combined estimate', type: 'line', x: ALPHA, y: ALPHA.map(rel), slot: 0 },
      { name: 'target only', type: 'line', x: [0, 1], y: [1, 1], dashed: true, muted: true },
      { name: 'source only', type: 'scatter', x: [1], y: [rel(1)], slot: 1 },
      { name: 'pooled, α = n_s/(n_s + n_t)', type: 'scatter', x: [pooled], y: [rel(pooled)], slot: 2 },
      { name: 'optimum α*', type: 'scatter', x: [star], y: [rel(star)], emphasis: true },
    ]
    return { series, star, pooled, rel }
  }, [delta, nt, ns])

  return (
    <Interactive
      title="Borrowing from a biased source"
      caption="Mean squared error of α·(source mean) + (1 − α)·(target mean), relative to using the target alone, for unit noise variance. Drag the vertical line or use the α slider. Below the dashed line the source helps; above it the source hurts (the axis stops at 3). Every α between 0 and 2α* helps, and pooling the two samples hurts once the bias δ is large compared with the target's standard error."
      controls={
        <>
          <ParamSlider label="bias δ / σ" value={delta} onChange={setDelta} min={0} max={1.5} step={0.01} />
          <ParamSlider label="target sample size n_t" value={nt} onChange={setNt} min={1} max={100} step={1} />
          <ParamSlider label="source sample size n_s" value={ns} onChange={setNs} min={10} max={2000} step={10} />
          <ParamSlider label="weight on source α" param={alpha} />
        </>
      }
      readout={
        <>
          <Readout label="relative MSE at α" value={formatNumber(r.rel(alpha.value))} />
          <Readout label="optimal α*" value={formatNumber(r.star)} />
          <Readout label="break-even 2α*" value={formatNumber(Math.min(2 * r.star, 1))} />
          <Readout label="relative MSE, pooled" value={formatNumber(r.rel(r.pooled))} />
        </>
      }
    >
      <XYChart
        series={r.series}
        xLabel="weight on source α"
        yLabel="MSE relative to target only"
        xRange={[0, 1]}
        yRange={[0, 3]}
        handles={[{ kind: 'x', at: alpha.value, label: 'α', onDrag: alpha.set }]}
      />
    </Interactive>
  )
}
