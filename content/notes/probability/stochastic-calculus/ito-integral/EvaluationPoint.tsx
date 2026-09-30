import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

const FINE = 2 ** 12
const MAX_PATHS = 30
/** Points drawn per running sum; finer grids are thinned for drawing only. */
const DRAWN = 512

const POINTS = [
  { key: 'left', sums: 'left-point sums (Itô)', mean: 'mean 0', limit: '½(W² − t): Itô', slot: 0 },
  { key: 'mid', sums: 'average-point sums (Stratonovich)', mean: 'mean t/2', limit: '½W²: Stratonovich', slot: 1 },
  { key: 'right', sums: 'right-point sums', mean: 'mean t', limit: '½(W² + t): right point', slot: 2 },
] as const

/** Brownian path p on [0, 2] on the fine grid. Path 0 keeps seed 3; path p > 0 has its own stream. */
function brownian(p: number): Float64Array {
  const { normal } = rng(p === 0 ? 3 : 3000 + p)
  const b = new Float64Array(FINE + 1)
  const sd = Math.sqrt(2 / FINE)
  for (let i = 1; i <= FINE; i++) b[i] = b[i - 1] + sd * normal()
  return b
}

/**
 * Running sums Σ W(τ_j) ΔW_j for Brownian paths on [0, 2], with τ_j the left end, the average of the two ends, or the
 * right end of each step. They converge to ½(W² − t), ½W² and ½(W² + t), whose means are 0, t/2 and t.
 */
export function EvaluationPoint() {
  const k = useParam(6, { min: 2, max: 12, step: 1 })
  const count = useParam(8, { min: 1, max: MAX_PATHS, step: 1 })
  const paths = useMemo(() => Array.from({ length: count.value }, (_, p) => brownian(p)), [count.value])

  const { series, finals } = useMemo(() => {
    const m = 2 ** k.value
    const stride = FINE / m
    const every = Math.max(1, Math.ceil(m / DRAWN))
    const many = paths.length > 1
    // finals[p][i]: sum at t = 2 on path p with evaluation point i.
    const finals: number[][] = []
    const sums: XYSeries[][] = [[], [], []]
    for (const path of paths) {
      const t: number[] = [0]
      const y: number[][] = [[0], [0], [0]]
      const s = [0, 0, 0]
      for (let j = 1; j <= m; j++) {
        const a = path[(j - 1) * stride]
        const b = path[j * stride]
        const d = b - a
        s[0] += a * d
        s[1] += ((a + b) / 2) * d
        s[2] += b * d
        if (j % every === 0 || j === m) {
          t.push((2 * j) / m)
          for (let i = 0; i < 3; i++) y[i].push(s[i])
        }
      }
      finals.push(s)
      POINTS.forEach((pt, i) => sums[i].push({ name: pt.sums, type: 'line', x: t, y: y[i], slot: pt.slot, thin: many }))
    }
    const out: XYSeries[] = sums.flat()
    if (many) {
      // Exact means of the three sums at every k: E[W_j ΔW_j] = 0, E[W_{j+1} ΔW_j] = Δt, and the average of the two.
      POINTS.forEach((pt, i) =>
        out.push({ name: pt.mean, type: 'line', x: [0, 2], y: [0, i], slot: pt.slot, dashed: true }),
      )
    } else {
      // One path: the limits ½(W² ∓ t) and ½W² on the fine grid, thinned for drawing.
      const path = paths[0]
      const ft: number[] = []
      const lim: number[][] = [[], [], []]
      for (let i = 0; i <= FINE; i += 16) {
        const ti = (2 * i) / FINE
        const w2 = path[i] ** 2
        ft.push(ti)
        lim[0].push((w2 - ti) / 2)
        lim[1].push(w2 / 2)
        lim[2].push((w2 + ti) / 2)
      }
      POINTS.forEach((pt, i) =>
        out.push({ name: pt.limit, type: 'line', x: ft, y: lim[i], slot: pt.slot, dashed: true }),
      )
    }
    return { series: out, finals }
  }, [paths, k.value])

  const many = count.value > 1
  const avg = (i: number) => finals.reduce((s, f) => s + f[i], 0) / finals.length
  const gap = finals.reduce((s, f) => s + f[2] - f[0], 0) / finals.length
  const pre = many ? 'mean ' : ''
  return (
    <Interactive
      title="Where the integrand is evaluated changes the integral"
      caption="Brownian paths on [0, 2]; the paths slider sets how many. For each path the three solid lines are the running sums Σ W(τⱼ)(W(tⱼ₊₁) − W(tⱼ)) over 2ᵏ steps, with τⱼ the left end of each step (Itô), the average of the two ends (Stratonovich) or the right end. With one path the dashed curves are ½(W² − t), ½W² and ½(W² + t), on which the three sums settle as k grows. With several paths each sum is a light line and the dashed lines are the exact means 0, t/2 and t: the left and right sums separate by t on average, and the average-point sum sits halfway between. The gap on each path is Σ(ΔW)², which tends to t."
      controls={
        <>
          <ParamSlider label="k (2ᵏ steps)" param={k} withArrows />
          <ParamSlider label="paths" param={count} withArrows format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label={`${pre}left sum at t = 2`} value={formatNumber(avg(0))} />
          <Readout label={`${pre}average-point sum`} value={formatNumber(avg(1))} />
          <Readout label={`${pre}right sum`} value={formatNumber(avg(2))} />
          <Readout label={`${pre}right − left = Σ(ΔW)²`} value={formatNumber(gap)} />
        </>
      }
    >
      <XYChart height={300} xLabel="t" yLabel="∫₀ᵗ W dW" series={series} xRange={[0, 2]} />
    </Interactive>
  )
}
