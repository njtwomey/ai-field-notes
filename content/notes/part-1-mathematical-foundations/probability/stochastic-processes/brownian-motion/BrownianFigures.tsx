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
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'

const STEP_CHOICES = ['4', '16', '64', '256', '4096'] as const
type Steps = (typeof STEP_CHOICES)[number]
/** Longest walk drawn as an exact step function; longer walks are sampled at this many points for drawing only. */
const DRAWN_STEPS = 256
const T = linspace(0, 1, 101)

/** Scaled simple random walks W(t) = S⌊nt⌋/√n on [0, 1]: jagged for small n, Brownian-looking for large n. */
export function ScaledWalks() {
  const [steps, setSteps] = useState<Steps>('16')
  const n = Number(steps)

  const paths = useParam(3, { min: 1, max: 30, step: 1 })
  const count = paths.value

  const series = useMemo((): XYSeries[] => {
    const many = count > 1
    const out: XYSeries[] = []
    const stride = Math.max(1, n / DRAWN_STEPS)
    for (let p = 0; p < count; p++) {
      // Each walk has its own stream, so adding walks leaves the existing ones unchanged.
      const { uniform } = rng(40 + p)
      const x = [0]
      const y = [0]
      let s = 0
      for (let k = 1; k <= n; k++) {
        s += uniform() < 0.5 ? 1 : -1
        if (stride === 1) {
          // The walk is constant between steps, so draw it as a step function.
          x.push(k / n, k / n)
          y.push(y[y.length - 1], s / Math.sqrt(n))
        } else if (k % stride === 0) {
          x.push(k / n)
          y.push(s / Math.sqrt(n))
        }
      }
      out.push({ name: many ? 'scaled walks' : 'scaled walk', type: 'line', x, y, slot: 0, thin: many })
    }
    out.push({ name: '+2√t', type: 'line', x: T, y: T.map((t) => 2 * Math.sqrt(t)), muted: true, dashed: true })
    out.push({ name: '−2√t', type: 'line', x: T, y: T.map((t) => -2 * Math.sqrt(t)), muted: true, dashed: true })
    return out
  }, [n, count])

  return (
    <Interactive
      title="Scaled random walks converge to Brownian motion"
      caption="Simple random walks with n steps of ±1, squeezed into time [0, 1] and scaled by 1/√n, drawn as light lines; the paths slider sets how many. At every n the value at time t has mean 0 and variance ⌊nt⌋/n ≈ t; the dashed curves are ±2√t, and about 95% of the walks lie between them at any t. As n grows the steps vanish and the paths take the rough, self-similar look of Brownian motion. Walks with more than 256 steps are drawn at 256 evenly spaced times."
      controls={
        <>
          <ParamChoice
            label="steps n"
            value={steps}
            onChange={setSteps}
            options={STEP_CHOICES.map((s) => ({ value: s, label: s }))}
          />
          <ParamSlider label="paths" param={paths} withArrows format={(v) => String(v)} />
        </>
      }
      readout={<Readout label="step size in space 1/√n" value={formatNumber(1 / Math.sqrt(n))} />}
    >
      <XYChart height={300} xLabel="t" yLabel="W(t)" series={series} xRange={[0, 1]} yRange={[-3, 3]} />
    </Interactive>
  )
}

const FINE = 2 ** 14

/**
 * One Brownian path on a fine grid, sampled on a coarser partition with 2^k intervals. The sum of squared increments
 * settles at t = 1 (quadratic variation); the sum of absolute increments grows like √(2^k) (infinite total variation).
 */
export function QuadraticVariation() {
  const [k, setK] = useState(4)
  const path = useMemo(() => {
    const { normal } = rng(11)
    const b = new Float64Array(FINE + 1)
    const sd = Math.sqrt(1 / FINE)
    for (let i = 1; i <= FINE; i++) b[i] = b[i - 1] + sd * normal()
    return b
  }, [])

  const m = 2 ** k
  const { series, qv, tv } = useMemo(() => {
    const stride = FINE / m
    let q = 0
    let v = 0
    const xs: number[] = []
    const ys: number[] = []
    for (let j = 0; j <= m; j++) {
      xs.push(j / m)
      ys.push(path[j * stride])
      if (j > 0) {
        const d = path[j * stride] - path[(j - 1) * stride]
        q += d * d
        v += Math.abs(d)
      }
    }
    // Draw the fine path thinned to 2048 points; it is the background the partition samples.
    const thin = 8
    const fx: number[] = []
    const fy: number[] = []
    for (let i = 0; i <= FINE; i += thin) {
      fx.push(i / FINE)
      fy.push(path[i])
    }
    const s: XYSeries[] = [
      { name: 'Brownian path', type: 'line', x: fx, y: fy, muted: true },
      { name: `partition with ${m} intervals`, type: 'line', x: xs, y: ys, slot: 0 },
    ]
    return { series: s, qv: q, tv: v }
  }, [path, m])

  return (
    <Interactive
      title="Quadratic variation of a Brownian path"
      caption="A Brownian path on [0, 1] and its values on a partition into 2ᵏ equal intervals. As the partition refines, the sum of squared increments settles at 1, the length of the time interval. The sum of absolute increments keeps growing like √(2ᵏ · 2/π): the path has infinite length."
      controls={<ParamSlider label="k (2ᵏ intervals)" value={k} onChange={setK} min={1} max={14} step={1} withArrows />}
      readout={
        <>
          <Readout label="Σ (ΔB)²" value={formatNumber(qv)} />
          <Readout label="Σ |ΔB|" value={formatNumber(tv)} />
          <Readout label="√(2ᵏ · 2/π)" value={formatNumber(Math.sqrt((m * 2) / Math.PI))} />
        </>
      }
    >
      <XYChart height={300} xLabel="t" yLabel="B(t)" series={series} xRange={[0, 1]} />
    </Interactive>
  )
}
