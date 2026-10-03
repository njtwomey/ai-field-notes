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

/** Schedules of Song et al. (2021): β(t) linear from 0.1 to 20 on [0, 1]; σ(t) = 0.01 · 5000ᵗ, from 0.01 to 50. */
const B_MIN = 0.1
const B_MAX = 20
const S_MIN = 0.01
const S_MAX = 50
const TS = linspace(0.002, 1, 250)
const intBeta = (t: number) => B_MIN * t + 0.5 * (B_MAX - B_MIN) * t * t

type Sde = 'vp' | 'subvp' | 've'
const KERNELS: Record<Sde, { label: string; m: (t: number) => number; s: (t: number) => number }> = {
  vp: { label: 'VP', m: (t) => Math.exp(-intBeta(t) / 2), s: (t) => Math.sqrt(1 - Math.exp(-intBeta(t))) },
  subvp: { label: 'sub-VP', m: (t) => Math.exp(-intBeta(t) / 2), s: (t) => 1 - Math.exp(-intBeta(t)) },
  ve: {
    label: 'VE',
    m: () => 1,
    s: (t) => Math.sqrt(Math.max(S_MIN ** 2 * (S_MAX / S_MIN) ** (2 * t) - S_MIN ** 2, 0)),
  },
}
/** Time grid of the sample paths, from x₀ at t = 0. */
const PATH_TS = [0, ...TS]
/** Largest |x| on the path panel for each SDE; VE paths reach about ±100 at t = 1. */
const PATH_RANGE: Record<Sde, [number, number]> = { vp: [-3.5, 3.5], subvp: [-3.5, 3.5], ve: [-120, 120] }

/**
 * One path of the forward SDE from x₀, sampled with the exact transition. Every one of the three SDEs is linear, so
 * x_t | x_u ~ N(r x_u, s(t)² − r² s(u)²) with r = m(t)/m(u): the variance at t is the variance at u, scaled by r²,
 * plus the variance added between u and t. `z` holds one standard normal per step.
 */
function forwardPath(k: (typeof KERNELS)[Sde], x0: number, z: Float64Array) {
  const y = [x0]
  for (let i = 1; i < PATH_TS.length; i++) {
    const u = PATH_TS[i - 1]
    const t = PATH_TS[i]
    const r = k.m(t) / k.m(u)
    const su = u === 0 ? 0 : k.s(u)
    y.push(r * y[i - 1] + Math.sqrt(Math.max(k.s(t) ** 2 - r * r * su * su, 0)) * z[i - 1])
  }
  return y
}

const STEP_CHOICES = ['50', '200', '1000'] as const
type Steps = (typeof STEP_CHOICES)[number]

/**
 * The transition kernel x_t | x₀ ~ N(m(t) x₀, s(t)²) of the three SDE families, with DDPM's discrete √ᾱ_i and
 * √(1 − ᾱ_i) for the matching linear schedule β_i = β(t_i)/N overlaid on the VP curves.
 */
export function NoiseSchedules() {
  const [sde, setSde] = useState<Sde>('vp')
  const [steps, setSteps] = useState<Steps>('50')
  const n = Number(steps)
  const x0 = useParam(1, { min: -2, max: 2, step: 0.05 })
  const count = useParam(10, { min: 1, max: 50, step: 1 })

  // One noise stream per path, shared by the three SDEs, so switching the SDE keeps the draws and raising the count
  // adds paths without changing the others.
  const noise = useMemo(
    () =>
      Array.from({ length: count.value }, (_, k) => {
        const { normal } = rng(61 * 1000 + k)
        return Float64Array.from({ length: PATH_TS.length - 1 }, () => normal())
      }),
    [count.value],
  )

  const pathSeries = useMemo<XYSeries[]>(() => {
    const k = KERNELS[sde]
    const mean = PATH_TS.map((t) => k.m(t) * x0.value)
    const sd = PATH_TS.map((t) => (t === 0 ? 0 : k.s(t)))
    const paths = noise.map((z) => ({ x: PATH_TS, y: forwardPath(k, x0.value, z) }))
    const x: number[] = []
    const y: number[] = []
    for (const p of paths) {
      x.push(...p.x, NaN)
      y.push(...p.y, NaN)
    }
    const band = [1, -1].flatMap((sign) => [...PATH_TS.map((_, i) => mean[i] + 2 * sign * sd[i]), NaN])
    return [
      { name: 'sample paths', type: 'line', x, y, slot: 2, thin: paths.length > 1 },
      { name: 'mean m(t) x₀', type: 'line', x: PATH_TS, y: mean, slot: 0 },
      { name: 'mean ± 2 s(t)', type: 'line', x: [...PATH_TS, NaN, ...PATH_TS, NaN], y: band, slot: 1 },
    ]
  }, [sde, x0.value, noise])

  const ddpm = useMemo(() => {
    const t: number[] = []
    const m: number[] = []
    const s: number[] = []
    let ab = 1
    let gap = 0
    const every = Math.max(1, Math.round(n / 50))
    for (let i = 1; i <= n; i++) {
      const beta = (B_MIN + ((B_MAX - B_MIN) * (i - 1)) / (n - 1)) / n
      ab *= 1 - beta
      gap = Math.max(gap, Math.abs(Math.sqrt(ab) - KERNELS.vp.m(i / n)))
      if (i % every === 0) {
        t.push(i / n)
        m.push(Math.sqrt(ab))
        s.push(Math.sqrt(1 - ab))
      }
    }
    return { t, m, s, gap }
  }, [n])

  const series = useMemo<XYSeries[]>(() => {
    const k = KERNELS[sde]
    const out: XYSeries[] = [
      { name: 'signal scale m(t)', type: 'line', x: TS, y: TS.map(k.m), slot: 0 },
      { name: 'noise scale s(t)', type: 'line', x: TS, y: TS.map(k.s), slot: 1 },
    ]
    if (sde === 'vp') {
      out.push({ name: `DDPM √ᾱᵢ, N = ${n}`, type: 'scatter', x: ddpm.t, y: ddpm.m, emphasis: true })
      out.push({ name: `DDPM √(1 − ᾱᵢ), N = ${n}`, type: 'scatter', x: ddpm.t, y: ddpm.s, muted: true })
    }
    return out
  }, [sde, n, ddpm])

  const k = KERNELS[sde]
  return (
    <Interactive
      title="How the three forward SDEs noise a data point"
      caption="Each forward SDE turns x₀ into x_t ~ N(m(t) x₀, s(t)²). The variance-preserving (VP) SDE shrinks the signal while the noise grows to 1; the sub-VP SDE uses the same shrinkage with less noise at every time; the variance-exploding (VE) SDE keeps the signal and grows the noise to 50. Schedules are those of Song et al. (2021); the y axis is logarithmic. In the VP view the dots are DDPM's √ᾱᵢ and √(1 − ᾱᵢ) for N steps of the matching discrete schedule: they approach the SDE curves as N grows. Below, sample paths of each forward SDE from the data point x₀ (drag it at t = 0), drawn as light lines, with the mean m(t) x₀ and the band of ±2 s(t); the paths slider sets how many. The VP and sub-VP paths forget x₀ and settle in a band of fixed width; the VE paths keep x₀ as their mean and spread to about ±100 (note the wider axis)."
      controls={
        <>
          <ParamChoice
            label="forward SDE"
            value={sde}
            onChange={setSde}
            options={(Object.keys(KERNELS) as Sde[]).map((s) => ({ value: s, label: KERNELS[s].label }))}
          />
          <ParamChoice
            label="DDPM steps N"
            value={steps}
            onChange={setSteps}
            options={STEP_CHOICES.map((s) => ({ value: s, label: s }))}
          />
          <ParamSlider label="data point x₀" param={x0} />
          <ParamSlider label="paths" param={count} withArrows format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="m(1)" value={formatNumber(k.m(1))} />
          <Readout label="s(1)" value={formatNumber(k.s(1))} />
          <Readout label="max |√ᾱᵢ − m(tᵢ)|" value={formatNumber(ddpm.gap)} />
        </>
      }
    >
      <XYChart height={300} xLabel="t" yLabel="scale" yLog series={series} xRange={[0, 1]} yRange={[0.001, 100]} />
      <XYChart
        height={240}
        xLabel="t"
        yLabel="x_t"
        series={pathSeries}
        xRange={[0, 1]}
        yRange={PATH_RANGE[sde]}
        handles={[{ kind: 'point', at: [0, x0.value], label: 'x₀', onDrag: ([, y]) => x0.set(y) }]}
      />
    </Interactive>
  )
}
