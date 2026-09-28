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
import { gaussPdf, histogram } from '../_shared/sde'

type Knob = 'constant' | 'steps' | 'ramp' | 'fade' | 'path'

const PATHS = 4000
const STEPS = 200

/** Each knob: its horizon T, the setting h(t) (for `path`, the knob is the path itself) and the exact variance. */
const KNOBS: Record<Knob, { label: string; T: number; h: (t: number) => number; variance: number }> = {
  constant: { label: 'constant 3', T: 2, h: () => 3, variance: 18 },
  steps: { label: '2, −1, 0.5', T: 2, h: (t) => (t < 0.5 ? 2 : t < 1.5 ? -1 : 0.5), variance: 3.125 },
  ramp: { label: 'ramp t', T: 1, h: (t) => t, variance: 1 / 3 },
  fade: { label: 'fading e^−(1−t)', T: 1, h: (t) => Math.exp(-(1 - t)), variance: (1 - Math.exp(-2)) / 2 },
  path: { label: 'the path W', T: 1, h: () => 0, variance: 0.5 },
}

/** Density of ½(W² − T) for W ~ N(0, T): the exact law of ∫₀ᵀ W dW. */
function itoWdWDensity(y: number, T: number): number {
  const s = 2 * y + T
  if (s <= 0) return 0
  const x = Math.sqrt(s)
  return (2 * gaussPdf(x, 0, T)) / x
}

/**
 * Simulated Itô integrals ∫ h dW for a knob set in advance (Gaussian with variance ∫ h² dt) or for the knob h = W
 * (not Gaussian: ½(W_T² − T)). Each path uses left-point sums over 200 steps.
 */
export function KnobIntegral() {
  const [knob, setKnob] = useState<Knob>('steps')
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const spec = KNOBS[knob]

  const sim = useMemo(() => {
    const { normal } = rng(seed.value)
    const dt = spec.T / STEPS
    const sd = Math.sqrt(dt)
    const values = new Float64Array(PATHS)
    let firstPath: number[] = []
    for (let p = 0; p < PATHS; p++) {
      let w = 0
      let sum = 0
      const trace = p === 0 ? [0] : []
      for (let j = 0; j < STEPS; j++) {
        const t = j * dt
        const dW = sd * normal()
        const h = knob === 'path' ? w : spec.h(t)
        sum += h * dW
        w += dW
        if (p === 0) trace.push(w)
      }
      values[p] = sum
      if (p === 0) firstPath = trace
    }
    let mean = 0
    for (const v of values) mean += v / PATHS
    let variance = 0
    for (const v of values) variance += (v - mean) ** 2 / (PATHS - 1)
    return { values, mean, variance, firstPath }
  }, [knob, seed.value, spec])

  const { knobSeries, histSeries, below } = useMemo(() => {
    const ts = Array.from({ length: STEPS + 1 }, (_, j) => (j * spec.T) / STEPS)
    const hs = knob === 'path' ? sim.firstPath : ts.map((t) => spec.h(t))
    const knobSeries: XYSeries[] = [
      {
        name: knob === 'path' ? 'h(t) = W(t), one sample path' : 'h(t)',
        type: 'line',
        x: ts,
        y: hs,
        slot: 0,
      },
    ]
    const sd = Math.sqrt(spec.variance)
    const lo = knob === 'path' ? -0.75 : -4 * sd
    const hi = knob === 'path' ? 3 : 4 * sd
    const hist = histogram(sim.values, lo, hi, 50)
    const grid = Array.from({ length: 241 }, (_, i) => lo + ((hi - lo) * i) / 240)
    const theory = grid.map((y) => (knob === 'path' ? itoWdWDensity(y, spec.T) : gaussPdf(y, 0, spec.variance)))
    const histSeries: XYSeries[] = [
      { name: `${PATHS} simulated integrals`, type: 'bar', x: hist.x, y: hist.y, muted: true },
      {
        name: knob === 'path' ? 'exact density of ½(W² − 1)' : `N(0, ${formatNumber(spec.variance)})`,
        type: 'line',
        x: grid,
        y: theory,
        slot: 1,
      },
    ]
    if (knob === 'path')
      histSeries.push({
        name: 'Gaussian with the same variance',
        type: 'line',
        x: grid,
        y: grid.map((y) => gaussPdf(y, 0, spec.variance)),
        slot: 2,
        dashed: true,
      })
    let below = 0
    for (const v of sim.values) if (v < -0.4) below++
    return { knobSeries, histSeries, below: below / PATHS }
  }, [knob, sim, spec])

  return (
    <Interactive
      title="Turn the knob: what the accumulated noise looks like"
      caption="White noise dW played through a volume knob h(t) for 200 steps, repeated on 4,000 paths. Top: the knob. Bottom: the histogram of the 4,000 values of ∫ h dW against the theory. For a knob set in advance the integral is Gaussian with variance ∫ h(t)² dt, whatever the knob's shape. For the knob h = W, which follows the path, the integral is ½(W₁² − 1): mean 0 and variance ½ as the isometry says, but skewed, never below −½, and not Gaussian (dashed)."
      controls={
        <>
          <ParamChoice
            label="knob h(t)"
            value={knob}
            onChange={setKnob}
            options={(Object.keys(KNOBS) as Knob[]).map((k) => ({ value: k, label: KNOBS[k].label }))}
          />
          <ParamSlider label="seed" param={seed} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="sample mean" value={formatNumber(sim.mean)} />
          <Readout label="sample variance" value={formatNumber(sim.variance)} />
          <Readout label="∫ h² dt (theory)" value={formatNumber(spec.variance)} />
          {knob === 'path' && <Readout label="share below −0.4 (exact 0.345)" value={formatNumber(below)} />}
        </>
      }
    >
      <div className="space-y-4">
        <XYChart height={180} xLabel="t" yLabel="h(t)" series={knobSeries} />
        <XYChart height={260} xLabel="∫ h dW" yLabel="density" series={histSeries} />
      </div>
    </Interactive>
  )
}
