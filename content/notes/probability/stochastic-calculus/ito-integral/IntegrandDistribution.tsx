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

type Integrand = 'constant' | 'steps' | 'ramp' | 'fade' | 'path'

const PATHS = 4000
const STEPS = 200

/** Each integrand: its horizon T, the function h(t) (unused for `path`, whose integrand is the path) and the exact variance. */
const INTEGRANDS: Record<Integrand, { label: string; T: number; h: (t: number) => number; variance: number }> = {
  constant: { label: 'constant 3', T: 2, h: () => 3, variance: 18 },
  steps: { label: '2, −1, 0.5', T: 2, h: (t) => (t < 0.5 ? 2 : t < 1.5 ? -1 : 0.5), variance: 3.125 },
  ramp: { label: 'ramp t', T: 1, h: (t) => t, variance: 1 / 3 },
  fade: { label: 'exponential e^−(1−t)', T: 1, h: (t) => Math.exp(-(1 - t)), variance: (1 - Math.exp(-2)) / 2 },
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
 * Simulated Itô integrals ∫ h dW for a deterministic integrand (Gaussian with variance ∫ h² dt) or for the integrand h = W
 * (not Gaussian: ½(W_T² − T)). Each path uses left-point sums over 200 steps.
 */
export function IntegrandDistribution() {
  const [integrand, setIntegrand] = useState<Integrand>('steps')
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const spec = INTEGRANDS[integrand]

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
        const h = integrand === 'path' ? w : spec.h(t)
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
  }, [integrand, seed.value, spec])

  const { integrandSeries, histSeries, below } = useMemo(() => {
    const ts = Array.from({ length: STEPS + 1 }, (_, j) => (j * spec.T) / STEPS)
    const hs = integrand === 'path' ? sim.firstPath : ts.map((t) => spec.h(t))
    const integrandSeries: XYSeries[] = [
      {
        name: integrand === 'path' ? 'h(t) = W(t), one sample path' : 'h(t)',
        type: 'line',
        x: ts,
        y: hs,
        slot: 0,
      },
    ]
    const sd = Math.sqrt(spec.variance)
    const lo = integrand === 'path' ? -0.75 : -4 * sd
    const hi = integrand === 'path' ? 3 : 4 * sd
    const hist = histogram(sim.values, lo, hi, 50)
    const grid = Array.from({ length: 241 }, (_, i) => lo + ((hi - lo) * i) / 240)
    const theory = grid.map((y) => (integrand === 'path' ? itoWdWDensity(y, spec.T) : gaussPdf(y, 0, spec.variance)))
    const histSeries: XYSeries[] = [
      { name: `${PATHS} simulated integrals`, type: 'bar', x: hist.x, y: hist.y, muted: true },
      {
        name: integrand === 'path' ? 'exact density of ½(W² − 1)' : `N(0, ${formatNumber(spec.variance)})`,
        type: 'line',
        x: grid,
        y: theory,
        slot: 1,
      },
    ]
    if (integrand === 'path')
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
    return { integrandSeries, histSeries, below: below / PATHS }
  }, [integrand, sim, spec])

  return (
    <Interactive
      title="The distribution of ∫ h dW for different integrands"
      caption="Left-point sums of ∫ h dW over 200 steps, on 4,000 simulated Brownian paths. Top: the integrand h(t). Bottom: the histogram of the 4,000 values against the theory. For a deterministic integrand the integral is Gaussian with variance ∫ h(t)² dt, whatever the integrand's shape. For the integrand h = W, which depends on the path, the integral is ½(W₁² − 1): mean 0 and variance ½ as the isometry says, but skewed, never below −½, and not Gaussian (dashed)."
      controls={
        <>
          <ParamChoice
            label="integrand h(t)"
            value={integrand}
            onChange={setIntegrand}
            options={(Object.keys(INTEGRANDS) as Integrand[]).map((k) => ({ value: k, label: INTEGRANDS[k].label }))}
          />
          <ParamSlider label="seed" param={seed} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="sample mean" value={formatNumber(sim.mean)} />
          <Readout label="sample variance" value={formatNumber(sim.variance)} />
          <Readout label="∫ h² dt (theory)" value={formatNumber(spec.variance)} />
          {integrand === 'path' && <Readout label="share below −0.4 (exact 0.345)" value={formatNumber(below)} />}
        </>
      }
    >
      <div className="space-y-4">
        <XYChart height={180} xLabel="t" yLabel="h(t)" series={integrandSeries} />
        <XYChart height={260} xLabel="∫ h dW" yLabel="density" series={histSeries} />
      </div>
    </Interactive>
  )
}
