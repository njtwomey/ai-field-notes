import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'

const D = 200
type Signal = 'spread' | 'decaying'

/** Coefficients of the true linear model over all D features, normalised to ‖β‖ = 1. */
function coefficients(signal: Signal): number[] {
  const raw = Array.from({ length: D }, (_, j) => (signal === 'spread' ? 1 : 1 / (j + 1)))
  const norm = Math.hypot(...raw)
  return raw.map((b) => b / norm)
}

/**
 * Expected test and training error of minimum-norm least squares on the first p of D independent standard Gaussian
 * features (Belkin, Hsu & Xu 2020). The unused features act as extra noise of variance ‖β_rest‖². Undefined within one
 * of the interpolation threshold p = n, where the expectation is infinite.
 */
function risk(p: number, n: number, sigma2: number, cum: number[]) {
  const used = cum[p]
  const noise = cum[D] - used + sigma2
  const train = p < n ? (noise * (n - p)) / n : 0
  if (p <= n - 2) return { test: noise * (1 + p / (n - p - 1)), train }
  if (p >= n + 2) return { test: used * (1 - n / p) + noise * (1 + n / (p - n - 1)), train }
  return { test: NaN, train }
}

/** Test error of least squares against the number of features used, across the interpolation threshold p = n. */
export function DoubleDescent() {
  const n = useParam(40, { min: 10, max: 100, step: 5 })
  const sigma = useParam(0.3, { min: 0, max: 1, step: 0.05 })
  const p = useParam(120, { min: 1, max: D, step: 1 })
  const [signal, setSignal] = useState<Signal>('spread')

  const cum = useMemo(() => {
    const beta = coefficients(signal)
    const out = [0]
    beta.forEach((b, j) => out.push(out[j] + b * b))
    return out
  }, [signal])
  const s2 = sigma.value ** 2

  const series = useMemo((): XYSeries[] => {
    const ps = Array.from({ length: D }, (_, i) => i + 1)
    const tests = ps.map((q) => risk(q, n.value, s2, cum).test)
    const trains = ps.map((q) => risk(q, n.value, s2, cum).train)
    const null0 = cum[D] + s2
    return [
      {
        name: 'test error',
        type: 'line',
        x: ps,
        y: tests.map((t) => (Number.isFinite(t) ? Math.min(t, 1e3) : NaN)),
        slot: 0,
      },
      { name: 'training error', type: 'line', x: ps, y: trains.map((t) => Math.max(t, 0.01)), slot: 1 },
      { name: 'predict zero', type: 'line', x: [1, D], y: [null0, null0], muted: true, dashed: true },
      { name: 'p = n', type: 'line', x: [n.value, n.value], y: [0.01, 100], emphasis: true, dashed: true },
    ]
  }, [n.value, s2, cum])

  const at = risk(p.value, n.value, s2, cum)
  const handles: Handle[] = [{ kind: 'x', at: p.value, label: 'p', onDrag: (x) => p.set(x) }]

  return (
    <Interactive
      title="Test error across the interpolation threshold"
      caption="Least squares on the first p of 200 independent Gaussian features, from n training points; the true model uses all 200 features. Below p = n the fit is ordinary least squares; above it, the fit is the minimum-norm interpolant and the training error is zero. The curves are exact expectations. The test error peaks where the model first interpolates, then falls again. With the signal spread over all features, the largest model beats every small one; with a decaying signal it does not. Drag the p line or use the sliders."
      controls={
        <>
          <ParamSlider label="features used p" param={p} format={(v) => String(v)} />
          <ParamSlider label="training points n" param={n} format={(v) => String(v)} />
          <ParamSlider label="noise σ" param={sigma} />
          <ParamChoice
            label="true coefficients"
            value={signal}
            onChange={setSignal}
            options={[
              { value: 'spread', label: 'spread evenly' },
              { value: 'decaying', label: 'decaying, βⱼ ∝ 1/j' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="p / n" value={formatNumber(p.value / n.value)} />
          <Readout label="expected test error" value={Number.isFinite(at.test) ? formatNumber(at.test) : '∞'} />
          <Readout label="expected training error" value={formatNumber(at.train)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="number of features p"
        yLabel="mean squared error"
        xRange={[1, D]}
        yRange={[0.01, 100]}
        yLog
        handles={handles}
        height={340}
      />
    </Interactive>
  )
}
