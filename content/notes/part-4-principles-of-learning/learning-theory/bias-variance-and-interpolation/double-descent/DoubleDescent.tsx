import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
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
  const state = useFigureState({
    p: int(120, { min: 1, max: D, step: 1, label: 'features used p', format: (v) => String(v) }),
    n: int(40, { min: 10, max: 100, step: 5, label: 'training points n', format: (v) => String(v) }),
    sigma: float(0.3, { min: 0, max: 1, step: 0.05, label: 'noise σ' }),
    signal: choice<Signal>(
      [
        { value: 'spread', label: 'spread evenly' },
        { value: 'decaying', label: 'decaying, βⱼ ∝ 1/j' },
      ],
      'spread',
      { label: 'true coefficients' },
    ),
  })

  const cum = useMemo(() => {
    const beta = coefficients(state.signal)
    const out = [0]
    beta.forEach((b, j) => out.push(out[j] + b * b))
    return out
  }, [state.signal])
  const s2 = state.sigma ** 2

  const series = useMemo(() => {
    const ps = Array.from({ length: D }, (_, i) => i + 1)
    const tests = ps.map((q) => risk(q, state.n, s2, cum).test)
    const trains = ps.map((q) => risk(q, state.n, s2, cum).train)
    const null0 = cum[D] + s2
    return [
      {
        name: 'test error',
        x: ps,
        y: tests.map((t) => (Number.isFinite(t) ? Math.min(t, 1e3) : NaN)),
        slot: 0,
      },
      { name: 'training error', x: ps, y: trains.map((t) => Math.max(t, 0.01)), slot: 1 },
      { name: 'predict zero', x: [1, D], y: [null0, null0], muted: true, dashed: true },
      { name: 'p = n', x: [state.n, state.n], y: [0.01, 100], emphasis: true, dashed: true },
    ] as const
  }, [state.n, s2, cum])

  const at = risk(state.p, state.n, s2, cum)

  const xAxis = useAxis({ label: 'number of features p', range: [1, D] })
  const yAxis = useAxis({ label: 'mean squared error', range: [0.01, 100], log: true })
  return (
    <Figure
      title="Test error across the interpolation threshold"
      state={state}
      caption="Least squares on the first p of 200 independent Gaussian features, from n training points; the true model uses all 200 features. Below p = n the fit is ordinary least squares; above it, the fit is the minimum-norm interpolant and the training error is zero. The curves are exact expectations. The test error peaks where the model first interpolates, then falls again. With the signal spread over all features, the largest model beats every small one; with a decaying signal it does not. Drag the p line or use the sliders."

      readouts={
        <>
          <Readout label="p / n" value={formatNumber(state.p / state.n)} />
          <Readout label="expected test error" value={Number.isFinite(at.test) ? formatNumber(at.test) : '∞'} />
          <Readout label="expected training error" value={formatNumber(at.train)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
        <Handle {...state.handle('p', { label: 'p' })} />
      </Plot>
    </Figure>
  )
}
