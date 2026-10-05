import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normalCdf } from 'aifn-compute/numerics/special'

/**
 * Explore-then-commit on two unit-variance Gaussian arms with gap Δ: the exact expected regret
 * mΔ + (T − 2m)Δ Φ(−Δ√(m/2)) and the Hoeffding-style bound mΔ + (T − 2m)Δ exp(−mΔ²/4), as functions of the number
 * m of exploration pulls per arm.
 */
export function EtcTradeoff() {
  const state = useFigureState({
    gap: slider(0.02, 1, 0.2, { step: 0.01, label: 'gap Δ' }),
    T: int(10000, { ge: 100, le: 100000, scale: 'log10', suggestions: [100, 1000, 10000, 100000], label: 'horizon T' }),
    // At most T/2; a larger value is read as T/2.
    m: int(50, { ge: 1, le: 50000, suggestions: [10, 50, 200, 1000], label: 'exploration pulls per arm m' }),
  })
  const gap = state.gap
  const T = state.T
  const mMax = Math.floor(T / 2)
  const mNow = Math.min(state.m, mMax)

  const curves = useMemo(() => {
    const d = gap
    const ms: number[] = []
    const step = Math.max(1, Math.floor(mMax / 400))
    for (let v = 1; v <= mMax; v += step) ms.push(v)
    const exact = (v: number) => v * d + (T - 2 * v) * d * normalCdf(-d * Math.sqrt(v / 2))
    const bound = (v: number) => v * d + (T - 2 * v) * d * Math.exp((-v * d * d) / 4)
    let best = 1
    for (let v = 1; v <= mMax; v++) if (exact(v) < exact(best)) best = v
    const formula = Math.min(mMax, Math.max(1, Math.ceil((4 / (d * d)) * Math.log((T * d * d) / 4))))
    return { ms, exact, bound, best, formula }
  }, [gap, T, mMax])

  const series = [
    { name: 'exact expected regret', x: curves.ms, y: curves.ms.map(curves.exact), slot: 0 },
    { name: 'upper bound', x: curves.ms, y: curves.ms.map(curves.bound), slot: 1, dashed: true },
    {
      name: 'exploration cost mΔ',
      x: [0, mMax],
      y: [0, mMax * gap],
      muted: true,
    },
    { name: 'chosen m', x: [mNow], y: [curves.exact(mNow)], emphasis: true },
  ] as const
  const wrong = normalCdf(-gap * Math.sqrt(mNow / 2))

  const xAxis = useAxis({ label: 'exploration pulls per arm m', range: [0, mMax] })
  const yAxis = useAxis({ label: 'expected regret', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="How long to explore"
      caption="Two arms with unit-variance Gaussian rewards and gap Δ. Explore-then-commit pulls each arm m times, then commits for the remaining T − 2m rounds to the arm with the higher sample mean. Drag along the chart, or type a value, to set m. Too little exploration commits to the wrong arm too often; too much wastes mΔ on the worse arm. The dashed curve is the bound used in the analysis, whose minimiser is the formula m."
      state={state}
      readouts={
        <>
          <Readout label="P(commit to the wrong arm)" value={formatNumber(wrong)} />
          <Readout label="regret at m" value={formatNumber(curves.exact(mNow))} />
          <Readout label="best m" value={`${curves.best} (regret ${formatNumber(curves.exact(curves.best))})`} />
          <Readout
            label="formula m = ⌈4 ln(TΔ²/4)/Δ²⌉"
            value={`${curves.formula} (regret ${formatNumber(curves.exact(curves.formula))})`}
          />
          <Readout label="never exploring, TΔ/2" value={formatNumber((T * gap) / 2)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Points {...series[3]} />
        <Handle kind="x" at={mNow} onDrag={(x) => state.set('m', Math.max(1, Math.min(mMax, Math.round(x))))} />
      </Plot>
    </Figure>
  )
}
