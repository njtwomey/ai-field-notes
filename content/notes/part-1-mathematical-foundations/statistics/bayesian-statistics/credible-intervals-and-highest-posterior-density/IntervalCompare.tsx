import { useMemo } from 'react'
import { Area, Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { betaHpd, betaPdf, betaQuantile } from '../_shared/beta'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const GRID = toFlat(linspace(0.0005, 0.9995, 500))

/**
 * Posterior Beta(1 + k, 1 + n − k) from a uniform prior. The equal-tailed interval cuts α/2 from each tail; the HPD
 * interval is the shortest interval with the same mass, and its endpoints have equal density.
 */
export function IntervalCompare() {
  const state = useFigureState({
    n: int(12, { min: 1, max: 60, step: 1, label: 'trials n' }),
    level: float(0.95, {
      min: 0.5,
      max: 0.99,
      step: 0.01,
      label: 'credible level',
      format: (v) => `${Math.round(100 * v)}%`,
    }),
    // The bound is the largest n; a k above the current n is read as k = n.
    k: int(1, { min: 0, max: 60, step: 1, label: 'successes k (at most n)' }),
  })
  const k = state.k

  const r = useMemo(() => {
    const kk = Math.min(k, state.n)
    const a = 1 + kk
    const b = 1 + state.n - kk
    const et: [number, number] = [betaQuantile((1 - state.level) / 2, a, b), betaQuantile((1 + state.level) / 2, a, b)]
    const hpd = betaHpd(a, b, state.level)
    const dens = GRID.map((x) => betaPdf(x, a, b))
    const top = Math.max(...dens)
    const cut = Math.min(betaPdf(Math.max(hpd[0], 1e-6), a, b), betaPdf(Math.min(hpd[1], 1 - 1e-6), a, b))
    const series = [
      { name: 'posterior density', x: GRID, y: dens, emphasis: true },
      {
        name: `${Math.round(100 * state.level)}% HPD interval`,
        x: GRID,
        y: GRID.map((x, i) => (x >= hpd[0] && x <= hpd[1] ? dens[i] : NaN)),
        slot: 0,
      },
      {
        name: 'equal-tailed interval',
        x: [et[0], et[0], NaN, et[1], et[1]],
        y: [0, top * 1.05, NaN, 0, top * 1.05],
        slot: 1,
        dashed: true,
      },
      { name: 'HPD density level', x: [0, 1], y: [cut, cut], muted: true },
    ] as const
    return { et, hpd, series, top, kk }
  }, [state.n, k, state.level])

  const xAxis = useAxis({ label: 'success probability θ', range: [0, 1] })
  const yAxis = useAxis({ label: 'posterior density', range: [0, r.top * 1.08] })
  return (
    <Figure
      title="Equal-tailed and highest-density intervals"
      state={state}
      caption="Posterior for a success probability after k successes in n trials with a uniform prior. The shaded region is the HPD interval: every point inside has higher density than every point outside, and its two ends sit on the same density level (grey line). The dashed lines mark the equal-tailed interval, which cuts equal probability from each tail. The two agree for a symmetric posterior (k = n/2) and differ most when k is near 0 or n."
      readouts={
        <>
          <Readout label="successes" value={`${r.kk} of ${state.n}`} />
          <Readout
            label="equal-tailed"
            value={`[${formatNumber(r.et[0])}, ${formatNumber(r.et[1])}], width ${formatNumber(r.et[1] - r.et[0])}`}
          />
          <Readout
            label="HPD"
            value={`[${formatNumber(r.hpd[0])}, ${formatNumber(r.hpd[1])}], width ${formatNumber(r.hpd[1] - r.hpd[0])}`}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...r.series[0]} />
        <Area {...r.series[1]} />
        <Curve {...r.series[2]} />
        <Curve {...r.series[3]} />
      </Plot>
    </Figure>
  )
}
