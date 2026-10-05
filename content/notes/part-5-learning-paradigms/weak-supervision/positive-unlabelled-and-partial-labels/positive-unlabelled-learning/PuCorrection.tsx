import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normalPdf } from 'aifn-compute/numerics/special'

const X_RANGE: [number, number] = [-5, 5]
const GRID = toFlat(linspace(-5, 5, 401))

/** First x on the grid where a curve crosses 1/2 from below, or null if it never does. */
function crossing(ys: number[]): number | null {
  for (let i = 1; i < ys.length; i++) if (ys[i - 1] < 0.5 && ys[i] >= 0.5) return GRID[i]
  return null
}

/**
 * Elkan and Noto's correction in one dimension. Positives are N(Δ/2, 1), negatives N(−Δ/2, 1). A fraction c of the
 * positives is labelled at random (SCAR). A classifier of labelled against unlabelled learns g(x) = c·f(x); dividing by
 * c recovers f. The estimate e₁ = E[g(x) | s = 1] equals c only when the classes do not overlap.
 */
export function PuCorrection() {
  const state = useFigureState({
    prior: float(0.4, { min: 0.1, max: 0.9, step: 0.05, label: 'class prior π' }),
    c: float(0.3, { min: 0.05, max: 1, step: 0.05, label: 'label frequency c' }),
    sep: float(3, { min: 0.5, max: 6, step: 0.25, label: 'class separation Δ' }),
  })
  const pi = state.prior
  const cv = state.c
  const d = state.sep

  const r = useMemo(() => {
    const pPos = GRID.map((x) => normalPdf(x - d / 2))
    const pNeg = GRID.map((x) => normalPdf(x + d / 2))
    const f = GRID.map((_, i) => (pi * pPos[i]) / (pi * pPos[i] + (1 - pi) * pNeg[i]))
    const g = f.map((v) => cv * v)
    // e₁ = E[g(x) | s = 1]; under SCAR the labelled points are distributed as the positives.
    const dx = GRID[1] - GRID[0]
    const e1 = GRID.reduce((s, _, i) => s + g[i] * pPos[i] * dx, 0)
    const corrected = g.map((v) => Math.min(1, v / e1))
    return { f, g, corrected, e1 }
  }, [pi, cv, d])

  const series = [
    { name: 'P(y = 1 | x), true', x: GRID, y: r.f, slot: 0 },
    { name: 'g(x) = P(s = 1 | x), labelled vs unlabelled', x: GRID, y: r.g, slot: 1 },
    { name: 'g(x) / e₁, Elkan–Noto estimate', x: GRID, y: r.corrected, slot: 2, dashed: true },
    { name: 'threshold 1/2', x: [...X_RANGE], y: [0.5, 0.5], muted: true, dashed: true },
  ] as const
  const bayes = crossing(r.f)
  const naive = crossing(r.g)
  const fixed = crossing(r.corrected)
  const fmt = (v: number | null) => (v === null ? 'none' : formatNumber(v))

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'probability', range: [0, 1.02] })
  return (
    <Figure
      title="Correcting a labelled-versus-unlabelled classifier"
      state={state}
      caption="Positives are N(Δ/2, 1) and negatives N(−Δ/2, 1), with class prior π. Only a fraction c of the positives carries a label, chosen at random. A classifier trained to separate labelled from unlabelled points learns g(x) = c·P(y = 1 | x): the same shape as the true posterior, scaled down by c. Thresholding g at one half misses most positives, or all of them when c < 1/2. Dividing by c restores the posterior. Elkan and Noto estimate c by e₁, the average of g over labelled points; e₁ equals c only when the classes do not overlap, so with overlap it is too small and the corrected curve overshoots."

      readouts={
        <>
          <Readout label="e₁ (estimate of c)" value={formatNumber(r.e1)} />
          <Readout label="true boundary" value={fmt(bayes)} />
          <Readout label="naive boundary, g = 1/2" value={fmt(naive)} />
          <Readout label="corrected boundary" value={fmt(fixed)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
      </Plot>
    </Figure>
  )
}
