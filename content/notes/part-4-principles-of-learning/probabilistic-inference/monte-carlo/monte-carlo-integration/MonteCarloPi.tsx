import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { stream, uniform } from 'aifn/foundation/random'

const MAX_SHOWN = 1500
const SIGMA = 4 * Math.sqrt((Math.PI / 4) * (1 - Math.PI / 4))
const ARC = toFlat(linspace(0, Math.PI / 2, 90))
const CHECKPOINTS = 160

/**
 * Estimate π as 4 × the fraction of uniform points in the unit square that fall inside the quarter disc. The right
 * panel shows the running estimate against log10(n), inside the band π ± 1.96 σ/√n that the central limit theorem
 * predicts.
 */
export function MonteCarloPi() {
  const state = useFigureState({
    logN: int(3, { min: 1, max: 5, step: 0.25, label: 'log₁₀ of sample size n' }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'random seed' }),
  })

  const r = useMemo(() => {
    const n = Math.round(10 ** state.logN)
    const g = stream(state.seed)
    const px: number[] = []
    const py: number[] = []
    const inside: number[] = []
    const tx: number[] = []
    const ty: number[] = []
    const marks = new Set(toFlat(linspace(0, state.logN, CHECKPOINTS)).map((l) => Math.round(10 ** l)))
    let hits = 0
    for (let i = 1; i <= n; i++) {
      const x = uniform(g)
      const y = uniform(g)
      const hit = x * x + y * y <= 1
      if (hit) hits++
      if (i <= MAX_SHOWN) {
        px.push(x)
        py.push(y)
        inside.push(hit ? 0 : 1)
      }
      if (marks.has(i)) {
        tx.push(Math.log10(i))
        ty.push((4 * hits) / i)
      }
    }
    const estimate = (4 * hits) / n
    const p = hits / n
    const se = (4 * Math.sqrt(p * (1 - p))) / Math.sqrt(n)
    return { n, px, py, inside, tx, ty, estimate, se }
  }, [state.logN, state.seed])

  const square = useMemo(
    () =>
      [
        {
          name: 'point',
          x: r.px,
          y: r.py,
          group: r.inside,
          groupNames: ['inside the quarter disc', 'outside'],
        },
        { name: 'x² + y² = 1', x: ARC.map(Math.cos), y: ARC.map(Math.sin), emphasis: true },
      ] as const,
    [r],
  )

  const running = useMemo(() => {
    const band = toFlat(linspace(0, state.logN, 120))
    const half = band.map((l) => (1.96 * SIGMA) / Math.sqrt(10 ** l))
    return [
      { name: 'running estimate', x: r.tx, y: r.ty, slot: 0 },
      { name: 'π', x: [0, state.logN], y: [Math.PI, Math.PI], emphasis: true, dashed: true },
      { name: 'π ± 1.96 σ/√n', x: band, y: band.map((_, i) => Math.PI + half[i]), muted: true },
      { name: 'π − 1.96 σ/√n', x: band, y: band.map((_, i) => Math.PI - half[i]), muted: true },
    ] as const
  }, [r, state.logN])

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'y', range: [0, 1], equal: xAxis })
  const xAxis2 = useAxis({ label: 'log₁₀ n', hold: 'union' })
  const yAxis2 = useAxis({ label: 'estimate of π', range: [2.4, 3.9] })
  return (
    <Figure
      title="Estimating π by Monte Carlo"
      state={state}
      caption={`Points are uniform on the unit square; 4 × the fraction inside the quarter disc estimates π. The left panel shows at most ${MAX_SHOWN} of the points. The right panel follows the estimate as samples accumulate, against log₁₀ n. The grey band is π ± 1.96 σ/√n with σ = 1.642: each tenfold increase in n narrows it by √10 ≈ 3.2, whatever the seed.`}

      readouts={
        <>
          <Readout label="n" value={r.n.toLocaleString('en-GB')} />
          <Readout label="estimate" value={formatNumber(r.estimate)} />
          <Readout label="error" value={formatNumber(r.estimate - Math.PI)} />
          <Readout label="standard error" value={formatNumber(r.se)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          <Points {...square[0]} />
          <Curve {...square[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve {...running[0]} />
          <Curve {...running[1]} />
          <Curve {...running[2]} />
          <Curve {...running[3]} />
        </Plot>
      </div>
    </Figure>
  )
}
