import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Points, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { stream, uniform } from 'aifn-compute/foundation/random'

const AXIS = toFlat(linspace(0, 1, 81))
/** Validation score: a narrow peak in the important hyperparameter u, a gentle slope in the unimportant v. */
const important = (u: number) => Math.exp(-((u - 0.63) ** 2) / (2 * 0.04 ** 2))
const score = (u: number, v: number) => 0.6 + 0.3 * important(u) + 0.03 * v

/** Grid search against random search with the same budget, when only one of two hyperparameters matters. */
export function GridVersusRandom() {
  const state = useFigureState({
    side: int(3, {
      min: 2,
      max: 8,
      step: 1,
      label: 'grid points per side (budget = side²)',
      format: (v) => `${v} (${v * v} trials)`,
    }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'random seed', format: (v) => String(v) }),
  })
  const budget = state.side ** 2

  const z = useMemo(() => AXIS.map((v) => AXIS.map((u) => score(u, v))), [])
  const designs = useMemo(() => {
    const ticks = Array.from({ length: state.side }, (_, i) => (i + 0.5) / state.side)
    const grid = ticks.flatMap((u) => ticks.map((v) => [u, v] as const))
    const g = stream(state.seed)
    const random = Array.from({ length: budget }, () => [uniform(g), uniform(g)] as const)
    return { grid, random }
  }, [state.side, state.seed, budget])

  const overlay = useMemo(
    () =>
      [
        { name: 'grid', x: designs.grid.map((p) => p[0]), y: designs.grid.map((p) => p[1]), slot: 1 },
        {
          name: 'random',
          x: designs.random.map((p) => p[0]),
          y: designs.random.map((p) => p[1]),
          slot: 2,
        },
      ] as const,
    [designs],
  )
  const projection = useMemo(
    () =>
      [
        { name: 'score at v = 0.5', x: AXIS, y: AXIS.map((u) => score(u, 0.5)), muted: true },
        {
          name: 'grid',
          x: designs.grid.map((p) => p[0]),
          y: designs.grid.map((p) => score(p[0], p[1])),
          slot: 1,
        },
        {
          name: 'random',
          x: designs.random.map((p) => p[0]),
          y: designs.random.map((p) => score(p[0], p[1])),
          slot: 2,
        },
      ] as const,
    [designs],
  )

  const best = (pts: readonly (readonly [number, number])[]) => Math.max(...pts.map(([u, v]) => score(u, v)))
  const xAxis = useAxis({ label: 'important hyperparameter u' })
  const yAxis = useAxis({ label: 'unimportant hyperparameter v' })
  const xAxis2 = useAxis({ label: 'important hyperparameter u', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'validation score', range: [0.55, 0.95] })
  return (
    <Figure
      title="Grid search and random search with the same budget"
      state={state}
      caption="The validation score depends strongly on hyperparameter u (a narrow peak at u = 0.63) and hardly at all on v. Left: the score over both, with a grid design and a random design of the same size. Right: every trial projected onto u. The grid tests only as many distinct values of u as it has points per side; random search tests a new value of u with every trial, so it is far more likely to land near the peak."

      readouts={
        <>
          <Readout label="distinct u, grid" value={String(state.side)} />
          <Readout label="distinct u, random" value={String(budget)} />
          <Readout label="best score, grid" value={formatNumber(best(designs.grid))} />
          <Readout label="best score, random" value={formatNumber(best(designs.random))} />
          <Readout label="optimum" value={formatNumber(score(0.63, 1))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Raster x={AXIS} y={AXIS} z={z} valueLabel={'validation score'} />
          <Points {...overlay[0]} live />
          <Points {...overlay[1]} live />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          <Curve {...projection[0]} />
          <Points {...projection[1]} />
          <Points {...projection[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
