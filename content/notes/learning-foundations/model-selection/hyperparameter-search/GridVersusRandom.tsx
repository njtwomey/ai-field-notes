import { useMemo } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type HeatmapOverlay,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'

const AXIS = linspace(0, 1, 81)
/** Validation score: a narrow peak in the important hyperparameter u, a gentle slope in the unimportant v. */
const important = (u: number) => Math.exp(-((u - 0.63) ** 2) / (2 * 0.04 ** 2))
const score = (u: number, v: number) => 0.6 + 0.3 * important(u) + 0.03 * v

/** Grid search against random search with the same budget, when only one of two hyperparameters matters. */
export function GridVersusRandom() {
  const side = useParam(3, { min: 2, max: 8, step: 1 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })
  const budget = side.value ** 2

  const z = useMemo(() => AXIS.map((v) => AXIS.map((u) => score(u, v))), [])
  const designs = useMemo(() => {
    const ticks = Array.from({ length: side.value }, (_, i) => (i + 0.5) / side.value)
    const grid = ticks.flatMap((u) => ticks.map((v) => [u, v] as const))
    const g = rng(seed.value)
    const random = Array.from({ length: budget }, () => [g.uniform(), g.uniform()] as const)
    return { grid, random }
  }, [side.value, seed.value, budget])

  const overlay = useMemo(
    (): HeatmapOverlay[] => [
      { name: 'grid', type: 'scatter', x: designs.grid.map((p) => p[0]), y: designs.grid.map((p) => p[1]), slot: 1 },
      {
        name: 'random',
        type: 'scatter',
        x: designs.random.map((p) => p[0]),
        y: designs.random.map((p) => p[1]),
        slot: 2,
      },
    ],
    [designs],
  )
  const projection = useMemo(
    (): XYSeries[] => [
      { name: 'score at v = 0.5', type: 'line', x: AXIS, y: AXIS.map((u) => score(u, 0.5)), muted: true },
      {
        name: 'grid',
        type: 'scatter',
        x: designs.grid.map((p) => p[0]),
        y: designs.grid.map((p) => score(p[0], p[1])),
        slot: 1,
      },
      {
        name: 'random',
        type: 'scatter',
        x: designs.random.map((p) => p[0]),
        y: designs.random.map((p) => score(p[0], p[1])),
        slot: 2,
      },
    ],
    [designs],
  )

  const best = (pts: readonly (readonly [number, number])[]) => Math.max(...pts.map(([u, v]) => score(u, v)))
  return (
    <Interactive
      title="Grid search and random search with the same budget"
      caption="The validation score depends strongly on hyperparameter u (a narrow peak at u = 0.63) and hardly at all on v. Left: the score over both, with a grid design and a random design of the same size. Right: every trial projected onto u. The grid tests only as many distinct values of u as it has points per side; random search tests a new value of u with every trial, so it is far more likely to land near the peak."
      controls={
        <>
          <ParamSlider
            label="grid points per side (budget = side²)"
            param={side}
            format={(v) => `${v} (${v * v} trials)`}
            withArrows
          />
          <ParamSlider label="random seed" param={seed} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="distinct u, grid" value={String(side.value)} />
          <Readout label="distinct u, random" value={String(budget)} />
          <Readout label="best score, grid" value={formatNumber(best(designs.grid))} />
          <Readout label="best score, random" value={formatNumber(best(designs.random))} />
          <Readout label="optimum" value={formatNumber(score(0.63, 1))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Heatmap
          x={AXIS}
          y={AXIS}
          z={z}
          xLabel="important hyperparameter u"
          yLabel="unimportant hyperparameter v"
          valueLabel="validation score"
          overlay={overlay}
          height={340}
        />
        <XYChart
          series={projection}
          xLabel="important hyperparameter u"
          yLabel="validation score"
          xRange={[0, 1]}
          yRange={[0.55, 0.95]}
          height={340}
        />
      </div>
    </Interactive>
  )
}
