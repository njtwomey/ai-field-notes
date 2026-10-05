import { useMemo } from 'react'
import { Area, Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normalPdf } from 'aifn-compute/numerics/special'

const XS = toFlat(linspace(20, 120, 400))

/**
 * Two groups with Gaussian scores. The total variance of a random student's score splits into the average variance
 * within the groups plus the variance of the group means.
 */
export function TotalVariance() {
  const state = useFigureState({
    weightA: float(0.3, { min: 0.05, max: 0.95, step: 0.05, label: 'share in group A' }),
    meanA: int(60, { min: 40, max: 100, step: 1, label: 'group A mean' }),
    meanB: int(80, { min: 40, max: 100, step: 1, label: 'group B mean' }),
    sdA: float(5, { min: 1, max: 12, step: 0.5, label: 'group A standard deviation' }),
    sdB: float(4, { min: 1, max: 12, step: 0.5, label: 'group B standard deviation' }),
  })

  const result = useMemo(() => {
    const weightB = 1 - state.weightA
    const mean = state.weightA * state.meanA + weightB * state.meanB
    const within = state.weightA * state.sdA ** 2 + weightB * state.sdB ** 2
    const between = state.weightA * (state.meanA - mean) ** 2 + weightB * (state.meanB - mean) ** 2
    const a = XS.map((x) => (state.weightA * normalPdf((x - state.meanA) / state.sdA)) / state.sdA)
    const b = XS.map((x) => (weightB * normalPdf((x - state.meanB) / state.sdB)) / state.sdB)
    const series = [
      { name: 'group A (weighted)', x: XS, y: a, slot: 0 },
      { name: 'group B (weighted)', x: XS, y: b, slot: 1 },
      { name: 'all students', x: XS, y: a.map((v, i) => v + b[i]), dashed: true, slot: 2 },
    ] as const
    return { mean, within, between, series }
  }, [state.weightA, state.meanA, state.meanB, state.sdA, state.sdB])

  const xAxis = useAxis({ label: 'score x', hold: 'union' })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Within-group and between-group variance"
      state={state}
      caption="Each group's scores are Gaussian. The dashed curve is the distribution of a randomly chosen student's score, a mixture of the two groups. Its variance is the average within-group variance plus the variance of the group means. Move the means apart and the between-group term grows; shrink the spreads and only the between-group term is left."

      readouts={
        <>
          <Readout label="E[X]" value={formatNumber(result.mean)} />
          <Readout label="E[var(X | Y)] (within)" value={formatNumber(result.within)} />
          <Readout label="var(E[X | Y]) (between)" value={formatNumber(result.between)} />
          <Readout label="var(X) (total)" value={formatNumber(result.within + result.between)} />
          <Readout
            label="share between groups"
            value={`${Math.round((100 * result.between) / (result.within + result.between))}%`}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Area {...result.series[0]} />
        <Area {...result.series[1]} />
        <Curve {...result.series[2]} />
      </Plot>
    </Figure>
  )
}
