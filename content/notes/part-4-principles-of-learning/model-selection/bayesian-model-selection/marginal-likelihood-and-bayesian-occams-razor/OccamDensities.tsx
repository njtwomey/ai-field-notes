import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { fromData, linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { softmax } from 'aifn-compute/numerics/special'
import { Normal } from 'aifn-compute/probability/distributions'

/** The candidate models: the prior sd τ of the mean (0 fixes μ = 0). */
const MODELS = [
  { tau: 0, name: 'M₀: μ = 0', slot: 0 },
  { tau: 0.5, name: 'τ = 0.5', slot: 1 },
  { tau: 2, name: 'τ = 2', slot: 2 },
  { tau: 20, name: 'τ = 20', slot: 3 },
] as const
const RANGE: [number, number] = [-3, 3]
const GRID = linspace(RANGE[0], RANGE[1], 241)
const GRID_X = Array.from(toFlat(GRID))
const fmt = (v: number) => formatNumber(Number(v.toPrecision(3)))

/**
 * The evidence of four models of a normal mean as densities over the sufficient statistic ȳ, with the observed ȳ
 * draggable: each model's evidence is its density at the observed value.
 */
export function OccamDensities() {
  const state = useFigureState({
    n: int(16, { ge: 2, le: 200, suggestions: [4, 16, 64], label: 'observations n' }),
    ybar: float(0.5, { min: RANGE[0], max: RANGE[1], step: 0.05, label: 'observed mean ȳ' }),
  })
  const { n, ybar } = state
  const curves = useMemo(
    () =>
      MODELS.map((m) => {
        const d = Normal(0, Math.sqrt(m.tau * m.tau + 1 / n))
        return { ...m, d, y: Array.from(toFlat(d.prob(GRID) as never)) }
      }),
    [n],
  )
  const logEv = curves.map((c) => Number(c.d.logProb(ybar)))
  const post = Array.from(toFlat(softmax(fromData(Float64Array.from(logEv), [logEv.length]))))
  const best = logEv.indexOf(Math.max(...logEv))
  const xAxis = useAxis({ label: 'sample mean ȳ', range: RANGE, nice: false })
  const yAxis = useAxis({ label: 'density p(ȳ | M)', range: [0, undefined], hold: 'union', key: `${n}` })
  return (
    <Figure
      title="Evidence as a density over data sets"
      state={state}
      caption="Four models of the mean μ of n observations from N(μ, 1): M₀ fixes μ = 0, and the others give μ a N(0, τ²) prior. Each curve is the model's prior predictive density of the sample mean, N(0, τ² + 1/n). It integrates to one over all possible data, so a wide prior (τ = 20) spreads its probability thinly and a narrow one piles it near zero. The evidence of each model is the height of its curve at the observed ȳ. Drag the vertical line to move the observed mean. Near zero the narrow models win; far from zero only the wider models predicted it, and the τ = 20 model never wins against τ = 2 inside this range, because it wastes probability on means that are never observed."
      readouts={
        <>
          {curves.map((c, i) => (
            <Readout key={c.name} label={`P(${c.name} | data)`} value={fmt(post[i])} />
          ))}
          <Readout label="largest evidence" value={curves[best].name} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {curves.map((c) => (
          <Curve key={c.name} name={c.name} x={GRID_X} y={c.y} slot={c.slot} />
        ))}
        <Handle
          kind="x"
          at={ybar}
          label="observed ȳ"
          onDrag={(x) => state.set('ybar', Math.max(RANGE[0], Math.min(RANGE[1], Math.round(x * 20) / 20)))}
        />
      </Plot>
    </Figure>
  )
}
