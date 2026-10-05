import { useMemo } from 'react'
import { Figure, formatNumber, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { softmax } from 'aifn-compute/numerics/special'
import { dirichlet } from 'aifn-compute/probability/samplers'
import { correlation } from 'aifn-compute/probability/stats'

const K = 5
const SAMPLES = 400

/** Topic proportions for five topics under a symmetric Dirichlet and under a logistic normal. */
export function CorrelatedProportions() {
  const state = useFigureState({
    alpha: slider(0.1, 3, 0.5, { step: 0.05, label: 'Dirichlet α' }),
    rho: slider(-0.95, 0.95, 0.8, { step: 0.05, label: 'logistic-normal correlation ρ' }),
    scale: slider(0.2, 2.5, 1, { step: 0.1, label: 'logistic-normal scale s' }),
  })
  const { alpha, rho, scale } = state

  const dir = useMemo(() => {
    const r = stream(5)
    const concentration = new Array<number>(K).fill(alpha)
    return Array.from({ length: SAMPLES }, () => toFlat(dirichlet(r, concentration)))
  }, [alpha])
  const logn = useMemo(() => {
    const r = stream(6)
    // η ~ N(0, s²Σ) with unit variances and correlation ρ between topics 1 and 2 only, via a 2 × 2 Cholesky factor.
    return Array.from({ length: SAMPLES }, () => {
      const e = Array.from({ length: K }, () => normal(r))
      const eta = e.map((v) => scale * v)
      eta[1] = scale * (rho * e[0] + Math.sqrt(1 - rho * rho) * e[1])
      return toFlat(softmax(tensor(eta)))
    })
  }, [rho, scale])

  const dirPoints = useMemo(() => ({ x: dir.map((t) => t[0]), y: dir.map((t) => t[1]) }), [dir])
  const lognPoints = useMemo(() => ({ x: logn.map((t) => t[0]), y: logn.map((t) => t[1]) }), [logn])
  const corrDir = correlation(dirPoints.x, dirPoints.y)
  const corrLogn = correlation(lognPoints.x, lognPoints.y)

  const xAxis = useAxis({ label: 'θ₁ (Dirichlet)', range: [0, 1] })
  const yAxis = useAxis({ label: 'θ₂', range: [0, 1] })
  const xAxis2 = useAxis({ label: 'θ₁ (logistic normal)', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'θ₂', range: [0, 1] })
  return (
    <Figure
      title="Topic correlations: Dirichlet against logistic normal"
      state={state}
      caption="Each point is the topic proportions of one simulated document with five topics, plotted as the share of topic 1 against the share of topic 2. Left: a symmetric Dirichlet, as in LDA. Whatever α is, the correlation between two proportions stays at −1/(K − 1) = −0.25: the concentration changes how spread the points are, never how the topics co-occur. Right: the logistic normal of the correlated topic model, with correlation ρ between the log-weights of topics 1 and 2. Positive ρ makes documents that use topic 1 also use topic 2, which the Dirichlet cannot express."
      readouts={
        <>
          <Readout label="corr(θ₁, θ₂), Dirichlet" value={formatNumber(corrDir)} />
          <Readout label="theory −1/(K − 1)" value="−0.25" />
          <Readout label="corr(θ₁, θ₂), logistic normal" value={formatNumber(corrLogn)} />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Points name="Dirichlet" {...dirPoints} slot={0} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Points name="logistic normal" {...lognPoints} slot={1} />
        </Plot>
      </div>
    </Figure>
  )
}
