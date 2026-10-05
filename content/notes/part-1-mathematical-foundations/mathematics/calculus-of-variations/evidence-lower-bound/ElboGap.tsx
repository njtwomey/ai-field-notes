import { useMemo } from 'react'
import { Area, Curve, Figure, float, formatNumber, Handle, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const LOG_2PI = Math.log(2 * Math.PI)
const normal = (z: number, m: number, v: number) => Math.exp(-((z - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v)
const ZS = toFlat(linspace(-4, 4, 321))

/**
 * The model z ~ N(0, 1), x | z ~ N(z, 1) with one observation x, and a Gaussian q(z) = N(m, s²). The exact posterior is
 * N(x/2, 1/2) and the evidence is N(x | 0, 2), so every term of log p(x) = ELBO(q) + KL(q ‖ p(z | x)) is in closed form.
 */
export function ElboGap() {
  const state = useFigureState({
    x: float(1.5, { min: -3, max: 3, step: 0.1, label: 'observation x' }),
    m: float(-1, { min: -3, max: 3, step: 0.05, label: 'mean of q' }),
    s: float(1.2, { min: 0.2, max: 2, step: 0.05, label: 'standard deviation of q' }),
  })

  const r = useMemo(() => {
    const v = state.s ** 2
    const expectedLogLik = -0.5 * LOG_2PI - 0.5 * ((state.x - state.m) ** 2 + v)
    const klPrior = 0.5 * (v + state.m ** 2 - 1 - Math.log(v))
    const elbo = expectedLogLik - klPrior
    const logEvidence = -0.5 * Math.log(4 * Math.PI) - state.x ** 2 / 4
    const [pm, pv] = [state.x / 2, 0.5]
    const klPosterior = 0.5 * Math.log(pv / v) + (v + (state.m - pm) ** 2) / (2 * pv) - 0.5
    return { expectedLogLik, klPrior, elbo, logEvidence, klPosterior, pm, pv }
  }, [state.x, state.m, state.s])

  const series = useMemo(
    () =>
      [
        { name: 'prior p(z)', x: ZS, y: ZS.map((z) => normal(z, 0, 1)), slot: 2, dashed: true },
        { name: 'posterior p(z | x)', x: ZS, y: ZS.map((z) => normal(z, r.pm, r.pv)), slot: 0 },
        { name: 'q(z)', x: ZS, y: ZS.map((z) => normal(z, state.m, state.s ** 2)), slot: 1 },
      ] as const,
    [r.pm, r.pv, state.m, state.s],
  )

  const xAxis = useAxis({ label: 'z', hold: 'union' })
  const yAxis = useAxis({ label: 'density', hold: 'union' })
  return (
    <Figure
      title="The gap between the ELBO and the evidence"
      state={state}
      caption="Prior z ~ N(0, 1), one observation x ~ N(z, 1), and a Gaussian approximation q. The evidence log p(x) does not depend on q. The ELBO is always below it, and the gap is exactly KL(q ‖ posterior). Move q with the slider or by dragging its mean, and widen or narrow it: the ELBO rises exactly as the KL falls, and the two meet when q equals the posterior N(x/2, 1/2)."

      readouts={
        <>
          <Readout label="log p(x)" value={formatNumber(r.logEvidence)} />
          <Readout label="ELBO" value={formatNumber(r.elbo)} />
          <Readout label="KL(q ‖ posterior)" value={formatNumber(r.klPosterior)} />
          <Readout label="ELBO + KL" value={formatNumber(r.elbo + r.klPosterior)} />
          <Readout label="E_q[log p(x | z)]" value={formatNumber(r.expectedLogLik)} />
          <Readout label="KL(q ‖ prior)" value={formatNumber(r.klPrior)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Area {...series[2]} />
        <Handle {...state.handle('m', { label: 'mean of q' })} />
      </Plot>
    </Figure>
  )
}
