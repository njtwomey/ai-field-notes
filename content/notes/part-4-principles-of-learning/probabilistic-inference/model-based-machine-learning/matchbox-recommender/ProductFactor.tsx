import { useMemo } from 'react'
import { Area, Curve, Figure, float, formatNumber, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { gaussPdf, grid } from '../_shared/gaussian'
import { exactS, vmp } from '../_shared/product'

const S = grid(-4, 4, 401)

/** The product factor z = s·t: the exact posterior of s, its moment-matched Gaussian and the variational solution. */
export function ProductFactor() {
  const state = useFigureState({
    mt: float(0.3, { min: -1.5, max: 1.5, step: 0.05, label: 'prior mean of t, mₜ', format: (v) => v.toFixed(2) }),
    z: float(1.5, { min: -3, max: 3, step: 0.1, label: 'observed product z', format: (v) => v.toFixed(1) }),
    noise: float(0.1, { min: 0.02, max: 2, step: 0.02, label: 'noise variance', format: (v) => v.toFixed(2) }),
  })

  const { series, exact, fit } = useMemo(() => {
    const p = { ms: 0, vs: 1, mt: state.mt, vt: 1, zObs: state.z, noiseVar: state.noise }
    const exact = exactS(p, S)
    const fit = vmp(p)
    const series = [
      { name: 'exact posterior of s', x: S, y: exact.density, slot: 0 },
      {
        name: 'moment-matched Gaussian (EP-style)',
        x: S,
        y: S.map((s) => gaussPdf(s, exact.mean, exact.variance)),
        slot: 1,
        dashed: true,
      },
      {
        name: 'variational q(s) (VMP)',
        x: S,
        y: S.map((s) => gaussPdf(s, fit.s.mean, fit.s.variance)),
        slot: 2,
      },
      { name: 'prior of s', x: S, y: S.map((s) => gaussPdf(s, 0, 1)), muted: true },
    ] as const
    return { series, exact, fit }
  }, [state.mt, state.z, state.noise])

  const xAxis = useAxis({ label: 'user trait s', range: [-4, 4] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="One user trait times one item trait"
      state={state}
      caption="A user trait s and an item trait t both start as N(0, 1) and N(m_t, 1); their product z = s·t is observed with Gaussian noise. The exact posterior of s is bimodal when the sign of t is uncertain: a positive z fits s and t both positive or both negative. The moment-matched Gaussian covers both modes and becomes wider than the prior. The variational solution picks one mode and is narrow. With m_t = 0 nothing prefers either sign, the variational solution is stuck at s = 0, and only a small asymmetry in the prior of t lets it choose."

      readouts={
        <>
          <Readout
            label="exact mean, sd"
            value={`${formatNumber(exact.mean)}, ${formatNumber(Math.sqrt(exact.variance))}`}
          />
          <Readout
            label="VMP mean, sd"
            value={`${formatNumber(fit.s.mean)}, ${formatNumber(Math.sqrt(fit.s.variance))}`}
          />
          <Readout label="VMP mean of t" value={formatNumber(fit.t.mean)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Area {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
      </Plot>
    </Figure>
  )
}
