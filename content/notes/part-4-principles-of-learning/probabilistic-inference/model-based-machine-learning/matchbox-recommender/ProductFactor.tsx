import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { gaussPdf, grid } from '../_shared/gaussian'
import { exactS, vmp } from '../_shared/product'

const S = grid(-4, 4, 401)

/** The product factor z = s·t: the exact posterior of s, its moment-matched Gaussian and the variational solution. */
export function ProductFactor() {
  const mt = useParam(0.3, { min: -1.5, max: 1.5, step: 0.05 })
  const z = useParam(1.5, { min: -3, max: 3, step: 0.1 })
  const noise = useParam(0.1, { min: 0.02, max: 2, step: 0.02 })

  const { series, exact, fit } = useMemo(() => {
    const p = { ms: 0, vs: 1, mt: mt.value, vt: 1, zObs: z.value, noiseVar: noise.value }
    const exact = exactS(p, S)
    const fit = vmp(p)
    const series: XYSeries[] = [
      { name: 'exact posterior of s', type: 'line', x: S, y: exact.density, slot: 0, area: true },
      {
        name: 'moment-matched Gaussian (EP-style)',
        type: 'line',
        x: S,
        y: S.map((s) => gaussPdf(s, exact.mean, exact.variance)),
        slot: 1,
        dashed: true,
      },
      {
        name: 'variational q(s) (VMP)',
        type: 'line',
        x: S,
        y: S.map((s) => gaussPdf(s, fit.s.mean, fit.s.variance)),
        slot: 2,
      },
      { name: 'prior of s', type: 'line', x: S, y: S.map((s) => gaussPdf(s, 0, 1)), muted: true },
    ]
    return { series, exact, fit }
  }, [mt.value, z.value, noise.value])

  return (
    <Interactive
      title="One user trait times one item trait"
      caption="A user trait s and an item trait t both start as N(0, 1) and N(m_t, 1); their product z = s·t is observed with Gaussian noise. The exact posterior of s is bimodal when the sign of t is uncertain: a positive z fits s and t both positive or both negative. The moment-matched Gaussian covers both modes and becomes wider than the prior. The variational solution picks one mode and is narrow. With m_t = 0 nothing prefers either sign, the variational solution is stuck at s = 0, and only a small asymmetry in the prior of t lets it choose."
      controls={
        <>
          <ParamSlider label="prior mean of t, mₜ" param={mt} format={(v) => v.toFixed(2)} />
          <ParamSlider label="observed product z" param={z} format={(v) => v.toFixed(1)} />
          <ParamSlider label="noise variance" param={noise} format={(v) => v.toFixed(2)} />
        </>
      }
      readout={
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
      <XYChart series={series} xLabel="user trait s" yLabel="density" xRange={[-4, 4]} yRange={[0, undefined]} />
    </Interactive>
  )
}
