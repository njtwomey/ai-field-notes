import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'

const LOG_2PI = Math.log(2 * Math.PI)
const normal = (z: number, m: number, v: number) => Math.exp(-((z - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v)
const ZS = linspace(-4, 4, 321)

/**
 * The model z ~ N(0, 1), x | z ~ N(z, 1) with one observation x, and a Gaussian q(z) = N(m, s²). The exact posterior is
 * N(x/2, 1/2) and the evidence is N(x | 0, 2), so every term of log p(x) = ELBO(q) + KL(q ‖ p(z | x)) is in closed form.
 */
export function ElboGap() {
  const x = useParam(1.5, { min: -3, max: 3, step: 0.1 })
  const m = useParam(-1, { min: -3, max: 3, step: 0.05 })
  const s = useParam(1.2, { min: 0.2, max: 2, step: 0.05 })

  const r = useMemo(() => {
    const v = s.value ** 2
    const expectedLogLik = -0.5 * LOG_2PI - 0.5 * ((x.value - m.value) ** 2 + v)
    const klPrior = 0.5 * (v + m.value ** 2 - 1 - Math.log(v))
    const elbo = expectedLogLik - klPrior
    const logEvidence = -0.5 * Math.log(4 * Math.PI) - x.value ** 2 / 4
    const [pm, pv] = [x.value / 2, 0.5]
    const klPosterior = 0.5 * Math.log(pv / v) + (v + (m.value - pm) ** 2) / (2 * pv) - 0.5
    return { expectedLogLik, klPrior, elbo, logEvidence, klPosterior, pm, pv }
  }, [x.value, m.value, s.value])

  const series: XYSeries[] = useMemo(
    () => [
      { name: 'prior p(z)', type: 'line', x: ZS, y: ZS.map((z) => normal(z, 0, 1)), slot: 2, dashed: true },
      { name: 'posterior p(z | x)', type: 'line', x: ZS, y: ZS.map((z) => normal(z, r.pm, r.pv)), slot: 0 },
      { name: 'q(z)', type: 'line', x: ZS, y: ZS.map((z) => normal(z, m.value, s.value ** 2)), slot: 1, area: true },
    ],
    [r.pm, r.pv, m.value, s.value],
  )
  const handles: Handle[] = [{ kind: 'x', at: m.value, label: 'mean of q', onDrag: m.set }]

  return (
    <Interactive
      title="The gap between the ELBO and the evidence"
      caption="Prior z ~ N(0, 1), one observation x ~ N(z, 1), and a Gaussian approximation q. The evidence log p(x) does not depend on q. The ELBO is always below it, and the gap is exactly KL(q ‖ posterior). Move q with the slider or by dragging its mean, and widen or narrow it: the ELBO rises exactly as the KL falls, and the two meet when q equals the posterior N(x/2, 1/2)."
      controls={
        <>
          <ParamSlider label="observation x" param={x} />
          <ParamSlider label="mean of q" param={m} />
          <ParamSlider label="standard deviation of q" param={s} />
        </>
      }
      readout={
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
      <XYChart series={series} xLabel="z" yLabel="density" handles={handles} height={300} />
    </Interactive>
  )
}
