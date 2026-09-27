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
import { grid, logNormalCdf, normalLogPdf, normalPdf, probitTilted, stepTilted, toNat, vFn, wFn } from '../_shared/ep'

const XS = grid(-6, 8, 351)
const TS = grid(-4, 4, 161)
const X_RANGE: [number, number] = [-6, 8]
const T_RANGE: [number, number] = [-4, 4]
const Y_DENSITY: [number | undefined, number | undefined] = [0, undefined]
const Y_FACTOR: [number | undefined, number | undefined] = [0, 1.6]
const Y_VW: [number | undefined, number | undefined] = [0, 2]
const V_CURVE = TS.map(vFn)
const W_CURVE = TS.map(wFn)

/**
 * A Gaussian cavity times a step (or probit) factor: the tilted distribution, its moment-matched Gaussian and the site
 * that EP stores, with the v and w functions that give the moments in closed form.
 */
export function TruncatedMoments() {
  const mu = useParam(-1, { min: -4, max: 5, step: 0.05 })
  const sd = useParam(1, { min: 0.3, max: 3, step: 0.05 })
  const noise = useParam(0, { min: 0, max: 2, step: 0.05 })
  const eps = useParam(0, { min: -3, max: 4, step: 0.05 })

  const r = useMemo(() => {
    const variance = sd.value ** 2
    const s2 = noise.value ** 2
    // A hard step is the probit factor with no noise; the two share one formula with σ² + s² in the denominator.
    const t = s2 > 0 ? probitTilted(mu.value, variance, 1, eps.value, s2) : stepTilted(mu.value, variance, eps.value)
    const z = (mu.value - eps.value) / Math.sqrt(variance + s2)
    const factor = XS.map((x) =>
      s2 > 0 ? Math.exp(logNormalCdf((x - eps.value) / Math.sqrt(s2))) : x > eps.value ? 1 : 0,
    )
    const cavity = XS.map((x) => normalPdf(x, mu.value, variance))
    const Z = Math.exp(t.logZ)
    const tilted = XS.map((_, i) => (cavity[i] * factor[i]) / Z)
    const q = XS.map((x) => normalPdf(x, t.mean, t.variance))
    // The site is Z q / cavity: it matches the factor where the cavity has mass and ignores it elsewhere.
    const site = XS.map((x) =>
      Math.min(5, Math.exp(t.logZ + normalLogPdf(x, t.mean, t.variance) - normalLogPdf(x, mu.value, variance))),
    )
    const nat = toNat(t)
    const siteTau = nat.tau - 1 / variance
    const siteNu = nat.nu - mu.value / variance
    return { t, z, factor, cavity, tilted, q, site, siteTau, siteNu, Z }
  }, [mu.value, sd.value, noise.value, eps.value])

  const densities: XYSeries[] = [
    { name: 'cavity', type: 'line', x: XS, y: r.cavity, muted: true },
    { name: 'tilted (exact)', type: 'line', x: XS, y: r.tilted, emphasis: true },
    { name: 'projection q', type: 'line', x: XS, y: r.q, slot: 0 },
  ]
  const factors: XYSeries[] = [
    { name: 'factor', type: 'line', x: XS, y: r.factor, emphasis: true },
    { name: 'site', type: 'line', x: XS, y: r.site, slot: 4, dashed: true },
    { name: 'cavity (scaled)', type: 'line', x: XS, y: r.cavity.map((c) => c * sd.value * 2.5), muted: true },
  ]
  const vw: XYSeries[] = [
    { name: 'v(t)', type: 'line', x: TS, y: V_CURVE, slot: 0 },
    { name: 'w(t)', type: 'line', x: TS, y: W_CURVE, slot: 1 },
    { name: 'current t', type: 'scatter', x: [r.z, r.z], y: [vFn(r.z), wFn(r.z)], emphasis: true },
  ]
  const handles: Handle[] = [{ kind: 'x', at: eps.value, onDrag: eps.set, label: 'threshold ε' }]

  return (
    <Interactive
      title="Moments of a Gaussian times a step"
      caption="The cavity N(μ, σ²) is multiplied by the factor 𝟙(x > ε), or by Φ((x − ε)/s) when the noise s is above zero. Left: the exact tilted density and the Gaussian q with its mean and variance. Right: the factor and the Gaussian site Z·q/cavity that EP stores in its place. Drag the threshold ε on either chart. Move μ below ε and the site narrows: it copies the factor only where the cavity has mass."
      controls={
        <>
          <ParamSlider label="cavity mean μ" param={mu} />
          <ParamSlider label="cavity sd σ" param={sd} />
          <ParamSlider label="threshold ε" param={eps} />
          <ParamSlider label="noise s (0 = hard step)" param={noise} />
        </>
      }
      readout={
        <>
          <Readout label="t" value={formatNumber(r.z)} />
          <Readout label="Z = Φ(t)" value={formatNumber(r.Z)} />
          <Readout label="v(t), w(t)" value={`${formatNumber(vFn(r.z))}, ${formatNumber(wFn(r.z))}`} />
          <Readout label="q mean, variance" value={`${formatNumber(r.t.mean)}, ${formatNumber(r.t.variance)}`} />
          <Readout
            label="site mean, variance"
            value={`${formatNumber(r.siteNu / r.siteTau)}, ${formatNumber(1 / r.siteTau)}`}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <XYChart
          series={densities}
          xLabel="x"
          yLabel="density"
          xRange={X_RANGE}
          yRange={Y_DENSITY}
          handles={handles}
          height={280}
        />
        <XYChart
          series={factors}
          xLabel="x"
          yLabel="factor value"
          xRange={X_RANGE}
          yRange={Y_FACTOR}
          handles={handles}
          height={280}
        />
      </div>
      <XYChart series={vw} xLabel="t = (μ − ε)/√(σ² + s²)" xRange={T_RANGE} yRange={Y_VW} height={220} />
    </Interactive>
  )
}
