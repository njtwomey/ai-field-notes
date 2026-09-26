import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { normalCdf } from '@/lib/math/special'
import { sampleBinomial, twoProportionZ } from '../_shared/ab'

const SRM_ALPHA = 0.001
/** 1 − Φ(z) loses all precision below about 1e-16, so smaller p-values are shown as this floor. */
const P_FLOOR = 1e-16
const LOSS_MAX = 0.3
const LOSSES = Array.from({ length: 61 }, (_, i) => (LOSS_MAX * i) / 60)

/** χ² goodness-of-fit test of observed arm sizes against an even split. With one degree of freedom χ² = z². */
function srmTest(na: number, nb: number): { chi2: number; p: number } {
  const chi2 = (na - nb) ** 2 / (na + nb)
  return { chi2, p: 2 * (1 - normalCdf(Math.sqrt(chi2))) }
}

/**
 * Both arms get the same page quality for everyone except slow-connection users in B, a share of whom leave before
 * being logged. The true effect is zero. Losing them shrinks arm B (a sample ratio mismatch) and removes low
 * converters from it, so B looks better than it is.
 */
export function SrmSimulator() {
  const [assigned, setAssigned] = useState(100000)
  const [slowShare, setSlowShare] = useState(0.3)
  const [slowRate, setSlowRate] = useState(0.04)
  const [fastRate, setFastRate] = useState(0.12)
  const loss = useParam(0.2, { min: 0, max: LOSS_MAX, step: 0.005 })
  const [seed, setSeed] = useState(1)

  // Expected values as a function of the loss rate: the bias in B − A and the SRM p-value.
  const curves = useMemo(() => {
    const rateA = slowShare * slowRate + (1 - slowShare) * fastRate
    const at = (l: number) => {
      const kept = slowShare * (1 - l)
      const nb = assigned * (kept + 1 - slowShare)
      const rateB = (kept * slowRate + (1 - slowShare) * fastRate) / (kept + 1 - slowShare)
      return { bias: rateB - rateA, srmP: Math.max(srmTest(assigned, nb).p, P_FLOOR) }
    }
    const values = LOSSES.map(at)
    return {
      bias: {
        name: 'expected bias',
        type: 'line',
        x: LOSSES.map((l) => 100 * l),
        y: values.map((v) => 100 * v.bias),
        slot: 1,
      },
      srm: {
        name: 'expected SRM p-value',
        type: 'line',
        x: LOSSES.map((l) => 100 * l),
        y: values.map((v) => v.srmP),
        slot: 0,
      },
    } satisfies Record<string, XYSeries>
  }, [assigned, slowShare, slowRate, fastRate])

  const r = useMemo(() => {
    const { uniform } = rng(seed)
    const slowA = sampleBinomial(assigned, slowShare, uniform)
    const slowB = sampleBinomial(assigned, slowShare, uniform)
    const keptSlowB = sampleBinomial(slowB, 1 - loss.value, uniform)
    const na = assigned
    const nb = keptSlowB + (assigned - slowB)
    const xa = sampleBinomial(slowA, slowRate, uniform) + sampleBinomial(assigned - slowA, fastRate, uniform)
    const xb = sampleBinomial(keptSlowB, slowRate, uniform) + sampleBinomial(assigned - slowB, fastRate, uniform)
    return { na, nb, srm: srmTest(na, nb), ab: twoProportionZ(xa, na, xb, nb, 0.05) }
  }, [assigned, slowShare, slowRate, fastRate, loss.value, seed])

  const handles: Handle[] = [{ kind: 'x', at: 100 * loss.value, label: 'loss', onDrag: (x) => loss.set(x / 100) }]
  const biasSeries: XYSeries[] = [
    curves.bias,
    { name: 'this experiment', type: 'scatter', x: [100 * loss.value], y: [100 * r.ab.diff], emphasis: true },
  ]
  const ci: Segment[] = [
    { from: [100 * loss.value, 100 * r.ab.ci[0]], to: [100 * loss.value, 100 * r.ab.ci[1]] },
    { from: [0, 0], to: [100 * LOSS_MAX, 0] },
  ]
  const srmSeries: XYSeries[] = [
    curves.srm,
    {
      name: `SRM threshold (p = ${SRM_ALPHA})`,
      type: 'line',
      x: [0, 100 * LOSS_MAX],
      y: [SRM_ALPHA, SRM_ALPHA],
      dashed: true,
      slot: 2,
    },
    {
      name: 'this experiment',
      type: 'scatter',
      x: [100 * loss.value],
      y: [Math.max(r.srm.p, P_FLOOR)],
      emphasis: true,
    },
  ]

  return (
    <Interactive
      title="Lost users, false wins"
      caption="The new page in arm B is exactly as good as the old one, but it loads slowly, and a share of slow-connection users leave before they are logged. Left: the lift that the A/B test then reports, against the loss rate, with this experiment's 95% interval; the true lift is zero. Right: the sample-ratio test's p-value on a log scale. Drag the line labelled loss. The ratio test flags the problem at losses too small to bias the result visibly."
      controls={
        <>
          <ParamSlider
            label="users assigned per arm"
            value={assigned}
            onChange={setAssigned}
            min={10000}
            max={500000}
            step={10000}
          />
          <ParamSlider label="loss of slow users in B" param={loss} />
          <ParamSlider
            label="share of slow users"
            value={slowShare}
            onChange={setSlowShare}
            min={0.05}
            max={0.6}
            step={0.05}
          />
          <ParamSlider
            label="slow users' conversion"
            value={slowRate}
            onChange={setSlowRate}
            min={0.01}
            max={0.2}
            step={0.01}
          />
          <ParamSlider
            label="fast users' conversion"
            value={fastRate}
            onChange={setFastRate}
            min={0.01}
            max={0.3}
            step={0.01}
          />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>Rerun</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="logged A, B" value={`${r.na.toLocaleString()}, ${r.nb.toLocaleString()}`} />
          <Readout label="B share" value={`${((100 * r.nb) / (r.na + r.nb)).toFixed(2)}%`} />
          <Readout
            label="SRM χ², p"
            value={`${formatNumber(r.srm.chi2)}, ${r.srm.p < P_FLOOR ? '< 1e-16' : formatNumber(r.srm.p)}`}
          />
          <Readout label="reported lift" value={`${(100 * r.ab.diff).toFixed(2)} pp`} />
          <Readout label="A/B p-value" value={formatNumber(r.ab.p)} />
          <Readout label="true lift" value="0 pp" />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={300}
          series={biasSeries}
          segments={ci}
          xRange={[0, 100 * LOSS_MAX]}
          xLabel="slow users lost from B (%)"
          yLabel="reported B − A (pp)"
          handles={handles}
        />
        <XYChart
          height={300}
          series={srmSeries}
          xRange={[0, 100 * LOSS_MAX]}
          yLog
          xLabel="slow users lost from B (%)"
          yLabel="SRM p-value"
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
