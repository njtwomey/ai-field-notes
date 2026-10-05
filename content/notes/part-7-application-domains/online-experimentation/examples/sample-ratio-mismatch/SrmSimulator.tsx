import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  type Segment,
  Segments,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform as drawUniform } from 'aifn-compute/foundation/random'
import { sampleBinomial, twoProportionZ } from '../_shared/ab'
import { normalCdf } from 'aifn-compute/numerics/special'

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
  const state = useFigureState({
    assigned: int(100000, { min: 10000, max: 500000, step: 10000, label: 'users assigned per arm' }),
    loss: float(0.2, { min: 0, max: LOSS_MAX, step: 0.005, label: 'loss of slow users in B' }),
    slowShare: float(0.3, { min: 0.05, max: 0.6, step: 0.05, label: 'share of slow users' }),
    slowRate: float(0.04, { min: 0.01, max: 0.2, step: 0.01, label: "slow users' conversion" }),
    fastRate: float(0.12, { min: 0.01, max: 0.3, step: 0.01, label: "fast users' conversion" }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  // Expected values as a function of the loss rate: the bias in B − A and the SRM p-value.
  const curves = useMemo(() => {
    const rateA = state.slowShare * state.slowRate + (1 - state.slowShare) * state.fastRate
    const at = (l: number) => {
      const kept = state.slowShare * (1 - l)
      const nb = state.assigned * (kept + 1 - state.slowShare)
      const rateB = (kept * state.slowRate + (1 - state.slowShare) * state.fastRate) / (kept + 1 - state.slowShare)
      return { bias: rateB - rateA, srmP: Math.max(srmTest(state.assigned, nb).p, P_FLOOR) }
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
    } satisfies Record<string, SeriesSpec>
  }, [state.assigned, state.slowShare, state.slowRate, state.fastRate])

  const r = useMemo(() => {
    const draws = stream(state.seed)
    const uniform = () => drawUniform(draws)
    const slowA = sampleBinomial(state.assigned, state.slowShare, uniform)
    const slowB = sampleBinomial(state.assigned, state.slowShare, uniform)
    const keptSlowB = sampleBinomial(slowB, 1 - state.loss, uniform)
    const na = state.assigned
    const nb = keptSlowB + (state.assigned - slowB)
    const xa =
      sampleBinomial(slowA, state.slowRate, uniform) + sampleBinomial(state.assigned - slowA, state.fastRate, uniform)
    const xb =
      sampleBinomial(keptSlowB, state.slowRate, uniform) +
      sampleBinomial(state.assigned - slowB, state.fastRate, uniform)
    return { na, nb, srm: srmTest(na, nb), ab: twoProportionZ(xa, na, xb, nb, 0.05) }
  }, [state.assigned, state.slowShare, state.slowRate, state.fastRate, state.loss, state.seed])

  const biasSeries: SeriesSpec[] = [
    curves.bias,
    { name: 'this experiment', type: 'scatter', x: [100 * state.loss], y: [100 * r.ab.diff], emphasis: true },
  ]
  const ci: Segment[] = [
    { from: [100 * state.loss, 100 * r.ab.ci[0]], to: [100 * state.loss, 100 * r.ab.ci[1]] },
    { from: [0, 0], to: [100 * LOSS_MAX, 0] },
  ]
  const srmSeries: SeriesSpec[] = [
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
      x: [100 * state.loss],
      y: [Math.max(r.srm.p, P_FLOOR)],
      emphasis: true,
    },
  ]

  const xAxis = useAxis({ label: 'slow users lost from B (%)', range: [0, 100 * LOSS_MAX] })
  const yAxis = useAxis({ label: 'reported B − A (pp)', hold: 'union' })
  const xAxis2 = useAxis({ label: 'slow users lost from B (%)', range: [0, 100 * LOSS_MAX] })
  const yAxis2 = useAxis({ label: 'SRM p-value', hold: 'union', log: true })
  return (
    <Figure
      title="Lost users, false wins"
      state={state}
      caption="The new page in arm B is exactly as good as the old one, but it loads slowly, and a share of slow-connection users leave before they are logged. Left: the lift that the A/B test then reports, against the loss rate, with this experiment's 95% interval; the true lift is zero. Right: the sample-ratio test's p-value on a log scale. Drag the line labelled loss. The ratio test flags the problem at losses too small to bias the result visibly."

      readouts={
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
        <Plot x={xAxis} y={yAxis} height={300}>
          {seriesLayers(biasSeries)}
          <Segments segments={ci} />
          <Handle kind="x" at={100 * state.loss} label="loss" onDrag={(x) => state.set('loss', x / 100)} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          {seriesLayers(srmSeries)}
          <Handle kind="x" at={100 * state.loss} label="loss" onDrag={(x) => state.set('loss', x / 100)} />
        </Plot>
      </div>
    </Figure>
  )
}
