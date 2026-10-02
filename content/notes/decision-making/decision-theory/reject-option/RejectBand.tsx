import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type Handle } from 'aifn-render'
import { normalCdf } from '@/lib/math/special'
import { linspace } from '@/lib/math'

const X = linspace(-3, 5, 241)
const D_GRID = linspace(0.005, 0.5, 100)
const logit = (p: number) => Math.log(p / (1 - p))

/**
 * Chow's reject option on a binormal problem: x is N(0, 1) for negatives and N(sep, 1) for positives, equal priors, so
 * the posterior is p(x) = σ(sep·x − sep²/2). A case is rejected when max(p, 1 − p) < 1 − d. The left chart shows the
 * posterior with the reject band; the right chart is the error-reject curve traced by varying d.
 */
export function RejectBand() {
  const d = useParam(0.2, { min: 0.01, max: 0.5, step: 0.005 })
  const sep = useParam(2, { min: 0.5, max: 4, step: 0.1 })

  // Scores where the posterior equals d and 1 − d.
  const band = (dd: number, s: number) => {
    const lo = (logit(dd) + (s * s) / 2) / s
    const hi = (logit(1 - dd) + (s * s) / 2) / s
    return { lo, hi }
  }
  const rates = (dd: number, s: number) => {
    const { lo, hi } = band(dd, s)
    const error = 0.5 * (1 - normalCdf(hi)) + 0.5 * normalCdf(lo - s)
    const reject = 0.5 * (normalCdf(hi) - normalCdf(lo)) + 0.5 * (normalCdf(hi - s) - normalCdf(lo - s))
    return { error, reject, lo, hi }
  }

  const post = useMemo(
    () => X.map((x) => 1 / (1 + Math.exp(-(sep.value * x - (sep.value * sep.value) / 2)))),
    [sep.value],
  )
  const curve = useMemo(() => {
    const pts = D_GRID.map((dd) => rates(dd, sep.value))
    return { reject: pts.map((q) => q.reject), error: pts.map((q) => q.error) }
  }, [sep.value])
  const at = rates(d.value, sep.value)

  const handles: Handle[] = [{ kind: 'y', at: 1 - d.value, label: 'accept above 1 − d', onDrag: (y) => d.set(1 - y) }]

  return (
    <Interactive
      title="Rejecting the least certain cases"
      caption="Left: the posterior P(y = 1 | x) for two unit-variance Gaussian classes with equal priors. Cases whose larger class probability is below 1 − d fall in the shaded band and are rejected. Drag the horizontal line at 1 − d, or use the slider. Right: the error rate against the reject rate as d varies from 0.5 (no rejection) towards 0. The marked point is the current d; the curve's slope there equals −d."
      controls={
        <>
          <ParamSlider label="rejection cost d" param={d} />
          <ParamSlider label="class separation" param={sep} />
        </>
      }
      readout={
        <>
          <Readout label="reject band on x" value={`[${formatNumber(at.lo)}, ${formatNumber(at.hi)}]`} />
          <Readout label="error rate" value={formatNumber(at.error)} />
          <Readout label="reject rate" value={formatNumber(at.reject)} />
          <Readout label="expected loss" value={formatNumber(at.error + d.value * at.reject)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <XYChart
          height={280}
          xLabel="score x"
          yLabel="P(y = 1 | x)"
          xRange={[-3, 5]}
          yRange={[0, 1]}
          handles={handles}
          series={[
            { name: 'posterior', type: 'line', x: X, y: post, slot: 0 },
            {
              name: 'reject band',
              type: 'line',
              x: [at.lo, at.lo, at.hi, at.hi],
              y: [0, 1, 1, 0],
              slot: 2,
              area: true,
            },
            { name: 'd', type: 'line', x: [-3, 5], y: [d.value, d.value], dashed: true, muted: true },
          ]}
        />
        <XYChart
          height={280}
          xLabel="reject rate"
          yLabel="error rate"
          xRange={[0, 1]}
          yRange={[0, undefined]}
          series={[
            { name: 'error-reject curve', type: 'line', x: curve.reject, y: curve.error, slot: 1 },
            { name: 'current d', type: 'scatter', x: [at.reject], y: [at.error], emphasis: true },
          ]}
        />
      </div>
    </Interactive>
  )
}
