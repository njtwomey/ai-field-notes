import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  Rug,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { Normal } from 'aifn-compute/probability/distributions'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { histogram } from 'aifn-compute/probability/stats'
import { drawSamples, logP, logQ, logRatio, type Shift } from '../_shared/ratio'

const LO = -4
const HI = 5
const X = toFlat(linspace(LO, HI, 361))
const EDGES = toFlat(linspace(LO, HI, 37))
const MIDS = EDGES.slice(1).map((e, i) => (e + EDGES[i]) / 2)

/** The quantity whose average under p the weighted q sample estimates. */
const TARGETS = {
  x: { label: 'x', f: (x: number) => x },
  square: { label: 'x²', f: (x: number) => x * x },
  tail: { label: '1{x > 1.5}', f: (x: number) => (x > 1.5 ? 1 : 0) },
} as const
type Target = keyof typeof TARGETS

/** E_p[f] for p = N(m, s²), from the engine's normal distribution. */
function expectation(target: Target, s: Shift) {
  const p = Normal(s.mean, s.sd)
  if (target === 'x') return p.mean() as number
  if (target === 'square') return (p.variance() as number) + (p.mean() as number) ** 2
  return 1 - (p.cdf(1.5) as number)
}

/**
 * A sample from q = N(0, 1), weighted by the true ratio r(x) = p(x)/q(x), stands in for a sample from
 * p = N(m, s²): its weighted histogram follows p, and weighted averages estimate expectations under p.
 */
export function ReweightingDemo() {
  const state = useFigureState({
    mean: float(1, { min: -2, max: 3, step: 0.05, label: 'mean of p', onChart: true }),
    sd: float(0.6, { min: 0.3, max: 1.6, step: 0.05, label: 'standard deviation of p' }),
    n: int(300, { min: 20, max: 2000, label: 'points drawn from q', suggestions: [50, 300, 2000] }),
    target: choice(
      (Object.keys(TARGETS) as Target[]).map((value) => ({ value, label: TARGETS[value].label })),
      'x',
      { label: 'average of' },
    ),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })
  const shift: Shift = useMemo(() => ({ mean: state.mean, sd: state.sd }), [state.mean, state.sd])

  const xq = useMemo(() => drawSamples(state.seed, 1, state.n, { mean: 0, sd: 1 }).xq, [state.seed, state.n])
  const weights = useMemo(() => xq.map((x) => Math.exp(logRatio(x, shift))), [xq, shift])

  const curves = useMemo(
    () => ({
      p: X.map((x) => Math.exp(logP(x, shift))),
      q: X.map((x) => Math.exp(logQ(x))),
      r: X.map((x) => Math.exp(logRatio(x, shift))),
    }),
    [shift],
  )

  const bars = useMemo(() => {
    const plain = histogram(xq, { bins: EDGES })
    const weighted = histogram(xq, { bins: EDGES, weights })
    return { plain: toFlat(plain.density), weighted: toFlat(weighted.density) }
  }, [xq, weights])

  const f = TARGETS[state.target].f
  const n = xq.length
  const sumW = weights.reduce((a, w) => a + w, 0)
  const sumW2 = weights.reduce((a, w) => a + w * w, 0)
  const plainMean = xq.reduce((a, x) => a + f(x), 0) / n
  const weightedMean = xq.reduce((a, x, i) => a + weights[i] * f(x), 0) / n
  const selfNormalised = xq.reduce((a, x, i) => a + weights[i] * f(x), 0) / sumW
  const ess = (sumW * sumW) / sumW2

  const stems = useMemo(
    () => xq.map((x, i) => ({ from: [x, 0] as const, to: [x, weights[i]] as const })),
    [xq, weights],
  )

  const xa = useAxis({ label: 'x', range: [LO, HI] })
  const ya = useAxis({ label: 'density', range: [0, undefined] })
  const xb = useAxis({ label: 'x', range: [LO, HI] })
  const yb = useAxis({ label: 'r(x) = p(x) / q(x)', range: [0, undefined] })
  const peak = Math.exp(logP(state.mean, shift))
  return (
    <Figure
      title="Reweighting a sample from q into a sample from p"
      state={state}
      caption="Left: the denominator q = N(0, 1), the numerator p, a histogram of the sample from q, and the same histogram with each point weighted by r(x). The weighted histogram follows p. Drag the peak of p to move it. Right: the ratio r(x) and the weight it gives each point of the sample, as a stem. When p is narrower than q the weights are bounded; when p is wider, or moved far from q, a few points in the tail of q carry most of the weight and the effective sample size collapses."
      readouts={
        <>
          <Readout
            label={`E_p[${TARGETS[state.target].label}], exact`}
            value={formatNumber(expectation(state.target, shift))}
          />
          <Readout label="plain average over the q sample" value={formatNumber(plainMean)} />
          <Readout label="weighted average (1/n) Σ r f" value={formatNumber(weightedMean)} />
          <Readout label="self-normalised Σ r f / Σ r" value={formatNumber(selfNormalised)} />
          <Readout label="effective sample size" value={`${formatNumber(ess)} of ${n}`} />
          <Readout label="largest weight" value={formatNumber(Math.max(...weights))} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya} height={300}>
          <Bars name="q sample" x={MIDS} y={bars.plain} edges={EDGES} slot={0} opacity={0.25} />
          <Bars name="q sample weighted by r" x={MIDS} y={bars.weighted} edges={EDGES} slot={1} opacity={0.35} />
          <Curve name="q(x)" x={X} y={curves.q} slot={0} />
          <Curve name="p(x)" x={X} y={curves.p} slot={1} />
          <Handle
            kind="point"
            at={[state.mean, peak]}
            label="p"
            slot={1}
            onDrag={([x]) => state.set('mean', Math.min(3, Math.max(-2, x)))}
          />
        </Plot>
        <Plot x={xb} y={yb} height={300}>
          <Segments name="weight of each point" segments={stems} slot={1} />
          <Curve name="r(x)" x={X} y={curves.r} emphasis />
          <Rug name="q sample" values={xq} slot={0} />
        </Plot>
      </div>
    </Figure>
  )
}
