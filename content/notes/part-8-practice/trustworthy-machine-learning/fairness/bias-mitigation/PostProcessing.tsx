import { useMemo } from 'react'
import {
  Button,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normalCdf, normalQuantile } from 'aifn/numerics/special'

/**
 * Binormal scores in each group: negatives ~ N(m, 1), positives ~ N(m + d, 1), with group B's scores offset by m = −0.5.
 * The ROC curve at false-positive rate f is TPR = Φ(d + Φ⁻¹(f)), reached by the threshold τ = m + Φ⁻¹(1 − f). The curve is concave, so the set of (FPR, TPR) pairs a
 * randomised threshold rule can reach is the region between the diagonal and the curve.
 */
const roc = (f: number, d: number) => (f <= 0 ? 0 : f >= 1 ? 1 : normalCdf(d + normalQuantile(f)))
const threshold = (f: number, offset: number) => offset + normalQuantile(1 - f)
const OFFSET_B = -0.5

const GRID = toFlat(linspace(0, 1, 201))

/** Accuracy of a derived predictor at (f, t) in a group with base rate π. */
const accuracy = (f: number, t: number, pi: number) => pi * t + (1 - pi) * (1 - f)

export function PostProcessing() {
  const state = useFigureState({
    dA: float(2.2, { min: 0.3, max: 3.5, step: 0.1, label: 'separation d, group A' }),
    dB: float(1.2, { min: 0.3, max: 3.5, step: 0.1, label: 'separation d, group B' }),
    piA: float(0.5, { min: 0.05, max: 0.95, step: 0.01, label: 'base rate, group A' }),
    piB: float(0.3, { min: 0.05, max: 0.95, step: 0.01, label: 'base rate, group B' }),
    fT: float(0.2, { min: 0, max: 1, step: 0.005, label: 'target FPR' }),
    tT: float(0.6, { min: 0, max: 1, step: 0.005, label: 'target TPR' }),
  })

  // The feasible set for both groups: t between f and the lower of the two ROC curves.
  const upper = (f: number) => Math.min(roc(f, state.dA), roc(f, state.dB))
  const f = state.fT
  const t = Math.min(Math.max(state.tT, f), upper(f))

  const curves = useMemo(() => {
    const yA = GRID.map((g) => roc(g, state.dA))
    const yB = GRID.map((g) => roc(g, state.dB))
    return { yA, yB, yMin: yA.map((y, i) => Math.min(y, yB[i])) }
  }, [state.dA, state.dB])

  // The linear program: maximise the equally weighted accuracy over the feasible set. The objective is linear in
  // (f, t) and increasing in t, so the optimum lies on the upper edge; a fine grid along it is exact enough.
  const optimum = useMemo(() => {
    let best = { f: 0, t: 0, acc: -Infinity }
    for (const g of toFlat(linspace(0, 1, 2001))) {
      const tt = Math.min(roc(g, state.dA), roc(g, state.dB))
      const acc = 0.5 * accuracy(g, tt, state.piA) + 0.5 * accuracy(g, tt, state.piB)
      if (acc > best.acc) best = { f: g, t: tt, acc }
    }
    return best
  }, [state.dA, state.dB, state.piA, state.piB])

  // Derived predictor for group a: with probability λ use the threshold τ that gives FPR f; otherwise predict positive
  // at random with probability f. Both parts have FPR f, and the TPR is λ·ROC(f) + (1 − λ)·f.
  const mix = (d: number) => {
    const top = roc(f, d)
    return top - f > 1e-9 ? (t - f) / (top - f) : 1
  }
  const lambdaA = mix(state.dA)
  const lambdaB = mix(state.dB)
  const accTarget = 0.5 * accuracy(f, t, state.piA) + 0.5 * accuracy(f, t, state.piB)

  const series = [
    { name: 'chance', x: [0, 1], y: [0, 1], dashed: true, muted: true },
    { name: 'group A ROC', x: GRID, y: curves.yA, slot: 0 },
    { name: 'group B ROC', x: GRID, y: curves.yB, slot: 1 },
    { name: 'feasible for both (upper edge)', x: GRID, y: curves.yMin, emphasis: true, dashed: true },
    { name: 'LP optimum', x: [optimum.f], y: [optimum.t], emphasis: true },
  ] as const
  const fmtThreshold = (x: number) => (Number.isFinite(x) ? formatNumber(x) : x > 0 ? '+∞' : '−∞')

  const xAxis = useAxis({ label: 'false-positive rate', range: [0, 1] })
  const yAxis = useAxis({ label: 'true-positive rate', range: [0, 1], equal: xAxis })
  return (
    <Figure
      title="Equalised odds by post-processing"
      purpose="Choose a target (FPR, TPR) pair and see the randomised thresholds that give both groups that operating point."
      state={state}
      caption="Scores in each group are binormal with separation d, and group B's scores run 0.5 lower than group A's. A randomised threshold rule can reach any (FPR, TPR) pair between the diagonal and its group's ROC curve, so both groups can share an operating point only below the dashed ink edge, the lower of the two curves. Drag the common operating point (or use the sliders); it is kept inside the shared region. The readout gives, per group, the threshold τ and the probability λ of applying it; otherwise the rule predicts positive at random with probability equal to the target FPR. The ink dot is the linear-program optimum for accuracy with equal group sizes."
      controls={
        <>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              state.set('fT', optimum.f)
              state.set('tT', optimum.t)
            }}
          >
            Move to optimum
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="common (FPR, TPR)" value={`(${formatNumber(f)}, ${formatNumber(t)})`} />
          <Readout label="group A: τ, λ" value={`${fmtThreshold(threshold(f, 0))}, ${formatNumber(lambdaA)}`} />
          <Readout label="group B: τ, λ" value={`${fmtThreshold(threshold(f, OFFSET_B))}, ${formatNumber(lambdaB)}`} />
          <Readout
            label="accuracy here / at optimum"
            value={`${formatNumber(accTarget)} / ${formatNumber(optimum.acc)}`}
          />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Plot x={xAxis} y={yAxis}>
          <Curve {...series[0]} />
          <Curve {...series[1]} />
          <Curve {...series[2]} />
          <Curve {...series[3]} />
          <Points {...series[4]} />
          <Handle
            kind="point"
            at={[f, t]}
            label="common operating point"
            onDrag={([x, y]) => {
              const fx = Math.min(1, Math.max(0, x))
              state.set('fT', fx)
              state.set('tT', Math.min(Math.max(y, fx), upper(fx)))
            }}
          />
        </Plot>
      </div>
    </Figure>
  )
}
