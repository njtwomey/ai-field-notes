import { useMemo } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { normalCdf, normalQuantile } from '@/lib/math/special'

/**
 * Binormal scores in each group: negatives ~ N(m, 1), positives ~ N(m + d, 1), with group B's scores offset by m = −0.5.
 * The ROC curve at false-positive rate f is TPR = Φ(d + Φ⁻¹(f)), reached by the threshold τ = m + Φ⁻¹(1 − f). The curve is concave, so the set of (FPR, TPR) pairs a
 * randomised threshold rule can reach is the region between the diagonal and the curve.
 */
const roc = (f: number, d: number) => (f <= 0 ? 0 : f >= 1 ? 1 : normalCdf(d + normalQuantile(f)))
const threshold = (f: number, offset: number) => offset + normalQuantile(1 - f)
const OFFSET_B = -0.5

const GRID = linspace(0, 1, 201)

/** Accuracy of a derived predictor at (f, t) in a group with base rate π. */
const accuracy = (f: number, t: number, pi: number) => pi * t + (1 - pi) * (1 - f)

export function PostProcessing() {
  const dA = useParam(2.2, { min: 0.3, max: 3.5, step: 0.1 })
  const dB = useParam(1.2, { min: 0.3, max: 3.5, step: 0.1 })
  const piA = useParam(0.5, { min: 0.05, max: 0.95, step: 0.01 })
  const piB = useParam(0.3, { min: 0.05, max: 0.95, step: 0.01 })
  const fT = useParam(0.2, { min: 0, max: 1, step: 0.005 })
  const tT = useParam(0.6, { min: 0, max: 1, step: 0.005 })

  // The feasible set for both groups: t between f and the lower of the two ROC curves.
  const upper = (f: number) => Math.min(roc(f, dA.value), roc(f, dB.value))
  const f = fT.value
  const t = Math.min(Math.max(tT.value, f), upper(f))

  const curves = useMemo(() => {
    const yA = GRID.map((g) => roc(g, dA.value))
    const yB = GRID.map((g) => roc(g, dB.value))
    return { yA, yB, yMin: yA.map((y, i) => Math.min(y, yB[i])) }
  }, [dA.value, dB.value])

  // The linear program: maximise the equally weighted accuracy over the feasible set. The objective is linear in
  // (f, t) and increasing in t, so the optimum lies on the upper edge; a fine grid along it is exact enough.
  const optimum = useMemo(() => {
    let best = { f: 0, t: 0, acc: -Infinity }
    for (const g of linspace(0, 1, 2001)) {
      const tt = Math.min(roc(g, dA.value), roc(g, dB.value))
      const acc = 0.5 * accuracy(g, tt, piA.value) + 0.5 * accuracy(g, tt, piB.value)
      if (acc > best.acc) best = { f: g, t: tt, acc }
    }
    return best
  }, [dA.value, dB.value, piA.value, piB.value])

  // Derived predictor for group a: with probability λ use the threshold τ that gives FPR f; otherwise predict positive
  // at random with probability f. Both parts have FPR f, and the TPR is λ·ROC(f) + (1 − λ)·f.
  const mix = (d: number) => {
    const top = roc(f, d)
    return top - f > 1e-9 ? (t - f) / (top - f) : 1
  }
  const lambdaA = mix(dA.value)
  const lambdaB = mix(dB.value)
  const accTarget = 0.5 * accuracy(f, t, piA.value) + 0.5 * accuracy(f, t, piB.value)

  const series: XYSeries[] = [
    { name: 'chance', type: 'line', x: [0, 1], y: [0, 1], dashed: true, muted: true },
    { name: 'group A ROC', type: 'line', x: GRID, y: curves.yA, slot: 0 },
    { name: 'group B ROC', type: 'line', x: GRID, y: curves.yB, slot: 1 },
    { name: 'feasible for both (upper edge)', type: 'line', x: GRID, y: curves.yMin, emphasis: true, dashed: true },
    { name: 'LP optimum', type: 'scatter', x: [optimum.f], y: [optimum.t], emphasis: true },
  ]
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [f, t],
      label: 'common operating point',
      onDrag: ([x, y]) => {
        const fx = Math.min(1, Math.max(0, x))
        fT.set(fx)
        tT.set(Math.min(Math.max(y, fx), upper(fx)))
      },
    },
  ]
  const fmtThreshold = (x: number) => (Number.isFinite(x) ? formatNumber(x) : x > 0 ? '+∞' : '−∞')

  return (
    <Interactive
      title="Equalised odds by post-processing"
      caption="Scores in each group are binormal with separation d, and group B's scores run 0.5 lower than group A's. A randomised threshold rule can reach any (FPR, TPR) pair between the diagonal and its group's ROC curve, so both groups can share an operating point only below the dashed ink edge, the lower of the two curves. Drag the common operating point (or use the sliders); it is kept inside the shared region. The readout gives, per group, the threshold τ and the probability λ of applying it; otherwise the rule predicts positive at random with probability equal to the target FPR. The ink dot is the linear-program optimum for accuracy with equal group sizes."
      controls={
        <>
          <ParamSlider label="separation d, group A" param={dA} />
          <ParamSlider label="separation d, group B" param={dB} />
          <ParamSlider label="base rate, group A" param={piA} />
          <ParamSlider label="base rate, group B" param={piB} />
          <ParamSlider label="target FPR" param={fT} />
          <ParamSlider label="target TPR" param={tT} />
          <ParamButton
            onClick={() => {
              fT.set(optimum.f)
              tT.set(optimum.t)
            }}
          >
            Move to optimum
          </ParamButton>
        </>
      }
      readout={
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
        <XYChart
          series={series}
          xLabel="false-positive rate"
          yLabel="true-positive rate"
          xRange={[0, 1]}
          yRange={[0, 1]}
          equalAspect
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
