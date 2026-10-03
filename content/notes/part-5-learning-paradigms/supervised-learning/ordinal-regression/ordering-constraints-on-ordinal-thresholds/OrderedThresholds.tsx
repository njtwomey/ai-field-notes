import { useMemo, useState } from 'react'
import { MathText } from 'aifn-render'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { sigmoid } from '@/lib/math'
import { useClassColors } from '../_shared/classColor'
import { softplus } from '../_shared/ordinal'

/** The worked example: class k (1-based) sits at score k − 1; classes 1, 3 and 4 have three examples each. */
const SCORES = [0, 1, 2, 3]
const K = SCORES.length
const X_RANGE: [number, number] = [-1, 4]
const MAX_N2 = 4
/** Amber: a warning that reads apart from the blue-to-red class colours. */
const CROSSED_SLOT = 3

const countsFor = (n2: number) => [3, n2, 3, 3]

/**
 * ∂L/∂θₖ for the logistic immediate-threshold loss: θₖ (0-based k) sees class k below it and class k + 1 above it.
 * It depends on θₖ alone and increases with it.
 */
const grad = (n: number[], k: number, t: number) =>
  -n[k] * sigmoid(SCORES[k] - t) + n[k + 1] * sigmoid(t - SCORES[k + 1])

function lossOf(n: number[], th: number[]) {
  let total = 0
  th.forEach((t, k) => {
    if (n[k] > 0) total += n[k] * softplus(SCORES[k] - t)
    if (n[k + 1] > 0) total += n[k + 1] * softplus(t - SCORES[k + 1])
  })
  return total
}

/** The root of an increasing function. */
function root(f: (t: number) => number) {
  let lo = -20
  let hi = 20
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2
    if (f(mid) < 0) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

/** A threshold fitted alone. With one of its two classes empty, the loss keeps falling and the fit runs off to ±∞. */
function separate(n: number[], k: number) {
  if (n[k] === 0) return -Infinity
  if (n[k + 1] === 0) return Infinity
  return root((t) => grad(n, k, t))
}

type Fit = { separate: number[]; constrained: number[]; mu: number[]; lossSeparate: number; lossConstrained: number }

/**
 * Both fits, exactly. The loss is a sum of one convex term per threshold, so the ordered optimum is found by pooling
 * adjacent violators: merge neighbouring blocks whose values are out of order and give each block the root of its
 * summed derivative. The multipliers then follow from stationarity, μₖ = −(∂₁L + … + ∂ₖL).
 */
function fit(n2: number): Fit {
  const n = countsFor(n2)
  const sep = Array.from({ length: K - 1 }, (_, k) => separate(n, k))
  const blocks: { ks: number[]; value: number }[] = []
  sep.forEach((value, k) => {
    blocks.push({ ks: [k], value })
    while (blocks.length > 1 && blocks[blocks.length - 2].value > blocks[blocks.length - 1].value) {
      const last = blocks.pop()!
      const prev = blocks.pop()!
      const ks = [...prev.ks, ...last.ks]
      blocks.push({ ks, value: root((t) => ks.reduce((s, j) => s + grad(n, j, t), 0)) })
    }
  })
  const constrained = blocks.flatMap((b) => b.ks.map(() => b.value))
  const mu: number[] = []
  let push = 0
  for (let k = 0; k < K - 2; k++) {
    push -= grad(n, k, constrained[k])
    // Clip the rounding error of an inactive constraint, whose multiplier is exactly zero.
    mu.push(push > 1e-9 ? push : 0)
  }
  return {
    separate: sep,
    constrained,
    mu,
    lossSeparate: lossOf(n, sep),
    lossConstrained: lossOf(n, constrained),
  }
}

const FITS = Array.from({ length: MAX_N2 + 1 }, (_, n2) => fit(n2))

const clampX = (t: number) => Math.min(X_RANGE[1], Math.max(X_RANGE[0], t))
const show = (t: number) => (t === Infinity ? '+∞' : t === -Infinity ? '−∞' : formatNumber(t))
const sub = (k: number) => '₀₁₂₃₄₅₆₇₈₉'[k]

/** A threshold drawn across the rows of the two classes it separates: θₖ spans rows k and k + 1. */
const thresholdLine = (t: number, k: number) => ({ x: [clampX(t), clampX(t)], y: [k + 0.6, k + 2.4] })

/**
 * The worked example with a shrinking class 2. The separate fits (dashed) cross once class 2 is small enough; the
 * constrained fit (solid) merges the two thresholds instead, and the multiplier of that constraint rises from zero.
 */
export function OrderedThresholds() {
  const [n2, setN2] = useState(1)
  const colors = useClassColors(K)
  const f = FITS[n2]
  const crossed = f.separate.slice(0, -1).map((t, k) => t > f.separate[k + 1])
  const active = f.mu.map((m) => m > 0)

  const points = useMemo<XYSeries[]>(
    () =>
      countsFor(n2).map((count, c) => ({
        name: `class ${c + 1}`,
        type: 'scatter' as const,
        x: Array.from({ length: count }, () => SCORES[c]),
        // Examples at the same score are stacked within their class's row, so the count is visible.
        y: Array.from({ length: count }, (_, j) => c + 1 + (j - (count - 1) / 2) * 0.2),
        color: colors[c],
      })),
    [n2, colors],
  )

  const lines = useMemo<XYSeries[]>(() => {
    const { separate: sep, constrained } = FITS[n2]
    const isCrossed = (k: number) => (k > 0 && sep[k - 1] > sep[k]) || (k < sep.length - 1 && sep[k] > sep[k + 1])
    return [
      ...sep.map((t, k) => ({
        name: isCrossed(k) ? 'separate fit, crossed' : 'separate fit',
        type: 'line' as const,
        dashed: true,
        ...(isCrossed(k) ? { slot: CROSSED_SLOT } : { muted: true }),
        ...thresholdLine(t, k),
      })),
      ...constrained.map((t, k) => ({
        name: 'constrained fit',
        type: 'line' as const,
        emphasis: true,
        ...thresholdLine(t, k),
      })),
    ]
  }, [n2])

  const bars = useMemo<XYSeries[]>(
    () => [{ name: 'μₖ', type: 'bar', x: FITS[n2].mu.map((_, k) => k + 1), y: FITS[n2].mu, slot: 0 }],
    [n2],
  )

  const crossedText = crossed.flatMap((c, k) =>
    c ? [`θ${sub(k + 1)} = ${show(f.separate[k])} > θ${sub(k + 2)} = ${show(f.separate[k + 1])}`] : [],
  )
  const mergedText = active.flatMap((a, k) =>
    a ? [`θ${sub(k + 1)} = θ${sub(k + 2)} = ${formatNumber(f.constrained[k])}, so class ${k + 2} is merged away`] : [],
  )
  const activeSet = active.flatMap((a, k) => (a ? [`θ${sub(k + 1)} = θ${sub(k + 2)}`] : []))

  return (
    <Interactive
      title="Ordering constraint and its multiplier"
      caption={
        <MathText text="The worked example: one row per class, with the examples at their scores; the slider sets how many examples class 2 has. Each dashed line is a threshold fitted on its own to the two classes it separates, drawn across their two rows. Each solid line is the constrained fit, with $\theta_1 \le \theta_2 \le \theta_3$ imposed. Move the slider left. With two or more class-2 examples the two fits agree and every multiplier is 0. With one, the dashed $\theta_1$ and $\theta_2$ cross, the solid ones merge instead, and the multiplier $\mu_1$ rises above 0. With none, the dashed pair runs off to $\pm\infty$ (drawn at the edges) and $\mu_1$ rises further. The bars show $\mu_1$ and $\mu_2$; $\mu_2$ stays 0 because $\theta_2 < \theta_3$ throughout." />
      }
      controls={
        <ParamSlider label="examples in class 2" value={n2} onChange={setN2} min={0} max={MAX_N2} step={1} withArrows />
      }
      readout={
        <>
          <Readout label="loss, separate fits" value={formatNumber(f.lossSeparate)} />
          <Readout label="loss, constrained" value={formatNumber(f.lossConstrained)} />
          <Readout label="active set" value={activeSet.length ? activeSet.join(', ') : 'none'} />
          {f.mu.map((m, k) => (
            <Readout key={k} label={`μ${sub(k + 1)}${active[k] ? ' (active)' : ''}`} value={formatNumber(m)} />
          ))}
          <Readout
            label="μₖ(θₖ₊₁ − θₖ)"
            value={f.mu.map((m, k) => formatNumber(m * (f.constrained[k + 1] - f.constrained[k]))).join(', ')}
          />
        </>
      }
    >
      <div className="flex flex-col gap-0.5 text-xs text-muted-foreground">
        <span>
          separate fits:{' '}
          {crossedText.length ? (
            <span className="font-medium text-foreground">{crossedText.join('; ')}, crossed</span>
          ) : (
            'ordered'
          )}
        </span>
        <span>
          constrained fit:{' '}
          {mergedText.length ? (
            <span className="font-medium text-foreground">{mergedText.join('; ')}</span>
          ) : (
            'ordered, no constraint active'
          )}
        </span>
      </div>
      <XYChart
        series={[...points, ...lines]}
        xRange={X_RANGE}
        yRange={[0.4, K + 0.6]}
        xLabel="score s"
        yLabel="class"
        height={280}
        ariaLabel="Examples of four classes on a score axis with separate and constrained thresholds"
      />
      <XYChart
        series={bars}
        xRange={[0.5, K - 1.5]}
        yRange={[0, 1]}
        integerX
        xLabel="constraint k"
        yLabel="μₖ"
        height={160}
        ariaLabel="Multipliers of the ordering constraints"
      />
    </Interactive>
  )
}
