import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'

type V = [number, number]
/** Symmetric 2×2 matrix stored as [s11, s12, s22]. */
type S = [number, number, number]

const BOX = 4
const GRID = linspace(-BOX, BOX, 51)
const RANGE: [number, number] = [0, 1]
const N_TEST = 4000

const covOf = (s1: number, s2: number, r: number): S => [s1 * s1, r * s1 * s2, s2 * s2]

function sampleClass(n: number, mu: V, c: S, g: ReturnType<typeof rng>): V[] {
  const l11 = Math.sqrt(c[0])
  const l21 = c[1] / l11
  const l22 = Math.sqrt(Math.max(c[2] - l21 * l21, 1e-12))
  return Array.from({ length: n }, () => {
    const z1 = g.normal()
    const z2 = g.normal()
    return [mu[0] + l11 * z1, mu[1] + l21 * z1 + l22 * z2]
  })
}

function meanOf(xs: V[]): V {
  return [xs.reduce((a, p) => a + p[0], 0) / xs.length, xs.reduce((a, p) => a + p[1], 0) / xs.length]
}

/** Scatter matrix Σ (x − μ)(x − μ)ᵀ, unnormalised. */
function scatterOf(xs: V[], mu: V): S {
  const s: S = [0, 0, 0]
  for (const p of xs) {
    const a = p[0] - mu[0]
    const b = p[1] - mu[1]
    s[0] += a * a
    s[1] += a * b
    s[2] += b * b
  }
  return s
}

/** log N(x | μ, Σ) up to the constant −log 2π shared by every class. */
function logGauss(x: V, mu: V, c: S): number {
  const det = c[0] * c[2] - c[1] * c[1]
  const a = x[0] - mu[0]
  const b = x[1] - mu[1]
  const quad = (c[2] * a * a - 2 * c[1] * a * b + c[0] * b * b) / det
  return -0.5 * quad - 0.5 * Math.log(det)
}

type Model = { mu: [V, V]; cov: [S, S] }

/** Posterior probability of class 1 under a two-class Gaussian model with equal priors. */
const posterior = (m: Model, x: V) =>
  1 / (1 + Math.exp(logGauss(x, m.mu[0], m.cov[0]) - logGauss(x, m.mu[1], m.cov[1])))

function fit(a: V[], b: V[]): { lda: Model; qda: Model } {
  const mu: [V, V] = [meanOf(a), meanOf(b)]
  const sa = scatterOf(a, mu[0])
  const sb = scatterOf(b, mu[1])
  const pooled = sa.map((v, i) => (v + sb[i]) / (a.length + b.length - 2)) as S
  const qa = sa.map((v) => v / (a.length - 1)) as S
  const qb = sb.map((v) => v / (b.length - 1)) as S
  return { lda: { mu, cov: [pooled, pooled] }, qda: { mu, cov: [qa, qb] } }
}

/**
 * Two Gaussian classes in the plane with draggable means. LDA fits one pooled covariance and draws a straight
 * boundary; QDA fits one covariance per class and draws a conic.
 */
export function DiscriminantExplorer() {
  const [muA, setMuA] = useState<V>([-1.2, -0.4])
  const [muB, setMuB] = useState<V>([1.2, 0.6])
  const sA1 = useParam(1, { min: 0.3, max: 2, step: 0.05 })
  const sA2 = useParam(0.6, { min: 0.3, max: 2, step: 0.05 })
  const rA = useParam(0.3, { min: -0.9, max: 0.9, step: 0.05 })
  const sB1 = useParam(0.5, { min: 0.3, max: 2, step: 0.05 })
  const sB2 = useParam(1.4, { min: 0.3, max: 2, step: 0.05 })
  const rB = useParam(-0.4, { min: -0.9, max: 0.9, step: 0.05 })
  const n = useParam(40, { min: 5, max: 300, step: 1 })
  const [shared, setShared] = useState(false)

  const covA = covOf(sA1.value, sA2.value, rA.value)
  const covB = shared ? covA : covOf(sB1.value, sB2.value, rB.value)
  const [a0, a1, a2] = covA
  const [b0, b1, b2] = covB

  const r = useMemo(() => {
    const cA: S = [a0, a1, a2]
    const cB: S = [b0, b1, b2]
    const g = rng(5)
    const trainA = sampleClass(n.value, muA, cA, g)
    const trainB = sampleClass(n.value, muB, cB, g)
    const { lda, qda } = fit(trainA, trainB)
    const truth: Model = { mu: [muA, muB], cov: [cA, cB] }
    const h = rng(99)
    const testA = sampleClass(N_TEST / 2, muA, cA, h)
    const testB = sampleClass(N_TEST / 2, muB, cB, h)
    const error = (m: Model) =>
      (testA.filter((x) => posterior(m, x) > 0.5).length + testB.filter((x) => posterior(m, x) <= 0.5).length) / N_TEST
    const surface = (m: Model) => GRID.map((y) => GRID.map((x) => posterior(m, [x, y])))
    return {
      trainA,
      trainB,
      ldaZ: surface(lda),
      qdaZ: surface(qda),
      ldaErr: error(lda),
      qdaErr: error(qda),
      bayesErr: error(truth),
    }
  }, [muA, muB, a0, a1, a2, b0, b1, b2, n.value])

  const overlay = useMemo((): HeatmapOverlay[] => {
    const pts = [...r.trainA, ...r.trainB]
    return [
      {
        name: 'points',
        type: 'scatter',
        x: pts.map((p) => p[0]),
        y: pts.map((p) => p[1]),
        group: pts.map((_, i) => (i < r.trainA.length ? 0 : 1)),
        groupNames: ['class 0', 'class 1'],
      },
      { name: 'true means', type: 'scatter', x: [muA[0], muB[0]], y: [muA[1], muB[1]], emphasis: true },
    ]
  }, [r, muA, muB])

  const clamp = (v: number) => Math.max(-BOX + 0.2, Math.min(BOX - 0.2, v))
  const handles: Handle[] = [
    { kind: 'point', at: muA, label: 'mean of class 0', onDrag: (p) => setMuA([clamp(p[0]), clamp(p[1])]) },
    { kind: 'point', at: muB, label: 'mean of class 1', onDrag: (p) => setMuB([clamp(p[0]), clamp(p[1])]) },
  ]

  return (
    <Interactive
      title="Linear and quadratic discriminant analysis"
      caption="Two Gaussian classes with equal priors; the diamonds are their true means, and both can be dragged on either chart. Shading is the fitted posterior probability of class 1. LDA pools the two sample covariances into one, so its boundary (where the shading is neutral) is a straight line. QDA fits a covariance per class and its boundary is a conic. When the true covariances differ, QDA approaches the Bayes error and LDA does not. With few points per class or a shared true covariance, QDA's extra parameters only add variance. Test errors use 4,000 fresh points."
      controls={
        <>
          <ParamSlider label="points per class" param={n} format={(v) => String(v)} />
          <ParamSwitch label="shared true covariance" checked={shared} onChange={setShared} />
          <ParamSlider label="class 0: sd of x₁" param={sA1} />
          <ParamSlider label="class 0: sd of x₂" param={sA2} />
          <ParamSlider label="class 0: correlation" param={rA} />
          {!shared && (
            <>
              <ParamSlider label="class 1: sd of x₁" param={sB1} />
              <ParamSlider label="class 1: sd of x₂" param={sB2} />
              <ParamSlider label="class 1: correlation" param={rB} />
            </>
          )}
        </>
      }
      readout={
        <>
          <Readout label="LDA test error" value={formatNumber(r.ldaErr)} />
          <Readout label="QDA test error" value={formatNumber(r.qdaErr)} />
          <Readout label="Bayes error" value={formatNumber(r.bayesErr)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <p className="mb-1 text-center text-xs text-muted-foreground">LDA: one pooled covariance</p>
          <Heatmap
            x={GRID}
            y={GRID}
            z={r.ldaZ}
            scale="diverging"
            range={RANGE}
            xLabel="x₁"
            yLabel="x₂"
            valueLabel="P(class 1 | x)"
            overlay={overlay}
            handles={handles}
            height={360}
          />
        </div>
        <div>
          <p className="mb-1 text-center text-xs text-muted-foreground">QDA: one covariance per class</p>
          <Heatmap
            x={GRID}
            y={GRID}
            z={r.qdaZ}
            scale="diverging"
            range={RANGE}
            xLabel="x₁"
            yLabel="x₂"
            valueLabel="P(class 1 | x)"
            overlay={overlay}
            handles={handles}
            height={360}
          />
        </div>
      </div>
    </Interactive>
  )
}
