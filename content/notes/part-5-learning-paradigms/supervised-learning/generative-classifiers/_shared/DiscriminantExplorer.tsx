import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream, type Stream } from 'aifn-compute/foundation/random'

type V = [number, number]
/** Symmetric 2×2 matrix stored as [s11, s12, s22]. */
type S = [number, number, number]

const BOX = 4
const GRID = toFlat(linspace(-BOX, BOX, 51))
const RANGE: [number, number] = [0, 1]
const N_TEST = 4000

const covOf = (s1: number, s2: number, r: number): S => [s1 * s1, r * s1 * s2, s2 * s2]

function sampleClass(n: number, mu: V, c: S, g: Stream): V[] {
  const l11 = Math.sqrt(c[0])
  const l21 = c[1] / l11
  const l22 = Math.sqrt(Math.max(c[2] - l21 * l21, 1e-12))
  return Array.from({ length: n }, () => {
    const z1 = normal(g)
    const z2 = normal(g)
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
/** How far a mean can be dragged from the origin. */
const LIM = BOX - 0.2

export function DiscriminantExplorer() {
  const state = useFigureState({
    n: int(40, { min: 5, max: 300, step: 1, suggestions: [10, 40, 100, 300], label: 'points per class' }),
    shared: setting(false, 'shared true covariance'),
    sA1: float(1, { min: 0.3, max: 2, step: 0.05, label: 'class 0: sd of x₁' }),
    sA2: float(0.6, { min: 0.3, max: 2, step: 0.05, label: 'class 0: sd of x₂' }),
    rA: slider(-0.9, 0.9, 0.3, { step: 0.05, label: 'class 0: correlation' }),
    sB1: float(0.5, { min: 0.3, max: 2, step: 0.05, label: 'class 1: sd of x₁', when: (v) => !v.shared }),
    sB2: float(1.4, { min: 0.3, max: 2, step: 0.05, label: 'class 1: sd of x₂', when: (v) => !v.shared }),
    rB: slider(-0.9, 0.9, -0.4, { step: 0.05, label: 'class 1: correlation', when: (v) => !v.shared }),
    ax: slider(-LIM, LIM, -1.2, { step: 0.01, onChart: true }),
    ay: slider(-LIM, LIM, -0.4, { step: 0.01, onChart: true }),
    bx: slider(-LIM, LIM, 1.2, { step: 0.01, onChart: true }),
    by: slider(-LIM, LIM, 0.6, { step: 0.01, onChart: true }),
  })
  const muA = useMemo((): V => [state.ax, state.ay], [state.ax, state.ay])
  const muB = useMemo((): V => [state.bx, state.by], [state.bx, state.by])

  const covA = covOf(state.sA1, state.sA2, state.rA)
  const covB = state.shared ? covA : covOf(state.sB1, state.sB2, state.rB)
  const [a0, a1, a2] = covA
  const [b0, b1, b2] = covB

  const r = useMemo(() => {
    const cA: S = [a0, a1, a2]
    const cB: S = [b0, b1, b2]
    const g = stream(5)
    const trainA = sampleClass(state.n, muA, cA, g)
    const trainB = sampleClass(state.n, muB, cB, g)
    const { lda, qda } = fit(trainA, trainB)
    const truth: Model = { mu: [muA, muB], cov: [cA, cB] }
    const h = stream(99)
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
  }, [muA, muB, a0, a1, a2, b0, b1, b2, state.n])

  const overlay = useMemo(() => {
    const pts = [...r.trainA, ...r.trainB]
    return [
      {
        name: 'points',
        x: pts.map((p) => p[0]),
        y: pts.map((p) => p[1]),
        group: pts.map((_, i) => (i < r.trainA.length ? 0 : 1)),
        groupNames: ['class 0', 'class 1'],
      },
      { name: 'true means', x: [muA[0], muB[0]], y: [muA[1], muB[1]], emphasis: true },
    ] as const
  }, [r, muA, muB])

  const xAxis = useAxis({ label: 'x₁' })
  const yAxis = useAxis({ label: 'x₂' })
  const xAxis2 = useAxis({ label: 'x₁' })
  const yAxis2 = useAxis({ label: 'x₂' })
  return (
    <Figure
      title="Linear and quadratic discriminant analysis"
      state={state}
      caption="Two Gaussian classes with equal priors; the diamonds are their true means, and both can be dragged on either chart. Shading is the fitted posterior probability of class 1. LDA pools the two sample covariances into one, so its boundary (where the shading is neutral) is a straight line. QDA fits a covariance per class and its boundary is a conic. When the true covariances differ, QDA approaches the Bayes error and LDA does not. With few points per class or a shared true covariance, QDA's extra parameters only add variance. Test errors use 4,000 fresh points."
      readouts={
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
          <Plot x={xAxis} y={yAxis} height={360}>
            <Raster x={GRID} y={GRID} z={r.ldaZ} scale={'diverging'} range={RANGE} valueLabel={'P(class 1 | x)'} />
            <Points {...overlay[0]} live />
            <Points {...overlay[1]} live />
            <Handle {...state.handle(['ax', 'ay'], { label: 'mean of class 0' })} />
            <Handle {...state.handle(['bx', 'by'], { label: 'mean of class 1' })} />
          </Plot>
        </div>
        <div>
          <p className="mb-1 text-center text-xs text-muted-foreground">QDA: one covariance per class</p>
          <Plot x={xAxis2} y={yAxis2} height={360}>
            <Raster x={GRID} y={GRID} z={r.qdaZ} scale={'diverging'} range={RANGE} valueLabel={'P(class 1 | x)'} />
            <Points {...overlay[0]} live />
            <Points {...overlay[1]} live />
            <Handle {...state.handle(['ax', 'ay'], { label: 'mean of class 0' })} />
            <Handle {...state.handle(['bx', 'by'], { label: 'mean of class 1' })} />
          </Plot>
        </div>
      </div>
    </Figure>
  )
}
