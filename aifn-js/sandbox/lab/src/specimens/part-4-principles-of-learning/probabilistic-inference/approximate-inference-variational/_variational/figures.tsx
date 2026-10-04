import { banana, gaussianMixtureTarget, gaussianTarget } from 'aifn-methods/data/targets'
import { type LogDensity } from 'aifn/inference/stochastic'
import { child, normal, stream } from 'aifn/foundation/random'
import { histogram } from 'aifn/probability/stats'
import { linspace, tensor, toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import {
  bbvi,
  elbo,
  fullRankGaussian,
  gradientVariance,
  meanFieldGaussian,
  type Baseline,
  type GradientEstimator,
} from 'aifn/inference/variational'
import { caviGaussianMixture, mixturePredictiveDensity } from 'aifn-methods/inference/mixture-models'
import { caviNormalGamma, normalGammaPosterior } from 'aifn-methods/inference/conjugate-models'
import { useMemo, useState } from 'react'
import { Player } from '@lab/controls'
import { Figure } from '@lab/layout'
import { choice, row, slider, useComputed, useFigureState } from '@lab/state'
import { Bars, Contours, Curve, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'
import { ElboPanel, formatValue, histogramBars } from '@lab/views'

const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))
const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))

type Box = { x: [number, number]; y: [number, number] }
const TARGET_NAMES = ['correlated Gaussian', 'banana', 'two-mode mixture'] as const
const TARGETS: Record<string, { target: LogDensity; box: Box }> = {
  'correlated Gaussian': {
    target: gaussianTarget(
      [0.5, -0.5],
      [
        [1, 0.85],
        [0.85, 1],
      ],
    ),
    box: { x: [-3.5, 4], y: [-4, 3.5] },
  },
  banana: { target: banana({ a: 1, b: 0.7 }), box: { x: [-3.5, 3.5], y: [-2.5, 6] } },
  'two-mode mixture': {
    target: gaussianMixtureTarget(
      [
        [-1.8, -1],
        [1.8, 1],
      ],
      0.7,
    ),
    box: { x: [-4.5, 4.5], y: [-3.5, 3.5] },
  },
}

// ── Black-box VI ───────────────────────────────────────────────────────────────────────────────────────────────────

const MEAN = meanFieldGaussian(2)
const FULL = fullRankGaussian(2)
const familyOf = (kind: 'mean-field' | 'full-rank') => (kind === 'full-rank' ? FULL : MEAN)

const STEPS = 600
const EVERY = 10

export function BlackBoxVi() {
  const state = useFigureState({
    model: row('1 · target and family', {
      which: choice(TARGET_NAMES, 'correlated Gaussian', { label: 'target' }),
      family: choice(['mean-field', 'full-rank'], 'mean-field', { label: 'family' }),
    }),
    optimiser: row('2 · gradient estimator and optimiser', {
      estimator: choice(['reparameterisation', 'score'], 'reparameterisation', { label: 'estimator' }),
      samples: slider(1, 50, 4, { label: 'draws per gradient S', step: 1 }),
      lr: slider(0.005, 0.2, 0.05, { label: 'Adam learning rate η₀ (η₀/(1 + t/100))', step: 0.005 }),
    }),
  })
  const { which, family } = state.model
  const { samples, lr } = state.optimiser
  const estimator = state.optimiser.estimator as GradientEstimator
  const [at, setAt] = useState(0)
  const spec = TARGETS[which]
  const surface = useMemo(() => {
    const x = grid(spec.box.x[0], spec.box.x[1], 80)
    const y = grid(spec.box.y[0], spec.box.y[1], 80)
    return { x, y, z: y.map((yi) => x.map((xj) => Math.exp(spec.target.logDensity(tensor([xj, yi])) as number))) }
  }, [spec])
  // 600 Adam steps plus a 500-draw ELBO at every kept step: rerun on release.
  const run = useComputed(
    () => {
      const tr = trace(
        bbvi(spec.target, { family, estimator, samples, stepSize: (t: number) => lr / (1 + t / 100) }),
        { mean0: [0, 0], sd0: 1 },
        STEPS,
        { stream: stream('bbvi'), every: EVERY, record: { elbo: (s) => s.elbo } },
      )
      // A low-noise ELBO at every kept step (500 draws each; the targets are normalised, so log Z = 0 and −ELBO = KL).
      const accurate = tr.steps.map(
        (s, i) => elbo(child(stream('elbo'), i), spec.target, familyOf(family), s.lambda, { samples: 500 }).value,
      )
      return { tr, accurate, noisy: toFlat(tr.series.elbo) }
    },
    [spec, family, estimator, samples, lr],
    { mode: 'release' },
  )
  const { tr, accurate, noisy } = run.value
  const k = Math.min(at, tr.steps.length - 1)
  const st = tr.steps[k]
  const qField = useMemo(
    () => surface.y.map((yi) => surface.x.map((xj) => Math.exp(familyOf(family).logDensity(st.lambda, [xj, yi])))),
    [surface, st, family],
  )
  const levels = useMemo(() => {
    const peak = Math.max(...qField.flat())
    return [0.1 * peak, 0.5 * peak, 0.9 * peak]
  }, [qField])
  const meanPath = useMemo(() => {
    const ms = tr.steps.slice(0, k + 1).map((s) => toFlat(s.mean))
    return { x: ms.map((m) => m[0]), y: ms.map((m) => m[1]) }
  }, [tr, k])
  const tx = useAxis({ label: 'θ₀', range: spec.box.x })
  const ty = useAxis({ label: 'θ₁', range: spec.box.y, equal: tx })
  const sx = useAxis({ label: 'step', range: [0, STEPS] })
  const ey = useAxis({ label: 'ELBO', key: `${which}${family}${estimator}`, hold: 'union' })
  return (
    <Figure
      title="Black-box VI fits a Gaussian to a posterior"
      purpose="Adam on stochastic ELBO gradients moves q towards the target; minimising KL(q ‖ p) makes q under-cover: too narrow along correlations with mean field, and on one mode of a mixture."
      defaultSize="L"
      state={state}
      controls={
        <Player
          label="3 · step"
          value={k}
          onChange={setAt}
          count={tr.steps.length}
          format={(i) => String(tr.index[i])}
        />
      }
      readouts={{
        'at this step': (
          <>
            <Readout label="ELBO (500 draws)" value={f3(accurate[k])} />
            <Readout label="KL(q ‖ p) = −ELBO" value={f3(-accurate[k])} />
            <Readout label="mean of q" value={toFlat(st.mean).map(f3).join(', ')} />
          </>
        ),
      }}
      caption="Play from the initial q = N(0, I). Left: the target density with contours of q at 10%, 50% and 90% of its peak and the path of q's mean. Right: the noisy ELBO the optimiser sees and a 500-draw estimate every 10 steps; the targets are normalised, so the gap to 0 is KL(q ‖ p). With the score estimator and S = 4 the path wanders; the reparameterisation estimator converges with S = 1."
    >
      <Plots cols={2} widths={[1.2, 1]}>
        <Plot x={tx} y={ty}>
          <Raster {...surface} valueLabel="p" colorBar={false} fillOpacity={0.7} />
          <Contours x={surface.x} y={surface.y} z={qField} levels={levels} labels={false} stale={run.stale} />
          <Curve name="mean of q" {...meanPath} slot={1} stale={run.stale} />
        </Plot>
        <Plot x={sx} y={ey}>
          <Curve name="ELBO (optimiser’s estimate)" x={tr.index} y={noisy} slot={0} thin stale={run.stale} />
          <Curve name="ELBO (500 draws)" x={tr.index} y={accurate} slot={1} stale={run.stale} />
          <Curve name="log Z = 0" x={[0, STEPS]} y={[0, 0]} emphasis dashed />
          <Points name="current" x={[tr.index[k]]} y={[accurate[k]]} emphasis live />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── Gradient variance ──────────────────────────────────────────────────────────────────────────────────────────────

const S_GRID = [1, 2, 3, 4, 8, 16, 32, 64]
const ESTIMATORS: { name: string; estimator: GradientEstimator; baseline: Baseline; min: number }[] = [
  { name: 'reparameterisation', estimator: 'reparameterisation', baseline: 'none', min: 1 },
  { name: 'score, no baseline', estimator: 'score', baseline: 'none', min: 1 },
  { name: 'score, leave-one-out baseline', estimator: 'score', baseline: 'leave-one-out', min: 2 },
  { name: 'score, control variate', estimator: 'score', baseline: 'control-variate', min: 8 },
]

export function GradientVariance() {
  const state = useFigureState({
    setup: row('1 · target and q', {
      which: choice(TARGET_NAMES, 'correlated Gaussian', { label: 'target' }),
      shift: slider(0, 3, 1, { label: "offset of q's mean (±s, ∓s)", step: 0.1 }),
    }),
  })
  const { which, shift } = state.setup
  // 100 repeats × 8 draw counts × 4 estimators: rerun on release.
  const computed = useComputed(
    () => {
      const target = TARGETS[which].target
      const lambda = MEAN.parameters([shift, -shift], 1)
      return ESTIMATORS.map((e, slot) => {
        const xs = S_GRID.filter((S) => S >= e.min)
        return {
          name: e.name,
          slot,
          x: xs,
          y: xs.map(
            (S) =>
              gradientVariance(child(stream('grad-var'), e.name, S), target, MEAN, lambda, {
                estimator: e.estimator,
                baseline: e.baseline,
                samples: S,
                repeats: 100,
              }).totalVariance,
          ),
        }
      })
    },
    [which, shift],
    { mode: 'release' },
  )
  const x = useAxis({ label: 'draws per estimate S', log: true })
  const y = useAxis({ label: 'total variance', log: true })
  return (
    <Figure
      title="The variance of the two ELBO gradient estimators"
      purpose="Both estimators are unbiased, but the reparameterisation gradient uses ∇ log p and has a variance several times below the score function's at every S; baselines close part of the gap."
      state={state}
      caption="Total variance Σᵢ Var ĝᵢ of each estimator of the ELBO gradient at a mean-field q = N((s, −s), I), from 100 repeats per point, against the number of draws S (both axes logarithmic). Every line falls like 1/S; the offsets between them are the point. The leave-one-out baseline needs S ≥ 2; the control variate estimates its scale from the other S − 1 draws, which is unstable below S = 8, so it starts there."
    >
      <Plot x={x} y={y}>
        {computed.value.map((l) => (
          <Curve key={l.name} name={l.name} x={l.x} y={l.y} slot={l.slot} showPoints stale={computed.stale} />
        ))}
      </Plot>
    </Figure>
  )
}

// ── CAVI on a Gaussian mixture ─────────────────────────────────────────────────────────────────────────────────────

const ITERS = 80

const XS = grid(-7, 9, 321)

export function CaviMixture() {
  const state = useFigureState({
    setup: row('1 · data and model', {
      n: slider(30, 1000, 300, { label: 'data points', step: 10 }),
      K: slider(1, 10, 6, { label: 'components K', step: 1 }),
      alpha0: choice([0.001, 0.1, 1, 10], 0.001, { label: 'Dirichlet concentration α₀' }),
    }),
  })
  const { n, K, alpha0 } = state.setup
  const [at, setAt] = useState(0)
  const data = useMemo(() => {
    const s = stream('mixture-data')
    const centres = [-3, 0.5, 4]
    const sds = [0.7, 0.5, 1]
    return Array.from({ length: n }, (_, i) => normal(child(s, i), centres[i % 3], sds[i % 3]))
  }, [n])
  const tr = useMemo(
    () =>
      trace(caviGaussianMixture(data, K, { alpha0 }), undefined, ITERS, {
        stream: stream('cavi-init'),
        record: { elbo: (s) => s.elbo },
      }),
    [data, K, alpha0],
  )
  const i = Math.min(at, tr.steps.length - 1)
  const st = tr.steps[i]
  const hist = useMemo(() => histogramBars(histogram(data, { bins: 50, range: [-7, 9] })), [data])
  const top = useMemo(() => {
    const pred = mixturePredictiveDensity(st, XS)
    const comps = toFlat(pred.components)
    const w = toFlat(st.weights)
    const components: { name: string; slot: number; y: number[] }[] = []
    for (let k = 0; k < K; k++)
      if (w[k] > 0.005)
        components.push({
          name: `component ${k} (E[π] = ${f3(w[k])})`,
          slot: k % 8,
          y: comps.slice(k * XS.length, (k + 1) * XS.length),
        })
    return { density: toFlat(pred.density), components }
  }, [st, K])
  const elbos = useMemo(() => toFlat(tr.series.elbo), [tr])
  const used = toFlat(st.weights).filter((w) => w > 0.01).length
  const x = useAxis({ label: 'x', range: [-7, 9] })
  const d = useAxis({ label: 'density', hold: 'union', key: n })
  const it = useAxis({ label: 'iteration' })
  const ey = useAxis({ label: 'ELBO' })
  return (
    <Figure
      title="CAVI on a Bayesian Gaussian mixture"
      purpose="Coordinate ascent alternates responsibilities and component posteriors; the ELBO rises at every iteration, and with a small Dirichlet concentration α₀ surplus components empty out."
      defaultSize="L"
      state={state}
      controls={<Player label="2 · iteration" value={i} onChange={setAt} count={tr.steps.length} />}
      readouts={{
        'at this iteration': (
          <>
            <Readout label="ELBO" value={f3(st.elbo)} />
            <Readout label="components with E[π] > 0.01" value={used} />
          </>
        ),
        run: <Readout label="stopped" value={`${tr.meta.stopped} after ${tr.meta.steps} iterations`} />,
      }}
      caption="Data from three Gaussians (means −3, 0.5, 4). Play from the random initialisation. Top: the data histogram, the predictive density (a mixture of Student t, Bishop eq. 10.81) and each component weighted by E[πₖ], at the player's iteration. Bottom: the ELBO, which never decreases. With α₀ = 0.001 and K = 6, three components carry the data; with α₀ = 10 all six stay in use."
    >
      <Plots rows={2} heights={[2, 1.2]}>
        <Plot x={x} y={d}>
          <Bars name="data" x={hist.x} y={hist.density} edges={hist.edges} muted />
          <Curve name="predictive density" x={XS} y={top.density} emphasis />
          {top.components.map((c) => (
            <Curve key={c.name} name={c.name} x={XS} y={c.y} slot={c.slot} />
          ))}
        </Plot>
        <Plot x={it} y={ey}>
          <Curve name="ELBO" x={tr.index} y={elbos} slot={0} showPoints />
          <Points name="current" x={[tr.index[i]]} y={[st.elbo]} emphasis live />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── CAVI for the Gaussian with unknown mean and precision ───────────────────────────────────────────────────────────

export function CaviNormalGamma() {
  const state = useFigureState({
    data: row('1 · data', { n: slider(2, 100, 10, { label: 'data points N', step: 1 }) }),
    start: row('2 · start', { tau0: slider(0.01, 5, 0.05, { label: 'starting E[τ]', step: 0.01 }) }),
  })
  const { n } = state.data
  const { tau0 } = state.start
  const x = useMemo(() => Array.from({ length: n }, (_, i) => normal(child(stream('ng'), i), 1.5, 0.8)), [n])
  const exact = useMemo(() => normalGammaPosterior(x), [x])
  const tr = useMemo(
    () => trace(caviNormalGamma(x), { expectedTau0: tau0 }, 30, { record: { elbo: (s) => s.elbo } }),
    [x, tau0],
  )
  const last = tr.steps[tr.steps.length - 1]
  const path = useMemo(() => {
    const pts = tr.steps.flatMap((s) => [
      [s.halfStep.muMean, s.halfStep.expectedTau],
      [s.muMean, s.expectedTau],
    ])
    return { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) }
  }, [tr])
  const elbos = useMemo(() => toFlat(tr.series.elbo), [tr])
  const mu = useAxis({ label: 'E[μ]' })
  const tau = useAxis({ label: 'E[τ]' })
  return (
    <>
      <Figure
        title="CAVI for a Gaussian with unknown mean and precision"
        purpose="The mean-field updates of q(μ) and q(τ) alternate along the axes of (E[μ], E[τ]) and settle on the exact posterior means."
        state={state}
        readouts={{
          'q against the exact posterior': (
            <>
              <Readout label="E_q[μ] / exact" value={`${f3(last.muMean)} / ${f3(exact.meanOfMu)}`} />
              <Readout label="Var_q[μ] / exact" value={`${f3(1 / last.muPrecision)} / ${f3(exact.varianceOfMu)}`} />
              <Readout label="E_q[τ] / exact" value={`${f3(last.expectedTau)} / ${f3(exact.meanOfTau)}`} />
            </>
          ),
        }}
        caption="Data: N draws from N(1.5, 0.8²); prior μ₀ = 0, λ₀ = 1, a₀ = b₀ = 1. The first update sets E[μ] (horizontal move), the second E[τ] (vertical move). The factorised q matches E[μ] exactly and under-states Var[μ], most visibly for small N."
      >
        <Plot x={mu} y={tau}>
          <Curve name="CAVI (E[μ], E[τ])" {...path} slot={0} showPoints />
          <Points name="exact posterior means" x={[exact.meanOfMu]} y={[exact.meanOfTau]} emphasis />
        </Plot>
      </Figure>
      <Figure
        title="Its ELBO rises to log p(x)"
        purpose="The ELBO climbs to just below log p(x); the gap is KL(q ‖ p(μ, τ | x)), which stays positive because the posterior does not factorise."
      >
        <ElboPanel elbo={elbos} steps={tr.index} logEvidence={exact.logEvidence} />
      </Figure>
    </>
  )
}
