import { banana, gaussianMixtureTarget, gaussianTarget, type Target } from 'aifn/mcmc'
import { normal, stream } from 'aifn/random'
import { histogram } from 'aifn/stats'
import { linspace, tensor, toFlat } from 'aifn/tensor'
import { trace } from 'aifn/trace'
import {
  bbvi,
  caviGaussianMixture,
  caviNormalGamma,
  elbo,
  fullRankGaussian,
  gradientVariance,
  meanFieldGaussian,
  mixturePredictiveDensity,
  normalGammaPosterior,
  type Baseline,
  type GradientEstimator,
} from 'aifn/vi'
import { useMemo, useState } from 'react'
import { Player, Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Heatmap, Panel, Readout, Subplots, XYChart, type HeatmapOverlay, type XYSeries } from '@lab/viz'
import { ElboView, formatValue } from '@lab/views'

const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))
const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))

type Box = { x: [number, number]; y: [number, number] }
const TARGETS: Record<string, { target: Target; box: Box }> = {
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
  const [which, setWhich] = useState('correlated Gaussian')
  const [family, setFamily] = useState<'mean-field' | 'full-rank'>('mean-field')
  const [estimator, setEstimator] = useState<GradientEstimator>('reparameterisation')
  const [samples, setSamples] = useState(4)
  const [lr, setLr] = useState(0.05)
  const [at, setAt] = useState(STEPS)
  const spec = TARGETS[which]
  const surface = useMemo(() => {
    const x = grid(spec.box.x[0], spec.box.x[1], 80)
    const y = grid(spec.box.y[0], spec.box.y[1], 80)
    return { x, y, z: y.map((yi) => x.map((xj) => Math.exp(spec.target.logDensity(tensor([xj, yi])) as number))) }
  }, [spec])
  const tr = useMemo(
    () =>
      trace(
        bbvi(spec.target, { family, estimator, samples, lr: (t: number) => lr / (1 + t / 100) }),
        { mean0: [0, 0], sd0: 1 },
        STEPS,
        {
          stream: stream('bbvi'),
          every: EVERY,
          record: { elbo: (s) => s.elbo },
        },
      ),
    [spec, family, estimator, samples, lr],
  )
  const k = Math.min(Math.round(at / EVERY), tr.steps.length - 1)
  const state = tr.steps[k]
  // A low-noise ELBO at every kept step (500 draws each; the targets are normalised, so log Z = 0 and −ELBO = KL).
  const accurate = useMemo(
    () =>
      tr.steps.map(
        (s, i) => elbo(spec.target, familyOf(family), s.lambda, stream('elbo').child(i), { samples: 500 }).value,
      ),
    [tr, spec, family],
  )
  const qField = useMemo(
    () => surface.y.map((yi) => surface.x.map((xj) => Math.exp(familyOf(family).logDensity(state.lambda, [xj, yi])))),
    [surface, state, family],
  )
  const contours = useMemo(() => {
    const peak = Math.max(...qField.flat())
    return { levels: [0.1 * peak, 0.5 * peak, 0.9 * peak], field: qField }
  }, [qField])
  const meanPath = useMemo<HeatmapOverlay[]>(() => {
    const ms = tr.steps.slice(0, k + 1).map((s) => toFlat(s.mean))
    return [{ name: 'mean of q', type: 'line', x: ms.map((m) => m[0]), y: ms.map((m) => m[1]), slot: 1 }]
  }, [tr, k])
  const elboSeries = useMemo<XYSeries[]>(() => {
    const x = tr.index
    return [
      { name: 'ELBO (optimiser’s estimate)', type: 'line', x, y: toFlat(tr.series.elbo), slot: 0, thin: true },
      { name: 'ELBO (500 draws)', type: 'line', x, y: accurate, slot: 1 },
      { name: 'log Z = 0', type: 'line', x: [0, STEPS], y: [0, 0], emphasis: true, dashed: true },
      { name: 'current', type: 'scatter', x: [tr.index[k]], y: [accurate[k]], emphasis: true },
    ]
  }, [tr, accurate, k])
  return (
    <Figure
      title="Black-box VI fits a Gaussian to a posterior"
      description="Adam on stochastic ELBO gradients moves q towards the target; minimising KL(q ‖ p) makes q under-cover: too narrow along correlations with mean field, and on one mode of a mixture."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · target and family">
            <Select label="target" value={which} onChange={setWhich} options={Object.keys(TARGETS)} />
            <Select label="family" value={family} onChange={setFamily} options={['mean-field', 'full-rank']} />
          </ControlRow>
          <ControlRow label="2 · gradient estimator and optimiser">
            <Select
              label="estimator"
              value={estimator}
              onChange={setEstimator}
              options={['reparameterisation', 'score']}
            />
            <Slider label="draws per gradient S" value={samples} onChange={setSamples} min={1} max={50} step={1} />
            <Slider
              label="Adam learning rate η₀ (η₀/(1 + t/100))"
              value={lr}
              onChange={setLr}
              min={0.005}
              max={0.2}
              step={0.005}
            />
          </ControlRow>
          <ControlRow label="3 · step">
            <Player value={at} onChange={setAt} count={STEPS + 1} />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="ELBO (500 draws)" value={f3(accurate[k])} />
          <Readout label="KL(q ‖ p) = −ELBO" value={f3(-accurate[k])} />
          <Readout label="mean of q" value={toFlat(state.mean).map(f3).join(', ')} />
        </>
      }
      caption="Top: the target density with contours of q at 10%, 50% and 90% of its peak and the path of q's mean. Bottom: the noisy ELBO the optimiser sees and a 500-draw estimate every 10 steps; the targets are normalised, so the gap to 0 is KL(q ‖ p). With the score estimator and S = 4 the path wanders; the reparameterisation estimator converges with S = 1."
    >
      <Subplots rows={2} heightRatios={[2, 1.2]}>
        <Panel>
          <Heatmap
            {...surface}
            xLabel="θ₀"
            yLabel="θ₁"
            valueLabel="p"
            colorBar={false}
            fillOpacity={0.7}
            equalAspect
            contours={contours}
            overlay={meanPath}
          />
        </Panel>
        <Panel>
          <XYChart series={elboSeries} xLabel="step" yLabel="ELBO" />
        </Panel>
      </Subplots>
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
  const [which, setWhich] = useState('correlated Gaussian')
  const [shift, setShift] = useState(1)
  const series = useMemo<XYSeries[]>(() => {
    const target = TARGETS[which].target
    const lambda = MEAN.parameters([shift, -shift], 1)
    return ESTIMATORS.map((e, slot) => {
      const xs = S_GRID.filter((S) => S >= e.min)
      return {
        name: e.name,
        type: 'line',
        slot,
        showPoints: true,
        x: xs,
        y: xs.map(
          (S) =>
            gradientVariance(target, MEAN, lambda, stream('grad-var').child(e.name, S), {
              estimator: e.estimator,
              baseline: e.baseline,
              samples: S,
              repeats: 100,
            }).totalVariance,
        ),
      }
    })
  }, [which, shift])
  return (
    <Figure
      title="The variance of the two ELBO gradient estimators"
      description="Both estimators are unbiased, but the reparameterisation gradient uses ∇ log p and has a variance several times below the score function's at every S; baselines close part of the gap."
      controls={
        <>
          <Select label="target" value={which} onChange={setWhich} options={Object.keys(TARGETS)} />
          <Slider label="offset of q's mean (±s, ∓s)" value={shift} onChange={setShift} min={0} max={3} step={0.1} />
        </>
      }
      caption="Total variance Σᵢ Var ĝᵢ of each estimator of the ELBO gradient at a mean-field q = N((s, −s), I), from 100 repeats per point, against the number of draws S (both axes logarithmic). Every line falls like 1/S; the offsets between them are the point. The leave-one-out baseline needs S ≥ 2; the control variate estimates its scale from the other S − 1 draws, which is unstable below S = 8, so it starts there."
    >
      <XYChart series={series} xLog yLog xLabel="draws per estimate S" yLabel="total variance" />
    </Figure>
  )
}

// ── CAVI on a Gaussian mixture ─────────────────────────────────────────────────────────────────────────────────────

const ITERS = 80

export function CaviMixture() {
  const [K, setK] = useState(6)
  const [alpha0, setAlpha0] = useState('0.001')
  const [n, setN] = useState(300)
  const [at, setAt] = useState(ITERS)
  const data = useMemo(() => {
    const s = stream('mixture-data')
    const centres = [-3, 0.5, 4]
    const sds = [0.7, 0.5, 1]
    return Array.from({ length: n }, (_, i) => normal(s.child(i), centres[i % 3], sds[i % 3]))
  }, [n])
  const tr = useMemo(
    () =>
      trace(caviGaussianMixture(data, K, { alpha0: Number(alpha0) }), {}, ITERS, {
        stream: stream('cavi-init'),
        record: { elbo: (s) => s.elbo },
      }),
    [data, K, alpha0],
  )
  const i = Math.min(at, tr.steps.length - 1)
  const state = tr.steps[i]
  const xs = useMemo(() => grid(-7, 9, 321), [])
  const top = useMemo<XYSeries[]>(() => {
    const h = histogram(data, { bins: 50, range: [-7, 9] })
    const pred = mixturePredictiveDensity(state, xs)
    const comps = toFlat(pred.components)
    const w = toFlat(state.weights)
    const out: XYSeries[] = [
      {
        name: 'data',
        type: 'bar',
        thin: true,
        muted: true,
        x: Array.from(h.counts, (_, k) => (h.edges[k] + h.edges[k + 1]) / 2),
        y: Array.from(h.density),
      },
      { name: 'predictive density', type: 'line', x: xs, y: toFlat(pred.density), emphasis: true },
    ]
    for (let k = 0; k < K; k++)
      if (w[k] > 0.005)
        out.push({
          name: `component ${k} (E[π] = ${f3(w[k])})`,
          type: 'line',
          x: xs,
          y: comps.slice(k * xs.length, (k + 1) * xs.length),
          slot: k % 8,
        })
    return out
  }, [data, state, xs, K])
  const bottom = useMemo<XYSeries[]>(
    () => [
      { name: 'ELBO', type: 'line', x: tr.index, y: toFlat(tr.series.elbo), slot: 0, showPoints: true },
      { name: 'current', type: 'scatter', x: [tr.index[i]], y: [state.elbo], emphasis: true },
    ],
    [tr, i, state],
  )
  const used = toFlat(state.weights).filter((w) => w > 0.01).length
  return (
    <Figure
      title="CAVI on a Bayesian Gaussian mixture"
      description="Coordinate ascent alternates responsibilities and component posteriors; the ELBO rises at every iteration, and with a small Dirichlet concentration α₀ surplus components empty out."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · data and model">
            <Slider label="data points" value={n} onChange={setN} min={30} max={1000} step={10} />
            <Slider label="components K" value={K} onChange={setK} min={1} max={10} step={1} />
            <Select
              label="Dirichlet concentration α₀"
              value={alpha0}
              onChange={setAlpha0}
              options={['0.001', '0.1', '1', '10']}
            />
          </ControlRow>
          <ControlRow label="2 · iteration">
            <Player value={at} onChange={setAt} count={ITERS + 1} />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="ELBO" value={f3(state.elbo)} />
          <Readout label="components with E[π] > 0.01" value={used} />
          <Readout label="stopped" value={`${tr.meta.stopped} after ${tr.meta.steps} iterations`} />
        </>
      }
      caption="Data from three Gaussians (means −3, 0.5, 4). Top: the data histogram, the predictive density (a mixture of Student t, Bishop eq. 10.81) and each component weighted by E[πₖ], at the player's iteration. Bottom: the ELBO, which never decreases. With α₀ = 0.001 and K = 6, three components carry the data; with α₀ = 10 all six stay in use."
    >
      <Subplots rows={2} heightRatios={[2, 1.2]}>
        <Panel>
          <XYChart series={top} xLabel="x" yLabel="density" rescaleOnChange={false} axisKey={`${n}`} holdFit="union" />
        </Panel>
        <Panel>
          <XYChart series={bottom} xLabel="iteration" yLabel="ELBO" />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ── CAVI for the Gaussian with unknown mean and precision ───────────────────────────────────────────────────────────

export function CaviNormalGamma() {
  const [n, setN] = useState(10)
  const [tau0, setTau0] = useState(0.05)
  const x = useMemo(() => Array.from({ length: n }, (_, i) => normal(stream('ng').child(i), 1.5, 0.8)), [n])
  const exact = useMemo(() => normalGammaPosterior(x), [x])
  const tr = useMemo(
    () => trace(caviNormalGamma(x), { expectedTau0: tau0 }, 30, { record: { elbo: (s) => s.elbo } }),
    [x, tau0],
  )
  const last = tr.steps[tr.steps.length - 1]
  const path = useMemo<XYSeries[]>(() => {
    const pts = tr.steps.flatMap((s) => [
      [s.halfStep.muMean, s.halfStep.expectedTau],
      [s.muMean, s.expectedTau],
    ])
    return [
      {
        name: 'CAVI (E[μ], E[τ])',
        type: 'line',
        x: pts.map((p) => p[0]),
        y: pts.map((p) => p[1]),
        slot: 0,
        showPoints: true,
      },
      { name: 'exact posterior means', type: 'scatter', x: [exact.meanOfMu], y: [exact.meanOfTau], emphasis: true },
    ]
  }, [tr, exact])
  return (
    <>
      <Figure
        title="CAVI for a Gaussian with unknown mean and precision"
        description="The mean-field updates of q(μ) and q(τ) alternate along the axes of (E[μ], E[τ]) and settle on the exact posterior means."
        controls={
          <>
            <Slider label="data points N" value={n} onChange={setN} min={2} max={100} step={1} />
            <Slider label="starting E[τ]" value={tau0} onChange={setTau0} min={0.01} max={5} step={0.01} />
          </>
        }
        readouts={
          <>
            <Readout label="E_q[μ] / exact" value={`${f3(last.muMean)} / ${f3(exact.meanOfMu)}`} />
            <Readout label="Var_q[μ] / exact" value={`${f3(1 / last.muPrecision)} / ${f3(exact.varianceOfMu)}`} />
            <Readout label="E_q[τ] / exact" value={`${f3(last.expectedTau)} / ${f3(exact.meanOfTau)}`} />
          </>
        }
        caption="Data: N draws from N(1.5, 0.8²); prior μ₀ = 0, λ₀ = 1, a₀ = b₀ = 1. The first update sets E[μ] (horizontal move), the second E[τ] (vertical move). The factorised q matches E[μ] exactly and under-states Var[μ], most visibly for small N."
      >
        <XYChart series={path} xLabel="E[μ]" yLabel="E[τ]" />
      </Figure>
      <ElboView
        title="Its ELBO rises to log p(x)"
        description="The gap to the exact log evidence is KL(q ‖ p(μ, τ | x)), which stays positive because the posterior does not factorise."
        elbo={toFlat(tr.series.elbo)}
        steps={tr.index}
        logEvidence={exact.logEvidence}
      />
    </>
  )
}
