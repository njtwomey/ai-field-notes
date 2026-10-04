/**
 * Differential privacy (`aifn/probability/privacy`): the error of a private mean against ε for the Laplace and
 * analytic Gaussian mechanisms and for randomised response; the total ε of many Gaussian releases under sequential,
 * advanced, Rényi and zero-concentrated accounting, with subsampling; and DP-SGD training a small MLP across noise
 * multipliers in the worker (`aifn-methods/neural/privacy`), test accuracy against the ε spent.
 */
import { useMemo, useState } from 'react'
import type { PrivateStudySnapshot } from 'aifn-methods/neural/privacy'
import { child, stream } from 'aifn/foundation/random'
import {
  advancedComposition,
  analyticGaussianSigma,
  gaussianEpsilon,
  gaussianMechanism,
  gaussianZcdp,
  laplaceMechanism,
  randomisedResponse,
  randomisedResponseEstimate,
  rdpSubsampledGaussian,
  rdpToEpsilon,
  zcdpToEpsilon,
} from 'aifn/probability/privacy'
import { Player, StatusText } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, float, int, row, slider, useComputed, useFigureState, useStreamed, type Task } from '@lab/state'
import { Button } from '@lab/ui/button'
import { Annotation, Curve, formatNumber, Histogram, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '∞')
const logGrid = (a: number, b: number, m: number) => Array.from({ length: m }, (_, i) => a * (b / a) ** (i / (m - 1)))

// ── 1 · ε against utility ────────────────────────────────────────────────────────────────────────────────────────────

const MECHANISMS = ['Laplace (central)', 'analytic Gaussian (central)', 'randomised response (local)'] as const

export function UtilitySpecimen() {
  const state = useFigureState({
    data: row('1 · survey', {
      n: int(1000, { ge: 10, le: 100000, suggestions: [100, 1000, 10000], label: 'people n' }),
      share: slider(0.05, 0.95, 0.3, { step: 0.01, label: 'true share of yes' }),
    }),
    privacy: row('2 · privacy', {
      epsilon: float(1, { gt: 0, scale: 'log10', suggestions: [0.1, 0.5, 1, 2, 5], label: 'ε shown below' }),
      delta: float(1e-5, { gt: 0, lt: 1, scale: 'log10', suggestions: [1e-3, 1e-5, 1e-7], label: 'δ (Gaussian)' }),
      releases: int(300, { ge: 20, le: 5000, suggestions: [100, 300, 1000], label: 'releases per ε' }),
    }),
  })
  const { n, share } = state.data
  const { epsilon, delta, releases } = state.privacy
  const eps = useMemo(() => logGrid(0.05, 10, 16), [])
  const bits = useMemo(() => {
    const ones = Math.round(share * n)
    return Float64Array.from({ length: n }, (_, i) => (i < ones ? 1 : 0))
  }, [n, share])
  const truth = bits.reduce((a, b) => a + b, 0) / n
  // Each mechanism releases the share of yes answers; the mean query has sensitivity 1/n.
  const release = (m: number, e: number, s: ReturnType<typeof stream>) => {
    if (m === 0) return laplaceMechanism([truth], 1 / n, e, s)[0]
    if (m === 1) return gaussianMechanism([truth], analyticGaussianSigma(1 / n, e, delta), s)[0]
    return randomisedResponseEstimate(randomisedResponse(bits, e, s), e)
  }
  const sweep = useComputed(
    () =>
      MECHANISMS.map((_, m) =>
        eps.map((e, k) => {
          const root = stream(`dp-utility-${m}-${k}`)
          let se = 0
          for (let r = 0; r < releases; r++) se += (release(m, e, child(root, r)) - truth) ** 2
          return Math.sqrt(se / releases)
        }),
      ),
    [n, share, delta, releases, eps],
    { mode: 'release' },
  )
  const draws = useComputed(
    () =>
      MECHANISMS.map((_, m) => {
        const root = stream(`dp-draws-${m}`)
        return Array.from({ length: releases }, (_, r) => release(m, epsilon, child(root, r)))
      }),
    [n, share, delta, releases, epsilon],
  )
  const laplaceSd = eps.map((e) => Math.SQRT2 / n / e)
  const gaussSd = eps.map((e) => analyticGaussianSigma(1 / n, e, delta))
  const eAxis = useAxis({ label: 'privacy budget ε', log: true, range: [0.01, 10] })
  const rAxis = useAxis({ label: 'RMSE of the released share', log: true, hold: 'union', key: `${n}-${delta}` })
  const vAxis = useAxis({
    label: `released share at ε = ${fmt(epsilon)}`,
    hold: 'union',
    key: `${n}-${share}-${epsilon}-${delta}`,
  })
  const cAxis = useAxis({ label: 'density', hold: 'union', key: `${n}-${share}-${epsilon}-${delta}` })
  const rmseAt = (m: number) => {
    const v = draws.value[m]
    return Math.sqrt(v.reduce((a, x) => a + (x - truth) ** 2, 0) / v.length)
  }
  return (
    <Figure
      title="Privacy costs accuracy: error against ε"
      purpose="Each mechanism adds noise scaled to how much one person can change the answer divided by ε, so halving ε doubles the error; a central curator adding noise once to the mean beats randomised response, where every person randomises their own answer, by a factor of order √n."
      state={state}
      defaultSize="L"
      readouts={{
        [`at ε = ${fmt(epsilon)}`]: (
          <>
            {MECHANISMS.map((name, m) => (
              <Readout key={name} label={`${name} RMSE`} value={fmt(rmseAt(m))} />
            ))}
            <Readout label="analytic Gaussian σ" value={fmt(analyticGaussianSigma(1 / n, epsilon, delta))} />
          </>
        ),
      }}
      caption={`A survey of ${n} yes/no answers with true share ${fmt(truth)}. Laplace and analytic Gaussian (at δ = ${delta}) release the share once with noise calibrated to its sensitivity 1/n; randomised response flips each answer with probability 1/(1 + e^ε) and debiases the average. Left: the RMSE over ${releases} seeded releases at each ε (lines) and the noise standard deviation the calibration predicts (dashed, central mechanisms). Right: the released values at the ε chosen above, with the truth (ink). Recomputed on release.`}
    >
      <Plots cols={2} widths={[55, 45]}>
        <Plot x={eAxis} y={rAxis}>
          {MECHANISMS.map((name, m) => (
            <Curve key={name} name={name} x={eps} y={sweep.value[m]} slot={m} showPoints stale={sweep.stale} />
          ))}
          <Curve name="Laplace sd √2/(nε)" x={eps} y={laplaceSd} slot={0} dashed width={1} silent />
          <Curve name="Gaussian σ" x={eps} y={gaussSd} slot={1} dashed width={1} silent />
          <Annotation x={epsilon} dashed text="ε" />
        </Plot>
        <Plot x={vAxis} y={cAxis}>
          {MECHANISMS.map((name, m) => (
            <Histogram key={name} name={name} values={draws.value[m]} slot={m} stale={draws.stale} />
          ))}
          <Annotation x={truth} text="truth" />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2 · Composition ──────────────────────────────────────────────────────────────────────────────────────────────────

export function CompositionSpecimen() {
  const state = useFigureState({
    mechanism: row('1 · each release', {
      sigma: slider(0.5, 50, 10, { step: 0.1, label: 'noise multiplier σ (Δ = 1)' }),
      q: float(0.01, { gt: 0, le: 1, scale: 'log10', suggestions: [0.001, 0.01, 0.1, 1], label: 'sampling rate q' }),
    }),
    target: row('2 · target', {
      delta: float(1e-5, { gt: 0, lt: 1, scale: 'log10', suggestions: [1e-3, 1e-5, 1e-7], label: 'total δ' }),
    }),
  })
  const { sigma, q } = state.mechanism
  const { delta } = state.target
  const ks = useMemo(() => logGrid(1, 10000, 25).map(Math.round), [])
  const curves = useComputed(() => {
    const perStep = rdpSubsampledGaussian(q, sigma, 1)
    const rdp = ks.map(
      (k) =>
        rdpToEpsilon(
          perStep.map((r) => r * k),
          delta,
        ).epsilon,
    )
    // Full-batch accounting (q = 1) for the classical theorems: each release (ε₀, δ₀)-DP at its exact profile.
    const sequential = ks.map((k) => k * gaussianEpsilon(1, sigma, delta / k))
    const advanced = ks.map(
      (k) => advancedComposition(gaussianEpsilon(1, sigma, delta / (2 * k)), delta / (2 * k), k, delta / 2).epsilon,
    )
    const zcdp = ks.map((k) => zcdpToEpsilon(k * gaussianZcdp(1, sigma), delta))
    const full = rdpSubsampledGaussian(1, sigma, 1)
    const rdpFull = ks.map(
      (k) =>
        rdpToEpsilon(
          full.map((r) => r * k),
          delta,
        ).epsilon,
    )
    return { rdp, sequential, advanced, zcdp, rdpFull }
  }, [sigma, q, delta, ks])
  const c = curves.value
  const kAxis = useAxis({ label: 'releases k', log: true, range: [1, 10000] })
  const epsAxis = useAxis({ label: `total ε at δ = ${delta}`, log: true, hold: 'union', key: `${sigma}-${q}-${delta}` })
  const at = (v: number[], k: number) => v[ks.indexOf(k)]
  return (
    <Figure
      title="Composition: how ε grows with the number of releases"
      purpose="Privacy losses add up over releases, but the plain sum k·ε₀ is pessimistic: advanced composition trades a √k term against a kε₀² one, so it helps only for many cheap releases; Rényi and zero-concentrated accounting of the Gaussian mechanism are tighter at every k; releasing on a random q-sample of the data shrinks each step's cost further (amplification by subsampling)."
      state={state}
      defaultSize="L"
      readouts={{
        'after 100 releases': (
          <>
            <Readout label="sequential k·ε₀" value={fmt(at(c.sequential, 100))} />
            <Readout label="advanced" value={fmt(at(c.advanced, 100))} />
            <Readout label="zCDP" value={fmt(at(c.zcdp, 100))} />
            <Readout label="RDP, full data" value={fmt(at(c.rdpFull, 100))} />
            <Readout label={`RDP, subsampled q = ${q}`} value={fmt(at(c.rdp, 100))} />
          </>
        ),
      }}
      caption={`Each release adds N(0, σ²) noise to a query of sensitivity 1; the total δ is fixed. Sequential composition charges every release its own ε₀ at δ/k (from the Gaussian mechanism's exact privacy profile); advanced composition (Dwork, Rothblum and Vadhan) at δ/2k each plus slack δ/2; zCDP adds ρ = 1/(2σ²) per release and converts once; Rényi DP adds each order's divergence and converts at the best order, as Opacus does. The last curve is RDP for the Poisson-subsampled Gaussian at rate q, the accounting behind DP-SGD; at q = 1 it equals the full-data curve. With little noise (σ near 1, ε₀ above 1) the kε₀(e^ε₀ − 1) term makes advanced composition worse than the plain sum. The subsampled curve's floor at small k is the conversion's limit at the largest order used (63), not the privacy loss itself.`}
    >
      <Plot x={kAxis} y={epsAxis}>
        <Curve name="sequential k·ε₀" x={ks} y={c.sequential} slot={0} stale={curves.stale} />
        <Curve name="advanced composition" x={ks} y={c.advanced} slot={1} stale={curves.stale} />
        <Curve name="zCDP" x={ks} y={c.zcdp} slot={2} stale={curves.stale} />
        <Curve name="Rényi DP (full data)" x={ks} y={c.rdpFull} slot={3} stale={curves.stale} />
        <Curve name={`Rényi DP, subsampled q = ${q}`} x={ks} y={c.rdp} slot={4} width={3} stale={curves.stale} />
      </Plot>
    </Figure>
  )
}

// ── 3 · DP-SGD ───────────────────────────────────────────────────────────────────────────────────────────────────────

type DpSetup = { n: number; noise: number; sigmas: string; steps: number; batch: number; clip: number; seed: number }

export function DpSgdSpecimen() {
  const state = useFigureState({
    data: row('1 · data', {
      n: int(500, { ge: 50, le: 5000, suggestions: [200, 500, 2000], label: 'training rows n' }),
      noise: slider(0, 0.5, 0.2, { step: 0.01, label: 'moons noise' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
    training: row('2 · DP-SGD', {
      steps: int(300, { ge: 1, le: 5000, suggestions: [100, 300, 1000], label: 'steps' }),
      batch: int(50, { ge: 1, le: 1000, suggestions: [10, 50, 200], label: 'expected batch qn' }),
      clip: float(1, { gt: 0, scale: 'log10', suggestions: [0.1, 1, 10], label: 'clipping norm C' }),
    }),
  })
  const current: DpSetup = {
    n: state.data.n,
    noise: state.data.noise,
    seed: state.data.seed,
    sigmas: '0,0.6,1,2,4',
    steps: state.training.steps,
    batch: state.training.batch,
    clip: state.training.clip,
  }
  const [trained, setTrained] = useState<DpSetup | null>(null)
  const stale = trained !== null && JSON.stringify(trained) !== JSON.stringify(current)
  const shown = trained ?? current
  const sigmas = shown.sigmas.split(',').map(Number)
  const task = useMemo((): Task<PrivateStudySnapshot> | null => {
    if (!trained) return null
    const data = (name: string, n: number) =>
      call('data/synthetic/moons', call('foundation/random/stream', `${name}-${trained.seed}`), {
        n,
        noise: trained.noise,
      })
    return call<PrivateStudySnapshot>(
      'applied/neural/privacy/privateTrainingStudy',
      data('dp-train', trained.n),
      data('dp-test', 1000),
      {
        noiseMultipliers: sigmas,
        hidden: 16,
        steps: trained.steps,
        batchSize: trained.batch,
        clipNorm: trained.clip,
        stepSize: 0.05,
        delta: 1 / (10 * trained.n),
        checkpoints: 30,
        seed: trained.seed,
      },
    )
  }, [trained]) // eslint-disable-line react-hooks/exhaustive-deps
  const run = useStreamed(task)
  const snap = run.value
  const [picked, setPicked] = useState<{ task: typeof task; index: number } | null>(null)
  const count = snap?.runs[0]?.at.length ?? 0
  const index = Math.min(picked?.task === task ? picked.index : 0, Math.max(0, count - 1))
  const progress = snap ? snap.done / Math.max(1, snap.total) : 0
  const eAxis = useAxis({ label: 'ε spent (δ = 1/10n)', log: true, range: [0.01, 1000] })
  const aAxis = useAxis({ label: 'test accuracy', range: [0.4, 1] })
  const sAxis = useAxis({ label: 'step', range: [0, shown.steps], key: shown.steps, integer: true })
  const lAxis = useAxis({ label: 'minibatch loss', hold: 'union', key: task })
  const baseline = snap?.runs.find((r) => r.noiseMultiplier === 0)
  const smooth = (v: number[]) => {
    const w = Math.max(1, Math.round(v.length / 60))
    return v.map((_, i) => {
      const part = v.slice(Math.max(0, i - w), i + 1).filter(Number.isFinite)
      return part.reduce((a, b) => a + b, 0) / Math.max(1, part.length)
    })
  }
  return (
    <Figure
      title="DP-SGD: accuracy against the privacy spent"
      purpose="DP-SGD clips each example's gradient to norm C and adds Gaussian noise σC to the sum, so one person's influence on every step is bounded and hidden; more noise buys a smaller ε at the price of accuracy, and ε grows with every step trained."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="3 · train">
          <div className="flex flex-wrap items-center gap-3">
            {run.running ? (
              <Button size="sm" variant="destructive" aria-label="Stop" onClick={run.stop}>
                Stop
              </Button>
            ) : (
              <Button
                size="sm"
                variant={!trained || stale ? 'default' : 'outline'}
                aria-label="Train"
                onClick={() => setTrained({ ...current })}
              >
                {trained ? 'Retrain' : 'Train'}
              </Button>
            )}
            <div className="h-1.5 w-40 overflow-hidden rounded bg-muted" aria-busy={run.running}>
              <div className="h-full bg-primary" style={{ width: `${100 * Math.min(1, progress)}%` }} />
            </div>
            <StatusText tone={trained && run.error ? 'error' : !trained || stale ? 'attention' : 'muted'}>
              {!trained
                ? 'Not trained yet: press Train to train one network per noise multiplier.'
                : run.error
                  ? `failed: ${run.error}`
                  : stale
                    ? 'Settings changed since this run: press Retrain.'
                    : run.stopped
                      ? `stopped after ${snap?.done ?? 0} steps`
                      : `${snap?.done ?? 0} / ${snap?.total ?? 0} steps over ${sigmas.length} runs${run.running ? '…' : ''}`}
            </StatusText>
          </div>
          <Player
            label="checkpoint"
            value={index}
            onChange={(i) => setPicked({ task, index: i })}
            count={Math.max(1, count)}
            format={(p) => `step ${snap?.runs[0]?.at[p] ?? 0}`}
          />
        </ControlRow>
      }
      readouts={{
        [`at step ${snap?.runs[0]?.at[index] ?? 0}`]: (
          <>
            {snap?.runs.map((r) => (
              <Readout
                key={r.noiseMultiplier}
                label={`σ = ${r.noiseMultiplier}`}
                value={
                  r.accuracy[index] === undefined ? '—' : `acc ${fmt(r.accuracy[index])}, ε ${fmt(r.epsilon[index])}`
                }
              />
            )) ?? <Readout label="runs" value="—" />}
          </>
        ),
      }}
      caption={`Data: aifn moons (seeded), ${shown.n} training and 1000 test points. One 2 → 16 → 1 tanh MLP, from the same initial weights, is trained by aifn privateTraining (Poisson sampling with q = ${shown.batch}/${shown.n}, per-example gradients by vmap(grad), clipping at C = ${shown.clip}, Adam) once per noise multiplier σ ∈ {${sigmas.join(', ')}}; σ = 0 clips without noise and has no privacy (ε = ∞, the dashed line). Left: each run's test accuracy against the ε its steps have spent (Rényi-DP accounting at δ = 1/10n), one point per checkpoint, the player's checkpoint marked. Right: the minibatch loss (smoothed). Press Train; the player opens at the untrained network.`}
    >
      <Plots cols={2}>
        <Plot x={eAxis} y={aAxis}>
          {snap?.runs
            .filter((r) => r.noiseMultiplier > 0)
            .map((r) => (
              <Curve
                key={r.noiseMultiplier}
                name={`σ = ${r.noiseMultiplier}`}
                x={r.epsilon.slice(1)}
                y={r.accuracy.slice(1)}
                slot={sigmas.indexOf(r.noiseMultiplier)}
                showPoints
              />
            ))}
          {baseline && (
            <Annotation
              y={baseline.accuracy[Math.min(index, baseline.accuracy.length - 1)]}
              dashed
              text="σ = 0 (no privacy)"
            />
          )}
          {snap && (
            <Points
              name="checkpoint"
              x={snap.runs.filter((r) => r.noiseMultiplier > 0 && r.epsilon[index] > 0).map((r) => r.epsilon[index])}
              y={snap.runs.filter((r) => r.noiseMultiplier > 0 && r.epsilon[index] > 0).map((r) => r.accuracy[index])}
              emphasis
              live
            />
          )}
        </Plot>
        <Plot x={sAxis} y={lAxis}>
          {snap?.runs.map((r) => (
            <Curve
              key={r.noiseMultiplier}
              name={`σ = ${r.noiseMultiplier}`}
              x={r.loss.map((_, t) => t + 1)}
              y={smooth(r.loss)}
              slot={sigmas.indexOf(r.noiseMultiplier)}
            />
          ))}
          {snap && <Annotation x={snap.runs[0].at[index]} dashed />}
        </Plot>
      </Plots>
    </Figure>
  )
}
