/**
 * Off-policy evaluation: logged contextual-bandit feedback from `aifn-methods/data` (`banditProblem`, `logBandit`),
 * whose target policy has a known true value, evaluated by IPS, clipped IPS, SNIPS, the direct method and doubly
 * robust from `aifn/learning/off-policy`. Logs are redrawn from seeded child streams, so the spread of each estimator
 * across logs is its sampling distribution. The page computes summaries of estimates only.
 */
import { useMemo } from 'react'
import { banditProblem, logBandit, type BanditProblem } from 'aifn-methods/data/synthetic'
import { child, stream } from 'aifn/foundation/random'
import {
  clippedIps,
  directMethod,
  doublyRobust,
  ips,
  snips,
  switchDoublyRobust,
  type BanditLog,
  type OffPolicyEstimate,
} from 'aifn/learning/off-policy'
import { mean, std, tensor, toFlat } from 'aifn/foundation/tensor'
import { Figure } from '@lab/layout'
import { choice, float, int, row, slider, useComputed, useFigureState, type AnyValues } from '@lab/state'
import {
  Annotation,
  Area,
  Curve,
  formatNumber,
  Histogram,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  useAxis,
} from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const ESTIMATORS = ['IPS', 'clipped IPS', 'SNIPS', 'DM', 'DR'] as const
type Name = (typeof ESTIMATORS)[number]

/** Every estimator on one log; the reward model and target come from the problem. */
function estimateAll(log: BanditLog, p: BanditProblem, clip: number): Record<Name, OffPolicyEstimate> {
  return {
    IPS: ips(log, p.target),
    'clipped IPS': clippedIps(log, p.target, { clip }),
    SNIPS: snips(log, p.target),
    DM: directMethod(log, p.target, p.rewardModel),
    DR: doublyRobust(log, p.target, p.rewardModel),
  }
}

const meanOf = (v: number[]) => mean(tensor(v)) as number
const sdOf = (v: number[]) => (v.length > 1 ? (std(tensor(v), null, false, 1) as number) : 0)

// ── 1 · Bias and variance as the logging policy sharpens ─────────────────────────────────────────────────────────────

export function OffPolicySweepSpecimen() {
  const state = useFigureState({
    data: row('1 · logs', {
      n: int(400, { ge: 20, le: 20000, suggestions: [100, 400, 2000], label: 'rounds per log n' }),
      logs: int(60, { ge: 5, le: 1000, suggestions: [30, 60, 200], label: 'logs drawn' }),
      modelError: slider(0, 2, 0.6, { step: 0.05, label: 'reward model error' }),
      modelBias: slider(-1.5, 1.5, 0.5, { step: 0.05, label: 'reward model bias (logit shift)' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
    estimators: row('2 · estimators', {
      clip: float(5, { gt: 0, scale: 'log10', suggestions: [2, 5, 20, 100], label: 'clip M' }),
    }),
  })
  const { n, logs, modelError, modelBias, seed } = state.data
  const { clip } = state.estimators
  const sweep = useComputed(
    () => {
      const sharpness = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6]
      const out: Record<Name, { mean: number[]; sd: number[]; rmse: number[] }> = Object.fromEntries(
        ESTIMATORS.map((e) => [e, { mean: [], sd: [], rmse: [] }]),
      ) as never
      const truth: number[] = []
      for (const beta of sharpness) {
        const p = banditProblem(stream(`ope/problem/${seed}`), {
          n,
          loggingSharpness: beta,
          exploration: 0.02,
          modelError,
          modelBias,
        })
        truth.push(p.trueValue)
        const values: Record<Name, number[]> = Object.fromEntries(ESTIMATORS.map((e) => [e, []])) as never
        for (let r = 0; r < logs; r++) {
          const log = logBandit(child(stream(`ope/logs/${seed}`), 'log', r), p)
          const est = estimateAll(log, p, clip)
          for (const e of ESTIMATORS) values[e].push(est[e].value)
        }
        for (const e of ESTIMATORS) {
          const m = meanOf(values[e])
          const sd = sdOf(values[e])
          out[e].mean.push(m)
          out[e].sd.push(sd)
          out[e].rmse.push(Math.sqrt((m - p.trueValue) ** 2 + sd * sd))
        }
      }
      return { sharpness, truth, out }
    },
    [n, logs, modelError, modelBias, seed, clip],
    { mode: 'release' },
  )
  const { sharpness, truth, out } = sweep.value
  const top = Math.max(...ESTIMATORS.flatMap((e) => out[e].mean.map((m, i) => m + out[e].sd[i])), ...truth)
  const low = Math.min(...ESTIMATORS.flatMap((e) => out[e].mean.map((m, i) => m - out[e].sd[i])), ...truth)
  const betaAxis = useAxis({ label: 'logging sharpness β₀ (0 = uniform)', range: [0, 6] })
  const valueAxis = useAxis({ label: 'estimate of V(π)', range: [low, top] })
  const rmseAxis = useAxis({ label: 'RMSE', range: [0, Math.max(...ESTIMATORS.flatMap((e) => out[e].rmse))] })
  const last = sharpness.length - 1
  return (
    <Figure
      title="Extreme propensities make IPS noisy; a model makes it biased"
      purpose="IPS is unbiased for any logging policy that covers the target's actions, but its variance grows with the importance weights; the direct method has little variance and the model's bias; doubly robust keeps IPS's unbiasedness at much lower variance."
      state={state}
      defaultSize="L"
      readouts={{
        [`at β₀ = ${sharpness[last]}`]: (
          <>
            <Readout label="true V(π)" value={fmt(truth[last])} />
            {ESTIMATORS.map((e) => (
              <Readout
                key={e}
                label={`${e} mean ± sd`}
                value={`${fmt(out[e].mean[last])} ± ${fmt(Math.abs(out[e].sd[last]) < 1e-12 ? 0 : out[e].sd[last], 2)}`}
              />
            ))}
          </>
        ),
      }}
      caption="Each estimator is run on the given number of logs drawn from the same contexts; the band is its mean ± one standard deviation across logs, the ink curve the target's true value. As the logging policy β₀ sharpens, the actions the target prefers become rare in the logs, their importance weights grow, and IPS spreads. Clipping at M cuts the spread but pulls the estimate down; SNIPS is stable but slightly biased; the direct method is as wrong as its reward model (change the model bias); doubly robust stays centred on the truth. Recomputed on release."
    >
      <Plots rows={2} heights={[60, 40]} hoverGroup>
        <Plot x={betaAxis} y={valueAxis}>
          {ESTIMATORS.map((e, k) => (
            <Area
              key={`${e}-band`}
              name={e}
              x={sharpness}
              y={out[e].mean.map((m, i) => m + out[e].sd[i])}
              base={out[e].mean.map((m, i) => m - out[e].sd[i])}
              slot={k}
              opacity={0.12}
              line={false}
              stale={sweep.stale}
            />
          ))}
          {ESTIMATORS.map((e, k) => (
            <Curve key={e} name={e} x={sharpness} y={out[e].mean} slot={k} stale={sweep.stale} />
          ))}
          <Curve name="true V(π)" x={sharpness} y={truth} emphasis width={2.5} />
        </Plot>
        <Plot x={betaAxis} y={rmseAxis}>
          {ESTIMATORS.map((e, k) => (
            <Curve key={e} name={e} x={sharpness} y={out[e].rmse} slot={k} showPoints stale={sweep.stale} />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2 · The clipping trade-off ───────────────────────────────────────────────────────────────────────────────────────

const methodOf = (v: AnyValues) => String(v.method)

export function ClippingSpecimen() {
  const state = useFigureState({
    data: row('1 · logs', {
      sharpness: slider(0, 6, 4, { step: 0.1, label: 'logging sharpness β₀' }),
      n: int(400, { ge: 20, le: 20000, suggestions: [100, 400, 2000], label: 'rounds per log n' }),
      logs: int(60, { ge: 5, le: 1000, suggestions: [30, 60, 200], label: 'logs drawn' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
    method: row('2 · estimator', {
      method: choice(
        [
          { value: 'clip', label: 'clipped IPS: weights capped at M' },
          { value: 'switch', label: 'switch-DR: the model above τ' },
        ],
        'clip',
        { label: 'estimator' },
      ),
      modelError: slider(0, 2, 0.6, { step: 0.05, label: 'reward model error', when: (v) => methodOf(v) === 'switch' }),
      modelBias: slider(-1.5, 1.5, 0.5, {
        step: 0.05,
        label: 'reward model bias (logit shift)',
        when: (v) => methodOf(v) === 'switch',
      }),
    }),
  })
  const { sharpness, n, logs, seed } = state.data
  const { method, modelError, modelBias } = state.method
  const sweep = useComputed(
    () => {
      const p = banditProblem(stream(`ope/problem/${seed}`), {
        n,
        loggingSharpness: sharpness,
        exploration: 0.02,
        modelError,
        modelBias,
      })
      const draws = Array.from({ length: logs }, (_, r) => logBandit(child(stream(`ope/logs/${seed}`), 'log', r), p))
      const caps = Array.from({ length: 25 }, (_, i) => 10 ** (-0.5 + (3.5 * i) / 24))
      const bias2: number[] = []
      const variance: number[] = []
      const mse: number[] = []
      for (const M of caps) {
        const v = draws.map((L) =>
          method === 'clip'
            ? clippedIps(L, p.target, { clip: M }).value
            : switchDoublyRobust(L, p.target, p.rewardModel, { tau: M }).value,
        )
        const m = meanOf(v)
        const s = sdOf(v)
        bias2.push((m - p.trueValue) ** 2)
        variance.push(s * s)
        mse.push((m - p.trueValue) ** 2 + s * s)
      }
      const plain = draws.map(
        (L) => (method === 'clip' ? ips(L, p.target) : doublyRobust(L, p.target, p.rewardModel)).value,
      )
      const plainMse = (meanOf(plain) - p.trueValue) ** 2 + sdOf(plain) ** 2
      const maxWeight = Math.max(...draws.flatMap((L) => Array.from(toFlat(ips(L, p.target).weights))))
      return { caps, bias2, variance, mse, plainMse, maxWeight }
    },
    [sharpness, n, logs, seed, method, modelError, modelBias],
    { mode: 'release' },
  )
  const { caps, bias2, variance, mse, plainMse, maxWeight } = sweep.value
  const best = mse.indexOf(Math.min(...mse))
  const capAxis = useAxis({
    label: method === 'clip' ? 'clip M' : 'threshold τ',
    log: true,
    range: [caps[0], caps[caps.length - 1]],
  })
  const errAxis = useAxis({ label: 'squared error', log: true, hold: 'union', key: `${method}/${sharpness}/${n}` })
  const plainName = method === 'clip' ? 'IPS (M = ∞)' : 'DR (τ = ∞)'
  return (
    <Figure
      title="Clipping trades variance for bias"
      purpose="Capping the importance weights at M removes the variance of the rare, heavily weighted rounds and adds a bias that grows as M falls; the mean squared error is least in between."
      state={state}
      readouts={
        <>
          <Readout label="best M" value={fmt(caps[best])} />
          <Readout label="MSE at best M" value={fmt(mse[best])} />
          <Readout label={`MSE of ${plainName}`} value={fmt(plainMse)} />
          <Readout label="largest weight seen" value={fmt(maxWeight)} />
        </>
      }
      caption="Squared bias, variance and their sum, the mean squared error, of the estimate over the drawn logs, for each cap (log axes). Above the largest weight nothing is clipped and the curves meet the unclipped estimator's error (dashed). With switch-DR, rounds whose weight exceeds τ fall back on the reward model, so its bias is the model's error on those actions: lower the model error to see the best τ fall. Recomputed on release."
    >
      <Plot x={capAxis} y={errAxis}>
        <Curve name="bias²" x={caps} y={bias2} slot={0} stale={sweep.stale} />
        <Curve name="variance" x={caps} y={variance} slot={1} stale={sweep.stale} />
        <Curve name="MSE" x={caps} y={mse} emphasis showPoints stale={sweep.stale} />
        <Annotation y={plainMse} text={plainName} dashed />
        <Points name="least MSE" x={[caps[best]]} y={[mse[best]]} slot={2} size={10} />
      </Plot>
    </Figure>
  )
}

// ── 3 · One log ──────────────────────────────────────────────────────────────────────────────────────────────────────

export function OneLogSpecimen() {
  const state = useFigureState({
    data: row('1 · log', {
      sharpness: slider(0, 6, 3, { step: 0.1, label: 'logging sharpness β₀' }),
      n: int(1000, { ge: 20, le: 50000, suggestions: [200, 1000, 5000], label: 'rounds n' }),
      modelError: slider(0, 2, 0.6, { step: 0.05, label: 'reward model error' }),
      modelBias: slider(-1.5, 1.5, 0.5, { step: 0.05, label: 'reward model bias (logit shift)' }),
      seed: slider(0, 50, 0, { step: 1, label: 'log seed' }),
    }),
  })
  const { sharpness, n, modelError, modelBias, seed } = state.data
  const { p, est } = useMemo(() => {
    const p = banditProblem(stream('ope/one/problem'), {
      n,
      loggingSharpness: sharpness,
      exploration: 0.02,
      modelError,
      modelBias,
    })
    const log = logBandit(child(stream('ope/one/logs'), 'log', seed), p)
    return { p, est: estimateAll(log, p, 5) }
  }, [n, sharpness, modelError, modelBias, seed])
  const weights = Array.from(toFlat(est.IPS.weights))
  const logW = weights.map((w) => Math.log10(Math.max(w, 1e-3)))
  const wAxis = useAxis({ label: 'log₁₀ importance weight', range: [-3, 2.5] })
  const countAxis = useAxis({ label: 'rounds', hold: 'union', key: n })
  const estAxis = useAxis({ label: 'estimator', categories: [...ESTIMATORS] })
  const vAxis = useAxis({ label: 'V̂(π) with 95% interval', hold: 'union', key: `${n}/${sharpness}/${modelError}` })
  return (
    <Figure
      title="One log: its importance weights and five estimates"
      purpose="A log's importance weights π/π₀ show how much of it the target policy can use: a heavy right tail and a small effective sample size mean a few rounds decide the IPS estimate."
      state={state}
      readouts={
        <>
          <Readout label="true V(π)" value={fmt(p.trueValue)} />
          <Readout label="logging policy's value" value={fmt(p.loggingValue)} />
          <Readout label="effective sample size" value={`${fmt(est.IPS.effectiveSampleSize)} of ${n}`} />
          <Readout label="largest weight" value={fmt(Math.max(...weights))} />
        </>
      }
      caption="Left: the importance weights of the logged actions (log scale; weights below 10⁻³ are drawn at −3). Right: each estimator's value on this log with its 95% normal interval, against the true value (dashed). Change the log seed to see a fresh log from the same policies: the IPS interval is wide and moves the most."
    >
      <Plots cols={2} widths={[50, 50]}>
        <Plot x={wAxis} y={countAxis}>
          <Histogram name="weights" values={logW} bins={40} range={[-3, 2.5]} normalize="count" slot={0} />
          <Annotation x={0} text="w = 1" dashed />
        </Plot>
        <Plot x={estAxis} y={vAxis}>
          <Segments
            name="95% interval"
            segments={ESTIMATORS.map((e, k) => ({ from: [k, est[e].lower] as const, to: [k, est[e].upper] as const }))}
            width={3}
          />
          <Points
            name="estimate"
            x={ESTIMATORS.map((_, k) => k)}
            y={ESTIMATORS.map((e) => est[e].value)}
            emphasis
            size={9}
          />
          <Annotation y={p.trueValue} text="true V(π)" dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
