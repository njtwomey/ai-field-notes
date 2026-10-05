import { useMemo, useState } from 'react'
import { banditProblem, logBandit, type BanditProblem } from 'aifn-methods/data/synthetic'
import { child, stream } from 'aifn-compute/foundation/random'
import { mean, std, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import {
  clippedIps,
  directMethod,
  doublyRobust,
  ips,
  snips,
  switchDoublyRobust,
  type BanditLog,
  type OffPolicyEstimate,
} from 'aifn-compute/learning/off-policy'
import {
  Annotation,
  ControlRow,
  Curve,
  Figure,
  Histogram,
  NumberSelector,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  Select,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const ESTIMATORS = ['IPS', 'clipped IPS', 'SNIPS', 'DM', 'DR'] as const
type Name = (typeof ESTIMATORS)[number]

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

// ── 1 · One log: importance weights & 5 estimators ───────────────────────────────────────────────────────────────

export function OneLogExplorer() {
  const [sharpness, setSharpness] = useState(3)
  const [n, setN] = useState(1000)
  const [modelError, setModelError] = useState(0.6)
  const [modelBias, setModelBias] = useState(0.5)
  const [seed, setSeed] = useState(0)

  const { p, est } = useMemo(() => {
    const prob = banditProblem(stream('ope/one/problem'), {
      n,
      loggingSharpness: sharpness,
      exploration: 0.02,
      modelError,
      modelBias,
    })
    const log = logBandit(child(stream('ope/one/logs'), 'log', seed), prob)
    return { p: prob, est: estimateAll(log, prob, 5) }
  }, [n, sharpness, modelError, modelBias, seed])

  const weights = Array.from(toFlat(est.IPS.weights))
  const logW = weights.map((w) => Math.log10(Math.max(w, 1e-3)))

  const wAxis = useAxis({ label: 'log₁₀ importance weight π(a|x) / π₀(a|x)', range: [-3, 2.5] })
  const countAxis = useAxis({ label: 'rounds', hold: 'union', key: n })
  const estAxis = useAxis({ label: 'estimator', categories: [...ESTIMATORS] })
  const vAxis = useAxis({ label: 'V̂(π) with 95% interval', hold: 'union', key: `${n}/${sharpness}/${modelError}` })

  return (
    <Figure
      title="One log: importance weights and five estimators"
      purpose="A log's importance weights π/π₀ show how well the logging policy explores the target policy's preferred actions. A heavy right tail and low effective sample size warn that a few rare actions dominate the IPS estimate."
      controls={
        <>
          <ControlRow label="Logging policy & data">
            <NumberSelector
              label="Logging sharpness β₀"
              value={sharpness}
              onChange={setSharpness}
              min={0}
              max={6}
              step={0.5}
              suggestions={[0, 1.5, 3, 5]}
            />
            <NumberSelector
              label="Logged rounds n"
              value={n}
              onChange={setN}
              min={200}
              max={5000}
              step={200}
              suggestions={[200, 500, 1000, 2000]}
            />
            <NumberSelector
              label="Log draw seed"
              value={seed}
              onChange={setSeed}
              min={0}
              max={20}
              step={1}
              suggestions={[0, 1, 2, 3]}
            />
          </ControlRow>
          <ControlRow label="Reward model quality">
            <NumberSelector
              label="Model noise error"
              value={modelError}
              onChange={setModelError}
              min={0}
              max={2}
              step={0.2}
              suggestions={[0, 0.4, 0.6, 1.2]}
            />
            <NumberSelector
              label="Model bias (logit shift)"
              value={modelBias}
              onChange={setModelBias}
              min={-1.5}
              max={1.5}
              step={0.25}
              suggestions={[-0.5, 0, 0.5, 1.0]}
            />
          </ControlRow>
        </>
      }
      readouts={{
        estimates: (
          <>
            <Readout label="True target value V(π)" value={fmt(p.trueValue)} />
            <Readout label="Logging policy value V(π₀)" value={fmt(p.loggingValue)} />
            <Readout label="Effective sample size" value={`${fmt(est.IPS.effectiveSampleSize)} / ${n}`} />
            <Readout label="Maximum weight seen" value={fmt(Math.max(...weights))} />
          </>
        ),
      }}
      caption="Left: distribution of logged importance weights on a log₁₀ scale. When the logging policy is sharp, weights exceed 10 or 100. Right: 95% confidence intervals for IPS, clipped IPS (capped at M = 5), SNIPS, the Direct Method (DM), and Doubly Robust (DR). Notice that Doubly Robust centers closely on true V(π) even with biased reward models."
    >
      <Plots cols={2} widths={[50, 50]}>
        <Plot x={wAxis} y={countAxis}>
          <Histogram name="weights" values={logW} bins={35} range={[-3, 2.5]} normalize="count" slot={0} />
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
            size={8}
          />
          <Annotation y={p.trueValue} text="true V(π)" dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2 · The clipping trade-off ───────────────────────────────────────────────────────────────────────────────────

export function ClippingTradeoffExplorer() {
  const [sharpness, setSharpness] = useState(3.5)
  const [method, setMethod] = useState<'clip' | 'switch'>('clip')
  const n = 300
  const logs = 30
  const seed = 0

  const sweep = useMemo(() => {
    const p = banditProblem(stream(`ope/problem/${seed}`), {
      n,
      loggingSharpness: sharpness,
      exploration: 0.02,
      modelError: 0.6,
      modelBias: 0.5,
    })
    const draws = Array.from({ length: logs }, (_, r) => logBandit(child(stream(`ope/logs/${seed}`), 'log', r), p))
    const caps = Array.from({ length: 20 }, (_, i) => 10 ** (-0.5 + (3.5 * i) / 19))
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
    return { caps, bias2, variance, mse, plainMse }
  }, [sharpness, method, n, logs, seed])

  const { caps, bias2, variance, mse, plainMse } = sweep
  const best = mse.indexOf(Math.min(...mse))
  const capAxis = useAxis({
    label: method === 'clip' ? 'Clipping threshold M' : 'Switch threshold τ',
    log: true,
    range: [caps[0], caps[caps.length - 1]],
  })
  const errAxis = useAxis({ label: 'Squared error', log: true, hold: 'union', key: `${method}/${sharpness}` })
  const plainName = method === 'clip' ? 'unclipped IPS (M = ∞)' : 'DR (τ = ∞)'

  return (
    <Figure
      title="Weight clipping trades variance for bias"
      purpose="Capping importance weights at M prevents rare rounds from blowing up estimator variance, at the cost of downward bias on value. The mean squared error is minimized at an intermediate threshold M*."
      controls={
        <>
          <ControlRow label="Estimator & logging">
            <Select
              label="Method"
              value={method}
              onChange={(v) => setMethod(v as 'clip' | 'switch')}
              options={[
                { value: 'clip', label: 'Clipped IPS (w ≤ M)' },
                { value: 'switch', label: 'Switch-DR (use model above τ)' },
              ]}
            />
            <NumberSelector
              label="Logging sharpness β₀"
              value={sharpness}
              onChange={setSharpness}
              min={1}
              max={5}
              step={0.5}
              suggestions={[1.5, 3, 4.5]}
            />
          </ControlRow>
        </>
      }
      readouts={{
        optimal: (
          <>
            <Readout label="Best threshold" value={fmt(caps[best])} />
            <Readout label="MSE at optimal threshold" value={fmt(mse[best])} />
            <Readout label={`MSE of ${plainName}`} value={fmt(plainMse)} />
          </>
        ),
      }}
      caption="Squared bias, variance, and MSE across 30 repeated log draws as a function of the weight threshold (log-log scale). Low thresholds suffer high bias; high thresholds suffer extreme variance. The minimum MSE marks the optimal bias-variance compromise."
    >
      <Plot x={capAxis} y={errAxis}>
        <Curve name="Bias²" x={caps} y={bias2} slot={0} />
        <Curve name="Variance" x={caps} y={variance} slot={1} />
        <Curve name="MSE" x={caps} y={mse} emphasis showPoints />
        <Annotation y={plainMse} text={plainName} dashed />
        <Points name="Optimal threshold" x={[caps[best]]} y={[mse[best]]} slot={2} size={9} />
      </Plot>
    </Figure>
  )
}
