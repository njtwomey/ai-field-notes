/**
 * Learning with expert advice: Hedge, fixed share and follow-the-leader played round by round on a loss sequence from
 * `aifn-applied/data` (`expertGame`), with the regret against the best expert so far drawn against Hedge's bound; and a
 * sweep of the learning rate that shows the two failure modes the tuned rate balances. Every number is computed by
 * `aifn/optim/online`; the page draws.
 */
import { useMemo } from 'react'
import { expertGame } from 'aifn-applied/data/synthetic'
import { stream } from 'aifn/foundation/random'
import { toFlat, toRows } from 'aifn/foundation/tensor'
import { run, trace } from 'aifn/foundation/trace'
import { fixedShare, hedge, hedgeRegretBound, hedgeTunedRate, type ExpertsState } from 'aifn/optim/online'
import { Player, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, float, int, row, slider, useComputed, useFigureState, type AnyValues } from '@lab/state'
import { Annotation, Bars, Curve, formatNumber, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const KINDS = [
  { value: 'stochastic', label: 'stochastic: one expert slightly better' },
  { value: 'switching', label: 'switching: the best expert changes' },
  { value: 'follow-the-leader trap', label: 'the follow-the-leader trap' },
  { value: 'random', label: 'uniform random losses' },
] as const
type Kind = (typeof KINDS)[number]['value']

const LEARNERS = [
  { value: 'hedge', label: 'Hedge' },
  { value: 'fixed-share', label: 'fixed share' },
  { value: 'ftl', label: 'follow the leader (η → ∞)' },
] as const
const RATES = [
  { value: 'tuned', label: 'tuned √(8 ln N / T)' },
  { value: 'anytime', label: 'anytime √(8 ln N / t)' },
  { value: 'constant', label: 'constant η' },
] as const
const learnerOf = (v: AnyValues) => String(v.learner)
const rateOf = (v: AnyValues) => String(v.rate)

/** The loss matrix of a game, as rows. */
function game(kind: Kind, rounds: number, experts: number, gap: number, seed: number) {
  const g = expertGame(stream(`online/game/${seed}`), { kind, rounds, experts, gap, switches: 3 })
  return { losses: toRows(g.losses) as number[][], best: Array.from(toFlat(g.best)) }
}

// ── 1 · Hedge, round by round ────────────────────────────────────────────────────────────────────────────────────────

export function HedgeRegretSpecimen() {
  const state = useFigureState({
    data: row('1 · losses', {
      kind: choice(KINDS, 'switching', { label: 'sequence' }),
      experts: int(8, { ge: 2, le: 200, suggestions: [2, 8, 32, 100], label: 'experts N' }),
      rounds: int(600, { ge: 10, le: 20000, suggestions: [200, 600, 2000, 10000], label: 'rounds T' }),
      gap: slider(0, 0.4, 0.15, { step: 0.01, label: 'gap of the best expert' }),
      seed: slider(0, 20, 1, { step: 1, label: 'seed' }),
    }),
    learner: row('2 · learner', {
      learner: choice(LEARNERS, 'hedge', { label: 'learner' }),
      rate: choice(RATES, 'tuned', { label: 'learning rate', when: (v) => learnerOf(v) !== 'ftl' }),
      eta: float(0.1, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.01, 0.1, 0.3, 1, 3],
        label: 'η',
        when: (v) => learnerOf(v) !== 'ftl' && rateOf(v) === 'constant',
      }),
      alpha: float(0.005, {
        gt: 0,
        lt: 1,
        scale: 'log10',
        suggestions: [0.001, 0.005, 0.02, 0.1],
        label: 'share α',
        when: (v) => learnerOf(v) === 'fixed-share',
      }),
    }),
  })
  const { kind, experts: N, rounds: T, gap, seed } = state.data
  const { learner, rate, eta, alpha } = state.learner

  const g = useMemo(() => game(kind as Kind, T, N, gap, seed), [kind, T, N, gap, seed])
  const etaOption = learner === 'ftl' ? 1e6 : rate === 'constant' ? eta : (rate as 'tuned' | 'anytime')
  const runOf = useComputed(() => {
    const alg =
      learner === 'fixed-share' ? fixedShare(g.losses, { eta: etaOption, alpha }) : hedge(g.losses, { eta: etaOption })
    const tr = trace(alg, undefined, T, { keep: 'all' })
    return tr.steps as ExpertsState[]
  }, [g, learner, etaOption, alpha, T])
  const steps = runOf.value
  const [at, setAt] = usePlayhead(steps.length)
  const s = steps[Math.min(at, steps.length - 1)]

  // Hedge's bound at each t for the rate in use (the tuned rate's bound is for the horizon T; the anytime rate has its
  // own bound; a constant rate gives ln N/η + ηt/8). Follow-the-leader has none.
  const bound = useMemo(() => {
    const ts = steps.map((st) => st.t)
    if (learner === 'ftl') return null
    if (rate === 'anytime') return { t: ts, y: ts.map((t) => hedgeRegretBound(Math.max(t, 1), N, 'anytime')) }
    const e = rate === 'constant' ? eta : hedgeTunedRate(T, N)
    return { t: ts, y: ts.map((t) => hedgeRegretBound(t, N, e)) }
  }, [steps, learner, rate, eta, N, T])

  const ts = steps.map((st) => st.t)
  const upto = at + 1
  const cut = <V,>(a: V[]) => a.slice(0, upto)
  const regret = steps.map((st) => st.regret)
  const learnerLoss = steps.map((st) => st.learnerLoss)
  const bestLoss = steps.map((st) => Math.min(...toFlat(st.cumulative)))
  const weights = Array.from(toFlat(s.weights))
  const expertsIdx = weights.map((_, i) => i)
  const leaderNow = g.best[Math.max(0, s.t - 1)] ?? -1

  // Axes are held to the whole run, so playing does not rescale them.
  const last = steps[steps.length - 1]
  const regretTop = Math.max(1, ...regret, ...(bound ? bound.y : []))
  const regretLow = Math.min(0, ...regret)
  const lossTop = Math.max(1, last.learnerLoss, ...toFlat(last.cumulative))
  const tAxis = useAxis({ label: 'round t', range: [0, T] })
  const regretAxis = useAxis({ label: 'regret Rₜ', range: [regretLow, regretTop] })
  const lossAxis = useAxis({ label: 'cumulative loss', range: [0, lossTop] })
  const expertAxis = useAxis({ label: 'expert i', range: [-0.6, N - 0.4] })
  const weightAxis = useAxis({ label: 'weight pₜ,ᵢ', range: [0, 1] })
  return (
    <Figure
      title="Hedge plays a softmax of the experts' cumulative losses"
      purpose="Hedge's weights p ∝ exp(−η L) move towards the experts with least loss; its regret against the best expert stays under ln N/η + ηT/8, while follow-the-leader can be made to lose every round."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="3 · rounds">
          <Player className="col-span-full" value={at} onChange={setAt} count={steps.length} label="t" />
        </ControlRow>
      }
      readouts={{
        [`round ${s.t}`]: (
          <>
            <Readout label="learner loss L̂ₜ" value={fmt(s.learnerLoss, 4)} />
            <Readout label="best expert loss" value={fmt(Math.min(...toFlat(s.cumulative)), 4)} />
            <Readout label="regret Rₜ" value={fmt(s.regret)} />
            <Readout label="Rₜ / t" value={fmt(s.t > 0 ? s.regret / s.t : 0)} />
            <Readout label="best so far" value={`expert ${s.best}`} />
            <Readout label="η" value={learner === 'ftl' ? '∞' : fmt(s.eta)} />
          </>
        ),
        horizon: (
          <>
            <Readout label="bound √(T ln N / 2)" value={fmt(hedgeRegretBound(T, N))} />
            <Readout label="final regret" value={fmt(steps[steps.length - 1].regret)} />
          </>
        ),
      }}
      caption="Play, or step with the arrows. Left: the regret Rₜ = L̂ₜ − minᵢ Lₜ,ᵢ against the best expert so far, with Hedge's bound for the rate in use (dashed). Right: the weights pₜ₊₁ the learner plays next; the ink-coloured bar is the expert that is truly best in the current regime. On the switching sequence Hedge needs a long time to move to the new best expert, because the old one's lead must be lost first; fixed share keeps every weight above α/N and switches quickly. On the follow-the-leader trap, follow-the-leader pays almost every round and its regret grows linearly, while Hedge stays under its bound."
    >
      <Plots rows={2} cols={2} heights={[55, 45]} widths={[60, 40]}>
        <Plot x={tAxis} y={regretAxis}>
          {bound && <Curve name="Hedge bound" x={bound.t} y={bound.y} dashed muted />}
          <Curve name="regret Rₜ" x={cut(ts)} y={cut(regret)} slot={0} stale={runOf.stale} />
          <Points name="now" x={[s.t]} y={[s.regret]} emphasis live />
          <Annotation y={0} dashed />
        </Plot>
        <Plot x={expertAxis} y={weightAxis}>
          <Bars name="weight pₜ₊₁" x={expertsIdx} y={weights} slot={1} />
          {leaderNow >= 0 && <Bars name="truly best now" x={[leaderNow]} y={[weights[leaderNow]]} emphasis />}
        </Plot>
        <Plot x={tAxis} y={lossAxis}>
          <Curve name="learner L̂ₜ" x={cut(ts)} y={cut(learnerLoss)} slot={0} />
          <Curve name="best expert minᵢ Lₜ,ᵢ" x={cut(ts)} y={cut(bestLoss)} emphasis />
        </Plot>
        <Plot x={expertAxis} y={lossAxis}>
          <Bars name="cumulative loss Lₜ,ᵢ" x={expertsIdx} y={Array.from(toFlat(s.cumulative))} slot={2} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2 · The learning rate ────────────────────────────────────────────────────────────────────────────────────────────

export function LearningRateSpecimen() {
  const state = useFigureState({
    data: row('1 · losses', {
      kind: choice(KINDS, 'follow-the-leader trap', { label: 'sequence' }),
      experts: int(8, { ge: 2, le: 200, suggestions: [2, 8, 32, 100], label: 'experts N' }),
      rounds: int(1000, { ge: 10, le: 20000, suggestions: [200, 1000, 5000], label: 'rounds T' }),
      seed: slider(0, 20, 1, { step: 1, label: 'seed' }),
    }),
  })
  const { kind, experts: N, rounds: T, seed } = state.data
  const g = useMemo(() => game(kind as Kind, T, N, 0.15, seed), [kind, T, N, seed])
  const sweep = useComputed(() => {
    const etas = Array.from({ length: 33 }, (_, i) => 10 ** (-3 + (4.5 * i) / 32))
    const regret = etas.map((eta) => run(hedge(g.losses, { eta }), undefined, T).regret)
    return { etas, regret, bound: etas.map((eta) => hedgeRegretBound(T, N, eta)) }
  }, [g, T, N])
  const tuned = hedgeTunedRate(T, N)
  const tunedRegret = useMemo(() => run(hedge(g.losses, { eta: tuned }), undefined, T).regret, [g, tuned, T])
  const etaAxis = useAxis({ label: 'learning rate η', log: true, range: [1e-3, 10 ** 1.5] })
  const { etas, regret, bound } = sweep.value
  // The bound grows without limit at both ends: the axis follows the regret, and the bound runs off the top.
  const regretAxis = useAxis({
    label: 'regret R_T',
    range: [Math.min(0, ...regret), 1.4 * Math.max(1, ...regret)],
  })
  return (
    <Figure
      title="The learning rate trades slow learning against instability"
      purpose="Hedge's regret bound ln N/η + ηT/8 has two terms: a small η stays near uniform and learns slowly, a large η chases the current leader. The tuned rate √(8 ln N/T) balances them."
      state={state}
      readouts={
        <>
          <Readout label="tuned η" value={fmt(tuned)} />
          <Readout label="regret at tuned η" value={fmt(tunedRegret)} />
          <Readout label="bound √(T ln N / 2)" value={fmt(hedgeRegretBound(T, N))} />
          <Readout label="least regret in the sweep" value={fmt(Math.min(...regret))} />
        </>
      }
      caption="Each point is one run of Hedge with a constant η over the same T rounds. The dashed curve is the bound ln N/η + ηT/8, which every run stays under; the vertical line marks the tuned rate, where the bound is least. On the follow-the-leader trap a large η copies the leader and pays nearly every round; on stochastic losses a large η costs little, because the leader rarely changes."
    >
      <Plot x={etaAxis} y={regretAxis}>
        <Curve name="bound ln N/η + ηT/8" x={etas} y={bound} dashed muted />
        <Curve name="regret R_T" x={etas} y={regret} slot={0} showPoints stale={sweep.stale} />
        <Annotation x={tuned} text="tuned η" dashed />
      </Plot>
    </Figure>
  )
}
