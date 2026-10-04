/**
 * Showcase: the feedback loop. aifn-methods `feedbackLoop` retrains four policies on the clicks their own slates
 * produced, round by round, in the worker; users click with the true probabilities of `implicitFeedback`'s taste
 * model. The page draws the exposure Gini coefficient and the expected click-through per round, and the exposure of
 * every item at the round on the player.
 */
import { useMemo, useState } from 'react'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { implicitFeedback } from 'aifn-methods/data/synthetic'
import { worldFromFactors, type FeedbackPolicy, type FeedbackSnapshot } from 'aifn-methods/retrieval/recommenders'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, float, int, row, useFigureState, type Task } from '@lab/state'
import { formatValue, TrainControls, useTrainedRun } from '@lab/views'
import { Curve, Handle, Plot, Plots, Readout, useAxis } from '@lab/viz'

const f3 = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—'
const POLICIES: { policy: FeedbackPolicy; name: string }[] = [
  { policy: 'popularity', name: 'popularity' },
  { policy: 'matrix-factorisation', name: 'implicit ALS' },
  { policy: 'random', name: 'random slates' },
  { policy: 'oracle', name: 'oracle (true probabilities)' },
]

type Settings = { users: number; items: number; rounds: number; slate: number; exploration: number; seed: number }

export function FeedbackLoopShowcase() {
  const state = useFigureState({
    world: row('1 · world', {
      users: int(60, { ge: 5, le: 200, suggestions: [30, 60, 120], label: 'users' }),
      items: int(80, { ge: 10, le: 300, suggestions: [40, 80, 160], label: 'items' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    loop: row('2 · loop', {
      rounds: int(30, { ge: 1, suggestions: [10, 30, 60], label: 'rounds' }),
      slate: int(5, { ge: 1, le: 20, suggestions: [3, 5, 10], label: 'slate size K' }),
      exploration: float(0, { ge: 0, le: 1, suggestions: [0, 0.2, 0.4], label: 'exploration ε (share of each slate)' }),
    }),
  })
  const settings: Settings = { ...state.world, ...state.loop }
  const trained = useTrainedRun(settings, (s): Task<FeedbackSnapshot> =>
    call<FeedbackSnapshot>('applied/retrieval/recommenders/feedbackLoop', {
      world: worldFromFactors(...worldArgs(s)),
      rounds: s.rounds,
      slate: s.slate,
      exploration: s.exploration,
      seed: s.seed,
    }),
  )
  const snap = trained.run.value ?? undefined
  const shown = trained.trained ?? settings
  const rounds = snap ? snap.round + 1 : 1
  const [picked, setPicked] = useState<{ run: unknown; round: number } | null>(null)
  const round = Math.min(picked && picked.run === trained.trained ? picked.round : 0, Math.max(0, rounds - 1))
  const pick = (r: number) => setPicked({ run: trained.trained, round: Math.round(r) })

  const rx = useAxis({ label: 'round', range: [0, shown.rounds], key: shown.rounds, integer: true })
  const gy = useAxis({ label: 'exposure Gini', range: [0, 1] })
  const cy = useAxis({ label: 'expected click-through of the slates', range: [0, 1] })
  const ix = useAxis({ label: 'items, most exposed first', range: [0, shown.items - 1], key: shown.items })
  const ey = useAxis({ label: 'share of exposure (cumulative)', range: [0, 1.02] })

  // Each policy's exposure up to the round shown: cumulative shares of items sorted by exposure (a Lorenz curve).
  const lorenz = useMemo(() => {
    if (!snap) return null
    return POLICIES.map(({ policy }) => {
      const h = snap.policies[policy]
      if (!h) return null
      const e = Float64Array.from(h.exposure).sort().reverse()
      const total = e.reduce((a, b) => a + b, 0) || 1
      let run = 0
      return Array.from(e, (v) => (run += v / total))
    })
  }, [snap])
  const empty = !trained.trained ? 'press Train to start' : !snap ? 'simulating…' : undefined
  const marker = snap ? <Handle kind="x" at={round} onDrag={pick} label={`round ${round}`} /> : null
  const at = (p: FeedbackPolicy, key: 'gini' | 'ctr') => snap?.policies[p]?.[key][round]
  return (
    <Figure
      title="The feedback loop: popularity concentration over rounds"
      purpose="A recommender retrained on the clicks its own slates produced shows more of what it showed before: under popularity ranking a few items take most of the exposure within a few rounds, while the taste of the users has not changed."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls
            run={trained as never}
            label="simulate"
            progress={snap ? snap.round / Math.max(1, snap.rounds) : 0}
            progressText={`${snap?.round ?? 0} / ${shown.rounds} rounds`}
          />
          <ControlRow label="rounds">
            <Player
              value={round}
              onChange={pick}
              count={Math.max(1, rounds)}
              label="round"
              format={(i) => `round ${i}`}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="round" value={round} />
          {POLICIES.map(({ policy, name }) => (
            <Readout key={policy} label={`Gini, ${name}`} value={f3(at(policy, 'gini'))} />
          ))}
          {POLICIES.map(({ policy, name }) => (
            <Readout key={`${policy}-ctr`} label={`CTR, ${name}`} value={f3(at(policy, 'ctr'))} />
          ))}
        </>
      }
      caption={
        <>
          aifn-methods <code>feedbackLoop</code> with {shown.users} users and {shown.items} items whose true click
          probabilities are σ(2 wᵤᵀvᵢ − 2) from <code>implicitFeedback</code>&apos;s factors. Round 0 shows every user
          the same random slate under each policy; from then on each policy ranks the items a user has not clicked and
          shows the top {shown.slate}
          {shown.exploration > 0
            ? `, ${Math.round(shown.exploration * shown.slate)} of them replaced by random items`
            : ''}
          , and every click joins that policy&apos;s log. Left: the Gini coefficient of cumulative exposure across items
          (0: equal, 1: one item). Middle: the slates&apos; expected click-through (the mean true probability of the
          items shown). Right: the cumulative share of exposure held by the most-exposed items at the end of the run.
          Drag the round marker or play the rounds from round 0.
        </>
      }
    >
      <Plots cols={3} scale={0.7}>
        <Plot x={rx} y={gy} title={empty ?? 'exposure concentration'}>
          {snap &&
            POLICIES.map(({ policy, name }, slot) =>
              snap.policies[policy] ? (
                <Curve
                  key={policy}
                  name={name}
                  x={snap.policies[policy].gini.map((_, i) => i)}
                  y={snap.policies[policy].gini}
                  slot={slot}
                />
              ) : null,
            )}
          {marker}
        </Plot>
        <Plot x={rx} y={cy} title={empty ?? 'expected click-through'}>
          {snap &&
            POLICIES.map(({ policy, name }, slot) =>
              snap.policies[policy] ? (
                <Curve
                  key={policy}
                  name={name}
                  x={snap.policies[policy].ctr.map((_, i) => i)}
                  y={snap.policies[policy].ctr}
                  slot={slot}
                />
              ) : null,
            )}
          {marker}
        </Plot>
        <Plot x={ix} y={ey} title={empty ?? 'who gets the exposure (end of run)'}>
          {lorenz &&
            POLICIES.map(({ policy, name }, slot) =>
              lorenz[slot] ? (
                <Curve key={policy} name={name} x={lorenz[slot]!.map((_, i) => i)} y={lorenz[slot]!} slot={slot} />
              ) : null,
            )}
        </Plot>
      </Plots>
    </Figure>
  )
}

/** The arguments of `worldFromFactors` for a world's settings (the same data the page draws). */
function worldArgs(s: Settings): Parameters<typeof worldFromFactors> {
  const d = implicitFeedback(stream(s.seed), { users: s.users, items: s.items })
  return [toFlat(d.userFactors), toFlat(d.itemFactors), d.users, d.items]
}
