import { useMemo, useState } from 'react'
import { type AttackMechanism, type BoostingAttackResult } from 'aifn-methods/theory/adaptive-holdout'
import {
  call,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
  type Task,
} from 'aifn-render'

const fmt = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—'

const NAMES: Record<AttackMechanism, string> = {
  exact: 'full-precision leaderboard',
  rounded: 'rounded leaderboard',
  ladder: 'Ladder',
  'parameter-free': 'parameter-free Ladder',
  thresholdout: 'Thresholdout',
}

/**
 * Blum and Hardt's boosting attack against three leaderboards on the same holdout and the same random submissions.
 * After each submission t the attacker submits the majority vote of the submissions it has kept; solid lines are what
 * each leaderboard reports for that vote, thin lines the vote's loss on fresh labels, whose expectation is ½.
 */
export function LeaderboardAttack() {
  const state = useFigureState({
    n: int(4000, { ge: 100, le: 20000, suggestions: [1000, 4000, 10000], label: 'holdout size n' }),
    k: int(1000, { ge: 10, le: 5000, suggestions: [200, 1000, 3000], label: 'submissions k' }),
    eta: float(0.01, { gt: 0, le: 0.2, suggestions: [0.005, 0.01, 0.02, 0.05], label: 'Ladder step η' }),
    third: choice(
      [
        { value: 'parameter-free', label: 'parameter-free Ladder' },
        { value: 'thresholdout', label: 'Thresholdout (T = 0.04, σ = 0.01)' },
        { value: 'rounded', label: 'leaderboard rounded to 2 decimals' },
      ],
      'parameter-free',
      { label: 'third leaderboard' },
    ),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { n, k, eta, third, seed } = state
  const mechanisms: AttackMechanism[] = ['exact', 'ladder', third as AttackMechanism]
  const run = useComputed(
    () => {
      const s = call('foundation/random/stream', `reusable-holdout/${seed}`)
      return mechanisms.map((mechanism) =>
        call<BoostingAttackResult>('theory/adaptive-holdout/boostingAttack', s, { n, k, mechanism, eta, digits: 2 }),
      ) as unknown as Task<BoostingAttackResult[]>
    },
    [n, k, eta, third, seed],
    { mode: 'worker', initial: [] as BoostingAttackResult[] },
  )
  const results = run.value
  const ready = results.length === 3 && results.every((r) => r.fresh.length === k)
  const ts = useMemo(() => Array.from({ length: k }, (_, i) => i + 1), [k])
  const [picked, setPicked] = useState<number | null>(null)
  const t = Math.min(k, Math.max(1, picked ?? k))

  const xAxis = useAxis({ label: 'submissions t' })
  const yAxis = useAxis({ label: 'loss of the majority vote', range: [0.3, 0.6] })
  return (
    <Figure
      title="The boosting attack against three leaderboards"
      state={state}
      readouts={
        ready ? (
          <>
            {results.map((r, i) => (
              <Readout
                key={mechanisms[i]}
                label={`${NAMES[mechanisms[i]]} at t = ${t}: reported / fresh`}
                value={`${fmt(r.reported[t - 1])} / ${fmt(r.fresh[t - 1])}`}
              />
            ))}
            <Readout
              label="submissions kept at t"
              value={results.map((r, i) => `${NAMES[mechanisms[i]]} ${r.kept[t - 1]}`).join(', ')}
            />
            {third === 'thresholdout' && <Readout label="Thresholdout budget spent" value={String(results[2].spent)} />}
          </>
        ) : (
          <Readout label="status" value={run.error ?? 'computing…'} />
        )
      }
      caption={`A holdout of n random binary labels and k submissions of random label vectors, each with true loss exactly ½. After each submission t the attacker submits the majority vote of the submissions it has kept: against the full-precision leaderboard, those scored at loss ½ or less; against the Ladders, those that moved the leaderboard; against Thresholdout, those whose answer was ½ or less. Solid lines: what each leaderboard reports for the vote. Thin lines in the same colour: the vote's loss on n fresh labels, the truth, whose expectation is ½ (dashed). The full-precision report falls like ½ − c√(t/n) while the fresh loss stays at ½; the Ladder's report stays within about η of ½. Every leaderboard sees the same holdout and the same submissions. Drag the vertical line to read off one submission.`}
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve name="true loss ½" x={[1, k]} y={[0.5, 0.5]} muted dashed />
        {ready &&
          results.map((r, i) => (
            <Curve
              key={`fresh-${mechanisms[i]}`}
              name={`${NAMES[mechanisms[i]]}: fresh loss`}
              x={ts}
              y={r.fresh}
              slot={i}
              thin
              stale={run.stale}
            />
          ))}
        {ready &&
          results.map((r, i) => (
            <Curve
              key={`reported-${mechanisms[i]}`}
              name={`${NAMES[mechanisms[i]]}: reported`}
              x={ts}
              y={r.reported}
              slot={i}
              stale={run.stale}
            />
          ))}
        <Handle kind="x" at={t} label={`t = ${t}`} onDrag={(x) => setPicked(Math.min(k, Math.max(1, Math.round(x))))} />
      </Plot>
    </Figure>
  )
}
