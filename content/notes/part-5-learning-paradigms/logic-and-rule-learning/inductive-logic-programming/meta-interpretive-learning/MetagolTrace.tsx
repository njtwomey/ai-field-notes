import { useMemo, useState } from 'react'
import { choice, cn, Figure, int, Player, Readout, useFigureState } from 'aifn-render'
import { metagol, type MilEvent, type MilProblem } from 'aifn-compute/logic/induction'
import { PARENT } from '../_shared/family'

const TASKS: Record<'ancestor' | 'grandparent' | 'greatgrandparent', Omit<MilProblem, 'background' | 'metarules'>> = {
  grandparent: {
    target: 'grandparent',
    positives: [
      ['ann', 'dee'],
      ['bob', 'gus'],
    ],
    negatives: [
      ['ann', 'bob'],
      ['ann', 'gus'],
    ],
  },
  ancestor: {
    target: 'ancestor',
    positives: [
      ['ann', 'bob'],
      ['ann', 'dee'],
      ['ann', 'gus'],
      ['cid', 'fay'],
    ],
    negatives: [
      ['bob', 'ann'],
      ['dee', 'bob'],
      ['cid', 'gus'],
    ],
  },
  greatgrandparent: {
    target: 'greatgrandparent',
    positives: [['ann', 'gus']],
    negatives: [
      ['ann', 'eve'],
      ['bob', 'gus'],
      ['ann', 'dee'],
    ],
  },
}

const schema = {
  task: choice(
    [
      { value: 'grandparent', label: 'grandparent' },
      { value: 'ancestor', label: 'ancestor (recursion)' },
      { value: 'greatgrandparent', label: 'great-grandparent (invention)' },
    ] as const,
    'ancestor',
    { label: 'target' },
  ),
  maxClauses: int(3, { label: 'clauses at most', min: 1, max: 4, suggestions: [1, 2, 3, 4] }),
}

const KIND: Record<MilEvent['kind'], string> = {
  bound: 'bound',
  fact: 'fact',
  fail: 'fail',
  reuse: 'reuse clause',
  'new-clause': 'new clause',
  bind: 'bind',
  invent: 'invent',
  reject: 'reject',
  accept: 'accept',
  none: 'no program',
}

/** Metagol proving the examples with the metarules ident and chain, and the program it builds on the way. */
export function MetagolTrace() {
  const state = useFigureState(schema)
  const key = `${state.task}-${state.maxClauses}`
  const [pos, setPos] = useState({ key, step: 0 })
  const run = useMemo(
    () =>
      metagol(
        { ...TASKS[state.task], background: { parent: PARENT }, metarules: ['ident', 'chain'] },
        { maxClauses: state.maxClauses, maxInvented: 1, maxEvents: 3000 },
      ),
    [state.task, state.maxClauses],
  )
  const last = run.events.length - 1
  const t = pos.key === key ? Math.min(pos.step, last) : 0
  const e = run.events[t]
  const recent = run.events.slice(Math.max(0, t - 9), t + 1)
  const task = TASKS[state.task]

  return (
    <Figure
      title="Learning a program by proving the examples"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      controls={
        <Player
          label="event"
          value={t}
          onChange={(k) => setPos({ key, step: k })}
          count={run.events.length}
          format={(k) => `${k} of ${last}`}
        />
      }
      equation={
        <div className="grid gap-3 text-left text-sm md:grid-cols-2">
          <div>
            <div className="text-xs text-muted-foreground">program so far (at most {e.clauses} clauses)</div>
            <pre className="font-mono text-xs leading-relaxed">{e.program.join('\n') || '(empty)'}</pre>
          </div>
          <div className="font-prose text-[15px] leading-snug">
            {e.goal && <div className="font-mono text-xs">goal {e.goal}</div>}
            {e.note}
          </div>
        </div>
      }
      readouts={
        <>
          <Readout
            label="examples"
            value={`+ ${task.positives.map(([a, b]) => `${a}→${b}`).join(', ')}   − ${task.negatives.map(([a, b]) => `${a}→${b}`).join(', ')}`}
          />
          <Readout
            label="result"
            value={
              run.program
                ? `${run.program.join(' ')} (${last} events)`
                : run.stopped === 'limit'
                  ? 'event limit reached'
                  : `no program of at most ${state.maxClauses} clauses`
            }
          />
        </>
      }
      caption="Background knowledge: the six parent facts of the family. Metarules: ident P(A, B) ← Q(A, B) and chain P(A, B) ← Q(A, C), R(C, B). Pick a target and the clause bound, then step through the search. The learner proves the positive examples one by one. A goal of a background predicate is proved by a fact; a goal of the target or an invented predicate by a clause already in the program or by a new clause from a metarule, whose open predicates (?) are bound as its body is proved. Programs are tried with 1 clause, then 2, and so on; a program that proves a negative example is rejected and the search backtracks."
    >
      <ol className="h-full overflow-auto font-mono text-xs leading-relaxed">
        {recent.map((x, i) => {
          const k = t - recent.length + 1 + i
          return (
            <li
              key={k}
              className={cn(
                k === t && 'font-semibold',
                (x.kind === 'fail' || x.kind === 'reject') && 'text-muted-foreground',
              )}
            >
              {k}. [{KIND[x.kind]}] {x.goal ? `${x.goal}: ` : ''}
              {x.note}
            </li>
          )
        })}
      </ol>
    </Figure>
  )
}
