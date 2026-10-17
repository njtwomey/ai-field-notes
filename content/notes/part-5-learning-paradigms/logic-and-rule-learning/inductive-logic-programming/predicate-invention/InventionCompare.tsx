import { useMemo } from 'react'
import { solveQuery } from 'aifn-compute/logic/resolution'
import { choice, Diagram, Figure, int, Readout, useFigureState, type DiagramEdge } from 'aifn-render'
import { metagol, type MilProblem, type MilResult } from 'aifn-compute/logic/induction'
import { EXTENDED_FEMALE, EXTENDED_PARENT, familyDiagram } from '../_shared/family'

type Pair = readonly [string, string]
const MOTHER = EXTENDED_PARENT.filter(([a]) => EXTENDED_FEMALE.includes(a))
const FATHER = EXTENDED_PARENT.filter(([a]) => !EXTENDED_FEMALE.includes(a))

const TASKS: Record<
  'grandparent' | 'greatgrandparent' | 'greatgreatgrandparent',
  { label: string; problem: Omit<MilProblem, 'metarules'> }
> = {
  grandparent: {
    label: 'grandparent from mother and father',
    problem: {
      background: { mother: MOTHER, father: FATHER },
      target: 'grandparent',
      positives: [
        ['ann', 'dee'],
        ['ann', 'eve'],
        ['ann', 'fay'],
        ['bob', 'gus'],
        ['eve', 'hal'],
        ['gus', 'ivy'],
      ],
      negatives: [
        ['ann', 'bob'],
        ['ann', 'gus'],
        ['dee', 'ann'],
        ['bob', 'fay'],
        ['eve', 'ivy'],
      ],
    },
  },
  greatgrandparent: {
    label: 'great-grandparent from parent',
    problem: {
      background: { parent: EXTENDED_PARENT },
      target: 'greatgrandparent',
      positives: [
        ['ann', 'gus'],
        ['bob', 'hal'],
        ['eve', 'ivy'],
      ],
      negatives: [
        ['ann', 'eve'],
        ['bob', 'gus'],
        ['ann', 'dee'],
        ['ann', 'hal'],
        ['bob', 'ivy'],
      ],
    },
  },
  greatgreatgrandparent: {
    label: 'great-great-grandparent from parent',
    problem: {
      background: { parent: EXTENDED_PARENT },
      target: 'greatgreatgrandparent',
      positives: [
        ['ann', 'hal'],
        ['bob', 'ivy'],
      ],
      negatives: [
        ['ann', 'gus'],
        ['bob', 'hal'],
        ['ann', 'ivy'],
        ['eve', 'ivy'],
        ['ann', 'eve'],
      ],
    },
  },
}

const schema = {
  task: choice(
    (Object.keys(TASKS) as (keyof typeof TASKS)[]).map((k) => ({ value: k, label: TASKS[k].label })),
    'grandparent',
    { label: 'target' },
  ),
  maxClauses: int(4, { label: 'clauses at most', min: 1, max: 4, suggestions: [2, 3, 4] }),
}

/** Literals in a program: every predicate application, heads included. */
const literals = (program: readonly string[]) => program.reduce((s, c) => s + (c.match(/\w+\(/g)?.length ?? 0), 0)

function Side({ title, run, max }: { title: string; run: MilResult; max: number }) {
  return (
    <div className="min-w-0">
      <div className="text-xs text-muted-foreground">{title}</div>
      <pre className="font-mono text-xs leading-relaxed whitespace-pre-wrap">
        {run.program
          ? run.program.join('\n')
          : run.stopped === 'limit'
            ? '(event limit reached)'
            : `(no program of at most ${max} clauses)`}
      </pre>
      <div className="text-xs">
        {run.program
          ? `${run.program.length} clauses, ${literals(run.program)} literals, ${run.events.length} search events`
          : `${run.events.length} search events`}
      </div>
    </div>
  )
}

/** The same task learnt by Metagol with and without an invented predicate. */
export function InventionCompare() {
  const state = useFigureState(schema)
  const { problem } = TASKS[state.task]
  const runs = useMemo(() => {
    const p: MilProblem = { ...problem, metarules: ['ident', 'chain'] }
    const opts = { maxClauses: state.maxClauses, maxEvents: 30000 }
    return {
      without: metagol(p, { ...opts, maxInvented: 0 }),
      with: metagol(p, { ...opts, maxInvented: 1 }),
    }
  }, [problem, state.maxClauses])

  // The extension of the invented predicate, by running the learnt program as Prolog.
  const invented = useMemo((): Pair[] => {
    const prog = runs.with.program
    if (!prog || !prog.some((c) => c.startsWith('inv1('))) return []
    const facts = Object.entries(problem.background)
      .flatMap(([name, pairs]) => pairs.map(([a, b]) => `${name}(${a}, ${b}).`))
      .join(' ')
    const r = solveQuery(`${facts}\n${prog.join('\n')}`, 'inv1(X, Y)', { maxSteps: 20000 })
    return r.solutions.map((s) => {
      const v = (n: string) => {
        const t = s.bindings.find((b) => b.name === n)!.value
        return t.kind === 'atom' ? t.name : '?'
      }
      return [v('X'), v('Y')] as const
    })
  }, [runs, problem])

  const spec = useMemo(() => {
    const isParent = (a: string, b: string) => EXTENDED_PARENT.some(([x, y]) => x === a && y === b)
    const extra: DiagramEdge[] = invented
      .filter(([a, b]) => !isParent(a, b))
      .map(([a, b]) => ({
        from: a,
        to: b,
        route: 'curve' as const,
        bend: 0.9,
        arrow: 'end' as const,
        tone: 2,
        highlight: true,
      }))
    const parentTone = invented.some(([a, b]) => isParent(a, b))
    return familyDiagram(
      {
        extra,
        edgeTone: (a, b) => (parentTone && invented.some(([x, y]) => x === a && y === b) ? 2 : undefined),
      },
      { extended: true },
    )
  }, [invented])

  return (
    <Figure
      title="One task, with and without an invented predicate"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      readouts={
        <>
          <Readout
            label="examples"
            value={`+ ${problem.positives.map(([a, b]) => `${a}→${b}`).join(', ')}   − ${problem.negatives.map(([a, b]) => `${a}→${b}`).join(', ')}`}
          />
          <Readout
            label="invented predicate inv1 holds of"
            value={invented.length ? invented.map(([a, b]) => `${a}→${b}`).join(', ') : 'none invented'}
          />
        </>
      }
      caption="Metagol with the metarules ident P(A, B) ← Q(A, B) and chain P(A, B) ← Q(A, C), R(C, B), on the family extended by Hal (son of Gus) and Ivy (daughter of Hal). Left: the program found when no predicate may be invented; right: the program found when one may. Iterative deepening returns a program with the fewest clauses either way. Pick a target and the clause bound. In the tree, the pairs of which the invented predicate inv1 holds are drawn in the third colour: the parent arrows themselves when inv1 is parent, curves when it is grandparent."
    >
      <div className="grid h-full gap-3 md:grid-cols-[minmax(0,0.7fr)_minmax(0,1.3fr)]">
        <div className="min-h-64">
          <Diagram spec={spec} ariaLabel="Family tree with the pairs of the invented predicate" />
        </div>
        <div className="grid h-full content-start gap-4 overflow-auto md:grid-cols-2">
          <Side title="invention off" run={runs.without} max={state.maxClauses} />
          <Side title="one invented predicate allowed" run={runs.with} max={state.maxClauses} />
        </div>
      </div>
    </Figure>
  )
}
