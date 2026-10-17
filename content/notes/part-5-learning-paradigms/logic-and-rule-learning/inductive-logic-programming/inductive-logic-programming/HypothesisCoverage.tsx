import { useMemo, useState } from 'react'
import { prologProgram, solveQuery } from 'aifn-compute/logic/resolution'
import {
  choice,
  cn,
  Diagram,
  Figure,
  Readout,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useFigureState,
  type DiagramEdge,
} from 'aifn-render'
import { FAMILY_TEXT, familyDiagram } from '../_shared/family'

interface Task {
  readonly label: string
  readonly target: string
  readonly positives: readonly (readonly [string, string])[]
  readonly negatives: readonly (readonly [string, string])[]
  /** Candidate hypotheses: programs as Prolog text, one clause per line. */
  readonly hypotheses: readonly string[]
  readonly initial: number
}

const TASKS: Record<'grandparent' | 'daughter' | 'ancestor', Task> = {
  grandparent: {
    label: 'grandparent',
    target: 'grandparent',
    positives: [
      ['ann', 'dee'],
      ['ann', 'eve'],
      ['ann', 'fay'],
      ['bob', 'gus'],
    ],
    negatives: [
      ['ann', 'bob'],
      ['cid', 'gus'],
      ['dee', 'ann'],
      ['ann', 'gus'],
      ['bob', 'fay'],
    ],
    hypotheses: [
      'grandparent(X, Y).',
      'grandparent(X, Y) :- parent(X, Y).',
      'grandparent(X, Y) :- parent(X, Z).',
      'grandparent(X, Y) :- parent(Z, Y).',
      'grandparent(X, Y) :- parent(X, Z), parent(Z, Y).',
      'grandparent(X, Y) :- parent(X, Z), parent(Z, Y), female(X).',
    ],
    initial: 2,
  },
  daughter: {
    label: 'daughter',
    target: 'daughter',
    positives: [
      ['cid', 'ann'],
      ['dee', 'bob'],
      ['eve', 'bob'],
      ['fay', 'cid'],
    ],
    negatives: [
      ['bob', 'ann'],
      ['gus', 'eve'],
      ['ann', 'cid'],
      ['dee', 'ann'],
    ],
    hypotheses: [
      'daughter(X, Y).',
      'daughter(X, Y) :- female(X).',
      'daughter(X, Y) :- parent(Y, X).',
      'daughter(X, Y) :- female(X), parent(Y, X).',
      'daughter(X, Y) :- female(Y), parent(Y, X).',
      'daughter(X, Y) :- female(X), parent(X, Y).',
    ],
    initial: 1,
  },
  ancestor: {
    label: 'ancestor (recursive)',
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
    hypotheses: [
      'ancestor(X, Y) :- parent(X, Z).',
      'ancestor(X, Y) :- parent(X, Y).',
      'ancestor(X, Y) :- parent(X, Z), parent(Z, Y).',
      'ancestor(X, Y) :- parent(X, Y).\nancestor(X, Y) :- parent(X, Z), parent(Z, Y).',
      'ancestor(X, Y) :- parent(X, Y).\nancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).',
    ],
    initial: 3,
  },
}

const schema = {
  task: choice(
    (Object.keys(TASKS) as (keyof typeof TASKS)[]).map((k) => ({ value: k, label: TASKS[k].label })),
    'grandparent',
    { label: 'target relation' },
  ),
}

/** Which candidate programs cover which examples, by running each example as a query against B ∪ H. */
export function HypothesisCoverage() {
  const state = useFigureState(schema)
  const task = TASKS[state.task]
  const [picked, setPicked] = useState<{ task: string; row: number } | null>(null)
  const row = picked && picked.task === state.task ? picked.row : task.initial

  const table = useMemo(
    () =>
      task.hypotheses.map((h) => {
        const program = prologProgram(`${FAMILY_TEXT}\n${h}`, { library: false })
        const covers = ([a, b]: readonly [string, string]) =>
          solveQuery(program, `${task.target}(${a}, ${b})`, { maxSolutions: 1, maxSteps: 5000 }).solutions.length > 0
        return { text: h, pos: task.positives.map(covers), neg: task.negatives.map(covers) }
      }),
    [task],
  )
  const current = table[row]
  const p = current.pos.filter(Boolean).length
  const n = current.neg.filter(Boolean).length

  const spec = useMemo(() => {
    const extra: DiagramEdge[] = [
      ...task.positives.map(([a, b], i) => ({
        from: a,
        to: b,
        route: 'curve' as const,
        bend: 0.9,
        arrow: 'end' as const,
        tone: 2,
        dashed: !current.pos[i],
        highlight: current.pos[i],
        state: current.pos[i] ? undefined : ('idle' as const),
      })),
      ...task.negatives.flatMap(([a, b], i) =>
        current.neg[i]
          ? [{ from: a, to: b, route: 'curve' as const, bend: -0.9, arrow: 'end' as const, tone: 1, highlight: true }]
          : [],
      ),
    ]
    return familyDiagram({ extra })
  }, [task, current])

  const verdict =
    p === task.positives.length && n === 0
      ? 'complete and consistent'
      : p === task.positives.length
        ? 'complete, not consistent'
        : n === 0
          ? 'consistent, not complete'
          : 'neither complete nor consistent'
  const example = ([a, b]: readonly [string, string]) => `${a}, ${b}`

  return (
    <Figure
      title="Candidate hypotheses and the examples they cover"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      readouts={
        <>
          <Readout label="positives covered" value={`${p} of ${task.positives.length}`} />
          <Readout label="negatives covered" value={`${n} of ${task.negatives.length}`} />
          <Readout label="selected H is" value={verdict} />
        </>
      }
      caption="Background knowledge B: the parent facts drawn as arrows from parent to child, and female facts (shaded people). Pick a target relation, then click a row of the table to choose a hypothesis H. A cell is ✓ when B ∪ H entails the example, decided by running it as a Prolog query. In the tree, a solid curve in the third colour is a positive example H covers and a dashed one a positive it misses; a curve in the second colour is a negative example H wrongly covers. A complete and consistent row covers every positive and no negative."
    >
      <div className="grid h-full gap-3 md:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <div className="min-h-56">
          <Diagram spec={spec} ariaLabel="Family tree with the examples covered by the selected hypothesis" />
        </div>
        <div className="overflow-auto">
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>hypothesis H</TableHead>
                {task.positives.map((e) => (
                  <TableHead key={`p${example(e)}`} className="text-center whitespace-nowrap">
                    + {example(e)}
                  </TableHead>
                ))}
                {task.negatives.map((e) => (
                  <TableHead key={`n${example(e)}`} className="text-center whitespace-nowrap">
                    − {example(e)}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {table.map((r, i) => (
                <TableRow
                  key={r.text}
                  className={cn('cursor-pointer', i === row && 'bg-muted font-semibold')}
                  onClick={() => setPicked({ task: state.task, row: i })}
                  aria-selected={i === row}
                >
                  <TableCell className="font-mono whitespace-pre">{r.text}</TableCell>
                  {r.pos.map((c, k) => (
                    <TableCell key={`p${k}`} className={cn('text-center', !c && 'text-muted-foreground')}>
                      {c ? '✓' : '·'}
                    </TableCell>
                  ))}
                  {r.neg.map((c, k) => (
                    <TableCell
                      key={`n${k}`}
                      className={cn('text-center', c ? 'font-semibold' : 'text-muted-foreground')}
                    >
                      {c ? '✓' : '·'}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </Figure>
  )
}
