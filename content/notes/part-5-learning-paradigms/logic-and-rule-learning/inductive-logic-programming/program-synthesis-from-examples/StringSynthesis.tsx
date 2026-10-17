import { useMemo } from 'react'
import {
  choice,
  cn,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useAxis,
  useFigureState,
} from 'aifn-render'
import {
  atomText,
  countByLength,
  runStringProgram,
  topPrograms,
  versionSpace,
  type StringExample,
} from 'aifn-methods/algorithms/program-synthesis'

const TASKS: Record<'names' | 'dates' | 'emails', { label: string; examples: StringExample[] }> = {
  names: {
    label: 'names to “Last, F.”',
    examples: [
      { input: 'jane doe', output: 'Doe, J.' },
      { input: 'alan turing', output: 'Turing, A.' },
      { input: 'grace brewster hopper', output: 'Hopper, G.' },
      { input: 'ada lovelace', output: 'Lovelace, A.' },
      { input: 'john von neumann', output: 'Neumann, J.' },
    ],
  },
  dates: {
    label: 'ISO dates to day/month/year',
    examples: [
      { input: '2026-10-17', output: '17/10/2026' },
      { input: '1999-01-01', output: '01/01/1999' },
      { input: '2024-02-29', output: '29/02/2024' },
      { input: '1969-07-20', output: '20/07/1969' },
    ],
  },
  emails: {
    label: 'e-mail addresses to names',
    examples: [
      { input: 'jane.doe@example.com', output: 'Jane Doe' },
      { input: 'alan.turing@cam.ac.uk', output: 'Alan Turing' },
      { input: 'grace.hopper@navy.mil', output: 'Grace Hopper' },
      { input: 'ada.lovelace@analytical.org', output: 'Ada Lovelace' },
    ],
  },
}

const schema = {
  task: choice(
    (Object.keys(TASKS) as (keyof typeof TASKS)[]).map((k) => ({ value: k, label: TASKS[k].label })),
    'names',
    { label: 'task' },
  ),
  given: int(1, { label: 'examples given', min: 1, max: 3, suggestions: [1, 2, 3] }),
  ranking: choice(
    [
      { value: 'fewest-atoms', label: 'fewest atoms' },
      { value: 'fewest-constants', label: 'fewest constant characters' },
    ] as const,
    'fewest-constants',
    { label: 'ranking' },
  ),
}

/** Programming by example: the version space of string programs as examples arrive, and the top-ranked programs. */
export function StringSynthesis() {
  const state = useFigureState(schema)
  const { examples } = TASKS[state.task]
  const given = examples.slice(0, state.given)
  const space = useMemo(() => versionSpace(given), [state.task, state.given]) // eslint-disable-line react-hooks/exhaustive-deps
  const counts = useMemo(() => countByLength(space), [space])
  const top = useMemo(() => topPrograms(space, 5, state.ranking), [space, state.ranking])
  const total = counts.reduce((s, x) => s + x, 0)
  const best = top[0]

  const lengths = counts.map((c, k) => ({ k, c })).filter(({ c }) => c > 0)
  const xa = useAxis({ label: 'atoms in the program', range: [0, 11], integer: true })
  const ya = useAxis({ label: 'consistent programs', log: true, range: [0.5, 2e5] })

  return (
    <Figure
      title="Programs consistent with the examples so far"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      readouts={
        <>
          <Readout label="consistent programs" value={formatNumber(total)} />
          <Readout label="version-space graph" value={`${space.nodes.length} nodes, ${space.edges.length} edges`} />
          <Readout label="top program" value={best ? best.atoms.map(atomText).join(' + ') : 'none'} />
        </>
      }
      caption="The language: a program concatenates atoms, each a constant string, a token of the input (a run of letters or digits, counted from the start, or from the end when negative) with a case change, or a token's first character. The version space of the given examples is a graph over output positions whose paths are the consistent programs; the plot counts them by length on a log scale. Pick a task, add examples one at a time, and switch the ranking: with one example the cheapest program by number of atoms is the output itself as a constant. Rows below the line are held out: the top program's output is compared with the expected one."
    >
      <div className="grid h-full gap-3 md:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <Plot x={xa} y={ya}>
          {lengths.length > 0 && (
            <Points name="programs of each length" x={lengths.map((l) => l.k)} y={lengths.map((l) => l.c)} slot={0} />
          )}
        </Plot>
        <div className="h-full overflow-auto">
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>input</TableHead>
                <TableHead>expected</TableHead>
                <TableHead>top program gives</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {examples.map((e, i) => {
                const out = best ? runStringProgram(best.atoms, e.input) : null
                const ok = out === e.output
                return (
                  <TableRow key={e.input} className={cn(i === state.given && 'border-t-2 border-foreground')}>
                    <TableCell className="font-mono">{e.input}</TableCell>
                    <TableCell className="font-mono">{e.output}</TableCell>
                    <TableCell className={cn('font-mono', !ok && 'font-semibold text-destructive')}>
                      {i < state.given ? 'given' : (out ?? '(undefined)')}
                      {i >= state.given && (ok ? ' ✓' : ' ✗')}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
          <div className="mt-3 text-xs text-muted-foreground">top programs ({state.ranking.replace('-', ' ')})</div>
          <ol className="font-mono text-xs leading-relaxed">
            {top.map((p, i) => (
              <li key={i}>
                {i + 1}. {p.atoms.map(atomText).join(' + ')}
              </li>
            ))}
          </ol>
        </div>
      </div>
    </Figure>
  )
}
