import { prologProgram, solveQuery } from 'aifn-compute/logic/resolution'
import { applySubstitution, parseProgram, parseTerm, termToString, unify } from 'aifn-compute/logic/terms'
import {
  choice,
  cn,
  Figure,
  Readout,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useFigureState,
} from 'aifn-render'

/** Background knowledge B: ground facts about a small family. */
const BACKGROUND = `parent(ann, mary). parent(ann, tom). parent(tom, eve). parent(tom, ian).
female(ann). female(mary). female(eve).`

const EXAMPLES = [
  { text: 'daughter(mary, ann)', positive: true },
  { text: 'daughter(eve, tom)', positive: true },
  { text: 'daughter(tom, ann)', positive: false },
  { text: 'daughter(eve, ann)', positive: false },
  { text: 'daughter(ian, tom)', positive: false },
]

const HYPOTHESES = [
  { value: 'top', label: 'daughter(X, Y).', clause: 'daughter(X, Y).' },
  { value: 'female', label: 'daughter(X, Y) :- female(X).', clause: 'daughter(X, Y) :- female(X).' },
  { value: 'child', label: 'daughter(X, Y) :- parent(Y, X).', clause: 'daughter(X, Y) :- parent(Y, X).' },
  {
    value: 'both',
    label: 'daughter(X, Y) :- female(X), parent(Y, X).',
    clause: 'daughter(X, Y) :- female(X), parent(Y, X).',
  },
  {
    value: 'reversed',
    label: 'daughter(X, Y) :- female(X), parent(X, Y).',
    clause: 'daughter(X, Y) :- female(X), parent(X, Y).',
  },
] as const

const schema = {
  hypothesis: choice(
    HYPOTHESES.map(({ value, label }) => ({ value, label })),
    'female',
    { label: 'hypothesis H' },
  ),
}

type HypothesisName = (typeof HYPOTHESES)[number]['value']
type Row = (typeof EXAMPLES)[number] & { covered: boolean; goals: { text: string; holds: boolean }[]; binding: string }

/** Each query needs one proof at most, in a handful of steps; the bound keeps every call short. */
const LIMITS = { maxSolutions: 1, maxSteps: 500, maxDepth: 20 }
const proves = (program: ReturnType<typeof prologProgram>, query: string) =>
  solveQuery(program, query, LIMITS).solutions.length > 0

// The rows of each hypothesis are computed once, on first use, and kept for the life of the page.
const CACHE = new Map<HypothesisName, Row[]>()
let backgroundProgram: ReturnType<typeof prologProgram> | null = null
function coverageRows(name: HypothesisName): Row[] {
  const cached = CACHE.get(name)
  if (cached) return cached
  const h = HYPOTHESES.find((x) => x.value === name) ?? HYPOTHESES[0]
  backgroundProgram ??= prologProgram(BACKGROUND, { library: false })
  const background = backgroundProgram
  const program = prologProgram(`${BACKGROUND}\n${h.clause}`, { library: false })
  const clause = parseProgram(h.clause).clauses[0]
  const rows = EXAMPLES.map((e) => {
    const covered = proves(program, e.text)
    // The clause's body under the substitution that maps its head onto the example, with each goal's truth in B.
    const theta = unify(clause.head, parseTerm(e.text))
    const goals = theta
      ? clause.body.map((g) => {
          const text = termToString(applySubstitution(g, theta))
          return { text, holds: proves(background, text) }
        })
      : []
    const binding = theta ? theta.map((b) => `${b.variable.name} ↦ ${termToString(b.value)}`).join(', ') : ''
    return { ...e, covered, goals, binding }
  })
  CACHE.set(name, rows)
  return rows
}

/** Coverage as entailment: an example is covered when B ∪ H proves it. */
export function CoverageByEntailment() {
  const state = useFigureState(schema)
  const rows = coverageRows(state.hypothesis)

  const p = rows.filter((r) => r.positive && r.covered).length
  const n = rows.filter((r) => !r.positive && r.covered).length
  const P = rows.filter((r) => r.positive).length
  const N = rows.length - P

  return (
    <Figure
      title="Coverage is entailment from the background knowledge"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="positives covered" value={`${p} of ${P}`} />
          <Readout label="negatives covered" value={`${n} of ${N}`} />
          <Readout
            label="H is"
            value={
              p === P && n === 0
                ? 'complete and consistent'
                : p === P
                  ? 'complete, not consistent'
                  : n === 0
                    ? 'consistent, not complete'
                    : 'neither'
            }
          />
        </>
      }
      caption={
        <>
          Background knowledge B: {BACKGROUND.replace(/\n/g, ' ')} Pick a hypothesis H for daughter(X, Y). An example is
          covered when B ∪ H ⊨ example, decided here by running the example as a Prolog query against B ∪ H. The column
          on the right shows the body of H under the substitution that maps its head onto the example, with each goal
          marked true or false in B. H is complete when it covers every positive and consistent when it covers no
          negative.
        </>
      }
    >
      <div className="h-full overflow-auto">
        <Table className="text-sm">
          <TableHeader>
            <TableRow>
              <TableHead>example</TableHead>
              <TableHead>label</TableHead>
              <TableHead>covered</TableHead>
              <TableHead>substitution θ</TableHead>
              <TableHead>body of H under θ</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => {
              const right = r.covered === r.positive
              return (
                <TableRow key={r.text}>
                  <TableCell className="font-mono">{r.text}</TableCell>
                  <TableCell>{r.positive ? 'positive' : 'negative'}</TableCell>
                  <TableCell className={cn(!right && 'font-semibold')}>
                    {r.covered ? 'yes' : 'no'}
                    {right ? '' : r.positive ? ' (missed)' : ' (wrongly covered)'}
                  </TableCell>
                  <TableCell className="font-mono text-xs">{r.binding}</TableCell>
                  <TableCell className="font-mono text-xs">
                    {r.goals.length === 0
                      ? 'true (no conditions)'
                      : r.goals.map((g, i) => (
                          <span key={i} className={cn(!g.holds && 'text-muted-foreground line-through')}>
                            {i > 0 ? ', ' : ''}
                            {g.text}
                          </span>
                        ))}
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>
    </Figure>
  )
}
