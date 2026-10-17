import { useMemo, useState } from 'react'
import { trace } from 'aifn-compute/foundation/trace'
import { foilProblem, foilSteps, type FoilState } from 'aifn-compute/logic/induction'
import {
  choice,
  cn,
  Diagram,
  Figure,
  Player,
  Readout,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  toggle,
  useFigureState,
  type DiagramEdge,
} from 'aifn-render'
import { FAMILY_TEXT, familyDiagram } from '../_shared/family'

const TASKS = {
  grandparent: 'grandparent(ann, dee). grandparent(ann, eve). grandparent(ann, fay). grandparent(bob, gus).',
  daughter: 'daughter(cid, ann). daughter(dee, bob). daughter(eve, bob). daughter(fay, cid).',
  ancestor:
    'ancestor(ann, bob). ancestor(ann, cid). ancestor(bob, dee). ancestor(bob, eve). ancestor(cid, fay). ancestor(eve, gus). ' +
    'ancestor(ann, dee). ancestor(ann, eve). ancestor(ann, fay). ancestor(bob, gus). ancestor(ann, gus).',
} as const

const schema = {
  task: choice(['grandparent', 'daughter', 'ancestor'] as const, 'grandparent', { label: 'target relation' }),
  recursion: toggle(true, { label: 'allow the target in bodies' }),
}

const fmt = (g: number) => (Number.isFinite(g) ? g.toFixed(2) : '−∞')

function explain(s: FoilState): string {
  const e = s.event
  switch (e.kind) {
    case 'start':
      return `Step 0: the clause ${s.currentText} covers every example: ${s.current.pos.length} positive and ${s.current.neg.length} negative tuples. Each step adds the literal with the highest FOIL gain.`
    case 'literal':
      return `Step ${s.t}: add ${e.chosen.text} (gain ${fmt(e.chosen.gain)}). The clause now has ${e.chosen.p} positive and ${e.chosen.n} negative tuples, from ${s.coveredPositives.length} positive and ${s.coveredNegatives.length} negative examples.`
    case 'clause':
      return `Step ${s.t}: the clause covers no negative, so it is learnt${e.clause.removed.length ? ` and simplified by dropping ${e.clause.removed.join(', ')}` : ''}: ${e.clause.text} Its ${e.clause.newlyCovered.length} positives are removed${s.uncovered.length ? `, and ${s.uncovered.length} remain for the next clause.` : '; none remain.'}`
    case 'stuck':
      return `Step ${s.t}: stopped: ${e.reason}.`
    case 'done':
      return `Step ${s.t}: every positive is covered.`
  }
}

/** FOIL learning a family relation: the candidate literals of each step with their gains, and the clause grown. */
export function FoilStepper() {
  const state = useFigureState(schema)
  const key = `${state.task}-${state.recursion}`
  const [pos, setPos] = useState({ key, step: 0 })

  const run = useMemo(() => {
    const problem = foilProblem(FAMILY_TEXT, TASKS[state.task])
    const states = trace(foilSteps(problem, { recursion: state.recursion }), undefined, 40, { keep: 'all' })
      .steps as FoilState[]
    const end = states.findIndex((s) => s.terminated)
    return { problem, states: end >= 0 ? states.slice(0, end + 1) : states }
  }, [state.task, state.recursion])
  const last = run.states.length - 1
  const t = pos.key === key ? Math.min(pos.step, last) : 0
  const s = run.states[t]
  // The candidates scored at this step (empty at the start and when a clause is committed).
  const scored = s.candidates
  const chosen = s.event.kind === 'literal' ? s.event.chosen.text : null
  const rows = scored.slice(0, 8)

  const spec = useMemo(() => {
    const { positives, negatives, constants } = run.problem
    const covP = new Set(s.coveredPositives)
    const uncovered = new Set(s.uncovered)
    const covN = s.coveredNegatives
    const name = (i: number) => constants[i]
    const extra: DiagramEdge[] = [
      ...positives.map(([a, b], i) => ({
        from: name(a),
        to: name(b),
        route: 'curve' as const,
        bend: 0.9,
        arrow: 'end' as const,
        tone: 2,
        dashed: !covP.has(i),
        highlight: covP.has(i),
        state: uncovered.has(i) ? undefined : ('idle' as const),
      })),
      // At most 12 covered negatives are drawn, so the tree stays readable at the first steps.
      ...covN.slice(0, 12).map((i) => ({
        from: name(negatives[i][0]),
        to: name(negatives[i][1]),
        route: 'curve' as const,
        bend: -0.9,
        arrow: 'end' as const,
        tone: 1,
        state: 'idle' as const,
      })),
    ]
    return familyDiagram({ extra })
  }, [run, s])

  return (
    <Figure
      title="FOIL growing clauses literal by literal"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      controls={
        <Player
          label="step"
          value={t}
          onChange={(k) => setPos({ key, step: k })}
          count={run.states.length}
          format={(k) => `${k} of ${last}`}
        />
      }
      equation={
        <div className="grid gap-2 text-left text-sm">
          <div className="font-mono">{s.currentText}</div>
          <div className="font-prose text-[15px] leading-snug">{explain(s)}</div>
        </div>
      }
      readouts={
        <>
          <Readout label="learnt clauses" value={s.clauses.map((c) => c.text).join('  ') || 'none yet'} />
          <Readout label="positives left" value={`${s.uncovered.length} of ${run.problem.positives.length}`} />
          <Readout label="closed-world negatives" value={run.problem.negatives.length} />
        </>
      }
      caption="Background knowledge: the family's parent facts (arrows) and female facts (shaded). Negatives are every pair of people that is not a positive (closed world). Step through the search. At each step FOIL scores every literal that uses at least one variable of the clause; the table lists the best eight with the positive tuples p and negative tuples n after the literal, the positive tuples t it extends, and the gain, and the chosen literal is bold. In the tree, curves in the third colour are positive examples (solid when covered by the clause being grown, faded once an earlier clause covers them), and curves in the second colour are negatives the clause still covers (at most 12 drawn)."
    >
      <div className="grid h-full gap-3 md:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)]">
        <div className="min-h-56">
          <Diagram spec={spec} ariaLabel="Family tree with the examples covered by the clause being grown" />
        </div>
        <div className="overflow-auto">
          {rows.length > 0 ? (
            <Table className="text-xs">
              <TableHeader>
                <TableRow>
                  <TableHead>literal</TableHead>
                  <TableHead className="text-right">p</TableHead>
                  <TableHead className="text-right">n</TableHead>
                  <TableHead className="text-right">t</TableHead>
                  <TableHead className="text-right">gain</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((c) => (
                  <TableRow key={c.text} className={cn(c.text === chosen && 'bg-muted font-semibold')}>
                    <TableCell className="font-mono">{c.text}</TableCell>
                    <TableCell className="text-right">{c.p}</TableCell>
                    <TableCell className="text-right">{c.n}</TableCell>
                    <TableCell className="text-right">{c.t}</TableCell>
                    <TableCell className="text-right">{fmt(c.gain)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-sm text-muted-foreground">
              No literal is scored at this step. Step on to see the candidates for the next literal.
            </p>
          )}
          {scored.length > rows.length && (
            <p className="mt-1 text-xs text-muted-foreground">
              {scored.length - rows.length} more candidates not shown.
            </p>
          )}
        </div>
      </div>
    </Figure>
  )
}
