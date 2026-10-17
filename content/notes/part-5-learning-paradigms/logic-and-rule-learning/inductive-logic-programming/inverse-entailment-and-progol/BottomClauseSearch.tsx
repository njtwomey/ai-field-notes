import { useMemo, useState } from 'react'
import { parseProgram, parseTerm, termToString, type Term } from 'aifn-compute/logic/terms'
import {
  choice,
  Diagram,
  Figure,
  Player,
  Readout,
  useFigureState,
  type DiagramEdge,
  type DiagramNode,
  type DiagramSpec,
  type ElementState,
  type Tone,
} from 'aifn-render'
import { bottomClause, bottomClauseSearch, parseMode, type BottomSearchStep } from 'aifn-compute/logic/induction'
import { FAMILY_TEXT } from '../_shared/family'

const FACTS: Term[] = parseProgram(FAMILY_TEXT).clauses.map((c) => c.head)
const POSITIVES = ['daughter(cid, ann)', 'daughter(dee, bob)', 'daughter(eve, bob)', 'daughter(fay, cid)']
const NEGATIVES = ['daughter(bob, ann)', 'daughter(gus, eve)', 'daughter(ann, cid)', 'daughter(dee, ann)']
const HEAD = parseMode('daughter(+person, +person)')
const BODY = ['female(+person)', 'parent(+person, -person)', 'parent(-person, +person)'].map(parseMode)

const schema = {
  seed: choice(
    POSITIVES.map((p) => ({ value: p, label: p })),
    POSITIVES[0],
    { label: 'seed example' },
  ),
}

const DECISION: Record<BottomSearchStep['decision'], string> = {
  refine: 'refined',
  consistent: 'covers no negative',
  pruned: 'pruned',
  leaf: 'nothing to add',
}
const keyOf = (s: readonly number[]) => s.join(',')

/** Progol's bottom clause for one seed example and the general-to-specific search of the clauses above it. */
export function BottomClauseSearch() {
  const state = useFigureState(schema)
  const [pos, setPos] = useState({ seed: state.seed as string, step: 0 })

  const run = useMemo(() => {
    const bottom = bottomClause(FACTS, parseTerm(state.seed), HEAD, BODY, 1)
    const steps = bottomClauseSearch(bottom, FACTS, POSITIVES.map(parseTerm), NEGATIVES.map(parseTerm))
    const k = bottom.literals.length
    const subsets: number[][] = []
    for (let m = 0; m < 1 << k; m++) subsets.push(Array.from({ length: k }, (_, i) => i).filter((i) => m & (1 << i)))
    return { bottom, steps, subsets }
  }, [state.seed])
  const last = run.steps.length - 1
  const t = pos.seed === state.seed ? Math.min(pos.step, last) : 0
  const s = run.steps[t]

  const spec: DiagramSpec = useMemo(() => {
    const visited = new Map<string, BottomSearchStep>()
    run.steps.slice(0, t + 1).forEach((x) => visited.set(keyOf(x.clause.subset), x))
    const levels = new Map<number, number[][]>()
    run.subsets.forEach((sub) => levels.set(sub.length, [...(levels.get(sub.length) ?? []), sub]))
    const place = new Map<string, { x: number; y: number }>()
    for (const [l, subs] of levels)
      subs.forEach((sub, i) => place.set(keyOf(sub), { x: (i - (subs.length - 1) / 2) * 1.9, y: l * 1.9 }))
    const best = s.best ? keyOf(s.best.subset) : null
    const current = keyOf(s.clause.subset)
    const nodes: DiagramNode[] = run.subsets.map((sub) => {
      const k = keyOf(sub)
      const v = visited.get(k)
      const tone: Tone = !v ? 'neutral' : v.decision === 'consistent' ? 2 : v.decision === 'pruned' ? 'neutral' : 0
      const st: ElementState = k === current ? 'active' : v ? 'done' : 'idle'
      return {
        id: `s${k}`,
        ...place.get(k)!,
        shape: 'pill' as const,
        w: 1.6,
        h: 0.75,
        label: sub.length ? sub.map((i) => i + 1).join(' ') : '∅',
        tone,
        state: st,
        filled: v?.decision === 'consistent',
        selected: k === best,
      }
    })
    const edges: DiagramEdge[] = []
    for (const sub of run.subsets)
      for (let i = 0; i < run.bottom.literals.length; i++)
        if (!sub.includes(i)) {
          const sup = [...sub, i].sort((a, b) => a - b)
          edges.push({
            from: `s${keyOf(sub)}`,
            to: `s${keyOf(sup)}`,
            route: 'straight',
            arrow: 'none',
            state: visited.has(keyOf(sup)) ? undefined : 'idle',
          })
        }
    return { unit: 40, nodes, edges }
  }, [run, t, s])

  const evaluated = run.steps.slice(0, t + 1).length
  return (
    <Figure
      title="The bottom clause and the search above it"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      controls={
        <Player
          label="clause"
          value={t}
          onChange={(k) => setPos({ seed: state.seed, step: k })}
          count={run.steps.length}
          format={(k) => `${k + 1} of ${last + 1}`}
        />
      }
      equation={
        <div className="grid gap-2 text-left text-sm md:grid-cols-2">
          <div>
            <div className="text-xs text-muted-foreground">bottom clause of {state.seed}</div>
            <ol className="font-mono text-xs leading-relaxed">
              <li>head {termToString(run.bottom.head)}</li>
              {run.bottom.literals.map((l, i) => (
                <li key={i}>
                  {i + 1}. {termToString(l.literal)}
                </li>
              ))}
            </ol>
          </div>
          <div>
            <div className="font-mono text-xs">{s.clause.text}</div>
            <div className="font-prose text-[15px] leading-snug">{s.note}</div>
          </div>
        </div>
      }
      readouts={
        <>
          <Readout label="clauses evaluated" value={`${evaluated} of ${run.subsets.length}`} />
          <Readout
            label="this clause"
            value={`${s.clause.positives.length}+ ${s.clause.negatives.length}−, score ${s.clause.score}, ${DECISION[s.decision]}`}
          />
          <Readout label="best so far (ringed)" value={s.best ? `${s.best.text} score ${s.best.score}` : 'none'} />
        </>
      }
      caption="Background knowledge: the family's parent and female facts. Modes: daughter(+person, +person) in the head; female(+person), parent(+person, −person) and parent(−person, +person) in the body. Pick a seed example: its bottom clause (top left) lists every literal the modes allow that the background makes true of it, with its constants replaced by variables. Each node of the lattice is a clause whose body is a subset of those literals (numbers), from the empty body at the top to the bottom clause at the bottom. Step through the breadth-first search, scored on 4 positive and 4 negative examples by P − N − L + 1 (L literals including the head). Visited clauses take the first colour, clauses that cover no negative are shaded in the third colour and not refined, and grey visited clauses were pruned; the best clause so far is ringed."
    >
      <Diagram
        spec={spec}
        ariaLabel="Lattice of clauses between the empty body and the bottom clause, with the search's visited clauses"
      />
    </Figure>
  )
}
