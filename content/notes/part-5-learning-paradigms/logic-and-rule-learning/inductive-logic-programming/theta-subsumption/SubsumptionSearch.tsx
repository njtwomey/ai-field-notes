import { useMemo, useState } from 'react'
import { thetaSubsumes, targetLiterals, thetaSubsumptionTrace } from 'aifn-compute/logic/induction'
import { parseProgram, termToString, type Term } from 'aifn-compute/logic/terms'
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
} from 'aifn-render'

const PAIRS = [
  {
    value: 'grandparent',
    label: 'grandparent onto a ground clause',
    c: 'grandparent(X, Y) :- parent(X, Z), parent(Z, Y).',
    d: 'grandparent(ann, dee) :- parent(ann, cid), parent(ann, bob), parent(bob, dee), female(dee).',
  },
  {
    value: 'daughter',
    label: 'daughter: no substitution',
    c: 'daughter(X, Y) :- female(X), parent(Y, X).',
    d: 'daughter(ann, cid) :- female(ann), female(cid), parent(ann, cid).',
  },
  {
    value: 'longer',
    label: 'a longer clause subsumes a shorter one',
    c: 'p(X) :- q(X, Y), q(X, Z).',
    d: 'p(A) :- q(A, B).',
  },
  {
    value: 'recursive',
    label: 'entailment without subsumption',
    c: 'p(f(X)) :- p(X).',
    d: 'p(f(f(Y))) :- p(Y).',
  },
  {
    value: 'triangle',
    label: 'a triangle in a 4-cycle (hard case)',
    c: 'cyclic :- e(X, Y), e(Y, Z), e(Z, X).',
    d: 'cyclic :- e(a, b), e(b, a), e(b, c), e(c, b), e(c, d), e(d, c), e(d, a), e(a, d).',
  },
] as const

const schema = {
  pair: choice(
    PAIRS.map(({ value, label }) => ({ value, label })),
    'grandparent',
    { label: 'clauses C and D' },
  ),
}

const clauseOf = (s: string) => {
  const c = parseProgram(s).clauses[0]
  return { head: c.head, body: c.body }
}
const show = (t: Term) => termToString(t)

/** The backtracking search for a substitution θ that maps every literal of C onto a literal of D. */
export function SubsumptionSearch() {
  const state = useFigureState(schema)
  const pair = PAIRS.find((p) => p.value === state.pair) ?? PAIRS[0]
  const [pos, setPos] = useState({ pair: pair.value as string, step: 0 })

  const run = useMemo(() => {
    const c = clauseOf(pair.c)
    const d = clauseOf(pair.d)
    const steps = thetaSubsumptionTrace(c, d, { text: show })
    return { c: [c.head, ...c.body], d: targetLiterals(c, d), steps, engine: thetaSubsumes(c, d) }
  }, [pair])
  const last = run.steps.length - 1
  const t = pos.pair === pair.value ? Math.min(pos.step, last) : 0
  const s = run.steps[t]

  const spec: DiagramSpec = useMemo(() => {
    const rowGap = 1.1
    const n = Math.max(run.c.length, run.d.length)
    const offset = (len: number) => ((n - len) * rowGap) / 2
    const mapped = new Set(s.assignment.map((j, i) => `${i}-${j}`))
    const nodes: DiagramNode[] = [
      ...run.c.map((lit, i) => ({
        id: `c${i}`,
        x: 0,
        y: offset(run.c.length) + i * rowGap,
        shape: 'box' as const,
        w: 5.4,
        h: 0.8,
        label: `${i === 0 ? 'head ' : ''}${show(lit)}`,
        tone: i < s.assignment.length ? 0 : ('ink' as const),
        state: s.literal === i ? ('active' as const) : i < s.assignment.length ? ('done' as const) : undefined,
      })),
      ...run.d.map((lit, j) => ({
        id: `d${j}`,
        x: 8,
        y: offset(run.d.length) + j * rowGap,
        shape: 'box' as const,
        w: 5.4,
        h: 0.8,
        label: `${j === 0 ? 'head ' : ''}${show(lit)}`,
        tone: s.assignment.includes(j) ? 0 : ('ink' as const),
        state: s.target === j ? ('active' as const) : undefined,
      })),
    ]
    const edges: DiagramEdge[] = s.assignment.map((j, i) => ({
      from: `c${i}:e`,
      to: `d${j}:w`,
      route: 'straight' as const,
      arrow: 'end' as const,
      tone: 0,
      highlight: true,
    }))
    if ((s.event === 'clash' || s.event === 'match') && s.literal !== null && s.target !== null) {
      if (!mapped.has(`${s.literal}-${s.target}`))
        edges.push({
          from: `c${s.literal}:e`,
          to: `d${s.target}:w`,
          route: 'straight',
          arrow: 'end',
          tone: 1,
          dashed: true,
          highlight: true,
        })
    }
    return { unit: 40, nodes, edges }
  }, [run, s])

  const theta = s.theta.length ? s.theta.map((b) => `${b.variable.name} ↦ ${show(b.value)}`).join(', ') : '∅'
  const verdict = run.steps[last].event === 'subsumes' ? 'C θ-subsumes D' : 'C does not θ-subsume D'

  return (
    <Figure
      title="Searching for a substitution"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      controls={
        <Player
          label="step"
          value={t}
          onChange={(k) => setPos({ pair: pair.value, step: k })}
          count={run.steps.length}
          format={(k) => `${k} of ${last}`}
        />
      }
      equation={<div className="text-left font-prose text-[15px] leading-snug">{s.note}</div>}
      readouts={
        <>
          <Readout label="θ so far" value={theta} />
          <Readout label="literals mapped" value={`${s.assignment.length} of ${run.c.length}`} />
          <Readout
            label="result"
            value={`${verdict} (${last} steps; engine agrees: ${run.engine === (run.steps[last].event === 'subsumes') ? 'yes' : 'no'})`}
          />
        </>
      }
      caption="C is on the left and D on the right, heads first. Pick a pair and step through the search. Each step tries to map the current literal of C (outlined) onto the next literal of D with the same predicate, binding C's variables by one-way matching; D's variables are renamed apart and act as constants. A solid arrow is a literal already mapped; a dashed arrow is the attempt of this step. When a literal of C has no target left, the search undoes the previous choice and tries that literal's next target."
    >
      <Diagram spec={spec} ariaLabel="Literals of C mapped onto literals of D by the substitution found so far" />
    </Figure>
  )
}
