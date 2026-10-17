import { useMemo, useState } from 'react'
import {
  choice,
  Diagram,
  Figure,
  Player,
  Readout,
  row,
  useFigureState,
  type DiagramEdge,
  type DiagramNode,
  type DiagramSpec,
  type ElementState,
  type Tone,
} from 'aifn-render'
import {
  allConjunctions,
  allInstances,
  conjunctionCoverage,
  conjunctionLevel,
  consistent,
  hasseEdges,
  sameConjunction,
  type AttributeSpace,
  type Conjunction,
  type LabelledInstance,
  bottomUpSearch,
  topDownSearch,
  type SearchStep,
} from 'aifn-compute/logic/induction'

const SPACE: AttributeSpace = {
  names: ['a', 'b', 'c'],
  values: [
    ['0', '1'],
    ['0', '1'],
    ['0', '1'],
  ],
}
const HYPOTHESES = allConjunctions(SPACE)
const EDGES = hasseEdges(HYPOTHESES, SPACE)
const INSTANCES = allInstances(SPACE)
const keyOf = (x: readonly number[]) => `e${x.join('')}`

/** A conjunction as compact KaTeX: a for a = 1, ā for a = 0, juxtaposed; ⊤ and ⊥ at the ends. */
function tex(h: Conjunction): string {
  if (h === null) return '$\\bot$'
  const lits = h.flatMap((v, i) => (v < 0 ? [] : [v === 1 ? SPACE.names[i] : `\\bar{${SPACE.names[i]}}`]))
  return lits.length ? `$${lits.join('')}$` : '$\\top$'
}
const plain = (h: Conjunction) =>
  tex(h)
    .replace(/\$/g, '')
    .replace(/\\bar\{(\w)\}/g, '¬$1')
    .replace('\\top', '⊤')
    .replace('\\bot', '⊥')

const LABEL = [
  { value: 'pos', label: '+' },
  { value: 'neg', label: '−' },
  { value: 'none', label: 'unused' },
] as const
type Label = (typeof LABEL)[number]['value']
const label = (initial: Label, x: readonly number[]) => choice(LABEL, initial, { label: `abc = ${x.join('')}` })

const schema = {
  examples: row('examples', {
    e000: label('none', [0, 0, 0]),
    e001: label('none', [0, 0, 1]),
    e010: label('neg', [0, 1, 0]),
    e011: label('none', [0, 1, 1]),
    e100: label('pos', [1, 0, 0]),
    e101: label('none', [1, 0, 1]),
    e110: label('pos', [1, 1, 0]),
    e111: label('none', [1, 1, 1]),
  }),
}

/** Place each hypothesis by its level in the order (number of tests) and its position within the level. */
function layout(): Record<number, { x: number; y: number }> {
  const levels = new Map<number, number[]>()
  HYPOTHESES.forEach((h, i) => {
    const l = conjunctionLevel(h, SPACE)
    levels.set(l, [...(levels.get(l) ?? []), i])
  })
  const out: Record<number, { x: number; y: number }> = {}
  for (const [l, ids] of levels) ids.forEach((id, k) => (out[id] = { x: (k - (ids.length - 1) / 2) * 1.9, y: l * 2.4 }))
  return out
}
const POS = layout()
const indexOf = (h: Conjunction) => HYPOTHESES.findIndex((g) => sameConjunction(g, h))

type Runs = { top: SearchStep[]; bottom: SearchStep[]; consistentIds: Set<number> }
// Both searches and the consistent set, per labelling of the eight instances (3⁸ labellings at most), computed on
// first use and kept. Each search takes at most 3 refinement steps or 8 generalisations.
const RUNS = new Map<string, Runs>()
function runsOf(key: string, examples: LabelledInstance[]): Runs {
  const cached = RUNS.get(key)
  if (cached) return cached
  const hasPositive = examples.some((e) => e.positive)
  const r: Runs = {
    top: hasPositive ? topDownSearch(SPACE, examples) : [],
    bottom: bottomUpSearch(examples),
    consistentIds: new Set(HYPOTHESES.flatMap((h, i) => (consistent(h, examples) ? [i] : []))),
  }
  if (RUNS.size >= 200) RUNS.clear()
  RUNS.set(key, r)
  return r
}

/** The generality order of conjunctions over three binary attributes, searched top-down and bottom-up. */
export function HasseSearch() {
  const state = useFigureState(schema)
  const [pos, setPos] = useState({ key: '', step: 0 })
  const [hover, setHover] = useState<number | null>(null)

  const examples: LabelledInstance[] = useMemo(
    () =>
      INSTANCES.flatMap((x) => {
        const v = (state.examples as Record<string, Label>)[keyOf(x)]
        return v === 'none' ? [] : [{ x, positive: v === 'pos' }]
      }),
    [state.examples],
  )
  const key = examples.map((e) => `${e.x.join('')}${e.positive ? '+' : '-'}`).join(',')
  const runs = runsOf(key, examples)
  const count = Math.max(runs.top.length, runs.bottom.length, 1)
  const t = pos.key === key ? Math.min(pos.step, count - 1) : 0
  const topNow = runs.top.slice(0, t + 1)
  const bottomNow = runs.bottom.slice(0, t + 1)
  const topIds = useMemo(() => runs.top.slice(0, t + 1).map((s) => indexOf(s.hypothesis)), [runs, t])
  const bottomIds = useMemo(() => runs.bottom.slice(0, t + 1).map((s) => indexOf(s.hypothesis)), [runs, t])

  const spec: DiagramSpec = useMemo(() => {
    const onTop = new Set(topIds)
    const onBottom = new Set(bottomIds)
    const current = new Set([topIds[topIds.length - 1], bottomIds[bottomIds.length - 1]])
    const nodes: DiagramNode[] = HYPOTHESES.map((h, i) => {
      const tone: Tone = onTop.has(i) && onBottom.has(i) ? 2 : onTop.has(i) ? 0 : onBottom.has(i) ? 1 : 'neutral'
      const st: ElementState = current.has(i) ? 'active' : onTop.has(i) || onBottom.has(i) ? 'done' : 'idle'
      return {
        id: `h${i}`,
        ...POS[i],
        shape: 'pill' as const,
        w: 1.6,
        h: 0.8,
        label: tex(h),
        tone,
        state: st,
        filled: runs.consistentIds.has(i),
      }
    })
    const pathEdges = (ids: number[], tone: Tone): DiagramEdge[] =>
      ids.slice(1).map((id, k) => ({
        from: `h${ids[k]}`,
        to: `h${id}`,
        tone,
        highlight: true,
        arrow: 'end' as const,
        route: 'straight' as const,
      }))
    const onPath = new Set(
      [
        ...topIds.slice(1).map((id, k) => [topIds[k], id]),
        ...bottomIds.slice(1).map((id, k) => [bottomIds[k], id]),
      ].flatMap(([a, b]) => [`${a}-${b}`, `${b}-${a}`]),
    )
    const edges: DiagramEdge[] = [
      ...EDGES.filter(([a, b]) => !onPath.has(`${a}-${b}`)).map(([a, b]) => ({
        from: `h${a}`,
        to: `h${b}`,
        route: 'straight' as const,
        arrow: 'none' as const,
        state: 'idle' as const,
      })),
      ...pathEdges(topIds, 0),
      ...pathEdges(bottomIds, 1),
    ]
    return { unit: 40, nodes, edges }
  }, [topIds, bottomIds, runs])

  const describe = (s: SearchStep | undefined) =>
    s ? `${plain(s.hypothesis)}: covers ${s.positives.length}+ ${s.negatives.length}−` : 'no positive example'
  const topStep = topNow[topNow.length - 1]
  const bottomStep = bottomNow[bottomNow.length - 1]
  const hoveredCov = hover === null ? null : conjunctionCoverage(HYPOTHESES[hover], examples)

  return (
    <Figure
      title="Searching the generality order from both ends"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      controls={
        <Player
          label="step"
          value={t}
          onChange={(k) => setPos({ key, step: k })}
          count={count}
          format={(k) => `${k} of ${count - 1}`}
        />
      }
      equation={
        <div className="grid gap-2 text-left text-sm md:grid-cols-2">
          <div>
            <div className="text-xs text-muted-foreground">top-down (general to specific)</div>
            <div>{describe(topStep)}</div>
            <div className="text-muted-foreground">{topStep?.note ?? ''}</div>
            {topStep && topStep.candidates.length > 0 && (
              <div className="font-mono text-xs">
                {topStep.candidates.map((c) => `${plain(c.hypothesis)} ${c.p}+${c.n}− (${c.score})`).join('   ')}
              </div>
            )}
          </div>
          <div>
            <div className="text-xs text-muted-foreground">bottom-up (specific to general)</div>
            <div>{describe(bottomStep)}</div>
            <div className="text-muted-foreground">{bottomStep?.note ?? ''}</div>
          </div>
        </div>
      }
      readouts={
        <>
          <Readout
            label="examples"
            value={`${examples.filter((e) => e.positive).length}+ ${examples.filter((e) => !e.positive).length}−`}
          />
          <Readout label="consistent hypotheses (shaded)" value={runs.consistentIds.size} />
          <Readout
            label={hover === null ? 'hover a hypothesis' : `hypothesis ${plain(HYPOTHESES[hover])}`}
            value={hoveredCov ? `covers ${hoveredCov.positives.length}+ ${hoveredCov.negatives.length}−` : '—'}
          />
        </>
      }
      caption="Each node is a conjunction of literals over three binary attributes (a means a = 1, ā means a = 0); ⊤ at the top covers every instance and ⊥ at the bottom covers none. An edge joins a hypothesis to a one-step specialisation below it. Label each of the eight instances positive, negative or unused, then step. The top-down search (first colour) starts at ⊤ and adds the literal with the best score, positives minus negatives covered, until no negative is covered. The bottom-up search (second colour) starts at ⊥ and generalises just enough to cover each positive in turn. Shaded nodes are consistent with the examples; nodes both searches visit take the third colour."
    >
      <Diagram
        spec={spec}
        ariaLabel="Hasse diagram of the 28 conjunctions over three binary attributes, with the paths of a top-down and a bottom-up search"
        onNodeHover={(id) => setHover(id === null ? null : Number(id.slice(1)))}
      />
    </Figure>
  )
}
