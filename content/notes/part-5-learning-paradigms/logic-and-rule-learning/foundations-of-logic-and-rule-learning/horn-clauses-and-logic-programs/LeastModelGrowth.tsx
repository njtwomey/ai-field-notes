import { useMemo, useState } from 'react'
import {
  clauseToString,
  parseProgram,
  PrologSyntaxError,
  substitutionToString,
  type Clause,
} from 'aifn-compute/logic/terms'
import {
  Button,
  circleLayout,
  cn,
  CodeEditor,
  Diagram,
  Figure,
  Player,
  Readout,
  Select,
  seriesColor,
  StatusText,
  useTheme,
  type DiagramEdge,
  type DiagramSpec,
  type Tone,
} from 'aifn-render'
import { herbrandBaseSize, leastHerbrandModel, type DerivedFact } from 'aifn-compute/logic/resolution'

const PRESETS = {
  family: {
    label: 'family: grandparent and ancestor',
    program: `% Facts: who is a parent of whom.
parent(ann, bob).
parent(bob, cid).
parent(bob, dee).
parent(cid, eve).

% Rules.
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).`,
  },
  cousins: {
    label: 'family: siblings and cousins',
    program: `parent(ann, bob).
parent(ann, cat).
parent(bob, dan).
parent(cat, eli).

sibling(X, Y) :- parent(P, X), parent(P, Y), X \\= Y.
cousin(X, Y) :- parent(P, X), parent(Q, Y), sibling(P, Q).`,
  },
  cycle: {
    label: 'paths in a graph with a cycle',
    program: `edge(a, b).
edge(b, c).
edge(c, d).
edge(d, a).

% On this cycle Prolog's depth-first search for path(a, e) never ends; forward chaining stops after 5 rounds.
path(X, Y) :- edge(X, Y).
path(X, Y) :- path(X, Z), edge(Z, Y).`,
  },
} as const
type PresetName = keyof typeof PRESETS

type Model =
  | { ok: false; message: string; line?: number; column?: number }
  | {
      ok: true
      clauses: Clause[]
      facts: readonly DerivedFact[]
      rounds: number
      fixpoint: boolean
      error: string | null
      base: number
      constants: string[]
    }

const MODELS = new Map<string, Model>()
/** The model of a program text, computed once per text (rounds and facts bounded: 30 rounds, 600 facts). */
function modelOf(source: string): Model {
  const cached = MODELS.get(source)
  if (cached) return cached
  const m = compute(source)
  if (MODELS.size >= 20) MODELS.clear()
  MODELS.set(source, m)
  return m
}

function compute(source: string): Model {
  let clauses: Clause[]
  try {
    clauses = parseProgram(source).clauses
  } catch (e) {
    if (e instanceof PrologSyntaxError) return { ok: false, message: e.message, line: e.line, column: e.column }
    throw e
  }
  const m = leastHerbrandModel(clauses, { maxRounds: 30, maxFacts: 600 })
  const hb = herbrandBaseSize(clauses)
  return { ok: true, clauses, ...m, base: hb.size, constants: hb.constants }
}

const args = (f: DerivedFact) => (f.term.kind === 'compound' ? f.term.args : [])
const name = (f: DerivedFact) => (f.term.kind === 'compound' ? f.term.functor : f.text)
const constantName = (t: DerivedFact['term']) =>
  t.kind === 'atom' ? t.name : t.kind === 'number' ? String(t.value) : ''

/** Positions of the constants: layered by the first binary relation when it has no cycle, else on a circle. */
function placeConstants(constants: string[], facts: readonly DerivedFact[]): Record<string, { x: number; y: number }> {
  const base = facts.find((f) => f.round === 1 && args(f).length === 2)
  const edges = base
    ? facts
        .filter((f) => f.round === 1 && name(f) === name(base))
        .map((f) => args(f).map(constantName) as [string, string])
    : []
  const depth: Record<string, number> = Object.fromEntries(constants.map((c) => [c, 0]))
  let acyclic = true
  for (let pass = 0; pass <= constants.length; pass++) {
    let changed = false
    for (const [a, b] of edges)
      if (depth[b] < depth[a] + 1) {
        depth[b] = depth[a] + 1
        changed = true
      }
    if (!changed) break
    if (pass === constants.length) acyclic = false
  }
  if (!acyclic || edges.length === 0) {
    const pts = circleLayout(constants.length, Math.max(1.6, constants.length * 0.45))
    return Object.fromEntries(constants.map((c, i) => [c, pts[i]]))
  }
  const levels = new Map<number, string[]>()
  for (const c of constants) levels.set(depth[c], [...(levels.get(depth[c]) ?? []), c])
  const out: Record<string, { x: number; y: number }> = {}
  for (const [d, cs] of levels) cs.forEach((c, i) => (out[c] = { x: (i - (cs.length - 1) / 2) * 2.6, y: d * 2 }))
  return out
}

/** Forward chaining: the least Herbrand model of a program, built one application of T_P at a time. */
export function LeastModelGrowth() {
  const [preset, setPreset] = useState<PresetName>('family')
  const [source, setSource] = useState<string>(PRESETS.family.program)
  const [committed, setCommitted] = useState<string>(PRESETS.family.program)
  const [round, setRound] = useState(0)
  const [selected, setSelected] = useState<number | null>(null)
  const { resolved: mode } = useTheme()

  const model = modelOf(committed)
  const commit = (src: string) => {
    setCommitted(src)
    setRound(0)
    setSelected(null)
  }
  const pick = (p: PresetName) => {
    setPreset(p)
    setSource(PRESETS[p].program)
    commit(PRESETS[p].program)
  }

  const facts = model.ok ? model.facts : []
  const last = model.ok ? model.rounds + (model.fixpoint ? 1 : 0) : 0
  const t = Math.min(round, last)
  const visible = useMemo(() => facts.filter((f) => f.round <= t), [facts, t])
  const predicates = useMemo(() => [...new Set(facts.filter((f) => args(f).length === 2).map(name))], [facts])
  const tone = (p: string): Tone => {
    const k = predicates.indexOf(p)
    return k >= 0 && k < 3 ? k : 'neutral'
  }
  const chosen = selected !== null && facts[selected] && facts[selected].round <= t ? facts[selected] : null

  const spec: DiagramSpec | null = useMemo(() => {
    if (!model.ok || model.constants.length === 0) return null
    const pos = placeConstants(model.constants, facts)
    const involved = new Set(chosen ? args(chosen).map(constantName) : [])
    const edges: DiagramEdge[] = visible
      .filter((f) => args(f).length === 2)
      .map((f) => {
        const [a, b] = args(f).map(constantName)
        const k = predicates.indexOf(name(f))
        return {
          from: a,
          to: b,
          route: k === 0 ? ('straight' as const) : ('curve' as const),
          bend: k === 0 ? undefined : 0.5 + 0.45 * k,
          tone: k >= 0 && k < 3 ? k : ('neutral' as const),
          dashed: k > 0,
          state: f.round === t ? ('active' as const) : ('done' as const),
          highlight: chosen === f,
        }
      })
    return {
      unit: 40,
      nodes: model.constants.map((c) => ({
        id: c,
        x: pos[c].x,
        y: pos[c].y,
        shape: 'pill' as const,
        w: 1.5,
        h: 0.8,
        label: c,
        tone: 'ink' as const,
        highlight: involved.has(c),
      })),
      edges,
    }
  }, [model, facts, visible, predicates, chosen, t])

  const byRound = new Map<number, { f: DerivedFact; i: number }[]>()
  facts.forEach((f, i) => {
    if (f.round <= t) byRound.set(f.round, [...(byRound.get(f.round) ?? []), { f, i }])
  })
  const newNow = facts.filter((f) => f.round === t).length
  const colour = (p: string) => {
    const k = tone(p)
    return typeof k === 'number' ? seriesColor(mode, k) : undefined
  }

  return (
    <Figure
      title="The least Herbrand model, one round at a time"
      defaultSize="L"
      hoverReadout={false}
      controls={
        <>
          <Select
            label="program"
            value={preset}
            onChange={pick}
            options={(Object.keys(PRESETS) as PresetName[]).map((k) => ({ value: k, label: PRESETS[k].label }))}
          />
          <Player
            label="round"
            value={t}
            onChange={setRound}
            count={last + 1}
            format={(r) => (r === 0 ? 'T_P ↑ 0 = ∅' : `T_P ↑ ${r}`)}
          />
        </>
      }
      equation={
        <div className="grid gap-3 text-left md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="flex flex-col gap-2">
            <CodeEditor
              value={source}
              onChange={setSource}
              language="prolog"
              label="logic program"
              onRun={() => commit(source)}
              errors={model.ok ? [] : [{ message: model.message, line: model.line, column: model.column }]}
            />
            <div className="flex items-center gap-2">
              <Button size="sm" variant={source !== committed ? 'default' : 'outline'} onClick={() => commit(source)}>
                Run (⌘↵)
              </Button>
              {!model.ok && <StatusText tone="error">{model.message}</StatusText>}
              {model.ok && model.error && <StatusText tone="error">{model.error}</StatusText>}
            </div>
          </div>
          <div className="flex max-h-80 flex-col gap-2 overflow-y-auto text-xs">
            {t === 0 && <div className="text-muted-foreground">Round 0: no facts yet. Step forward to apply T_P.</div>}
            {model.ok && model.fixpoint && t > model.rounds && (
              <div className="font-semibold">
                Round {t}: T_P adds nothing new, so T_P ↑ {model.rounds} is a fixpoint: the least Herbrand model.
              </div>
            )}
            {[...byRound.entries()].map(([r, list]) => (
              <div key={r}>
                <div className="mb-0.5 text-muted-foreground">
                  round {r}: {r === 1 ? 'the facts of the program' : `${list.length} new from the rules`}
                </div>
                <div className="flex flex-wrap gap-1">
                  {list.map(({ f, i }) => (
                    <button
                      key={i}
                      type="button"
                      onClick={() => setSelected(selected === i ? null : i)}
                      className={cn(
                        'rounded border px-1.5 py-0.5 font-mono',
                        r === t && 'font-semibold',
                        selected === i && 'bg-muted',
                      )}
                      style={{ borderColor: colour(name(f)) }}
                    >
                      {f.text}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            {chosen && model.ok && (
              <div className="mt-1 rounded border p-2">
                <div className="font-mono font-semibold">{chosen.text}</div>
                {chosen.premises.length === 0 ? (
                  <div>a fact of the program (clause {chosen.clause + 1})</div>
                ) : (
                  <>
                    <div>
                      clause {chosen.clause + 1}:{' '}
                      <span className="font-mono">{clauseToString(model.clauses[chosen.clause])}</span>
                    </div>
                    <div>
                      substitution: <span className="font-mono">{substitutionToString(chosen.substitution)}</span>
                    </div>
                    <div>
                      from: <span className="font-mono">{chosen.premises.map((p) => facts[p].text).join(', ')}</span>{' '}
                      (rounds {chosen.premises.map((p) => facts[p].round).join(', ')})
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      }
      readouts={
        <>
          <Readout label="facts in T_P ↑ round" value={visible.length} />
          <Readout label="new this round" value={t === 0 ? 0 : newNow} />
          <Readout label="Herbrand base" value={model.ok ? model.base : '—'} />
          <Readout
            label="status"
            value={
              !model.ok || model.error
                ? 'stopped'
                : t === last && model.fixpoint
                  ? 'fixpoint: the least model'
                  : 'growing'
            }
          />
        </>
      }
      caption="Choose a program, or edit it and press Run. Round 0 is the empty interpretation. Each round applies every rule to the facts known before it and adds the instances of the rule heads whose bodies are all known facts. The diagram draws the constants and the facts of two-place predicates as arrows; the first relation is solid and derived relations are curved and dashed, with this round's facts emphasised. Click a fact to see the clause, the substitution and the earlier facts that derived it. The run stops when a round adds nothing: that set is the least Herbrand model."
    >
      {spec ? (
        <Diagram
          spec={spec}
          ariaLabel="The constants of the program, with arrows for the two-place facts derived so far"
        />
      ) : (
        <StatusText>No constants to draw.</StatusText>
      )}
    </Figure>
  )
}
