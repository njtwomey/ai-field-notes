/**
 * Showcase: Prolog. A program and a query are run by aifn's `sldSteps` (SLD resolution, depth first with
 * backtracking) and traced with every state kept; `sldTree` reads the search tree out of the trace. The figure plays
 * the tree from the query (step 0): each step adds one node (a clause used, a built-in run, an alternative resumed)
 * or closes one (a success, a failure). The lab only lays the tree out and colours it by status.
 */
import { useMemo, useState } from 'react'
import { treeFromParents } from 'aifn/graph'
import { trace } from 'aifn/foundation/trace'
import {
  formatSolution,
  prologProgram,
  sldSteps,
  sldTree,
  solveQuery,
  type SldEvent,
  type SldNode,
  type SldNodeStatus,
  type SldSolution,
  type SldState,
  type SolveResult,
} from 'aifn/logic/resolution'
import {
  clauseToString,
  parseQuery,
  PrologSyntaxError,
  listItems,
  substitutionToString,
  termToString,
  type Query,
  type Term,
} from 'aifn/logic/terms'
import { Button, CodeEditor, Player, Select, StatusText, type CodeError } from '@lab/controls'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { int, pinField, useFigureState, usePinned } from '@lab/state'
import { Input } from '@lab/ui/input'
import { cn } from '@lab/lib/utils'
import { TreeView } from '@lab/views'
import { Readout } from '@lab/viz'
import type { Tone } from '@lab/diagram'
import { PRESETS, type PresetName } from './programs'

/** Steps drawn by default; the reader can raise it (a longer search is finished without drawing, with a note). */
const DRAWN_STEPS = 1200
const MAX_DEPTH = 60
/** Steps allowed when a search too long to draw is finished without a trace (e.g. Einstein's puzzle, ~39k steps). */
const FULL_STEPS = 400_000
const FULL_DEPTH = 2000
/** Above this many nodes the tree draws dots without labels. */
const LABELLED_NODES = 40
const NO_NODES: SldNode[] = []

const show = (t: Term) => termToString(t)
const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s)

const STATUS_TONE: Record<SldNodeStatus, Tone> = {
  open: 'ink',
  expanded: 'ink',
  success: 2,
  failure: 7,
  pruned: 'neutral',
  limit: 3,
  error: 7,
}
const STATUS_TEXT: Record<SldNodeStatus, string> = {
  open: 'open: its first goal is next',
  expanded: 'expanded',
  success: 'success: no goals left',
  failure: 'failure',
  pruned: 'pruned by a cut',
  limit: 'cut off at the depth limit',
  error: 'error',
}

/** A node's status as of step t (statuses in the tree are as of the last step). */
function statusAt(n: SldNode, t: number, hasChildren: boolean): SldNodeStatus {
  if (n.closed !== null && n.closed <= t) return n.status
  return hasChildren ? 'expanded' : 'open'
}

/** One step in words. */
function explain(e: SldEvent, s: SldState, program: ReturnType<typeof prologProgram>): string {
  switch (e.kind) {
    case 'start':
      return `Step 0: the query is the first resolvent: ${s.goals!.map((g) => show(g.term)).join(', ')}. Prolog always works on the leftmost goal.`
    case 'resolve': {
      const c = program.clauses[e.clause]
      const where = program.sources[e.clause] === 'library' ? 'library clause' : `clause ${e.clause + 1}`
      return `Step ${s.t}: ${e.retry ? 'backtrack, then try ' : 'resolve '}${show(e.goal)} with ${where}, ${clauseToString(c)} Unifying the goal with the head gives ${substitutionToString(e.unifier)}; the body replaces the goal.${e.alternatives > 0 ? ` ${e.alternatives} more clause${e.alternatives > 1 ? 's' : ''} wait${e.alternatives > 1 ? '' : 's'} at this choice point.` : ''}`
    }
    case 'builtin':
      return `Step ${s.t}: built-in ${show(e.goal)}: ${e.note}${e.unifier.length ? `, ${substitutionToString(e.unifier)}` : ''}.`
    case 'alternative':
      return `Step ${s.t}: backtrack to node ${e.parent} and resume: ${e.note}.`
    case 'success':
      return `Step ${s.t}: no goals left, so the query is proved: ${formatSolution(s.solutions[e.solution])}. Prolog then backtracks to look for more.`
    case 'fail':
      return `Step ${s.t}: failure at node ${e.node}: ${e.reason}. Prolog backtracks to the most recent choice point.`
    case 'error':
      return `Step ${s.t}: error: ${e.message}.`
    case 'exhausted':
      return `Step ${s.t}: no choice points are left: the search is complete with ${s.solutions.length} solution${s.solutions.length === 1 ? '' : 's'}.`
  }
}

type Run =
  | { ok: false; errors: CodeError[]; message: string }
  | {
      ok: true
      program: ReturnType<typeof prologProgram>
      query: Query
      states: SldState[]
      nodes: SldNode[]
      message: string
      /** The whole search, run without a trace, when the drawn part stopped at the step limit. */
      full: SolveResult | null
    }

function runQuery(source: string, queryText: string, drawn: number): Run {
  let program
  try {
    program = prologProgram(source)
  } catch (e) {
    if (e instanceof PrologSyntaxError)
      return { ok: false, errors: [{ message: e.message, line: e.line, column: e.column }], message: e.message }
    throw e
  }
  let query: Query
  try {
    query = parseQuery(queryText)
  } catch (e) {
    return { ok: false, errors: [], message: `query: ${(e as Error).message}` }
  }
  const run = trace(sldSteps(program, query, { maxDepth: MAX_DEPTH }), undefined, drawn, { keep: 'all' })
  const states = run.steps as SldState[]
  const last = states[states.length - 1]
  const n = last.solutions.length
  const message =
    last.stopped === 'error'
      ? `Error: ${last.error}`
      : last.stopped === 'exhausted'
        ? `Search complete: ${n} solution${n === 1 ? '' : 's'} in ${last.t} steps.${last.depthLimited ? ` Some branches were cut off at depth ${MAX_DEPTH}.` : ''}`
        : `Stopped at the step limit (${drawn} steps) with ${n} solution${n === 1 ? '' : 's'}; the search was not finished.`
  if (last.stopped === 'error' || last.stopped === 'exhausted')
    return { ok: true, program, query, states, nodes: sldTree(states, program.clauses), message, full: null }
  // Too long to draw: the tree shows the first `drawn` steps and the same search is finished without a trace.
  const full = solveQuery(program, query, { maxSteps: FULL_STEPS, maxDepth: FULL_DEPTH })
  const m = full.solutions.length
  const fullMessage =
    full.stopped === 'error'
      ? `The tree shows the first ${drawn.toLocaleString()} steps. Finishing the search failed: ${full.message}`
      : full.stopped === 'steps'
        ? `The tree shows the first ${drawn.toLocaleString()} steps. The full search stopped at ${FULL_STEPS.toLocaleString()} steps with ${m} solution${m === 1 ? '' : 's'}, unfinished.`
        : `Search longer than the steps drawn: the tree shows the first ${drawn.toLocaleString()} steps (raise “steps drawn” to play further). Run to the end without drawing, the search found ${m} solution${m === 1 ? '' : 's'} in ${full.steps.toLocaleString()} steps.`
  return { ok: true, program, query, states, nodes: sldTree(states, program.clauses), message: fullMessage, full }
}

/**
 * A solution: as text, except that a binding to a list of compound terms of one arity (a row of houses, h(Colour,
 * Nation, …)) is laid out as a table, one row per term.
 */
function SolutionView({ solution }: { solution: SldSolution }) {
  const tables = solution.bindings.flatMap((b) => {
    if (b.value.kind !== 'compound') return []
    const { items, tail } = listItems(b.value)
    if (tail.kind !== 'atom' || items.length < 2) return []
    const first = items[0]
    if (first.kind !== 'compound') return []
    const ok = items.every(
      (x) => x.kind === 'compound' && x.functor === first.functor && x.args.length === first.args.length,
    )
    return ok ? [{ name: b.name, rows: items as Extract<Term, { kind: 'compound' }>[] }] : []
  })
  if (tables.length === 0) return <>{formatSolution(solution)}</>
  return (
    <div className="flex flex-col gap-1">
      {tables.map(({ name, rows }) => (
        <div key={name}>
          <div className="text-muted-foreground">
            {name} = [{rows[0].functor}(…), …]
          </div>
          <table className="mt-0.5 border-collapse">
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="border-b border-border last:border-0">
                  <td className="pr-3 text-muted-foreground">{i + 1}</td>
                  {r.args.map((a, j) => (
                    <td key={j} className="pr-3">
                      {termToString(a)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  )
}

/** The details of one node: how it was reached, the goals left, and the query's bindings there. */
function NodeDetails({ node, run, t }: { node: SldNode; run: Extract<Run, { ok: true }>; t: number }) {
  const names = run.query.variableNames
  const bindings = names
    .map((name, id) => ({ name, value: node.answer[id] }))
    .filter((b) => !b.name.startsWith('_') && b.value !== undefined)
  const status = statusAt(
    node,
    t,
    run.nodes.some((m) => m.parent === node.id && m.created <= t),
  )
  const clause = node.clause === null ? null : run.program.clauses[node.clause]
  return (
    <div className="flex flex-col gap-1.5 text-xs">
      <div className="flex flex-wrap gap-x-4">
        <span className="font-medium">node {node.id}</span>
        <span className="text-muted-foreground">depth {node.depth}</span>
        <span className="text-muted-foreground">made at step {node.created}</span>
        <span>{STATUS_TEXT[status]}</span>
      </div>
      {node.goal && (
        <div>
          <span className="text-muted-foreground">goal resolved </span>
          <span className="font-mono">{show(node.goal)}</span>
        </div>
      )}
      {clause && (
        <div>
          <span className="text-muted-foreground">
            {run.program.sources[node.clause!] === 'library' ? 'library clause ' : `clause ${node.clause! + 1} `}
          </span>
          <span className="font-mono">{clauseToString(clause)}</span>
        </div>
      )}
      {node.head && (
        <div>
          <span className="text-muted-foreground">renamed head </span>
          <span className="font-mono">{show(node.head)}</span>
        </div>
      )}
      {node.status !== 'pruned' && (
        <div>
          <span className="text-muted-foreground">most general unifier </span>
          <span className="font-mono">
            {node.unifier.length ? substitutionToString(node.unifier) : '{} (nothing bound)'}
          </span>
        </div>
      )}
      <div>
        <span className="text-muted-foreground">{node.note === 'the query' ? 'query ' : 'how '}</span>
        {node.note}
      </div>
      {node.status !== 'pruned' && (
        <div>
          <span className="text-muted-foreground">goals left </span>
          <span className="font-mono">{node.goals.length ? node.goals.map(show).join(', ') : '□ (none: a proof)'}</span>
        </div>
      )}
      {node.status !== 'pruned' && bindings.length > 0 && (
        <div>
          <span className="text-muted-foreground">query bindings here </span>
          <span className="font-mono">{bindings.map((b) => `${b.name} = ${show(b.value)}`).join(', ')}</span>
        </div>
      )}
      {node.reason && status !== 'open' && status !== 'expanded' && (
        <div className="text-muted-foreground">{node.reason}</div>
      )}
    </div>
  )
}

export function PrologShowcase() {
  const figure = useFigureState({
    drawn: int(DRAWN_STEPS, {
      ge: 100,
      le: 50_000,
      suggestions: [1200, 5000, 20_000, 40_000],
      label: 'steps drawn (large trees draw slowly)',
    }),
    pin: pinField(),
  })
  const [preset, setPreset] = useState<PresetName>('family')
  const [source, setSource] = useState<string>(PRESETS.family.program)
  const [queryText, setQueryText] = useState<string>(PRESETS.family.queries[0])
  const [committed, setCommitted] = useState({ source: PRESETS.family.program, query: PRESETS.family.queries[0] })
  const [step, setStep] = useState(0)
  const run = useMemo(() => runQuery(committed.source, committed.query, figure.drawn), [committed, figure.drawn])
  const commit = (src: string, q: string) => {
    setCommitted({ source: src, query: q })
    setStep(0)
    figure.set('pin', -1)
  }
  const pickPreset = (name: PresetName) => {
    setPreset(name)
    setSource(PRESETS[name].program)
    setQueryText(PRESETS[name].queries[0])
    commit(PRESETS[name].program, PRESETS[name].queries[0])
  }
  const stale = committed.source !== source || committed.query !== queryText

  const states = run.ok ? run.states : []
  const last = Math.max(0, states.length - 1)
  const t = Math.min(step, last)
  const state = states[t] as SldState | undefined
  const nodes = run.ok ? run.nodes : NO_NODES
  const pins = usePinned(figure.pin, (v) => figure.set('pin', v), {
    valid: (v) => v < nodes.length && nodes[v].created <= t,
  })
  const current =
    state && state.event.kind !== 'exhausted'
      ? state.goals === null && 'node' in state.event
        ? state.event.node
        : state.node
      : null
  const focus = pins.focus ?? current

  const tree = useMemo(
    () =>
      nodes.length
        ? treeFromParents(
            nodes.map((n) => n.parent),
            { edge: (c) => ({ label: nodes[c].clause !== null ? String(nodes[c].clause! + 1) : '' }) },
          )
        : null,
    [nodes],
  )
  const view = useMemo(() => {
    const created = new Set(nodes.filter((n) => n.created <= t).map((n) => n.id))
    const hasChildren = new Set(nodes.filter((n) => n.created <= t && n.parent !== null).map((n) => n.parent!))
    const status = nodes.map((n) => statusAt(n, t, hasChildren.has(n.id)))
    const compact = nodes.length > LABELLED_NODES
    const labels = nodes.map((n, i) => {
      if (compact) return ''
      const s = status[i]
      const goals = n.goals.length ? n.goals.slice(0, 2).map(show).join(', ') + (n.goals.length > 2 ? ', …' : '') : '□'
      if (s === 'success') {
        const sol = n.solution === null ? null : run.ok ? run.states[last].solutions[n.solution] : null
        return clip(`□ ${sol ? formatSolution(sol) : 'proved'}`, 30)
      }
      if (s === 'failure') return clip(`✗ ${goals}`, 30)
      if (s === 'pruned') return '✂'
      if (s === 'limit') return '…'
      if (s === 'error') return '!'
      return clip(goals, 34)
    })
    const solutionsSoFar = state?.solutions ?? []
    return { created, status, compact, labels, solutionsSoFar }
  }, [nodes, t, state, run, last])

  const editorErrors = run.ok ? [] : run.errors
  const preset0 = PRESETS[preset]
  const solutions = view.solutionsSoFar
  const allSolutions = run.ok ? run.states[last].solutions.length : 0

  return (
    <Figure
      title="Prolog: logic as a program"
      purpose="A Prolog program is facts and rules; a query asks whether something follows from them. Prolog answers by resolution: it replaces the leftmost goal by the body of a clause whose head unifies with it, and when it runs out of clauses it backtracks to the last choice point. Every step is one node of the search tree."
      state={figure}
      defaultSize="XL"
      hoverReadout={false}
      controls={
        <>
          <Select
            label="program"
            value={preset}
            onChange={pickPreset}
            options={(Object.keys(PRESETS) as PresetName[]).map((k) => ({ value: k, label: PRESETS[k].label }))}
          />
          <div className="flex items-end">
            <Button size="sm" variant={stale ? 'default' : 'outline'} onClick={() => commit(source, queryText)}>
              Run query (⌘↵)
            </Button>
          </div>
          <Player
            label="step"
            value={t}
            onChange={setStep}
            count={states.length || 1}
            format={(p) => `${p} of ${last}`}
          />
        </>
      }
      equation={
        <div className="grid gap-3 text-left md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <CodeEditor
            value={source}
            onChange={setSource}
            language="prolog"
            errors={editorErrors}
            onRun={() => commit(source, queryText)}
            label="Prolog program"
          />
          <div className="flex flex-col gap-2">
            <label className="flex flex-col gap-1 text-xs">
              <span className="text-muted-foreground">query (?-)</span>
              <Input
                aria-label="query"
                className="h-8 font-mono text-sm"
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commit(source, queryText)
                }}
              />
            </label>
            <div className="flex flex-wrap gap-1.5">
              {preset0.queries.map((q) => (
                <Button
                  key={q}
                  aria-label={`query ${q}`}
                  size="sm"
                  variant={committed.query === q && committed.source === source ? 'secondary' : 'outline'}
                  className="h-7 font-mono text-xs"
                  onClick={() => {
                    setQueryText(q)
                    commit(source, q)
                  }}
                >
                  ?- {q}
                </Button>
              ))}
            </div>
            {run.ok ? (
              <StatusText
                tone={
                  run.states[last].stopped === 'error'
                    ? 'error'
                    : run.states[last].stopped === null
                      ? 'attention'
                      : undefined
                }
              >
                {run.message}
                {stale ? ' The program or query has changed: run it again.' : ''}
              </StatusText>
            ) : (
              <StatusText tone="error">{run.message}</StatusText>
            )}
            {run.ok && run.full && run.full.solutions.length > 0 && (
              <div className="text-xs">
                <div className="mb-1 text-muted-foreground">
                  solutions of the whole search (run without drawing; {run.full.solutions.length})
                </div>
                <ol className="flex flex-col gap-1 font-mono">
                  {run.full.solutions.slice(0, 20).map((s, i) => (
                    <li key={i}>
                      {i + 1}. <SolutionView solution={s} />
                    </li>
                  ))}
                </ol>
              </div>
            )}
            <div className="text-xs">
              <div className="mb-1 text-muted-foreground">
                solutions found by step {t} ({solutions.length} of {allSolutions})
              </div>
              {solutions.length === 0 ? (
                <div className="text-muted-foreground">none yet</div>
              ) : (
                <ol className="flex flex-col gap-0.5 font-mono">
                  {solutions.map((s, i) => (
                    <li key={i}>
                      <button
                        type="button"
                        className={cn('rounded px-1 text-left hover:bg-muted', pins.pinned === s.node && 'bg-muted')}
                        onClick={() => pins.toggle(s.node)}
                        onMouseEnter={() => pins.hover(s.node)}
                        onMouseLeave={() => pins.hover(null)}
                      >
                        {i + 1}. {formatSolution(s)}
                        <span className="ml-2 text-muted-foreground">
                          (node {s.node}, step {s.step})
                        </span>
                      </button>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          </div>
        </div>
      }
      readouts={
        state && {
          [`step ${t}`]: (
            <>
              <Readout label="nodes" value={view.created.size} />
              <Readout label="choice points" value={state.choices.length} />
              <Readout label="depth" value={state.depth} />
              <Readout label="solutions" value={state.solutions.length} />
              <Readout label="clauses renamed" value={state.renames} />
            </>
          ),
        }
      }
      caption={
        <>
          Edit the program or the query and press Run (or ⌘↵ in the editor, Enter in the query box); the chips are
          queries to try. Step 0 is the query. Play or step: each step either resolves the leftmost goal with the next
          clause whose head unifies with it (the edge label is the clause number; a node shows the first two goals
          left), runs a built-in such as is, \=, ! or \+, or closes a branch. Green □ nodes are proofs (a solution), red
          ✗ nodes are failures, dashed ✂ nodes are clauses a cut removed before they were tried, and the ringed node is
          the one being worked on. Unification makes two terms equal by binding variables (X = tom); a clause&apos;s
          variables are renamed first (Y_3), so each use is fresh. Backtracking is depth first: after a success or
          failure Prolog returns to the most recent node with clauses left, which is why the tree grows left to right.
          Hover a node to see its goal, clause, unifier and bindings; click to pin it (again or Escape to unpin). Large
          trees draw dots without labels.
        </>
      }
    >
      <Dashboard>
        <DashboardRow fit="width" minHeight={80}>
          <DashboardCell>
            <div className="font-prose text-[15px] leading-snug">
              {run.ok && state ? explain(state.event, state, run.program) : 'Fix the program to run the query.'}
            </div>
          </DashboardCell>
        </DashboardRow>
        <DashboardRow ratio={2.2}>
          <DashboardCell>
            {tree && run.ok ? (
              <TreeView
                tree={tree}
                ariaLabel="SLD search tree"
                height="fill"
                hidden={(i) => !view.created.has(i)}
                nodeLabels={view.labels}
                nodeTone={(i) => STATUS_TONE[view.status[i]]}
                shape={view.compact ? 'dot' : (i) => (view.labels[i].length > 2 ? 'box' : 'circle')}
                nodeState={(i) => (i === current ? 'active' : undefined)}
                node={(i) => (view.status[i] === 'pruned' ? { dashed: true } : {})}
                nodeSummary={false}
                selected={pins.pinned}
                onSelect={(i) => pins.toggle(i)}
                onNodeHover={(i) => pins.hover(i)}
              />
            ) : (
              <StatusText tone="error">No tree: the program did not parse.</StatusText>
            )}
          </DashboardCell>
        </DashboardRow>
        <DashboardRow fit="width">
          <DashboardCell>
            {run.ok && focus !== null && nodes[focus] && nodes[focus].created <= t ? (
              <NodeDetails node={nodes[focus]} run={run} t={t} />
            ) : (
              <StatusText>Hover or click a node to see its goal, clause, unifier and bindings.</StatusText>
            )}
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
