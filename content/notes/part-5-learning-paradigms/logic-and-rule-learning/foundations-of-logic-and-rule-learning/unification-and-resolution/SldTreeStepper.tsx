import { useMemo, useState } from 'react'
import { treeFromParents } from 'aifn-compute/graph'
import { trace } from 'aifn-compute/foundation/trace'
import {
  formatSolution,
  prologProgram,
  sldSteps,
  sldTree,
  type SldEvent,
  type SldNode,
  type SldNodeStatus,
  type SldState,
} from 'aifn-compute/logic/resolution'
import { clauseToString, substitutionToString, termToString, type Term } from 'aifn-compute/logic/terms'
import { choice, Figure, Player, Readout, TreeView, useFigureState, type Tone } from 'aifn-render'

const SOURCE = `parent(ann, bob).
parent(bob, cid).
parent(bob, dee).
parent(cid, eve).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).`

const QUERIES = ['grandparent(ann, W)', 'ancestor(ann, eve)', 'ancestor(X, eve)', 'grandparent(cid, W)'] as const

const STATUS_TONE: Record<SldNodeStatus, Tone> = {
  open: 'ink',
  expanded: 'ink',
  success: 2,
  failure: 1,
  pruned: 'neutral',
  limit: 'neutral',
  error: 1,
}

const show = (t: Term) => termToString(t)
const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s)

/** A node's status as of step t (the tree's statuses are those at the last step). */
function statusAt(n: SldNode, t: number, hasChildren: boolean): SldNodeStatus {
  if (n.closed !== null && n.closed <= t) return n.status
  return hasChildren ? 'expanded' : 'open'
}

/** One step in words. */
function explain(e: SldEvent, s: SldState, clauses: ReturnType<typeof prologProgram>['clauses']): string {
  switch (e.kind) {
    case 'start':
      return `Step 0: the query is the first resolvent: ${s.goals!.map((g) => show(g.term)).join(', ')}. SLD resolution always selects the leftmost goal.`
    case 'resolve':
      return `Step ${s.t}: ${e.retry ? 'backtrack and ' : ''}resolve ${show(e.goal)} with clause ${e.clause + 1}, ${clauseToString(clauses[e.clause])} Its variables are renamed apart, the head ${show(e.head)} unifies with the goal by ${e.unifier.length ? substitutionToString(e.unifier) : 'the empty substitution'}, and the body replaces the goal.${e.alternatives > 0 ? ` ${e.alternatives} more clause${e.alternatives > 1 ? 's' : ''} wait${e.alternatives > 1 ? '' : 's'} at this choice point.` : ''}`
    case 'builtin':
      return `Step ${s.t}: built-in ${show(e.goal)}: ${e.note}.`
    case 'alternative':
      return `Step ${s.t}: backtrack to node ${e.parent}: ${e.note}.`
    case 'success':
      return `Step ${s.t}: the resolvent is empty, so the query is proved with ${formatSolution(s.solutions[e.solution])}. The search backtracks for more answers.`
    case 'fail':
      return `Step ${s.t}: failure: ${e.reason}. The search backtracks to the most recent choice point.`
    case 'error':
      return `Step ${s.t}: error: ${e.message}.`
    case 'exhausted':
      return `Step ${s.t}: no choice point is left. The search is complete with ${s.solutions.length} answer${s.solutions.length === 1 ? '' : 's'}.`
  }
}

const schema = { query: choice(QUERIES, 'grandparent(ann, W)', { label: 'query' }) }

// The program is read once, and each query's trace and tree are computed on first use and kept: at most 400 steps,
// branches deeper than 30 steps cut off (every query here finishes in under 50 steps).
const PROGRAM = prologProgram(SOURCE, { library: false })
type Run = { states: SldState[]; nodes: SldNode[]; tree: ReturnType<typeof treeFromParents> }
const RUNS = new Map<string, Run>()
function runOf(query: string): Run {
  const cached = RUNS.get(query)
  if (cached) return cached
  const states = trace(sldSteps(PROGRAM, query, { maxDepth: 30 }), undefined, 400, { keep: 'all' }).steps as SldState[]
  const nodes = sldTree(states, PROGRAM.clauses)
  const tree = treeFromParents(
    nodes.map((n) => n.parent),
    { edge: (c) => ({ label: nodes[c].clause !== null ? String(nodes[c].clause! + 1) : '' }) },
  )
  const r = { states, nodes, tree }
  RUNS.set(query, r)
  return r
}

/** The SLD tree of a query against the family program, grown one resolution step at a time. */
export function SldTreeStepper() {
  const state = useFigureState(schema)
  const [pos, setPos] = useState({ query: state.query as string, step: 0 })
  const program = PROGRAM
  const run = runOf(state.query)
  const last = run.states.length - 1
  const t = pos.query === state.query ? Math.min(pos.step, last) : 0
  const s = run.states[t]
  const [hover, setHover] = useState<number | null>(null)

  const view = useMemo(() => {
    const created = new Set(run.nodes.filter((n) => n.created <= t).map((n) => n.id))
    const parents = new Set(run.nodes.filter((n) => n.created <= t && n.parent !== null).map((n) => n.parent!))
    const status = run.nodes.map((n) => statusAt(n, t, parents.has(n.id)))
    const labels = run.nodes.map((n, i) => {
      if (status[i] === 'success') {
        const sol = n.solution === null ? null : run.states[last].solutions[n.solution]
        return clip(`□ ${sol ? formatSolution(sol) : ''}`, 28)
      }
      const goals = n.goals.length ? n.goals.slice(0, 2).map(show).join(', ') + (n.goals.length > 2 ? ', …' : '') : '□'
      return status[i] === 'failure' ? clip(`✗ ${goals}`, 34) : clip(goals, 40)
    })
    return { created, status, labels }
  }, [run, t, last])

  const current = s.event.kind !== 'exhausted' ? (s.goals === null && 'node' in s.event ? s.event.node : s.node) : null
  const focus = hover !== null && view.created.has(hover) ? run.nodes[hover] : null

  return (
    <Figure
      title="SLD resolution of a query"
      state={state}
      defaultSize="L"
      hoverReadout={false}
      controls={
        <Player
          label="step"
          value={t}
          onChange={(k) => setPos({ query: state.query, step: k })}
          count={run.states.length}
          format={(k) => `${k} of ${last}`}
        />
      }
      equation={
        <div className="grid gap-3 text-left text-sm md:grid-cols-[minmax(0,0.8fr)_minmax(0,1fr)]">
          <pre className="font-mono text-xs leading-relaxed">
            {SOURCE.split('\n')
              .map((l, i) => `${i + 1}  ${l}`)
              .join('\n')}
          </pre>
          <div className="font-prose text-[15px] leading-snug">{explain(s.event, s, program.clauses)}</div>
        </div>
      }
      readouts={
        <>
          <Readout label="nodes" value={view.created.size} />
          <Readout label="choice points" value={s.choices.length} />
          <Readout label="answers so far" value={s.solutions.map(formatSolution).join('; ') || 'none'} />
          {focus && (
            <Readout
              label={`node ${focus.id}`}
              value={`${focus.goal ? `resolved ${show(focus.goal)}` : 'the query'}${focus.clause !== null ? ` with clause ${focus.clause + 1}` : ''}; unifier ${focus.unifier.length ? substitutionToString(focus.unifier) : '{}'}; goals left ${focus.goals.length ? focus.goals.map(show).join(', ') : '□'}`}
            />
          )}
        </>
      }
      caption="Pick a query and step through its proof search. Step 0 is the query. Each step resolves the leftmost goal with the next clause whose head unifies with it (the edge label is the clause number, and each node shows the first two goals left), or closes a branch. A node with no goals left (□) is a proof and gives an answer; a node whose leftmost goal unifies with no clause head (✗) fails, and the search backtracks to the most recent node with clauses left. Hover a node to read its goal, clause and unifier."
    >
      <TreeView
        tree={run.tree}
        ariaLabel="SLD tree of the query"
        hidden={(i) => !view.created.has(i)}
        nodeLabels={view.labels}
        nodeTone={(i) => STATUS_TONE[view.status[i]]}
        shape={(i) => (view.labels[i].length > 2 ? 'box' : 'circle')}
        nodeState={(i) => (i === current ? 'active' : undefined)}
        nodeSummary={false}
        onNodeHover={setHover}
      />
    </Figure>
  )
}
