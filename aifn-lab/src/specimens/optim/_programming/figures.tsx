import { Player } from '@lab/controls'
import { Figure } from '@lab/layout'
import { choice, setting, slider, useFigureState } from '@lab/state'
import { formatValue, ProgramView, TableauView, TreeView } from '@lab/views'
import {
  Curve,
  Handle,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  useAxis,
  Vectors,
  type Handle as HandleSpec,
} from '@lab/viz'
import {
  activeSet,
  branchAndBound,
  branchAndBoundTree,
  dynamicProgram,
  gomory,
  hungarianSteps,
  linearInteriorPoint,
  linprog,
  lpCentralPath,
  quadraticInteriorPoint,
  quadprog,
  simplex,
  type BranchNode,
  type LinearProgram,
  type NodeSelection,
  type SimplexRule,
} from 'aifn/optim/programming'
import { knapsack, knapsackProgram, needlemanWunsch, smithWaterman } from 'aifn-applied/algorithms/dynamic-programming'
import { integers, stream } from 'aifn/foundation/random'
import { toFlat, toRows } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { useMemo, useState, type ReactNode } from 'react'
import { centroid, clipLine, closedPath, lattice, polygonOf, type Pt, type Segment } from './geometry'

// ---------------------------------------------------------------------------------------------------------------------
// A 2-D linear program shared by the simplex and interior-point figures: maximise u·x over a hexagon.

const LP_A = [
  [1, 1],
  [-1, 2],
  [2, 1],
  [1, 0],
]
const LP_B = [6, 8, 10, 4.5]
const BOX: [number, number] = [-0.5, 6.5]
const NON_NEGATIVE = [
  [-1, 0],
  [0, -1],
]

/** A position that follows a trace: reset to 0 when the trace changes length below it. */
function usePosition(count: number): [number, (p: number) => void] {
  const [position, setPosition] = useState(0)
  return [Math.min(position, Math.max(0, count - 1)), setPosition]
}

/** Constraint lines as segments across the view box. */
function constraintSegments(A: readonly (readonly number[])[], b: readonly number[], box = BOX): Segment[] {
  return A.flatMap((a, i) => {
    const s = clipLine(a, b[i], box)
    return s ? [s] : []
  })
}

/** The objective direction as a vector from the region's centre, with a handle at its tip that sets the angle. */
function useDirection(theta: number, setDegrees: (d: number) => void, centre: Pt) {
  return useMemo(() => {
    const tip: Pt = [centre[0] + 1.4 * Math.cos(theta), centre[1] + 1.4 * Math.sin(theta)]
    const handle: HandleSpec = {
      kind: 'point',
      at: tip,
      label: 'objective direction',
      onDrag: ([x, y]) => {
        const a = Math.atan2(y - centre[1], x - centre[0])
        setDegrees(Math.round((((a * 180) / Math.PI + 360) % 360) * 10) / 10)
      },
    }
    return { vectors: [{ from: centre, to: tip, label: 'u' }], handle }
  }, [theta, setDegrees, centre])
}

const pathOf = (points: readonly (readonly number[])[]) => ({ x: points.map((p) => p[0]), y: points.map((p) => p[1]) })

/** A 2-D constraint set on a square box: constraint lines, the feasible polygon and whatever is drawn over it. */
function RegionPlot({
  box,
  region,
  segments,
  children,
  labels = ['x₁', 'x₂'],
}: {
  box: [number, number]
  region: readonly Pt[]
  segments: readonly Segment[]
  children?: ReactNode
  labels?: [string, string]
}) {
  const ring = useMemo(() => closedPath(region), [region])
  const x = useAxis({ label: labels[0], range: box })
  const y = useAxis({ label: labels[1], range: box, equal: x })
  return (
    <Plot x={x} y={y}>
      <Segments segments={segments} />
      <Curve name="feasible region" x={ring.x} y={ring.y} slot={0} />
      {children}
    </Plot>
  )
}

const PIVOT_RULES = [
  { value: 'bland', label: "Bland's rule" },
  { value: 'dantzig', label: "Dantzig's rule" },
] as const

export function SimplexSpecimen() {
  const fig = useFigureState({
    degrees: slider(0, 360, 55, { step: 1, label: 'objective direction u (degrees)' }),
    rule: choice(PIVOT_RULES, 'bland', { label: 'pivot rule' }),
    phase1: setting(false, 'add x + y ≥ 2 (needs phase 1)'),
  })
  const { degrees, phase1 } = fig
  const rule = fig.rule as SimplexRule
  const theta = (degrees * Math.PI) / 180
  const { problem, A, b } = useMemo(() => {
    const A = phase1 ? [...LP_A, [-1, -1]] : LP_A
    const b = phase1 ? [...LP_B, -2] : LP_B
    const problem: LinearProgram = { c: [-Math.cos(theta), -Math.sin(theta)], A_ub: A, b_ub: b }
    return { problem, A, b }
  }, [theta, phase1])
  const region = useMemo(() => polygonOf([...A, ...NON_NEGATIVE], [...b, 0, 0]), [A, b])
  const t = useMemo(() => trace(simplex(problem, { rule }), {}, 100), [problem, rule])
  const [position, setPosition] = usePosition(t.steps.length)
  const state = t.steps[position]
  const vertices = useMemo(() => t.steps.map((s) => toFlat(s.x)), [t])
  const centre = useMemo(() => centroid(region), [region])
  const setDegrees = fig.bind('degrees').set
  const { vectors, handle } = useDirection(theta, setDegrees, centre)
  const segments = useMemo(() => constraintSegments(A, b), [A, b])
  const walked = useMemo(() => pathOf(vertices.slice(0, position + 1)), [vertices, position])
  const label = (column: number) => (column >= 0 ? state.labels[column] : '—')
  return (
    <Figure
      title="Simplex pivots on a 2-D linear program"
      purpose="Each pivot moves from one vertex of the feasible polygon to a neighbour along an edge that improves u·x; when the origin is infeasible, phase 1 first finds a vertex by minimising the artificial variables."
      state={fig}
      defaultSize="L"
      controls={
        <div className="col-span-full">
          <Player value={position} onChange={setPosition} count={t.steps.length} label="step" />
        </div>
      }
      readouts={{
        pivot: (
          <>
            <Readout label="phase" value={state.phase} />
            <Readout label="event" value={state.event} />
            <Readout label="next entering" value={label(state.entering)} />
            <Readout label="next leaving" value={state.leaving >= 0 ? label(state.basis.data[state.leaving]) : '—'} />
            <Readout
              label={state.phase === 1 ? 'sum of artificials' : 'u·x'}
              value={formatValue(state.phase === 1 ? state.objective : -state.objective)}
            />
            <Readout label="degenerate" value={String(state.degenerate)} />
            <Readout label="status" value={state.status} />
          </>
        ),
        tableau: (
          <TableauView
            tableau={state.tableau}
            basis={state.basis}
            labels={state.labels}
            entering={state.entering}
            leaving={state.leaving}
            ratios={state.ratios}
          />
        ),
      }}
      caption="Drag the arrow's tip to turn the objective. Step through the pivots with the player; the tableau highlights the entering column, the leaving row and the pivot. Switch on x + y ≥ 2: the origin leaves the region and phase 1 runs first."
    >
      <RegionPlot box={BOX} region={region} segments={segments}>
        <Curve name="vertices visited" x={walked.x} y={walked.y} slot={1} showPoints live />
        <Points name="current basic solution" x={[vertices[position][0]]} y={[vertices[position][1]]} emphasis live />
        <Vectors vectors={vectors} live />
        <Handle {...handle} />
      </RegionPlot>
    </Figure>
  )
}

const MUS = Array.from({ length: 60 }, (_, k) => 10 ** (1.5 - (7 * k) / 59))

export function InteriorPointSpecimen() {
  const fig = useFigureState({ degrees: slider(0, 360, 55, { step: 1, label: 'objective direction u (degrees)' }) })
  const theta = (fig.degrees * Math.PI) / 180
  const problem = useMemo<LinearProgram>(
    () => ({ c: [-Math.cos(theta), -Math.sin(theta)], A_ub: LP_A, b_ub: LP_B }),
    [theta],
  )
  const region = useMemo(() => polygonOf([...LP_A, ...NON_NEGATIVE], [...LP_B, 0, 0]), [])
  const t = useMemo(() => trace(linearInteriorPoint(problem), {}, 60), [problem])
  const path = useMemo(() => lpCentralPath(problem, MUS), [problem])
  const [position, setPosition] = usePosition(t.steps.length)
  const state = t.steps[position]
  const iterates = useMemo(() => t.steps.map((s) => toFlat(s.x)), [t])
  const centre = useMemo(() => centroid(region), [region])
  const { vectors, handle } = useDirection(theta, fig.bind('degrees').set, centre)
  const segments = useMemo(() => constraintSegments(LP_A, LP_B), [])
  const central = useMemo(() => pathOf(toRows(path.x)), [path])
  const walked = useMemo(() => pathOf(iterates.slice(0, position + 1)), [iterates, position])
  return (
    <Figure
      title="Interior-point method and the central path"
      purpose="The central path x(μ) minimises −u·x − μ Σ log(slacks) and runs from the analytic centre (large μ) to the optimal vertex (μ → 0); Mehrotra's predictor–corrector follows it, cutting μ by orders of magnitude per step."
      state={fig}
      controls={
        <div className="col-span-full">
          <Player value={position} onChange={setPosition} count={t.steps.length} label="iteration" />
        </div>
      }
      readouts={{
        barrier: (
          <>
            <Readout label="μ" value={formatValue(state.mu)} />
            <Readout label="σ" value={formatValue(state.sigma)} />
            <Readout label="α primal" value={formatValue(state.alphaPrimal)} />
            <Readout label="α dual" value={formatValue(state.alphaDual)} />
          </>
        ),
        convergence: (
          <>
            <Readout label="primal residual" value={formatValue(state.primalResidual)} />
            <Readout label="relative gap" value={formatValue(state.gap)} />
            <Readout label="u·x" value={formatValue(-state.objective)} />
          </>
        ),
      }}
      caption="Drag the arrow's tip to turn the objective: the central path bends to its new optimal vertex. Step through the iterations from an infeasible start to see μ fall."
    >
      <RegionPlot box={BOX} region={region} segments={segments}>
        <Curve name="central path x(μ)" x={central.x} y={central.y} slot={2} dashed />
        <Curve name="Mehrotra iterates" x={walked.x} y={walked.y} slot={1} showPoints live />
        <Points name="current iterate" x={[iterates[position][0]]} y={[iterates[position][1]]} emphasis live />
        <Vectors vectors={vectors} live />
        <Handle {...handle} />
      </RegionPlot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Integer programming: branch and bound, and Gomory cuts, on max 5x + 8y s.t. x + y ≤ 6, 5x + 9y ≤ 45.

const IP_A = [
  [1, 1],
  [5, 9],
]
const IP_B = [6, 45]
const IP: LinearProgram = { c: [-5, -8], A_ub: IP_A, b_ub: IP_B }
const IP_BOX: [number, number] = [-0.5, 7]

/** Palette slot by how a node ended: integral leaves and pruned leaves differ; open nodes stay neutral. */
const STATUS_TONE: Partial<Record<BranchNode['status'], number>> = {
  integral: 2,
  infeasible: 3,
  bound: 3,
  unbounded: 3,
}
const STATUS_NOTE: Record<BranchNode['status'], string> = {
  open: 'open',
  branched: '',
  integral: 'integral',
  infeasible: 'infeasible',
  bound: 'pruned: bound',
  unbounded: 'unbounded',
}

const branchLabel = (n: BranchNode) =>
  n.branch
    ? `x${n.branch.variable + 1} ${n.branch.direction === 'down' ? '≤' : '≥'} ${n.branch.direction === 'down' ? Math.floor(n.branch.value) : Math.ceil(n.branch.value)}`
    : 'root'

const NODE_SELECTIONS = ['depth-first', 'best-bound', 'breadth-first'] as const

export function BranchAndBoundSpecimen() {
  const fig = useFigureState({ strategy: choice(NODE_SELECTIONS, 'depth-first', { label: 'node selection' }) })
  const strategy = fig.strategy as NodeSelection
  const t = useMemo(() => trace(branchAndBound(IP, { strategy }), {}, 200), [strategy])
  // Opens on the finished search: the tree's size is the point; play from the start to watch it grow.
  const [chosen, setPosition] = useState<number | null>(null)
  const position = Math.min(chosen ?? t.steps.length - 1, t.steps.length - 1)
  const state = t.steps[position]
  // `current` is the processed node's id (read into a local so it is not mistaken for a ref).
  const { nodes, current: processed } = state
  const searchTree = useMemo(() => branchAndBoundTree(nodes), [nodes])
  const region = useMemo(() => polygonOf([...IP_A, ...NON_NEGATIVE], [...IP_B, 0, 0]), [])
  const grid = useMemo(() => lattice(0, 7), [])
  const node = processed >= 0 ? nodes[processed] : nodes[0]
  const bounds = useMemo(() => {
    const [lx, ly] = toFlat(node.lower).map((v) => Math.max(v, IP_BOX[0]))
    const [ux, uy] = toFlat(node.upper).map((v) => Math.min(v, IP_BOX[1]))
    return { x: [lx, ux, ux, lx, lx], y: [ly, ly, uy, uy, ly] }
  }, [node])
  return (
    <Figure
      title="Branch and bound"
      purpose="Each node solves the LP relaxation within its bounds, then branches on a fractional variable or is pruned (infeasible, or its bound cannot beat the incumbent); the order nodes are taken in decides how many are needed."
      description="Maximise 5x + 8y subject to x + y ≤ 6 and 5x + 9y ≤ 45 over the integers."
      state={fig}
      defaultSize="L"
      controls={
        <div className="col-span-full">
          <Player
            value={position}
            onChange={setPosition}
            count={t.steps.length}
            label="node processed"
            startReason="the finished search tree is the point: how many nodes each selection rule needs"
          />
        </div>
      }
      readouts={{
        node: (
          <>
            <Readout label="node" value={processed >= 0 ? `${processed} (${branchLabel(node)})` : '—'} />
            <Readout label="outcome" value={processed >= 0 ? node.status : '—'} />
            <Readout label="node bound" value={formatValue(-node.bound)} />
          </>
        ),
        search: (
          <>
            <Readout label="incumbent" value={state.incumbent ? formatValue(-state.incumbentValue) : 'none'} />
            <Readout label="best bound" value={formatValue(-state.bestBound)} />
            <Readout label="open nodes" value={state.open.length} />
          </>
        ),
      }}
      caption="Values are shown for the maximisation (the solver minimises −5x − 8y). Each tree node shows its relaxation's value, a bound on what its subtree can reach; edges show the branching constraint (down branch left). Integral leaves and pruned ones (∅: infeasible) take different colours; open nodes are dimmed. Play to the end with each node selection and compare how many nodes they need."
    >
      <div className="grid h-full gap-4 md:grid-cols-2">
        <TreeView
          tree={searchTree}
          height="fill"
          ariaLabel="Branch-and-bound search tree"
          nodeLabels={(v) =>
            nodes[v].status === 'infeasible' ? '$\\varnothing$' : `$${+(-nodes[v].bound).toFixed(2)}$`
          }
          nodeNotes={(v) => STATUS_NOTE[nodes[v].status] || undefined}
          shape="pill"
          nodeTone={(v) => STATUS_TONE[nodes[v].status]}
          nodeState={(v) => (v === processed ? 'active' : nodes[v].status === 'open' ? 'idle' : 'done')}
          edgeState={(c) => (c === processed ? 'active' : nodes[c].status === 'open' ? 'idle' : 'done')}
        />
        <RegionPlot box={IP_BOX} region={region} segments={NO_SEGMENTS} labels={['x', 'y']}>
          <Points name="integer points" x={grid.x} y={grid.y} muted />
          <Curve name="node's bounds" x={bounds.x} y={bounds.y} slot={1} dashed />
          {node.x && <Points name="node's LP optimum" x={[node.x.data[0]]} y={[node.x.data[1]]} emphasis />}
          {state.incumbent && (
            <Points name="incumbent" x={[state.incumbent.data[0]]} y={[state.incumbent.data[1]]} slot={2} />
          )}
        </RegionPlot>
      </div>
    </Figure>
  )
}

const NO_SEGMENTS: Segment[] = []

export function GomorySpecimen() {
  const t = useMemo(() => trace(gomory(IP), {}, 30), [])
  const [position, setPosition] = usePosition(t.steps.length)
  const state = t.steps[position]
  const grid = useMemo(() => lattice(0, 7), [])
  const region = useMemo(() => {
    const cuts = state.cuts.map((c) => toFlat(c.a))
    return polygonOf([...IP_A, ...NON_NEGATIVE, ...cuts], [...IP_B, 0, 0, ...state.cuts.map((c) => c.b)])
  }, [state])
  const optima = useMemo(() => pathOf(t.steps.slice(0, position + 1).map((s) => toFlat(s.x))), [t, position])
  const segments = useMemo(
    () =>
      state.cuts.flatMap((c) => {
        const s = clipLine(toFlat(c.a), c.b, IP_BOX)
        return s ? [s] : []
      }),
    [state],
  )
  const last = state.cuts[state.cuts.length - 1]
  const cutText = last
    ? `${toFlat(last.a)
        .map((v, j) => `${formatValue(Number(v.toPrecision(4)))}·x${j + 1}`)
        .join(' + ')} ≤ ${formatValue(Number(last.b.toPrecision(4)))}`
    : '—'
  return (
    <Figure
      title="Gomory fractional cuts"
      purpose="Each Gomory cut, read from a tableau row whose basic variable is fractional, removes the current LP vertex and no integer point; after enough cuts the LP optimum is integral."
      description="The integer program of the branch-and-bound figure, solved by cutting planes; dual simplex pivots restore optimality after each cut."
      controls={
        <div className="col-span-full">
          <Player value={position} onChange={setPosition} count={t.steps.length} label="cuts" />
        </div>
      }
      readouts={
        <>
          <Readout label="cuts" value={state.cuts.length} />
          <Readout label="latest cut" value={cutText} />
          <Readout label="dual pivots" value={state.dualPivots} />
          <Readout label="5x + 8y" value={formatValue(-state.objective)} />
          <Readout label="status" value={state.status} />
        </>
      }
      caption="Step through the cuts: the cut lines (thin) shave the relaxation (the polygon) towards the integer hull, and the LP optimum (ink) moves until it lands on an integer point."
    >
      <RegionPlot box={IP_BOX} region={region} segments={segments} labels={['x', 'y']}>
        <Points name="integer points" x={grid.x} y={grid.y} muted />
        <Curve name="LP optimum after each cut" x={optima.x} y={optima.y} slot={1} showPoints />
        <Points name="current LP optimum" x={[state.x.data[0]]} y={[state.x.data[1]]} emphasis />
      </RegionPlot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Dynamic programming tables.

/** A table (rows × columns) as raster data with row 0 at the top: y is minus the row index. */
function tableGrid(rows: number[][]) {
  const m = rows.length
  const n = rows[0]?.length ?? 0
  return {
    x: Array.from({ length: n }, (_, j) => j),
    y: Array.from({ length: m }, (_, i) => i - (m - 1)),
    z: Array.from({ length: m }, (_, i) => rows[m - 1 - i]),
  }
}

/** Row axis labels for a table drawn with row 0 at the top (y = −row). */
const rowFormat = (v: number) => String(Math.round(-v))

/** A table as a raster with its overlays; rows run down from 0. */
function TablePlot({
  grid,
  labels,
  valueLabel,
  scale,
  range,
  children,
}: {
  grid: ReturnType<typeof tableGrid>
  labels: [string, string]
  valueLabel: string
  scale?: 'sequential' | 'diverging'
  range?: [number, number]
  children?: ReactNode
}) {
  const x = useAxis({ label: labels[0] })
  const y = useAxis({ label: labels[1], format: rowFormat, equal: x })
  return (
    <Plot x={x} y={y}>
      <Raster x={grid.x} y={grid.y} z={grid.z} scale={scale} range={range} valueLabel={valueLabel} />
      {children}
    </Plot>
  )
}

const VALUES = [6, 10, 12, 7, 3, 9]
const WEIGHTS = [1, 2, 3, 2, 1, 4]

export function KnapsackSpecimen() {
  const fig = useFigureState({ capacity: slider(1, 13, 8, { step: 1, label: 'capacity C' }) })
  const capacity = fig.capacity
  const program = useMemo(() => knapsackProgram(VALUES, WEIGHTS, capacity), [capacity])
  const t = useMemo(() => trace(dynamicProgram(program), {}, VALUES.length + 1), [program])
  const result = useMemo(() => knapsack(VALUES, WEIGHTS, capacity), [capacity])
  const [position, setPosition] = usePosition(t.steps.length)
  const state = t.steps[position]
  const grid = useMemo(() => tableGrid(toRows(state.table)), [state])
  const traceback = useMemo(() => {
    const cells = toRows(result.path)
    return { x: cells.map((c) => c[1]), y: cells.map((c) => -c[0]) }
  }, [result])
  const chosen = toFlat(result.take).flatMap((k, i) => (k ? [`item ${i + 1}`] : []))
  return (
    <Figure
      title="0/1 knapsack table"
      purpose="Cell (i, c) is the best value from the first i items within capacity c, T[i][c] = max(T[i−1][c], T[i−1][c − wᵢ] + vᵢ), so the table fills row by row and the traceback reads the chosen items back."
      state={fig}
      controls={
        <div className="col-span-full">
          <Player value={position} onChange={setPosition} count={t.steps.length} label="rows filled" />
        </div>
      }
      readouts={{
        items: (
          <>
            <Readout label="values" value={VALUES.join(', ')} />
            <Readout label="weights" value={WEIGHTS.join(', ')} />
          </>
        ),
        answer: (
          <>
            <Readout label="best value" value={state.converged ? result.value : '…'} />
            <Readout label="chosen" value={state.converged ? chosen.join(', ') : '…'} />
            <Readout label="weight used" value={state.converged ? result.weight : '…'} />
          </>
        ),
      }}
      caption="Play to fill the rows; once the table is full the traceback path marks the items taken (a step down and left takes item i). The colour scale is held at the final best value."
    >
      <TablePlot grid={grid} labels={['capacity c', 'items i']} valueLabel="best value" range={[0, result.value]}>
        {state.converged && <Curve name="traceback" x={traceback.x} y={traceback.y} slot={1} showPoints />}
      </TablePlot>
    </Figure>
  )
}

const PAIRS = {
  'GATTACA / GCATGCU': ['GATTACA', 'GCATGCU'],
  'TGTTACGG / GGTTGACTA': ['TGTTACGG', 'GGTTGACTA'],
  'kitten / sitting': ['kitten', 'sitting'],
  'PAWHEAE / HEAGAWGHEE': ['PAWHEAE', 'HEAGAWGHEE'],
} as const
type PairName = keyof typeof PAIRS

const PAIR_NAMES = Object.keys(PAIRS) as PairName[]

export function AlignmentSpecimen() {
  const fig = useFigureState({
    pair: choice(PAIR_NAMES, 'TGTTACGG / GGTTGACTA', { label: 'sequences' }),
    mode: choice(['global', 'local'], 'local', { label: 'alignment' }),
    match: slider(0, 5, 3, { step: 1, label: 'match' }),
    mismatch: slider(-5, 0, -3, { step: 1, label: 'mismatch' }),
    gap: slider(-5, 0, -2, { step: 1, label: 'gap' }),
  })
  const { mode, match, mismatch, gap } = fig
  const [a, b] = PAIRS[fig.pair as PairName]
  const result = useMemo(() => {
    const scoring = { match, mismatch, gap }
    return mode === 'global' ? needlemanWunsch(a, b, scoring) : smithWaterman(a, b, scoring)
  }, [a, b, mode, match, mismatch, gap])
  const grid = useMemo(() => tableGrid(toRows(result.table)), [result])
  const traceback = useMemo(() => {
    const cells = toRows(result.path)
    return { x: cells.map((c) => c[1]), y: cells.map((c) => -c[0]) }
  }, [result])
  return (
    <Figure
      title="Sequence alignment"
      purpose="F[i][j] = max(F[i−1][j−1] + s(aᵢ, bⱼ), F[i−1][j] + gap, F[i][j−1] + gap) scores the best alignment of two prefixes; local alignment adds 0 as an option, so it can restart anywhere and keeps the best substring match."
      state={fig}
      readouts={
        <>
          <Readout label="score" value={result.score} />
          <Readout label="a" value={<span className="whitespace-pre">{String(result.alignedA)}</span>} />
          <Readout label="b" value={<span className="whitespace-pre">{String(result.alignedB)}</span>} />
        </>
      }
      caption={`Columns are positions in b (${b}), rows positions in a (${a}); row and column 0 are the empty prefixes. The traceback runs from the best cell (local) or the corner (global) back to its start. Switch to global: the whole of both sequences must be aligned and negative scores appear.`}
    >
      <TablePlot grid={grid} labels={['j (b)', 'i (a)']} valueLabel="score" scale="diverging">
        <Curve name="traceback" x={traceback.x} y={traceback.y} slot={1} showPoints />
      </TablePlot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Assignment.

function costMatrix(seed: number, n: number): number[][] {
  const s = stream(`hungarian/${seed}`)
  return Array.from({ length: n }, () => Array.from({ length: n }, () => 1 + integers(s, 9)))
}

export function HungarianSpecimen() {
  const fig = useFigureState({ seed: slider(1, 30, 1, { step: 1, label: 'seed' }) })
  const cost = useMemo(() => costMatrix(fig.seed, 5), [fig.seed])
  const t = useMemo(() => trace(hungarianSteps(cost), {}, 500), [cost])
  const [position, setPosition] = usePosition(t.steps.length)
  const state = t.steps[position]
  const grid = useMemo(() => tableGrid(toRows(state.reduced)), [state])
  const marks = useMemo(() => {
    const cells = (cols: number[]) => cols.flatMap((j, i) => (j >= 0 ? [[j, -i]] : []))
    const s = cells(toFlat(state.starred))
    const p = cells(toFlat(state.primed))
    const path = state.last === 'augment' ? toRows(state.path) : []
    return {
      star: { x: s.map((c) => c[0]), y: s.map((c) => c[1]) },
      prime: { x: p.map((c) => c[0]), y: p.map((c) => c[1]) },
      path: { x: path.map((c) => c[1]), y: path.map((c) => -c[0]) },
    }
  }, [state])
  const covered = (v: typeof state.rowCovered) =>
    toFlat(v)
      .flatMap((c, i) => (c ? [i + 1] : []))
      .join(', ') || 'none'
  let total = 0
  toFlat(state.starred).forEach((j, i) => j >= 0 && (total += cost[i][j]))
  return (
    <Figure
      title="Hungarian algorithm"
      purpose="Reducing rows and columns keeps the optimal assignment and creates zeros; the algorithm stars independent zeros, covers lines and creates new zeros until a full set of starred zeros is an optimal assignment."
      description="Munkres' steps on a 5 × 5 cost matrix: reduce, star, cover, prime, augment along alternating paths, adjust by the smallest uncovered value."
      state={fig}
      controls={
        <div className="col-span-full">
          <Player value={position} onChange={setPosition} count={t.steps.length} label="step" />
        </div>
      }
      readouts={{
        step: (
          <>
            <Readout label="last step" value={state.last} />
            <Readout label="next" value={state.next} />
            <Readout label="adjustment" value={state.last === 'adjust' ? formatValue(state.delta) : '—'} />
          </>
        ),
        covers: (
          <>
            <Readout label="covered rows" value={covered(state.rowCovered)} />
            <Readout label="covered columns" value={covered(state.columnCovered)} />
            <Readout label="cost of starred" value={total} />
          </>
        ),
      }}
      caption="The raster is the reduced matrix cost − u − v; starred zeros are the current assignment, primed zeros the candidates, and an augmenting path (when one is found) swaps them."
    >
      <TablePlot grid={grid} labels={['column', 'row']} valueLabel="reduced cost">
        <Points name="starred zeros" x={marks.star.x} y={marks.star.y} slot={1} />
        <Points name="primed zeros" x={marks.prime.x} y={marks.prime.y} slot={2} />
        {marks.path.x.length > 0 && (
          <Curve name="augmenting path" x={marks.path.x} y={marks.path.y} slot={3} showPoints />
        )}
      </TablePlot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Quadratic programming and the result view.

const QP_A = [
  [-1, 2],
  [1, 2],
  [1, -2],
  [-1, 0],
  [0, -1],
]
const QP_B = [2, 6, 2, 0, 0]
const QP_BOX: [number, number] = [-0.5, 4.5]

const QP_METHODS = ['active set', 'interior point'] as const
const RING = Array.from({ length: 73 }, (_, k) => (k * 2 * Math.PI) / 72)

export function ActiveSetSpecimen() {
  const fig = useFigureState({
    method: choice(QP_METHODS, 'active set', { label: 'method' }),
    tx: slider(-0.5, 4.5, 1, { onChart: true, step: 0.05, label: 't₁' }),
    ty: slider(-0.5, 4.5, 2.5, { onChart: true, step: 0.05, label: 't₂' }),
  })
  const { method, tx, ty } = fig
  const problem = useMemo(
    () => ({
      Q: [
        [2, 0],
        [0, 2],
      ],
      c: [-2 * tx, -2 * ty],
      A: QP_A,
      b: QP_B,
    }),
    [tx, ty],
  )
  const iterates = useMemo(() => {
    if (method === 'active set') {
      const t = trace(activeSet(problem), { x0: [2, 0] }, 50)
      return t.steps.map((s) => ({
        x: toFlat(s.x),
        status: s.status as string,
        text: `${s.event}${s.changed >= 0 ? ` constraint ${s.changed + 1}` : ''}; working set {${toFlat(s.working)
          .map((i) => i + 1)
          .join(', ')}}`,
      }))
    }
    const t = trace(quadraticInteriorPoint(problem), {}, 50)
    return t.steps.map((s) => ({ x: toFlat(s.x), status: s.status as string, text: `μ = ${formatValue(s.mu)}` }))
  }, [problem, method])
  const [position, setPosition] = usePosition(iterates.length)
  const current = iterates[position]
  const region = useMemo(() => polygonOf(QP_A, QP_B), [])
  const segments = useMemo(() => constraintSegments(QP_A, QP_B, QP_BOX), [])
  const level = useMemo(() => {
    const r = Math.hypot(current.x[0] - tx, current.x[1] - ty)
    return { x: RING.map((a) => tx + r * Math.cos(a)), y: RING.map((a) => ty + r * Math.sin(a)) }
  }, [current, tx, ty])
  const walked = useMemo(() => pathOf(iterates.slice(0, position + 1).map((i) => i.x)), [iterates, position])
  return (
    <Figure
      title="Quadratic programming: active set and interior point"
      purpose="The active-set method moves along faces of the polygon, adding blocking constraints and dropping those with negative multipliers; the interior-point method approaches the same minimiser through the interior."
      description="Minimise ‖x − t‖² over a polygon (Nocedal and Wright, Example 16.3)."
      state={fig}
      controls={
        <div className="col-span-full">
          <Player value={position} onChange={setPosition} count={iterates.length} label="step" />
        </div>
      }
      readouts={
        <>
          <Readout label="t" value={`(${tx}, ${ty})`} />
          <Readout label="x" value={`(${formatValue(current.x[0])}, ${formatValue(current.x[1])})`} />
          <Readout label="step" value={current.text} />
          <Readout label="status" value={current.status} />
        </>
      }
      caption="Drag t, the unconstrained minimiser: inside the polygon the answer is t itself, outside it is the nearest point of the polygon. The dashed circle is the level set through the current iterate; at the solution it touches the polygon."
    >
      <RegionPlot box={QP_BOX} region={region} segments={segments}>
        <Curve name="level set through x" x={level.x} y={level.y} slot={2} dashed live />
        <Curve name="iterates" x={walked.x} y={walked.y} slot={1} showPoints live />
        <Points name="current" x={[current.x[0]]} y={[current.x[1]]} emphasis live />
        <Handle {...fig.handle(['tx', 'ty'], { label: 'unconstrained minimiser t' })} />
      </RegionPlot>
    </Figure>
  )
}

const WYNDOR: LinearProgram = {
  c: [-3, -5],
  A_ub: [
    [1, 0],
    [0, 2],
    [3, 2],
  ],
  b_ub: [4, 12, 18],
}

const CASES = {
  'LP, simplex': () => linprog(WYNDOR),
  'LP, interior point': () => linprog(WYNDOR, { method: 'interior-point' }),
  'LP with free and bounded variables': () =>
    linprog({
      c: [1, 2, -1],
      A_eq: [[1, 1, 1]],
      b_eq: [3],
      A_ub: [[1, -1, 0]],
      b_ub: [-1],
      bounds: [
        [null, null],
        [0, 4],
        [-2, 2],
      ],
    }),
  'infeasible LP': () => linprog({ c: [1], A_ub: [[1], [-1]], b_ub: [1, -2] }),
  'unbounded LP': () => linprog({ c: [-1, 0], A_ub: [[1, -1]], b_ub: [1] }),
  'QP, active set': () =>
    quadprog({
      Q: [
        [2, 0],
        [0, 2],
      ],
      c: [-2, -5],
      A: QP_A,
      b: QP_B,
    }),
  // Q = diag(2, −2) curves down along x₂: the active set meets negative curvature on its working set and stops.
  'nonconvex QP, active set': () =>
    quadprog({
      Q: [
        [2, 0],
        [0, -2],
      ],
      c: [-2, -5],
      A: QP_A,
      b: QP_B,
    }),
  'QP, interior point': () =>
    quadprog(
      {
        Q: [
          [2, 0],
          [0, 2],
        ],
        c: [-2, -5],
        A: QP_A,
        b: QP_B,
      },
      { method: 'interior-point' },
    ),
} as const
type CaseName = keyof typeof CASES

const CASE_NAMES = Object.keys(CASES) as CaseName[]

export function ProgramViewSpecimen() {
  const fig = useFigureState({ name: choice(CASE_NAMES, 'LP, simplex', { label: 'problem' }) })
  const name = fig.name as CaseName
  const result = useMemo(() => CASES[name](), [name])
  return (
    <Figure
      title="LP and QP results"
      purpose="One result view for every LP and QP solver: variables, objective, duals (scipy's convention), slacks, active constraints and the optimality checks, including infeasible and unbounded outcomes."
      state={fig}
      hoverReadout={false}
      caption="The Wyndor Glass problem by simplex and by interior point gives the same optimum and duals; the infeasible and unbounded cases report their status instead of a solution, and the active set stops as nonconvex when Q has negative curvature on its working set."
    >
      <ProgramView result={result} />
    </Figure>
  )
}
