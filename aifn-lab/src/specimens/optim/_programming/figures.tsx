import { Player, Select, Slider, Switch } from '@lab/controls'
import { Figure } from '@lab/layout'
import { formatValue, ProgramView, TableauView, TreeView } from '@lab/views'
import {
  Heatmap,
  Readout,
  Readouts,
  XYChart,
  type Handle,
  type HeatmapOverlay,
  type Segment,
  type XYSeries,
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
import { useMemo, useState } from 'react'
import { centroid, clipLine, closedPath, lattice, polygonOf, type Pt } from './geometry'

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
function useDirection(theta: number, setTheta: (t: number) => void, centre: Pt) {
  return useMemo(() => {
    const tip: Pt = [centre[0] + 1.4 * Math.cos(theta), centre[1] + 1.4 * Math.sin(theta)]
    const handle: Handle = {
      kind: 'point',
      at: tip,
      label: 'objective direction',
      onDrag: ([x, y]) => {
        const a = Math.atan2(y - centre[1], x - centre[0])
        setTheta(Math.round((((a * 180) / Math.PI + 360) % 360) * 10) / 10)
      },
    }
    return { vectors: [{ from: centre, to: tip, label: 'u' }], handles: [handle] }
  }, [theta, setTheta, centre])
}

const pathOf = (points: readonly (readonly number[])[]) => ({ x: points.map((p) => p[0]), y: points.map((p) => p[1]) })

export function SimplexSpecimen() {
  const [degrees, setDegrees] = useState(55)
  const [rule, setRule] = useState<SimplexRule>('bland')
  const [phase1, setPhase1] = useState(false)
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
  const { vectors, handles } = useDirection(theta, setDegrees, centre)
  const segments = useMemo(() => constraintSegments(A, b), [A, b])
  const series = useMemo<XYSeries[]>(() => {
    const walked = pathOf(vertices.slice(0, position + 1))
    return [
      { name: 'feasible region', type: 'line', ...closedPath(region), slot: 0 },
      { name: 'vertices visited', type: 'line', ...walked, slot: 1, showPoints: true },
      {
        name: 'current basic solution',
        type: 'scatter',
        x: [vertices[position][0]],
        y: [vertices[position][1]],
        emphasis: true,
      },
    ]
  }, [region, vertices, position])
  const label = (column: number) => (column >= 0 ? state.labels[column] : '—')
  return (
    <div className="space-y-4">
      <Figure
        title="Simplex pivots on a 2-D linear program"
        description="Each pivot moves from one vertex of the feasible polygon to a neighbour along an edge that improves u·x. With the extra constraint x + y ≥ 2 the origin is infeasible, and phase 1 first minimises the sum of the artificial variables."
        controls={
          <>
            <Slider
              label="objective direction u (degrees)"
              value={degrees}
              onChange={setDegrees}
              min={0}
              max={360}
              step={1}
            />
            <Select
              label="pivot rule"
              value={rule}
              onChange={setRule}
              options={[
                { value: 'bland', label: "Bland's rule" },
                { value: 'dantzig', label: "Dantzig's rule" },
              ]}
            />
            <Switch label="add x + y ≥ 2 (needs phase 1)" checked={phase1} onChange={setPhase1} />
            <Player value={position} onChange={setPosition} count={t.steps.length} label="step" />
          </>
        }
        readouts={
          <Readouts>
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
          </Readouts>
        }
        caption="Drag the arrow's tip to turn the objective. Step through the pivots with the player; the tableau below highlights the entering column, the leaving row and the pivot."
      >
        <XYChart
          series={series}
          segments={segments}
          vectors={vectors}
          handles={handles}
          xRange={BOX}
          yRange={BOX}
          equalAspect
          xLabel="x₁"
          yLabel="x₂"
        />
      </Figure>
      <TableauView
        tableau={state.tableau}
        basis={state.basis}
        labels={state.labels}
        entering={state.entering}
        leaving={state.leaving}
        ratios={state.ratios}
      />
    </div>
  )
}

const MUS = Array.from({ length: 60 }, (_, k) => 10 ** (1.5 - (7 * k) / 59))

export function InteriorPointSpecimen() {
  const [degrees, setDegrees] = useState(55)
  const theta = (degrees * Math.PI) / 180
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
  const { vectors, handles } = useDirection(theta, setDegrees, centre)
  const segments = useMemo(() => constraintSegments(LP_A, LP_B), [])
  const series = useMemo<XYSeries[]>(() => {
    const central = pathOf(toRows(path.x))
    const walked = pathOf(iterates.slice(0, position + 1))
    return [
      { name: 'feasible region', type: 'line', ...closedPath(region), slot: 0 },
      { name: 'central path x(μ)', type: 'line', ...central, slot: 2, dashed: true },
      { name: 'Mehrotra iterates', type: 'line', ...walked, slot: 1, showPoints: true },
      {
        name: 'current iterate',
        type: 'scatter',
        x: [iterates[position][0]],
        y: [iterates[position][1]],
        emphasis: true,
      },
    ]
  }, [region, path, iterates, position])
  return (
    <Figure
      title="Interior-point method and the central path"
      description="The central path x(μ) minimises −u·x − μ Σ log(slacks) and runs from the analytic centre (large μ) to the optimal vertex (μ → 0). Mehrotra's predictor–corrector iterates follow it from an infeasible start, cutting μ by orders of magnitude per step."
      controls={
        <>
          <Slider
            label="objective direction u (degrees)"
            value={degrees}
            onChange={setDegrees}
            min={0}
            max={360}
            step={1}
          />
          <Player value={position} onChange={setPosition} count={t.steps.length} label="iteration" />
        </>
      }
      readouts={
        <Readouts>
          <Readout label="μ" value={formatValue(state.mu)} />
          <Readout label="σ" value={formatValue(state.sigma)} />
          <Readout label="α primal" value={formatValue(state.alphaPrimal)} />
          <Readout label="α dual" value={formatValue(state.alphaDual)} />
          <Readout label="primal residual" value={formatValue(state.primalResidual)} />
          <Readout label="relative gap" value={formatValue(state.gap)} />
          <Readout label="u·x" value={formatValue(-state.objective)} />
        </Readouts>
      }
      caption="Drag the arrow's tip to turn the objective: the central path bends to its new optimal vertex. Step through the iterations to see μ fall."
    >
      <XYChart
        series={series}
        segments={segments}
        vectors={vectors}
        handles={handles}
        xRange={BOX}
        yRange={BOX}
        equalAspect
        xLabel="x₁"
        yLabel="x₂"
      />
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

export function BranchAndBoundSpecimen() {
  const [strategy, setStrategy] = useState<NodeSelection>('depth-first')
  const t = useMemo(() => trace(branchAndBound(IP, { strategy }), {}, 200), [strategy])
  // Start at the end, so the whole search tree shows; changing the strategy keeps the reader's step where it fits.
  const [chosen, setPosition] = useState<number | null>(null)
  const position = Math.min(chosen ?? t.steps.length - 1, t.steps.length - 1)
  const state = t.steps[position]
  // `current` is the processed node's id (read into a local so it is not mistaken for a ref).
  const { nodes, current: processed } = state
  const searchTree = useMemo(() => branchAndBoundTree(nodes), [nodes])
  const region = useMemo(() => polygonOf([...IP_A, ...NON_NEGATIVE], [...IP_B, 0, 0]), [])
  const grid = useMemo(() => lattice(0, 7), [])
  const node = processed >= 0 ? nodes[processed] : nodes[0]
  const regionSeries = useMemo<XYSeries[]>(() => {
    const [lx, ly] = toFlat(node.lower).map((v) => Math.max(v, IP_BOX[0]))
    const [ux, uy] = toFlat(node.upper).map((v) => Math.min(v, IP_BOX[1]))
    const series: XYSeries[] = [
      { name: 'integer points', type: 'scatter', ...grid, muted: true },
      { name: 'LP relaxation', type: 'line', ...closedPath(region), slot: 0 },
      { name: "node's bounds", type: 'line', x: [lx, ux, ux, lx, lx], y: [ly, ly, uy, uy, ly], slot: 1, dashed: true },
    ]
    if (node.x)
      series.push({
        name: "node's LP optimum",
        type: 'scatter',
        x: [node.x.data[0]],
        y: [node.x.data[1]],
        emphasis: true,
      })
    if (state.incumbent)
      series.push({
        name: 'incumbent',
        type: 'scatter',
        x: [state.incumbent.data[0]],
        y: [state.incumbent.data[1]],
        slot: 2,
      })
    return series
  }, [node, state, region, grid])
  return (
    <Figure
      title="Branch and bound"
      description="Maximise 5x + 8y subject to x + y ≤ 6 and 5x + 9y ≤ 45 over the integers. Each node solves the LP relaxation within its bounds, then branches on a fractional variable or is pruned (infeasible, or its bound cannot beat the incumbent)."
      controls={
        <>
          <Select
            label="node selection"
            value={strategy}
            onChange={setStrategy}
            options={['depth-first', 'best-bound', 'breadth-first']}
          />
          <Player value={position} onChange={setPosition} count={t.steps.length} label="node processed" />
        </>
      }
      readouts={
        <Readouts>
          <Readout label="node" value={processed >= 0 ? `${processed} (${branchLabel(node)})` : '—'} />
          <Readout label="outcome" value={processed >= 0 ? node.status : '—'} />
          <Readout label="node bound" value={formatValue(-node.bound)} />
          <Readout label="incumbent" value={state.incumbent ? formatValue(-state.incumbentValue) : 'none'} />
          <Readout label="best bound" value={formatValue(-state.bestBound)} />
          <Readout label="open nodes" value={state.open.length} />
        </Readouts>
      }
      caption="Values are shown for the maximisation (the solver minimises −5x − 8y). Each tree node shows its relaxation's value, a bound on what its subtree can reach; edges show the branching constraint (down branch left). Green leaves are integral, orange ones pruned (∅: infeasible); open nodes are dimmed and show their parent's bound. Compare how many nodes depth-first and best-bound need."
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
        <XYChart series={regionSeries} xRange={IP_BOX} yRange={IP_BOX} equalAspect xLabel="x" yLabel="y" />
      </div>
    </Figure>
  )
}

export function GomorySpecimen() {
  const t = useMemo(() => trace(gomory(IP), {}, 30), [])
  const [position, setPosition] = usePosition(t.steps.length)
  const state = t.steps[position]
  const grid = useMemo(() => lattice(0, 7), [])
  const series = useMemo<XYSeries[]>(() => {
    const cuts = state.cuts.map((c) => toFlat(c.a))
    const A = [...IP_A, ...NON_NEGATIVE, ...cuts]
    const b = [...IP_B, 0, 0, ...state.cuts.map((c) => c.b)]
    const optima = pathOf(t.steps.slice(0, position + 1).map((s) => toFlat(s.x)))
    return [
      { name: 'integer points', type: 'scatter', ...grid, muted: true },
      { name: 'relaxation with cuts', type: 'line', ...closedPath(polygonOf(A, b)), slot: 0 },
      { name: 'LP optimum after each cut', type: 'line', ...optima, slot: 1, showPoints: true },
      { name: 'current LP optimum', type: 'scatter', x: [state.x.data[0]], y: [state.x.data[1]], emphasis: true },
    ]
  }, [state, t, position, grid])
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
      description="The same integer program solved by cutting planes: each cut is read from a row of the optimal tableau whose basic variable is fractional. It removes the current vertex and no integer point; dual simplex pivots then restore optimality."
      controls={<Player value={position} onChange={setPosition} count={t.steps.length} label="cuts" />}
      readouts={
        <Readouts>
          <Readout label="cuts" value={state.cuts.length} />
          <Readout label="latest cut" value={cutText} />
          <Readout label="dual pivots" value={state.dualPivots} />
          <Readout label="5x + 8y" value={formatValue(-state.objective)} />
          <Readout label="status" value={state.status} />
        </Readouts>
      }
    >
      <XYChart series={series} segments={segments} xRange={IP_BOX} yRange={IP_BOX} equalAspect xLabel="x" yLabel="y" />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Dynamic programming tables.

/** A table (rows × columns) as heatmap data with row 0 at the top: y is minus the row index. */
function tableGrid(rows: number[][]) {
  const m = rows.length
  const n = rows[0]?.length ?? 0
  return {
    x: Array.from({ length: n }, (_, j) => j),
    y: Array.from({ length: m }, (_, i) => i - (m - 1)),
    z: Array.from({ length: m }, (_, i) => rows[m - 1 - i]),
  }
}

const VALUES = [6, 10, 12, 7, 3, 9]
const WEIGHTS = [1, 2, 3, 2, 1, 4]

export function KnapsackSpecimen() {
  const [capacity, setCapacity] = useState(8)
  const program = useMemo(() => knapsackProgram(VALUES, WEIGHTS, capacity), [capacity])
  const t = useMemo(() => trace(dynamicProgram(program), {}, VALUES.length + 1), [program])
  const result = useMemo(() => knapsack(VALUES, WEIGHTS, capacity), [capacity])
  const [position, setPosition] = usePosition(t.steps.length)
  const state = t.steps[position]
  const grid = useMemo(() => tableGrid(toRows(state.table)), [state])
  const overlay = useMemo<HeatmapOverlay[]>(() => {
    if (!state.converged) return []
    const cells = toRows(result.path)
    return [
      {
        name: 'traceback',
        type: 'line',
        x: cells.map((c) => c[1]),
        y: cells.map((c) => -c[0]),
        showPoints: true,
        slot: 1,
      },
    ]
  }, [state, result])
  const chosen = toFlat(result.take).flatMap((k, i) => (k ? [`item ${i + 1}`] : []))
  return (
    <Figure
      title="0/1 knapsack table"
      description="Cell (i, c) is the best value from the first i items within capacity c: T[i][c] = max(T[i−1][c], T[i−1][c − wᵢ] + vᵢ). Each step fills one row; the traceback reads the chosen items back from the full table."
      controls={
        <>
          <Slider label="capacity C" value={capacity} onChange={setCapacity} min={1} max={13} step={1} />
          <Player value={position} onChange={setPosition} count={t.steps.length} label="rows filled" />
        </>
      }
      readouts={
        <Readouts>
          <Readout label="values" value={VALUES.join(', ')} />
          <Readout label="weights" value={WEIGHTS.join(', ')} />
          <Readout label="best value" value={state.converged ? result.value : '…'} />
          <Readout label="chosen" value={state.converged ? chosen.join(', ') : '…'} />
          <Readout label="weight used" value={state.converged ? result.weight : '…'} />
        </Readouts>
      }
    >
      <Heatmap
        {...grid}
        overlay={overlay}
        xLabel="capacity c"
        yLabel="−items i"
        valueLabel="best value"
        range={[0, result.value]}
      />
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

export function AlignmentSpecimen() {
  const [pair, setPair] = useState<PairName>('TGTTACGG / GGTTGACTA')
  const [mode, setMode] = useState<'global' | 'local'>('local')
  const [match, setMatch] = useState(3)
  const [mismatch, setMismatch] = useState(-3)
  const [gap, setGap] = useState(-2)
  const [a, b] = PAIRS[pair]
  const result = useMemo(() => {
    const scoring = { match, mismatch, gap }
    return mode === 'global' ? needlemanWunsch(a, b, scoring) : smithWaterman(a, b, scoring)
  }, [a, b, mode, match, mismatch, gap])
  const grid = useMemo(() => tableGrid(toRows(result.table)), [result])
  const overlay = useMemo<HeatmapOverlay[]>(() => {
    const cells = toRows(result.path)
    return [
      {
        name: 'traceback',
        type: 'line',
        x: cells.map((c) => c[1]),
        y: cells.map((c) => -c[0]),
        showPoints: true,
        slot: 1,
      },
    ]
  }, [result])
  return (
    <Figure
      title="Sequence alignment"
      description="Needleman–Wunsch (global) and Smith–Waterman (local) fill F[i][j] = max(F[i−1][j−1] + s(aᵢ, bⱼ), F[i−1][j] + gap, F[i][j−1] + gap), with 0 as a fourth option for local alignment. The traceback from the end cell spells out the alignment."
      controls={
        <>
          <Select label="sequences" value={pair} onChange={setPair} options={Object.keys(PAIRS) as PairName[]} />
          <Select label="alignment" value={mode} onChange={setMode} options={['global', 'local']} />
          <Slider label="match" value={match} onChange={setMatch} min={0} max={5} step={1} />
          <Slider label="mismatch" value={mismatch} onChange={setMismatch} min={-5} max={0} step={1} />
          <Slider label="gap" value={gap} onChange={setGap} min={-5} max={0} step={1} />
        </>
      }
      readouts={
        <Readouts>
          <Readout label="score" value={result.score} />
          <Readout label="a" value={<span className="whitespace-pre">{String(result.alignedA)}</span>} />
          <Readout label="b" value={<span className="whitespace-pre">{String(result.alignedB)}</span>} />
        </Readouts>
      }
      caption={`Columns are positions in b (${b}), rows positions in a (${a}); row and column 0 are the empty prefixes.`}
    >
      <Heatmap {...grid} overlay={overlay} scale="diverging" xLabel="j (b)" yLabel="−i (a)" valueLabel="score" />
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
  const [seed, setSeed] = useState(1)
  const cost = useMemo(() => costMatrix(seed, 5), [seed])
  const t = useMemo(() => trace(hungarianSteps(cost), {}, 500), [cost])
  const [position, setPosition] = usePosition(t.steps.length)
  const state = t.steps[position]
  const grid = useMemo(() => tableGrid(toRows(state.reduced)), [state])
  const overlay = useMemo<HeatmapOverlay[]>(() => {
    const star = toFlat(state.starred)
    const prime = toFlat(state.primed)
    const cells = (cols: number[]) => cols.flatMap((j, i) => (j >= 0 ? [[j, -i]] : []))
    const s = cells(star)
    const p = cells(prime)
    const out: HeatmapOverlay[] = [
      { name: 'starred zeros', type: 'scatter', x: s.map((c) => c[0]), y: s.map((c) => c[1]), slot: 1 },
      { name: 'primed zeros', type: 'scatter', x: p.map((c) => c[0]), y: p.map((c) => c[1]), slot: 2 },
    ]
    if (state.last === 'augment') {
      const path = toRows(state.path)
      out.push({
        name: 'augmenting path',
        type: 'line',
        x: path.map((c) => c[1]),
        y: path.map((c) => -c[0]),
        slot: 3,
        showPoints: true,
      })
    }
    return out
  }, [state])
  const covered = (t: typeof state.rowCovered) =>
    toFlat(t)
      .flatMap((v, i) => (v ? [i + 1] : []))
      .join(', ') || 'none'
  let total = 0
  toFlat(state.starred).forEach((j, i) => j >= 0 && (total += cost[i][j]))
  return (
    <Figure
      title="Hungarian algorithm"
      description="Munkres' steps on a 5 × 5 cost matrix: reduce rows and columns, star independent zeros, cover their columns, prime uncovered zeros and augment along alternating paths, and create new zeros by adjusting with the smallest uncovered value."
      controls={
        <>
          <Slider label="seed" value={seed} onChange={setSeed} min={1} max={30} step={1} />
          <Player value={position} onChange={setPosition} count={t.steps.length} label="step" />
        </>
      }
      readouts={
        <Readouts>
          <Readout label="last step" value={state.last} />
          <Readout label="next" value={state.next} />
          <Readout label="covered rows" value={covered(state.rowCovered)} />
          <Readout label="covered columns" value={covered(state.columnCovered)} />
          <Readout label="adjustment" value={state.last === 'adjust' ? formatValue(state.delta) : '—'} />
          <Readout label="cost of starred" value={total} />
        </Readouts>
      }
      caption="The heatmap is the reduced matrix cost − u − v; starred zeros are the current assignment."
    >
      <Heatmap {...grid} overlay={overlay} xLabel="column" yLabel="−row" valueLabel="reduced cost" />
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

export function ActiveSetSpecimen() {
  const [target, setTarget] = useState<Pt>([1, 2.5])
  const [method, setMethod] = useState<'active set' | 'interior point'>('active set')
  const problem = useMemo(
    () => ({
      Q: [
        [2, 0],
        [0, 2],
      ],
      c: [-2 * target[0], -2 * target[1]],
      A: QP_A,
      b: QP_B,
    }),
    [target],
  )
  const iterates = useMemo(() => {
    if (method === 'active set') {
      const t = trace(activeSet(problem), { x0: [2, 0] }, 50)
      return t.steps.map((s) => ({
        x: toFlat(s.x),
        text: `${s.event}${s.changed >= 0 ? ` constraint ${s.changed + 1}` : ''}; working set {${toFlat(s.working)
          .map((i) => i + 1)
          .join(', ')}}`,
      }))
    }
    const t = trace(quadraticInteriorPoint(problem), {}, 50)
    return t.steps.map((s) => ({ x: toFlat(s.x), text: `μ = ${formatValue(s.mu)}` }))
  }, [problem, method])
  const [position, setPosition] = usePosition(iterates.length)
  const current = iterates[position]
  const region = useMemo(() => polygonOf(QP_A, QP_B), [])
  const segments = useMemo(() => constraintSegments(QP_A, QP_B, QP_BOX), [])
  const series = useMemo<XYSeries[]>(() => {
    const r = Math.hypot(current.x[0] - target[0], current.x[1] - target[1])
    const angles = Array.from({ length: 73 }, (_, k) => (k * 2 * Math.PI) / 72)
    return [
      { name: 'feasible region', type: 'line', ...closedPath(region), slot: 0 },
      {
        name: 'level set through x',
        type: 'line',
        x: angles.map((a) => target[0] + r * Math.cos(a)),
        y: angles.map((a) => target[1] + r * Math.sin(a)),
        slot: 2,
        dashed: true,
      },
      {
        name: 'iterates',
        type: 'line',
        ...pathOf(iterates.slice(0, position + 1).map((i) => i.x)),
        slot: 1,
        showPoints: true,
      },
      { name: 'current', type: 'scatter', x: [current.x[0]], y: [current.x[1]], emphasis: true },
    ]
  }, [region, iterates, position, current, target])
  const handles = useMemo<Handle[]>(
    () => [
      {
        kind: 'point',
        at: target,
        label: 'unconstrained minimiser',
        onDrag: ([x, y]) => setTarget([Math.round(x * 20) / 20, Math.round(y * 20) / 20]),
      },
    ],
    [target],
  )
  return (
    <Figure
      title="Quadratic programming: active set and interior point"
      description="Minimise ‖x − t‖² over a polygon (Nocedal and Wright, Example 16.3). The active-set method moves along the working set's faces, adding blocking constraints and dropping those with negative multipliers; the interior-point method approaches through the interior."
      controls={
        <>
          <Select label="method" value={method} onChange={setMethod} options={['active set', 'interior point']} />
          <Player value={position} onChange={setPosition} count={iterates.length} label="step" />
        </>
      }
      readouts={
        <Readouts>
          <Readout label="t" value={`(${target[0]}, ${target[1]})`} />
          <Readout label="x" value={`(${formatValue(current.x[0])}, ${formatValue(current.x[1])})`} />
          <Readout label="step" value={current.text} />
        </Readouts>
      }
      caption="Drag t, the unconstrained minimiser. The dashed circle is the level set through the current iterate."
    >
      <XYChart
        series={series}
        segments={segments}
        handles={handles}
        xRange={QP_BOX}
        yRange={QP_BOX}
        equalAspect
        xLabel="x₁"
        yLabel="x₂"
      />
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

export function ProgramViewSpecimen() {
  const [name, setName] = useState<CaseName>('LP, simplex')
  const result = useMemo(() => CASES[name](), [name])
  return (
    <div className="space-y-4">
      <div className="max-w-sm">
        <Select label="problem" value={name} onChange={setName} options={Object.keys(CASES) as CaseName[]} />
      </div>
      <ProgramView result={result} />
    </div>
  )
}
