import { useMemo } from 'react'
import { choice, Curve, Figure, float, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import {
  circleLayout,
  drawingRange,
  FixedHeight,
  graphLayers,
  useSlicedRun,
  type Drawing,
} from '../_shared/figure-helpers'
import { DATA, LARGE, READOUTS, readoutRun, TEST, TRAIN, type Checkpoint, type Setup, type Task } from './readout-run'

const PANEL = 260
const TASKS: { value: Task; label: string }[] = [
  { value: 'family', label: 'family: cycle, star or tree' },
  { value: 'size', label: `size: at least ${LARGE} nodes or not` },
]

/** One drawn example of each family, side by side: the first graph of that family in the data. */
const EXAMPLES: Drawing[] = [0, 1, 2].map((f, k) => {
  const i = DATA.family.indexOf(f)
  const g = DATA.graphs[i]
  const n = g.nodes
  const cx = 3 * k
  const pos: [number, number][] =
    f === 1
      ? [[cx, 0], ...circleLayout(n - 1, cx, 0, 1)]
      : f === 0
        ? circleLayout(n, cx, 0, 1)
        : treeLayout(
            g.edges.map((e) => [e.from, e.to] as [number, number]),
            n,
            cx,
          )
  return { pos, edges: g.edges.map((e) => [e.from, e.to] as const) }
})

/** A layered drawing of a rooted tree (root 0 at the top), scaled into a 2 by 2 box centred at x = cx. */
function treeLayout(edges: [number, number][], n: number, cx: number): [number, number][] {
  const children: number[][] = Array.from({ length: n }, () => [])
  for (const [a, b] of edges) children[Math.min(a, b)].push(Math.max(a, b))
  const depth = new Array<number>(n).fill(0)
  const order: number[] = []
  const visit = (v: number) => {
    order.push(v)
    for (const c of children[v]) {
      depth[c] = depth[v] + 1
      visit(c)
    }
  }
  visit(0)
  // Leaves get consecutive slots left to right; an inner node sits over the middle of its children.
  const slot = new Array<number>(n).fill(0)
  let next = 0
  const place = (v: number): number => {
    if (children[v].length === 0) return (slot[v] = next++)
    const xs = children[v].map(place)
    return (slot[v] = (Math.min(...xs) + Math.max(...xs)) / 2)
  }
  place(0)
  const width = Math.max(1, next - 1)
  const height = Math.max(1, Math.max(...depth))
  return order
    .slice()
    .sort((a, b) => a - b)
    .map((v) => [cx - 1 + (2 * slot[v]) / width, 1 - (2 * depth[v]) / height])
}

/** The share of test graphs with the commoner label of a task: the accuracy of always predicting it. */
function majority(task: Task): number {
  const labels = task === 'family' ? DATA.family.slice(TRAIN) : DATA.size.slice(TRAIN).map((n) => (n >= LARGE ? 1 : 0))
  const counts = [0, 1, 2].map((c) => labels.filter((y) => y === c).length)
  return Math.max(...counts) / TEST
}

const percent = (x?: number) => (x === undefined ? '…' : `${Math.round(100 * x)}%`)

/** Four readouts of the same message passing network, trained on cycles, stars and trees with constant features. */
export function ReadoutComparison() {
  const state = useFigureState({
    task: choice(TASKS, 'size', { label: 'task' }),
    steps: int(150, { ge: 10, le: 2000, suggestions: [50, 150, 500], label: 'Adam steps' }),
    rate: float(0.02, { gt: 0, le: 0.5, suggestions: [0.005, 0.02, 0.05], label: 'learning rate' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { task, steps, rate, seed } = state
  const setup = useMemo<Setup>(() => ({ task, steps, rate, seed }), [task, steps, rate, seed])
  const run = useSlicedRun(setup, readoutRun)
  const cps: Checkpoint[] = run.checkpoints
  const last = cps.at(-1)

  const range = drawingRange(EXAMPLES, 0.35)
  const gx = useAxis({ range: range.x, nice: false, zoom: false })
  const gy = useAxis({ range: range.y, nice: false, zoom: false, equal: gx })
  const sx = useAxis({ label: 'Adam step', range: [0, steps], nice: false })
  const ay = useAxis({ label: 'test accuracy', range: [0, 1.02], nice: false })
  const chance = majority(task)

  return (
    <Figure
      title="Which readout can count nodes"
      state={state}
      defaultSize="L"
      caption={`${TRAIN + TEST} small graphs, a third each cycles, stars and random trees with 6 to 13 nodes; ${TRAIN} train the network and ${TEST} test it. Every node starts with the same feature, 1, so only the structure carries information. The network is two GIN-0 layers of width 16 with tanh, a readout over each graph's nodes, and a one-hidden-layer classifier; it is trained four times from the same start, once per readout, by full-batch Adam. The family task asks whether a graph is a cycle, a star or a tree; the size task asks whether it has at least ${LARGE} nodes. On the family task every readout reaches high accuracy, because the three families differ in their mix of node degrees. On the size task only the sum readout succeeds: every node of a cycle has the same embedding at every layer, so the mean, the max and the attention-weighted mean give every cycle the same graph vector whatever its length. The dashed line is the accuracy of always predicting the commoner test label. Change the task, the training length, the rate or the seed; the curves fill in as training runs.`}
      readouts={
        <>
          {READOUTS.map((r) => (
            <Readout key={r} label={`${r} readout, test accuracy`} value={percent(last?.test[r])} />
          ))}
          <Readout label="step" value={last ? `${last.step}${run.done ? '' : ', training…'}` : '…'} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[2fr_3fr]">
        <FixedHeight height={PANEL}>
          <Plot x={gx} y={gy} bare title="one graph of each family">
            {EXAMPLES.flatMap((d, k) => graphLayers(d, { id: `example-${k}`, size: 9 }))}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sx} y={ay} title="accuracy on the test graphs">
            {READOUTS.map((r, k) => (
              <Curve key={r} name={r} slot={k} x={cps.map((c) => c.step)} y={cps.map((c) => c.test[r])} />
            ))}
            <Curve name="majority label" x={[0, steps]} y={[chance, chance]} dashed muted />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
