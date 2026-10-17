import { useMemo, useState } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  toggle,
  useAxis,
  useFigureState,
  useScaleColor,
  Vectors,
  type Vector,
} from 'aifn-render'
import { add, mul, slice, tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { messagePassing } from 'aifn-compute/nn/graph'
import type { Graph } from 'aifn-compute/graph'

// The worked example's graph: a triangle 1-2-3 with a tail 3-4 (nodes 0-3 here), and a bond order per edge.
const N = 4
const EDGES: [number, number, number][] = [
  [0, 1, 1],
  [0, 2, 2],
  [1, 2, 1],
  [2, 3, 2],
]
const POS: [number, number][] = [
  [0, 1],
  [0, -1],
  [1.6, 0],
  [3.2, 0],
]
const H0 = [1, 2, 3, 4]
const ROUNDS = 3
const AGGREGATORS = ['sum', 'mean', 'max'] as const
type Aggregator = (typeof AGGREGATORS)[number]

const graphOf = (weighted: boolean): Graph => ({
  kind: 'graph',
  nodes: N,
  edges: EDGES.map(([from, to, w]) => (weighted ? { from, to, weight: w } : { from, to })),
  directed: false,
})

// Message M(h_v, h_w, w_vw) = w_vw h_w, read from the rows [h_w, h_v, w_vw]; update U(h_v, m_v) = h_v + m_v.
const message = (x: Value): Value => mul(slice(x, null, [0, 1]), slice(x, null, [2, 3]))
const update = (x: Value): Value => add(slice(x, null, [0, 1]), slice(x, null, [1, 2]))
const aggregateOnly = (x: Value): Value => slice(x, null, [1, 2])

const neighbours = (v: number) => EDGES.flatMap(([a, b, w]) => (a === v ? [{ u: b, w }] : b === v ? [{ u: a, w }] : []))

/** Hop distance from v to every node (breadth-first search on four nodes). */
function hops(v: number): number[] {
  const d = new Array<number>(N).fill(Infinity)
  d[v] = 0
  const queue = [v]
  while (queue.length) {
    const x = queue.shift()!
    for (const { u } of neighbours(x)) if (d[u] === Infinity) (queue.push(u), (d[u] = d[x] + 1))
  }
  return d
}

/** Message passing on a four-node graph, one phase at a time: messages along the edges, then the update. */
export function MessagePassingStepper() {
  const state = useFigureState({
    aggregate: choice<Aggregator>([...AGGREGATORS], 'sum', { label: 'aggregator' }),
    node: choice([1, 2, 3, 4], 3, { label: 'inspect node' }),
    weighted: toggle(false, { label: 'edge features (bond order)' }),
  })
  const [step, setStep] = useState(0)

  // The whole trace up front: h^(t) for t = 0..ROUNDS, and the sum, mean and max received by every node per round.
  const trace = useMemo(() => {
    const g = graphOf(state.weighted)
    const h: Tensor[] = [tensor(H0.map((v) => [v]))]
    const received: Record<Aggregator, number[]>[] = []
    for (let t = 0; t < ROUNDS; t++) {
      const prev = h[t]
      received.push(
        Object.fromEntries(
          AGGREGATORS.map((a) => [
            a,
            toFlat(messagePassing(g, prev, message, aggregateOnly, { aggregate: a }) as Tensor),
          ]),
        ) as Record<Aggregator, number[]>,
      )
      h.push(messagePassing(g, prev, message, update, { aggregate: state.aggregate }) as Tensor)
    }
    return { h: h.map((x) => toFlat(x)), received }
  }, [state.aggregate, state.weighted])

  const v = state.node - 1
  const round = Math.ceil(step / 2) // the round being shown (0 at the start)
  const sending = step % 2 === 1 // odd steps: messages of round `round`, computed from h^(round-1)
  const shown = trace.h[sending ? round - 1 : round]
  const reached = sending ? round - 1 : round // rounds completed
  const dist = useMemo(() => hops(v), [v])

  const colour = useScaleColor('sequential')
  const [lo, hi] = [Math.min(...shown), Math.max(...shown)]
  const colours = shown.map((x) => colour(hi > lo ? (x - lo) / (hi - lo) : 0.5))

  const arrows: Vector[] = sending
    ? neighbours(v).map(({ u, w }): Vector => {
        const [x0, y0] = POS[u]
        const [x1, y1] = POS[v]
        const len = Math.hypot(x1 - x0, y1 - y0)
        const cut = 0.22 / len // stop short of both nodes
        const m = (state.weighted ? w : 1) * trace.h[round - 1][u]
        return {
          from: [x0 + (x1 - x0) * cut, y0 + (y1 - y0) * cut],
          to: [x1 - (x1 - x0) * cut, y1 - (y1 - y0) * cut],
          label: formatNumber(m),
          labelAt: 'middle',
          slot: 1,
          width: 2,
        }
      })
    : []

  const field = POS.filter((_, i) => dist[i] <= reached)
  const xAxis = useAxis({ label: '', range: [-0.6, 3.8], nice: false })
  const yAxis = useAxis({ label: '', range: [-1.5, 1.5], nice: false, equal: xAxis })
  const received = round > 0 ? trace.received[round - 1] : null
  const phase =
    step === 0
      ? 'Step 0: the initial features h⁽⁰⁾.'
      : sending
        ? `Round ${round}, messages: every neighbour w of node ${state.node} sends M = ${state.weighted ? 'w_vw · ' : ''}h_w⁽${round - 1}⁾.`
        : `Round ${round}, update: every node sets h_v⁽${round}⁾ = h_v⁽${round - 1}⁾ + m_v⁽${round}⁾.`

  return (
    <Figure
      title="Message passing, one phase at a time"
      state={state}
      caption={`${phase} The graph is the worked example's: a triangle 1–2–3 with a tail 3–4, scalar features h⁽⁰⁾ = (1, 2, 3, 4), message M = h_w (times the bond order w_vw when edge features are on) and update h_v + m_v. Node colour shows the current value; the dark rings mark the nodes whose initial features have reached the inspected node after the rounds completed so far, which grow by one hop per round. The readouts give the sum, mean and max of the messages the inspected node receives in the round shown, whichever aggregator the update uses: sum separates nodes 1 and 2 from the rest by count, mean and max do not count neighbours.`}
      controls={
        <Player
          value={step}
          onChange={setStep}
          count={2 * ROUNDS + 1}
          label="phase"
          format={(s) => (s === 0 ? 'start' : `round ${Math.ceil(s / 2)}: ${s % 2 === 1 ? 'messages' : 'update'}`)}
        />
      }
      readouts={
        <>
          {AGGREGATORS.map((a) => (
            <Readout
              key={a}
              label={`${a} received by node ${state.node}${a === state.aggregate ? ' (used)' : ''}`}
              value={received ? formatNumber(received[a][v]) : '·'}
            />
          ))}
          <Readout label={`h of node ${state.node}`} value={formatNumber(shown[v])} />
          <Readout label="sum readout over nodes" value={formatNumber(shown.reduce((s, x) => s + x, 0))} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Segments
            name="edges"
            segments={EDGES.map(([a, b]) => ({ from: POS[a], to: POS[b] }))}
            width={state.weighted ? 2 : 1.5}
          />
          {field.length > 0 && (
            <Points
              name={`reached node ${state.node}`}
              x={field.map((p) => p[0])}
              y={field.map((p) => p[1])}
              emphasis
              size={34}
            />
          )}
          <Points
            name="node value"
            x={POS.map((p) => p[0])}
            y={POS.map((p) => p[1])}
            colors={colours}
            size={26}
            labels={shown.map((x, i) => `${i + 1}: ${formatNumber(x)}`)}
          />
          {arrows.length > 0 && <Vectors name="messages" vectors={arrows} />}
        </Plot>
      </div>
    </Figure>
  )
}
