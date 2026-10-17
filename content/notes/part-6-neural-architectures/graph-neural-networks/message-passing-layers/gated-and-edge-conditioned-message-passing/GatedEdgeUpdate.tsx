import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  useAxis,
  useFigureState,
  Vectors,
  type Vector,
} from 'aifn-render'
import {
  add,
  matmul,
  mul,
  slice,
  tensor,
  toArray,
  zeros,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { GruCell } from 'aifn-compute/nn/layers'
import { messagePassing } from 'aifn-compute/nn/graph'
import type { Graph } from 'aifn-compute/graph'

// A three-atom fragment O–C–N: C is node 0, O node 1, N node 2. Bonds C–O and C–N have a bond order b of 1 or 2.
const ATOMS = ['C', 'O', 'N']
const POS: [number, number][] = [
  [0, 0],
  [-1.3, 0.75],
  [1.3, 0.75],
]
const H0 = [
  [1, 0],
  [0, 1],
  [0.5, -0.5],
]
// Edge-conditioned message matrix A(b) = Θ0 + b Θ1, stored transposed because node states are rows.
const THETA0_T = tensor([
  [0.5, 0],
  [0, 0.5],
])
const THETA1_T = tensor([
  [0, 0.5],
  [-0.5, 0],
])
// A GRU with every weight matrix the identity and zero biases (gates in PyTorch's order r, z, n).
const CELL = GruCell(2, 2)
const EYE3 = tensor([
  [1, 0, 1, 0, 1, 0],
  [0, 1, 0, 1, 0, 1],
])
const GRU_PARAMS = { inputWeight: EYE3, hiddenWeight: EYE3, bias: zeros([6]), hiddenBias: zeros([6]) }

// Rows [h_u, h_v, b] -> A(b) h_u, as rows.
const message = (x: Value): Value => {
  const hu = slice(x, null, [0, 2])
  return add(matmul(hu, THETA0_T), mul(slice(x, null, [4, 5]), matmul(hu, THETA1_T)))
}
const gated = (x: Value): Value => CELL.step(GRU_PARAMS, slice(x, null, [2, 4]), { h: slice(x, null, [0, 2]) }).h
const additive = (x: Value): Value => add(slice(x, null, [0, 2]), slice(x, null, [2, 4]))
const received = (x: Value): Value => slice(x, null, [2, 4])

type Bond = 1 | 2
type Update = 'gru' | 'add'
const BOND = (label: string) =>
  choice<Bond>(
    [
      { value: 1, label: 'single' },
      { value: 2, label: 'double' },
    ],
    label === 'C–O' ? 2 : 1,
    { label: `bond ${label}` },
  )

/** Offsets for drawing a bond as one line or two parallel lines. */
function bondSegments(a: [number, number], b: [number, number], order: Bond) {
  if (order === 1) return [{ from: a, to: b }]
  const [dx, dy] = [b[0] - a[0], b[1] - a[1]]
  const len = Math.hypot(dx, dy)
  const [nx, ny] = [(-dy / len) * 0.06, (dx / len) * 0.06]
  return [-1, 1].map((s) => ({
    from: [a[0] + s * nx, a[1] + s * ny] as [number, number],
    to: [b[0] + s * nx, b[1] + s * ny] as [number, number],
  }))
}

/** Edge-conditioned messages on a three-atom fragment, and a GRU update against a plain sum over many rounds. */
export function GatedEdgeUpdate() {
  const state = useFigureState({
    co: BOND('C–O'),
    cn: BOND('C–N'),
    update: choice<Update>(
      [
        { value: 'gru', label: 'GRU cell (gated)' },
        { value: 'add', label: 'h + m (no gate)' },
      ],
      'gru',
      { label: 'update' },
    ),
    rounds: int(8, { ge: 1, le: 40, suggestions: [4, 8, 20], label: 'rounds T' }),
  })
  const [pos, setPos] = useState({ key: '', t: 0 })
  const key = `${state.co}-${state.cn}-${state.update}-${state.rounds}`
  const t = pos.key === key ? Math.min(pos.t, state.rounds) : 0

  // The trace up front: states h^(t) for t = 0..T and the messages each atom receives in round t + 1.
  const trace = useMemo(() => {
    const g: Graph = {
      kind: 'graph',
      nodes: 3,
      edges: [
        { from: 0, to: 1, weight: state.co },
        { from: 0, to: 2, weight: state.cn },
      ],
      directed: false,
    }
    const states: number[][][] = [H0]
    let h: Tensor = tensor(H0)
    for (let k = 0; k < state.rounds; k++) {
      h = messagePassing(g, h, message, state.update === 'gru' ? gated : additive) as Tensor
      states.push(toArray(h) as number[][])
    }
    // Each message into C at every round, one per neighbour, read from the state before the round.
    const into = states.map((s) =>
      [1, 2].map((u) => {
        const b = u === 1 ? state.co : state.cn
        const one = messagePassing(
          { kind: 'graph', nodes: 3, edges: [{ from: u, to: 0, weight: b }], directed: true },
          tensor(s),
          message,
          received,
        ) as Tensor
        return (toArray(one) as number[][])[0]
      }),
    )
    return { states, into }
  }, [state.co, state.cn, state.update, state.rounds])

  const now = trace.states[t]
  const [mO, mN] = trace.into[t]
  const mC: [number, number] = [mO[0] + mN[0], mO[1] + mN[1]]
  const arrows: Vector[] = [
    { from: [0, 0], to: [mO[0], mO[1]], label: 'from O', slot: 1 },
    { from: [0, 0], to: [mN[0], mN[1]], label: 'from N', slot: 2 },
    { from: [0, 0], to: mC, label: 'm_C' },
  ]
  const largest = Math.max(...now.flat().map(Math.abs))
  const fmt = (v: number[]) => `(${formatNumber(v[0])}, ${formatNumber(v[1])})`

  const mx = useAxis({ label: '', range: [-1.8, 1.8], nice: false })
  const my = useAxis({ label: '', range: [-0.6, 1.4], nice: false, equal: mx })
  const sx = useAxis({ label: 'state 1', range: [-2.5, 2.5] })
  const sy = useAxis({ label: 'state 2', range: [-2.5, 2.5], equal: sx })
  const box: [number, number][] = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
    [-1, -1],
  ]

  return (
    <Figure
      title="Edge-conditioned messages and a gated update"
      state={state}
      defaultSize="L"
      caption={`Left: an O–C–N fragment whose two bonds can be single (b = 1) or double (b = 2); each atom shows its state h after round ${t}. Each message is A(b) h_u with A(b) = Θ0 + b Θ1, so changing a bond changes the matrix that carries the neighbour's state. Right: the two-dimensional states over the rounds, one trajectory per atom; the arrows are the messages C receives in the next round and their sum m_C. The GRU update makes each new state a convex combination of the old state and a tanh, so states that start in the dashed square stay there for any number of rounds; the update h + m has no gate and, with these matrices, its states grow geometrically and leave the plot within a few rounds.`}
      controls={
        <Player
          value={t}
          onChange={(v) => setPos({ key, t: v })}
          count={state.rounds + 1}
          label="round"
          format={(v) => `after round ${v}`}
        />
      }
      readouts={
        <>
          <Readout label="m_C, sum of messages into C" value={fmt(mC)} />
          <Readout label="h_C" value={fmt(now[0])} />
          <Readout label="largest |entry| of any state" value={formatNumber(largest)} />
        </>
      }
    >
      <Plots cols={2} widths={[2, 3]}>
        <Plot x={mx} y={my} title="O–C–N fragment">
          <Segments
            name="bonds"
            segments={[...bondSegments(POS[0], POS[1], state.co), ...bondSegments(POS[0], POS[2], state.cn)]}
            width={2}
          />
          <Points
            name="atoms"
            x={POS.map((p) => p[0])}
            y={POS.map((p) => p[1])}
            group={[0, 1, 2]}
            groupNames={ATOMS}
            size={18}
            labels={ATOMS.map((a, i) => `${a} ${fmt(now[i])}`)}
          />
        </Plot>
        <Plot x={sx} y={sy} title="states over rounds">
          <Curve name="square |h| ≤ 1" x={box.map((p) => p[0])} y={box.map((p) => p[1])} muted dashed width={1} />
          {ATOMS.map((a, i) => (
            <Curve
              key={a}
              name={`${a} trajectory`}
              slot={i}
              x={trace.states.slice(0, t + 1).map((s) => s[i][0])}
              y={trace.states.slice(0, t + 1).map((s) => s[i][1])}
              showPoints
              width={1.5}
            />
          ))}
          <Points name="states now" x={now.map((s) => s[0])} y={now.map((s) => s[1])} emphasis size={9} />
          <Vectors name="messages into C" vectors={arrows} />
        </Plot>
      </Plots>
    </Figure>
  )
}
