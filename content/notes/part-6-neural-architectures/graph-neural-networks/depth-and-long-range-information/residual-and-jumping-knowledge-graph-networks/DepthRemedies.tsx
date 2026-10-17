import { useMemo } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Plots,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { dirichletEnergy } from 'aifn-compute/nn/graph'
import { propagationStack, type PropagationVariant } from 'aifn-methods/neural/graph'
import { plantedPartition, readoutAccuracy } from '../_shared/communities'

const DEPTH = 64
const LAYERS = Array.from({ length: DEPTH + 1 }, (_, k) => k)
/** The jumping-knowledge readout fits (layers + 1) times as many weights, so it takes a larger ridge penalty. */
const JK_L2 = 2

const SERIES: { key: PropagationVariant | 'jk'; name: string; slot: number }[] = [
  { key: 'plain', name: 'plain Â H', slot: 0 },
  { key: 'residual', name: 'residual ½(H + Â H)', slot: 1 },
  { key: 'initial-residual', name: 'initial residual (GCNII, W = I)', slot: 2 },
  { key: 'dropedge', name: 'DropEdge', slot: 3 },
  { key: 'jk', name: 'jumping knowledge (concat)', slot: 4 },
]

/** Accuracy and Dirichlet energy against depth for weight-free stacks with each remedy for oversmoothing. */
export function DepthRemedies() {
  const state = useFigureState({
    alpha: float(0.1, {
      min: 0,
      max: 1,
      step: 0.05,
      suggestions: [0.05, 0.1, 0.2, 0.5],
      label: 'initial-residual weight α',
    }),
    drop: slider(0, 0.9, 0.3, { step: 0.05, label: 'DropEdge probability' }),
    pOut: slider(0, 0.15, 0.04, { step: 0.01, label: 'edge probability across communities' }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })

  const run = useMemo(() => {
    const data = plantedPartition(state.seed, 0.25, state.pOut)
    const stacks: Record<PropagationVariant, number[][][]> = {
      plain: propagationStack(data.graph, data.x, { depth: DEPTH }),
      residual: propagationStack(data.graph, data.x, { depth: DEPTH, variant: 'residual' }),
      'initial-residual': propagationStack(data.graph, data.x, {
        depth: DEPTH,
        variant: 'initial-residual',
        alpha: state.alpha,
      }),
      dropedge: propagationStack(data.graph, data.x, {
        depth: DEPTH,
        variant: 'dropedge',
        dropProbability: state.drop,
        stream: stream(state.seed + 1),
      }),
      pairnorm: [],
    }
    const accuracy: Record<string, number[]> = {}
    const energy: Record<string, number[]> = {}
    for (const s of SERIES) {
      if (s.key === 'jk') continue
      accuracy[s.key] = stacks[s.key].map((h) => readoutAccuracy(h, data.y))
      energy[s.key] = stacks[s.key].map((h) => dirichletEnergy(data.graph, h))
    }
    // Jumping knowledge: the readout sees the concatenation of layers 0..K of the plain stack.
    accuracy.jk = LAYERS.map((K) => {
      const rows = data.x.map((_, i) => stacks.plain.slice(0, K + 1).flatMap((layer) => layer[i]))
      return readoutAccuracy(rows, data.y, JK_L2)
    })
    return { accuracy, energy }
  }, [state.seed, state.pOut, state.alpha, state.drop])

  const ax = useAxis({ label: 'depth L (layers)', range: [0, DEPTH], integer: true })
  const ay = useAxis({ label: 'accuracy on unlabelled nodes', range: [0, 1] })
  const ex = useAxis({ label: 'depth L (layers)', range: [0, DEPTH], integer: true })
  const ey = useAxis({ label: 'Dirichlet energy', log: true })
  const at = (key: string, k: number) => formatNumber(run.accuracy[key][k])

  return (
    <Figure
      title="Remedies for oversmoothing against depth"
      state={state}
      defaultSize="L"
      readouts={
        <>
          {SERIES.map((s) => (
            <Readout
              key={s.key}
              label={`${s.name}: accuracy at L = 4 / 64`}
              value={`${at(s.key, 4)} / ${at(s.key, 64)}`}
            />
          ))}
        </>
      }
      caption="The planted-partition graph of the oversmoothing note: 40 nodes in two communities, 2-D features, 8 labelled nodes. Each stack applies L weight-free layers; a ridge readout fitted to the labelled nodes scores the last layer on the other 32 (penalty 0.05; the jumping-knowledge readout sees all L + 1 layers side by side and uses penalty 2, since it fits L + 1 times as many weights). Left: accuracy against depth. Right: the Dirichlet energy of the last layer on a log scale (jumping knowledge has no single last layer). The averaged residual and DropEdge decay more slowly than the plain stack but reach the same collapse. The initial residual converges to personalised PageRank features, whose energy levels off, so its accuracy stays flat at any depth. Set α to 0 to recover the plain stack and to 1 to ignore the graph."
    >
      <Plots cols={2}>
        <Plot x={ax} y={ay} title="accuracy against depth">
          {SERIES.map((s) => (
            <Curve key={s.key} name={s.name} x={LAYERS} y={run.accuracy[s.key]} slot={s.slot} />
          ))}
          <Curve name="chance" x={[0, DEPTH]} y={[0.5, 0.5]} dashed muted />
        </Plot>
        <Plot x={ex} y={ey} title="energy of the last layer">
          {SERIES.filter((s) => s.key !== 'jk').map((s) => (
            <Curve key={s.key} name={s.name} x={LAYERS} y={run.energy[s.key]} slot={s.slot} />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}
