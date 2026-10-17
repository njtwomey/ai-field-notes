import { useMemo, useState } from 'react'
import {
  Annotation,
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
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { adjacencySpectrum, dirichletEnergy } from 'aifn-compute/nn/graph'
import { propagationStack } from 'aifn-methods/neural/graph'
import { LABELLED, plantedPartition, readoutAccuracy } from '../_shared/communities'

const DEPTH = 64
const LAYERS = Array.from({ length: DEPTH + 1 }, (_, k) => k)
type Variant = 'plain' | 'pairnorm'

/** Weight-free GCN propagation on a planted partition: features, Dirichlet energy and readout accuracy by layer. */
export function SmoothingDepth() {
  const state = useFigureState({
    shown: choice<Variant>(
      [
        { value: 'plain', label: 'plain: Â H' },
        { value: 'pairnorm', label: 'PairNorm after each Â H' },
      ],
      'plain',
      { label: 'features drawn' },
    ),
    pOut: slider(0, 0.15, 0.04, { step: 0.01, label: 'edge probability across communities' }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })
  const [layer, setLayer] = useState(2)

  const run = useMemo(() => {
    const data = plantedPartition(state.seed, 0.25, state.pOut)
    const { values, contraction } = adjacencySpectrum(data.graph)
    const stacks = {
      plain: propagationStack(data.graph, data.x, { depth: DEPTH, variant: 'plain' }),
      pairnorm: propagationStack(data.graph, data.x, { depth: DEPTH, variant: 'pairnorm' }),
    }
    const energy = {
      plain: stacks.plain.map((h) => dirichletEnergy(data.graph, h)),
      pairnorm: stacks.pairnorm.map((h) => dirichletEnergy(data.graph, h)),
    }
    const accuracy = {
      plain: stacks.plain.map((h) => readoutAccuracy(h, data.y)),
      pairnorm: stacks.pairnorm.map((h) => readoutAccuracy(h, data.y)),
    }
    const bound = LAYERS.map((k) => energy.plain[0] * contraction ** (2 * k))
    return { data, mu2: toFlat(values)[1], contraction, stacks, energy, accuracy, bound }
  }, [state.seed, state.pOut])

  const k = Math.min(layer, DEPTH)
  const h = run.stacks[state.shown][k]
  const fx = useAxis({ label: 'feature 1' })
  const fy = useAxis({ label: 'feature 2' })
  const lx = useAxis({ label: 'layer', range: [0, DEPTH], integer: true })
  const ex = useAxis({ label: 'layer', range: [0, DEPTH], integer: true })
  const ey = useAxis({ label: 'Dirichlet energy', log: true })
  const ay = useAxis({ label: 'accuracy', range: [0, 1] })
  const labelled = new Set(LABELLED)
  const groupNames = ['community 1', 'community 2']

  return (
    <Figure
      title="Oversmoothing on a planted partition"
      state={state}
      defaultSize="L"
      controls={<Player value={k} onChange={setLayer} count={DEPTH + 1} label="layer" />}
      readouts={
        <>
          <Readout label="μ₂ of Â" value={formatNumber(run.mu2)} />
          <Readout label="λ = max(|μ₂|, |μ_N|)" value={formatNumber(run.contraction)} />
          <Readout label="edges inside communities" value={formatNumber(run.data.homophily)} />
          <Readout label={`energy at layer ${k} (plain)`} value={formatNumber(run.energy.plain[k])} />
          <Readout label={`accuracy at layer ${k} (plain)`} value={formatNumber(run.accuracy.plain[k])} />
          <Readout label={`accuracy at layer ${k} (PairNorm)`} value={formatNumber(run.accuracy.pairnorm[k])} />
        </>
      }
      caption="Forty nodes in two communities of 20 (edge probability 0.25 inside a community), each with a 2-D feature whose mean depends on its community. Every layer multiplies the features by the GCN matrix Â, with no weights. Left: the features at the layer chosen with the player (axes refit at every layer; ringed nodes are the 8 labelled ones). Middle: the Dirichlet energy on a log scale, with the bound E₀λ^(2k) dashed. Right: accuracy on the 32 unlabelled nodes of a ridge readout fitted to the labelled ones (penalty 0.05, the analogue of weight decay). A few layers average away the noise inside each community and accuracy rises; the energy then falls geometrically, the features collapse onto one line through the origin and plain accuracy drops towards chance. PairNorm keeps the energy fixed, but its features also end on one line, here the community direction. Raise the cross-community edge probability to bring μ₂ down and the collapse forward."
    >
      <Plots cols={3}>
        <Plot x={fx} y={fy} title={`features at layer ${k}`}>
          <Points
            name="labelled"
            x={h.filter((_, i) => labelled.has(i)).map((r) => r[0])}
            y={h.filter((_, i) => labelled.has(i)).map((r) => r[1])}
            emphasis
            size={16}
          />
          <Points
            name="nodes"
            x={h.map((r) => r[0])}
            y={h.map((r) => r[1])}
            group={run.data.y}
            groupNames={groupNames}
            size={9}
          />
        </Plot>
        <Plot x={ex} y={ey} title="energy by layer">
          <Curve name="plain" x={LAYERS} y={run.energy.plain} slot={0} />
          <Curve name="PairNorm" x={LAYERS} y={run.energy.pairnorm} slot={1} />
          <Curve name="bound E₀λ^(2k)" x={LAYERS} y={run.bound} dashed muted />
          <Annotation x={k} dashed />
        </Plot>
        <Plot x={lx} y={ay} title="readout accuracy">
          <Curve name="plain" x={LAYERS} y={run.accuracy.plain} slot={0} />
          <Curve name="PairNorm" x={LAYERS} y={run.accuracy.pairnorm} slot={1} />
          <Curve name="chance" x={[0, DEPTH]} y={[0.5, 0.5]} dashed muted />
          <Annotation x={k} dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
