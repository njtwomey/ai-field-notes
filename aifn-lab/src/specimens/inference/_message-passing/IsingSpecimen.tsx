import { beliefPropagation } from 'aifn/inference/message-passing'
import { factorGraphGibbs, gibbsMarginals } from 'aifn/inference/stochastic'
import { isingLattice } from 'aifn-applied/inference/lattice-models'
import { stream } from 'aifn/foundation/random'
import { run } from 'aifn/foundation/trace'
import { useMemo } from 'react'
import { Slider, useParam } from '@lab/controls'
import { Columns, Figure } from '@lab/layout'
import { ChartSize, Heatmap, Readout, formatNumber } from '@lab/viz'

const SIDE = 5
const AXIS = Array.from({ length: SIDE }, (_, i) => i + 1)
/** A field that is positive on the left half and negative on the right, so the marginals vary across the grid. */
const field = (h: number) =>
  Array.from({ length: SIDE * SIDE }, (_, i) => (i % SIDE < SIDE / 2 - 0.5 ? h : -h) * (1 - 0.15 * (i % SIDE)))

const rows = (xs: ArrayLike<number>) =>
  Array.from({ length: SIDE }, (_, r) => Array.from({ length: SIDE }, (_, c) => xs[r * SIDE + c]))

/** Loopy BP's beliefs on a 5 × 5 Ising grid against the marginals estimated by Gibbs sampling. */
export function IsingSpecimen() {
  const J = useParam(0.4, { min: -1, max: 1.2, step: 0.05 })
  const h = useParam(0.3, { min: 0, max: 1, step: 0.05 })
  const damping = useParam(0, { min: 0, max: 0.9, step: 0.05 })
  const sweeps = useParam(2000, { min: 200, max: 10000, step: 100 })
  const graph = useMemo(() => isingLattice(SIDE, SIDE, J.value, field(h.value)), [J.value, h.value])
  const bp = useMemo(() => beliefPropagation(graph, { damping: damping.value, maxSteps: 500 }), [graph, damping.value])
  const gibbs = useMemo(
    () => gibbsMarginals(run(factorGraphGibbs(graph), undefined, sweeps.value, { stream: stream('ising') })),
    [graph, sweeps.value],
  )
  const pb = bp.marginals.map((m) => m.data[1])
  const pg = gibbs.map((m) => m.data[1])
  const diff = pb.map((p, i) => p - pg[i])
  const worst = Math.max(...diff.map(Math.abs))
  const heat = (z: number[][], scale: 'sequential' | 'diverging', range: [number, number], label: string) => (
    <ChartSize scale={0.9}>
      <Heatmap x={AXIS} y={AXIS} z={z} scale={scale} range={range} valueLabel={label} xLabel="column" yLabel="row" />
    </ChartSize>
  )
  return (
    <Figure
      title="Loopy BP against Gibbs on a 5 × 5 Ising grid"
      description="With loops, BP's fixed point only approximates the marginals; Gibbs sampling estimates them without that bias."
      defaultSize="L"
      controls={
        <>
          <Slider label="coupling J" param={J} />
          <Slider label="field strength h" param={h} />
          <Slider label="damping" param={damping} />
          <Slider label="Gibbs sweeps" param={sweeps} />
        </>
      }
      readouts={
        <>
          <Readout label="BP sweeps" value={bp.converged ? bp.sweeps : `no convergence in ${bp.sweeps}`} />
          <Readout label="largest |BP − Gibbs|" value={formatNumber(worst)} />
          <Readout label="Bethe log Z" value={formatNumber(bp.logZ)} />
        </>
      }
      caption="Spins xᵢ ∈ {−1, +1} with coupling J between neighbours and a field pushing the left half up and the right half down. Left: loopy BP's p(xᵢ = +1) after flooding updates. Middle: the fraction of Gibbs sweeps each spin spent at +1. Right: their difference. For weak coupling the two agree to within the Gibbs noise; strong coupling (J near 1) makes BP overconfident near the boundary, because evidence goes round the grid's loops and is counted again. Strong negative coupling can stop undamped BP converging; damping restores convergence."
    >
      <Columns
        panels={[
          { title: 'loopy BP p(x = +1)', body: heat(rows(pb), 'sequential', [0, 1], 'p(x = +1)') },
          { title: 'Gibbs p(x = +1)', body: heat(rows(pg), 'sequential', [0, 1], 'p(x = +1)') },
          { title: 'BP − Gibbs', body: heat(rows(diff), 'diverging', [-0.2, 0.2], 'difference') },
        ]}
      />
    </Figure>
  )
}
