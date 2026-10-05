import { beliefPropagation } from 'aifn-compute/inference/message-passing'
import { factorGraphGibbs, gibbsMarginals } from 'aifn-compute/inference/stochastic'
import { isingLattice } from 'aifn-methods/inference/lattice-models'
import { stream } from 'aifn-compute/foundation/random'
import { run } from 'aifn-compute/foundation/trace'
import { useMemo } from 'react'
import { Figure } from 'aifn-render/layout'
import { row, slider, useComputed, useFigureState } from 'aifn-render/state'
import { Plot, Plots, Raster, Readout, formatNumber, useAxis } from 'aifn-render/viz'

const SIDE = 5
const AXIS = Array.from({ length: SIDE }, (_, i) => i + 1)
/** A field that is positive on the left half and negative on the right, so the marginals vary across the grid. */
const field = (h: number) =>
  Array.from({ length: SIDE * SIDE }, (_, i) => (i % SIDE < SIDE / 2 - 0.5 ? h : -h) * (1 - 0.15 * (i % SIDE)))

const rows = (xs: ArrayLike<number>) =>
  Array.from({ length: SIDE }, (_, r) => Array.from({ length: SIDE }, (_, c) => xs[r * SIDE + c]))

/** Loopy BP's beliefs on a 5 × 5 Ising grid against the marginals estimated by Gibbs sampling. */
export function IsingSpecimen() {
  const state = useFigureState({
    model: row('1 · model', {
      J: slider(-1, 1.2, 0.4, { label: 'coupling J', step: 0.05 }),
      h: slider(0, 1, 0.3, { label: 'field strength h', step: 0.05 }),
    }),
    inference: row('2 · inference', {
      damping: slider(0, 0.9, 0, { label: 'BP damping', step: 0.05 }),
      sweeps: slider(200, 10000, 2000, { label: 'Gibbs sweeps', step: 100 }),
    }),
  })
  const { J, h } = state.model
  const { damping, sweeps } = state.inference
  const graph = useMemo(() => isingLattice(SIDE, SIDE, J, field(h)), [J, h])
  const bp = useMemo(() => beliefPropagation(graph, { damping, maxSteps: 500 }), [graph, damping])
  // Thousands of Gibbs sweeps: rerun on release.
  const gibbs = useComputed(
    () => gibbsMarginals(run(factorGraphGibbs(graph), undefined, sweeps, { stream: stream('ising') })),
    [graph, sweeps],
    { mode: 'release' },
  )
  const view = useMemo(() => {
    const pb = bp.marginals.map((m) => m.data[1])
    const pg = gibbs.value.map((m) => m.data[1])
    const diff = pb.map((p, i) => p - pg[i])
    return { pb: rows(pb), pg: rows(pg), diff: rows(diff), worst: Math.max(...diff.map(Math.abs)) }
  }, [bp, gibbs.value])
  const col = useAxis({ label: 'column' })
  const row_ = useAxis({ label: 'row', equal: col })
  return (
    <Figure
      title="Loopy BP against Gibbs on a 5 × 5 Ising grid"
      purpose="With loops, BP's fixed point only approximates the marginals: strong coupling makes it overconfident near the field's boundary, where Gibbs sampling, unbiased, disagrees."
      defaultSize="L"
      state={state}
      readouts={{
        comparison: (
          <>
            <Readout label="BP sweeps" value={bp.converged ? bp.sweeps : `no convergence in ${bp.sweeps}`} />
            <Readout label="largest |BP − Gibbs|" value={formatNumber(view.worst)} />
            <Readout label="Bethe log Z" value={formatNumber(bp.logZ)} />
          </>
        ),
      }}
      caption="Spins xᵢ ∈ {−1, +1} with coupling J between neighbours and a field pushing the left half up and the right half down. Left: loopy BP's p(xᵢ = +1) after flooding updates. Middle: the fraction of Gibbs sweeps each spin spent at +1. Right: their difference (diverging, ±0.2). For weak coupling the two agree to within the Gibbs noise; strong coupling (J near 1) makes BP overconfident near the boundary, because evidence goes round the grid's loops and is counted again. Strong negative coupling can stop undamped BP converging; damping restores convergence."
    >
      <Plots cols={3}>
        <Plot x={col} y={row_} title="loopy BP p(x = +1)">
          <Raster x={AXIS} y={AXIS} z={view.pb} range={[0, 1]} valueLabel="p(x = +1)" />
        </Plot>
        <Plot x={col} y={row_} title="Gibbs p(x = +1)">
          <Raster x={AXIS} y={AXIS} z={view.pg} range={[0, 1]} valueLabel="p(x = +1)" stale={gibbs.stale} />
        </Plot>
        <Plot x={col} y={row_} title="BP − Gibbs">
          <Raster
            x={AXIS}
            y={AXIS}
            z={view.diff}
            scale="diverging"
            range={[-0.2, 0.2]}
            valueLabel="difference"
            stale={gibbs.stale}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
