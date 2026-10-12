import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { kernelMatrix, medianHeuristic, mmd2, twoDimensionalPair, witness } from '../_shared/twoSample'

const BOX = 4
const G = toFlat(linspace(-BOX, BOX, 41))
const GRID = G.flatMap((y) => G.map((x) => [x, y]))
const clamp = (v: number) => Math.min(BOX - 0.5, Math.max(-BOX + 0.5, v))

/**
 * The MMD witness function for a sample from P = N(0, I) and a sample from Q, a unit-variance Gaussian with a
 * draggable mean and a correlation ρ. The witness is positive where P's kernel mean embedding exceeds Q's and
 * negative where Q's does; MMD is the size of that difference.
 */
export function WitnessField() {
  const state = useFigureState({
    mx: float(1, { min: -3, max: 3, step: 0.05, label: 'mean of Q, x₁', onChart: true }),
    my: float(0, { min: -3, max: 3, step: 0.05, label: 'mean of Q, x₂', onChart: true }),
    rho: float(0, { min: -0.95, max: 0.95, step: 0.05, label: 'correlation of Q, ρ' }),
    median: setting(true, { label: 'kernel width by the median heuristic' }),
    logSigma: float(0, { min: -1, max: 1, step: 0.02, label: 'log₁₀ kernel width σ' }),
    n: int(100, { min: 10, max: 300, label: 'points in each sample', suggestions: [20, 100, 300] }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })

  const data = useMemo(
    () => twoDimensionalPair(state.seed, state.n, state.n, [state.mx, state.my], state.rho),
    [state.seed, state.n, state.mx, state.my, state.rho],
  )
  const pooled = useMemo(() => [...data.x, ...data.y], [data])
  const heuristic = useMemo(() => medianHeuristic(pooled), [pooled])
  const sigma = state.median ? heuristic : 10 ** state.logSigma

  const stats = useMemo(() => {
    const K = kernelMatrix(pooled, sigma)
    const N = pooled.length
    const ix = Array.from({ length: data.x.length }, (_, i) => i)
    const iy = Array.from({ length: data.y.length }, (_, i) => data.x.length + i)
    return { unbiased: mmd2(K, N, ix, iy, true), biased: mmd2(K, N, ix, iy, false) }
  }, [pooled, sigma, data])

  const field = useMemo(() => {
    const w = witness(data.x, data.y, GRID, sigma)
    return G.map((_, r) => w.slice(r * G.length, (r + 1) * G.length))
  }, [data, sigma])
  const top = Math.max(...field.flat().map(Math.abs))

  const xa = useAxis({ label: 'x₁', range: [-BOX, BOX] })
  const ya = useAxis({ label: 'x₂', range: [-BOX, BOX], equal: xa })
  return (
    <Figure
      title="The witness function"
      state={state}
      caption="A sample from P = N(0, I) and a sample from Q, over the witness function: the mean kernel similarity to P's points minus the mean similarity to Q's. Red marks where P is denser at the kernel's scale, blue where Q is. Drag the marker to move Q's mean. With the means equal, set ρ away from 0: both marginals stay N(0, 1), yet the witness picks out the diagonal along which Q stretches. A very narrow kernel makes the witness a set of spikes on single points; a very wide one smears it into a slope."
      readouts={
        <>
          <Readout label="σ" value={formatNumber(sigma)} />
          <Readout label="median heuristic" value={formatNumber(heuristic)} />
          <Readout label="MMD², unbiased" value={formatNumber(stats.unbiased)} />
          <Readout label="MMD², biased" value={formatNumber(stats.biased)} />
        </>
      }
    >
      <Plot x={xa} y={ya} height={380}>
        <Raster x={G} y={G} z={field} scale="diverging" range={[-top, top]} valueLabel="witness" fillOpacity={0.8} />
        <Points name="sample from P" x={data.x.map((p) => p[0])} y={data.x.map((p) => p[1])} slot={0} thin />
        <Points name="sample from Q" x={data.y.map((p) => p[0])} y={data.y.map((p) => p[1])} slot={1} thin />
        <Handle
          kind="point"
          at={[state.mx, state.my]}
          label="mean of Q"
          slot={1}
          onDrag={([x, y]) => {
            state.set('mx', clamp(x))
            state.set('my', clamp(y))
          }}
        />
      </Plot>
    </Figure>
  )
}
