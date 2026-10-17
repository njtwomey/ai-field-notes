import { useContext, useMemo, type ReactNode } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  FrameContext,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  Rug,
  Segments,
  useAxis,
  useFigureState,
  useScaleColor,
} from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  EDGE_SEGMENTS,
  EIGENVALUES,
  graphFourier,
  inverseGraphFourier,
  N,
  POSITIONS,
  XS,
  YS,
} from '../_shared/sensorGraph'

const LAMBDA_MAX = EIGENVALUES[N - 1]
// A smooth signal over the layout: slow waves in x and y.
const CLEAN = POSITIONS.map(([x, y]) => Math.cos((Math.PI * x) / 5) + 0.5 * Math.cos((Math.PI * y) / 4))
const CLEAN_HAT = graphFourier(CLEAN)
const GRID = Array.from({ length: 161 }, (_, i) => (i / 160) * (LAMBDA_MAX + 0.4))
const rms = (a: readonly number[], b: readonly number[]) =>
  Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length)

type Kind = 'ideal' | 'heat'
const gain = (kind: Kind, cutoff: number, lambda: number) =>
  kind === 'ideal' ? (lambda <= cutoff ? 1 : 0) : Math.exp(-lambda / cutoff)

function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/**
 * Denoising a graph signal by a spectral filter: transform with the Laplacian eigenvectors, multiply each coefficient
 * by a gain g(λ), transform back.
 */
export function LowPassFilter() {
  const state = useFigureState({
    noise: float(0.5, { min: 0, max: 1.5, step: 0.05, label: 'noise σ' }),
    seed: int(1, { ge: 1, le: 999, label: 'noise draw' }),
    kind: choice<Kind>(
      [
        { value: 'ideal', label: 'ideal low-pass 1[λ ≤ λc]' },
        { value: 'heat', label: 'heat kernel exp(−λ/λc)' },
      ],
      'ideal',
      { label: 'filter' },
    ),
    cutoff: float(1, { min: 0.05, max: 7.5, step: 0.05, label: 'cut-off λc' }),
  })
  const { noise, seed, kind, cutoff } = state
  const noisy = useMemo(() => {
    const e = toFlat(normals(stream(`gnn-foundations/low-pass/${seed}`), [N]))
    return CLEAN.map((v, i) => v + noise * e[i])
  }, [noise, seed])
  const noisyHat = useMemo(() => graphFourier(noisy), [noisy])
  const gains = EIGENVALUES.map((l) => gain(kind, cutoff, l))
  const filteredHat = noisyHat.map((c, k) => gains[k] * c)
  const filtered = inverseGraphFourier(filteredHat)
  const kept = gains.reduce((s, g) => s + g, 0)

  const colour = useScaleColor('diverging')
  const bound = 2
  const paint = (x: readonly number[]) => x.map((v) => colour(Math.max(0, Math.min(1, (v / bound + 1) / 2))))

  const gx = useAxis({ label: '', range: [-0.6, 5.6], nice: false })
  const gy = useAxis({ label: '', range: [-0.6, 4.6], nice: false, equal: gx })
  const lx = useAxis({ label: 'eigenvalue λ', range: [-0.2, LAMBDA_MAX + 0.4], nice: false })
  const cy = useAxis({ label: '|x̂ₖ|', range: [0, undefined], hold: 'union', key: `${noise}/${seed}` })
  const hy = useAxis({ label: 'gain g(λ)', range: [-0.05, 1.1], nice: false })

  return (
    <Figure
      title="Low-pass filtering a graph signal"
      state={state}
      hoverReadout={false}
      caption={`A smooth signal on the ${N}-node graph plus Gaussian noise of standard deviation σ (top left). The filter computes the graph Fourier coefficients x̂ = Uᵀx, multiplies coefficient k by the gain g(λₖ), and transforms back (top right). Bottom left: the size of each coefficient at its eigenvalue, for the noisy signal (grey), after filtering (coloured) and for the clean signal (dots). The clean signal's energy sits almost entirely in the two lowest non-constant frequencies; the noise spreads evenly over all ${N}. Bottom right: the gain; drag the vertical line to move the cut-off λc. A cut-off just above the smooth part removes most of the noise; a cut-off of 0 keeps only the mean, and one beyond the largest eigenvalue keeps everything.`}
      readouts={
        <>
          <Readout label="RMS error, noisy signal" value={formatNumber(Number(rms(noisy, CLEAN).toPrecision(3)))} />
          <Readout label="RMS error, filtered" value={formatNumber(Number(rms(filtered, CLEAN).toPrecision(3)))} />
          <Readout
            label="frequencies kept (sum of gains)"
            value={`${formatNumber(Number(kept.toPrecision(3)))} of ${N}`}
          />
        </>
      }
    >
      <FixedHeight height={250}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Plot x={gx} y={gy} title="noisy signal x" bare>
            <Segments segments={EDGE_SEGMENTS} muted width={1.5} />
            <Points name="node" x={XS} y={YS} colors={paint(noisy)} size={13} />
          </Plot>
          <Plot x={gx} y={gy} title="filtered signal U g(Λ) Uᵀ x" bare>
            <Segments segments={EDGE_SEGMENTS} muted width={1.5} />
            <Points name="node" x={XS} y={YS} colors={paint(filtered)} size={13} />
          </Plot>
          <Plot x={lx} y={cy} title="graph Fourier coefficients">
            <Bars name="noisy" x={EIGENVALUES} y={noisyHat.map(Math.abs)} width={0.05} muted />
            <Bars name="filtered" x={EIGENVALUES} y={filteredHat.map(Math.abs)} width={0.05} slot={0} />
            <Points name="clean" x={EIGENVALUES} y={CLEAN_HAT.map(Math.abs)} emphasis size={5} />
          </Plot>
          <Plot x={lx} y={hy} title="filter gain">
            <Curve name="g(λ)" x={GRID} y={GRID.map((l) => gain(kind, cutoff, l))} slot={0} />
            <Rug values={EIGENVALUES} />
            <Handle
              kind="x"
              at={cutoff}
              label="λc"
              onDrag={(x) => state.set('cutoff', Math.max(0.05, Math.min(7.5, Math.round(x * 20) / 20)))}
            />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
