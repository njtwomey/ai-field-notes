import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'
import { type Tensor, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { causalConvolution, implicitFilter } from 'aifn-compute/nn/sequence'
import { convolutionMultiplies } from 'aifn-methods/neural/sequence'

const L = 512
const STEPS = Array.from({ length: L }, (_, t) => t)

/** A seeded test input: a slow sinusoid plus noise, with a burst near the start. */
const U = (() => {
  const r = stream(21)
  return STEPS.map((t) => Math.sin((2 * Math.PI * t) / 128) + 0.4 * normal(r) + (t >= 20 && t < 28 ? 3 : 0))
})()
const U_TENSOR = tensor(U)

const LENGTHS = Array.from({ length: 13 }, (_, i) => 2 ** (i + 4))
const COSTS = LENGTHS.map(convolutionMultiplies)
const CROSSOVER = LENGTHS.find((_, i) => COSTS[i].fft < COSTS[i].direct) ?? NaN

/**
 * An implicit long filter: a small network with sine activations maps the positional encoding of each lag t to a
 * filter value, and an exponential window shapes it. The filter, as long as the input, is applied by FFT convolution
 * and by the direct sum, which agree; the multiply counts of the two methods are compared against length.
 */
export function ImplicitFilter() {
  const state = useFigureState({
    decay: float(6, { min: 0, max: 30, step: 0.5, label: 'window decay α' }),
    bias: float(0.02, { min: 0, max: 0.3, step: 0.01, label: 'window bias' }),
    features: int(8, { min: 1, max: 32, label: 'encoding features K', suggestions: [2, 8, 32] }),
    frequency: float(10, { min: 1, max: 30, step: 0.5, label: 'sine frequency ω' }),
    seed: int(1, { min: 0, max: 99, label: 'network seed', suggestions: [1, 2, 3] }),
  })
  const { decay, bias, features, frequency, seed } = state

  const result = useMemo(() => {
    const f = implicitFilter(L, { features, width: 32, depth: 2, frequency, decay, bias }, stream(seed))
    const filter = toFlat(f.filter)
    const scale = Math.max(...toFlat(f.raw).map(Math.abs))
    const fft = toFlat(causalConvolution(f.filter, U_TENSOR, { method: 'fft' }) as Tensor)
    const direct = toFlat(causalConvolution(f.filter, U_TENSOR) as Tensor)
    const error = Math.max(...fft.map((v, t) => Math.abs(v - direct[t])))
    return {
      filter,
      envelope: toFlat(f.window).map((w) => w * scale),
      fft,
      direct,
      error,
    }
  }, [decay, bias, features, frequency, seed])

  const at = convolutionMultiplies(L)

  const tAxis = useAxis({ label: 'lag t', range: [0, L - 1] })
  const hAxis = useAxis({ label: 'h_t', hold: 'union', key: seed })
  const kAxis = useAxis({ label: 'position t', range: [0, L - 1] })
  const yAxis = useAxis({ label: 'y_t', hold: 'union', key: seed })
  const nAxis = useAxis({ label: 'sequence length L', log: true, range: [16, 65536] })
  const cAxis = useAxis({ label: 'multiplies', log: true })

  return (
    <Figure
      title="An implicit filter as long as the sequence"
      state={state}
      caption="A two-layer network with sine activations maps the positional encoding of each lag t (K complex exponentials) to a filter value h_t, and the window exp(−αt/L) + bias shapes it (dashed envelope). Raise K or the sine frequency for filters with more high-frequency content; raise the decay to concentrate the filter on recent inputs. The 512-tap filter is applied to the input by FFT (line) and by the direct sum (dots): the outputs agree to rounding error. The last panel counts multiplies for one causal convolution: L(L + 1)/2 for the direct sum, about 3M log₂M for FFTs padded to M ≥ 2L."
      readouts={
        <>
          <Readout label="max |FFT − direct|" value={result.error.toExponential(1)} />
          <Readout
            label="multiplies at L = 512, direct / FFT"
            value={`${formatNumber(at.direct)} / ${formatNumber(at.fft)}`}
          />
          <Readout label="FFT cheaper from L =" value={`${CROSSOVER}`} />
        </>
      }
    >
      <Plots rows={3} heights={[1, 1, 1]}>
        <Plot x={tAxis} y={hAxis}>
          <Curve name="window × max|FFN|" x={STEPS} y={result.envelope} muted dashed />
          <Curve name="filter h_t" x={STEPS} y={result.filter} slot={0} />
        </Plot>
        <Plot x={kAxis} y={yAxis}>
          <Curve name="input v" x={STEPS} y={U} muted />
          <Curve name="FFT convolution" x={STEPS} y={result.fft} slot={1} />
          <Points name="direct sum" x={STEPS} y={result.direct} slot={2} size={2} />
        </Plot>
        <Plot x={nAxis} y={cAxis}>
          <Curve name="direct, L(L + 1)/2" x={LENGTHS} y={COSTS.map((c) => c.direct)} slot={0} showPoints />
          <Curve name="FFT, padded" x={LENGTHS} y={COSTS.map((c) => c.fft)} slot={1} showPoints />
        </Plot>
      </Plots>
    </Figure>
  )
}
