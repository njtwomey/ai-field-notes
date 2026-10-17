import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { convolutionAlgorithmCosts, winogradRelativeError } from 'aifn-methods/neural/convolutional'

const KERNELS = Array.from({ length: 15 }, (_, i) => i + 1)

const megabytes = (numbers: number) => `${formatNumber((4 * numbers) / 2 ** 20)} MB`
const count = (v: number) => (v >= 1e9 ? `${formatNumber(v / 1e9)} G` : `${formatNumber(v / 1e6)} M`)

/**
 * Real multiplications per image of one stride-1, same-padded layer against kernel size, by the direct sum (and
 * im2col, which does the same products), FFT convolution over the whole map, and Winograd F(m × m, k × k).
 */
export function AlgorithmCosts() {
  const state = useFigureState({
    size: int(56, { ge: 4, le: 512, suggestions: [7, 14, 28, 56, 112, 224], label: 'feature map side H = W' }),
    channels: int(64, { ge: 1, le: 2048, suggestions: [3, 16, 64, 128, 256, 512], label: 'channels C = C′' }),
    batch: int(32, { ge: 1, le: 1024, suggestions: [1, 8, 32, 128], label: 'batch N' }),
    tile: choice<'2' | '4' | '6'>(
      [
        { value: '2', label: 'm = 2' },
        { value: '4', label: 'm = 4' },
        { value: '6', label: 'm = 6' },
      ],
      '2',
      { label: 'Winograd outputs per tile side' },
    ),
    precomputed: setting(false, 'kernel transforms kept (inference)'),
    kernel: slider(1, 15, 3, { step: 1, onChart: true }),
  })
  const m = Number(state.tile)
  const k = Math.round(state.kernel)

  const curves = useMemo(() => {
    const per = (v: number) => v / state.batch
    const costs = KERNELS.map((kk) =>
      convolutionAlgorithmCosts({
        height: state.size,
        width: state.size,
        inChannels: state.channels,
        outChannels: state.channels,
        kernel: kk,
        batch: state.batch,
        tile: m,
        precomputedKernels: state.precomputed,
      }),
    )
    return {
      costs,
      direct: costs.map((c) => per(c.direct.multiplies)),
      fft: costs.map((c) => per(c.fft.multiplies)),
      winograd: costs.map((c) => per(c.winograd.multiplies)),
    }
  }, [state.size, state.channels, state.batch, m, state.precomputed])

  const error = useMemo(() => winogradRelativeError(m, k, stream(`winograd-error/${m}/${k}`), 8), [m, k])
  const c = curves.costs[k - 1]
  const perImage = (v: number) => v / state.batch

  const xAxis = useAxis({ label: 'kernel size k', range: [0.5, 15.5], integer: true })
  const yAxis = useAxis({
    label: 'real multiplications per image',
    log: true,
    hold: 'union',
    key: `${state.size}-${state.channels}-${state.batch}-${state.precomputed}`,
  })
  return (
    <Figure
      title="Multiplications of four ways to compute one layer"
      state={state}
      caption="One stride-1 layer with same padding on an H × H map from C to C′ = C channels, for kernel sizes 1 to 15. The direct sum and im2col do the same k²CC′H² multiplications; im2col only reorganises them into one matrix product. FFT convolution transforms every input map, kernel and output map once over the zero-padded map and multiplies spectra pointwise; its cost barely depends on k and jumps when H + k − 1 passes a power of two. Winograd F(m × m, k × k) needs (m + k − 1)² products per tile of m × m outputs. Drag the guide to a kernel size. At 3 × 3 the products alone are 2.25 times fewer than direct for m = 2 and 4 times fewer for m = 4, and the transforms take back a little of the saving. The float32 error in the readouts grows by orders of magnitude with the tile, which is why only small tiles and kernels are used. At batch 1 the FFT pays for one kernel transform per channel pair, C C′ = 4,096 of them at the defaults, and loses to the direct sum at 3 × 3; a larger batch shares those transforms."
      readouts={
        <>
          <Readout label={`direct and im2col, k = ${k}`} value={count(perImage(c.direct.multiplies))} />
          <Readout
            label="FFT"
            value={`${count(perImage(c.fft.multiplies))} (${formatNumber(c.direct.multiplies / c.fft.multiplies)}× fewer), size ${c.fft.size[0]} × ${c.fft.size[1]}`}
          />
          <Readout
            label={`Winograd F(${m} × ${m}, ${k} × ${k})`}
            value={`${count(perImage(c.winograd.multiplies))} (${formatNumber(c.direct.multiplies / c.winograd.multiplies)}× fewer), tile ${c.winograd.tileSize} × ${c.winograd.tileSize}`}
          />
          <Readout label="im2col workspace per batch (float32)" value={megabytes(c.im2col.workspace)} />
          <Readout label="FFT workspace per batch" value={megabytes(c.fft.workspace)} />
          <Readout label="Winograd workspace per batch" value={megabytes(c.winograd.workspace)} />
          <Readout
            label="float32 relative error: Winograd / direct"
            value={`${error.winograd.toExponential(1)} / ${error.direct.toExponential(1)}`}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} ariaLabel="Multiplications per image against kernel size">
        <Curve name="direct and im2col" x={KERNELS} y={curves.direct} slot={0} showPoints />
        <Curve name="FFT over the whole map" x={KERNELS} y={curves.fft} slot={1} showPoints />
        <Curve name={`Winograd, m = ${m}`} x={KERNELS} y={curves.winograd} slot={2} showPoints />
        <Points
          name={`k = ${k}`}
          x={[k, k, k]}
          y={[curves.direct[k - 1], curves.fft[k - 1], curves.winograd[k - 1]]}
          emphasis
          live
        />
        <Handle
          kind="x"
          at={k}
          label="kernel size"
          onDrag={(x) => state.set('kernel', Math.min(15, Math.max(1, Math.round(x))))}
        />
      </Plot>
    </Figure>
  )
}
