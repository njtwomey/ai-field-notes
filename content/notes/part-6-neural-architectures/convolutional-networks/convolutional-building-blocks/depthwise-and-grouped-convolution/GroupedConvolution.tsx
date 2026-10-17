import { useMemo } from 'react'
import {
  Bars,
  choice,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Raster,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { conv2d, convolutionCost } from 'aifn-compute/nn/functional'
import { normals, stream } from 'aifn-compute/foundation/random'
import { sub, tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

/** Channels drawn in the connectivity panel. */
const SHOWN = 16
const KERNEL = 3

const BARS = ['standard k × k', 'grouped k × k, g groups', 'depthwise k × k', 'depthwise + 1 × 1 (separable)']

/**
 * Grouped convolution as a block-diagonal connection pattern between input and output channels, checked against a full
 * convolution whose kernel is zero outside the blocks, and its cost against a standard and a depthwise separable layer.
 */
export function GroupedConvolution() {
  const state = useFigureState({
    groups: choice<1 | 2 | 4 | 8 | 16>([1, 2, 4, 8, 16], 4, { label: 'groups g (of 16 channels drawn)' }),
    channels: int(256, {
      ge: 16,
      le: 4096,
      suggestions: [64, 128, 256, 512, 1024],
      label: 'channels C = C′ for the cost',
    }),
    kernel: int(3, { ge: 1, le: 11, suggestions: [3, 5, 7], label: 'kernel k for the cost' }),
    size: int(56, { ge: 1, le: 512, suggestions: [7, 14, 28, 56, 112], label: 'feature map side' }),
    output: slider(0, SHOWN - 1, 5, { step: 1, onChart: true }),
  })
  const g = state.groups
  const per = SHOWN / g
  const o = state.output
  const first = Math.floor(o / per) * per

  // Connectivity: output channel o reads input channel c when both are in the same group.
  const z = useMemo(
    () =>
      Array.from({ length: SHOWN }, (_, r) =>
        Array.from({ length: SHOWN }, (_, c) => (Math.floor(r / per) === Math.floor(c / per) ? 1 : 0)),
      ),
    [per],
  )
  const edges = useMemo(() => {
    const s: { from: [number, number]; to: [number, number] }[] = []
    for (let b = 0; b <= g; b++) {
      const at = b * per - 0.5
      s.push({ from: [at, -0.5], to: [at, SHOWN - 0.5] }, { from: [-0.5, at], to: [SHOWN - 0.5, at] })
    }
    return s
  }, [g, per])

  // Equivalence check on a seeded random 16-channel 6 × 6 input: grouped kernels against the block-diagonal full kernel.
  const check = useMemo(() => {
    const s = stream('grouped-convolution/check')
    const x = normals(s, [1, SHOWN, 6, 6])
    const w = toFlat(normals(s, [SHOWN, per, KERNEL, KERNEL]))
    const dense = new Array(SHOWN * SHOWN * KERNEL * KERNEL).fill(0)
    for (let r = 0; r < SHOWN; r++) {
      const block = Math.floor(r / per)
      for (let ci = 0; ci < per; ci++) {
        const c = block * per + ci
        for (let t = 0; t < KERNEL * KERNEL; t++) {
          dense[(r * SHOWN + c) * KERNEL * KERNEL + t] = w[(r * per + ci) * KERNEL * KERNEL + t]
        }
      }
    }
    const grouped = conv2d(x, tensor(w, [SHOWN, per, KERNEL, KERNEL]), { groups: g })
    const full = conv2d(x, tensor(dense, [SHOWN, SHOWN, KERNEL, KERNEL]))
    return Math.max(...toFlat(sub(grouped, full) as Tensor).map(Math.abs))
  }, [g, per])

  const C = state.channels
  const k = state.kernel
  const side = state.size
  const pad = Math.floor((k - 1) / 2)
  const divisible = C % g === 0
  const cost = useMemo(() => {
    const common = { inChannels: C, outChannels: C, kernel: k, padding: pad, bias: false, input: [side, side] }
    const standard = convolutionCost(common)
    const grouped = divisible ? convolutionCost({ ...common, groups: g }) : undefined
    const depthwise = convolutionCost({ ...common, kind: 'depthwise' })
    const pointwise = convolutionCost({ ...common, kernel: 1, padding: 0 })
    return { standard, grouped, depthwise, pointwise }
  }, [C, k, pad, g, divisible, side])
  const macs = [
    cost.standard.multiplies,
    cost.grouped?.multiplies ?? 0,
    cost.depthwise.multiplies,
    cost.depthwise.multiplies + cost.pointwise.multiplies,
  ].map((v) => v / 1e6)
  const separable = cost.depthwise.weights + cost.pointwise.weights

  const xAxis = useAxis({ label: 'input channel c', range: [-0.5, SHOWN - 0.5], integer: true, zoom: false })
  const yAxis = useAxis({
    label: 'output channel',
    range: [-0.5, SHOWN - 0.5],
    integer: true,
    inverse: true,
    zoom: false,
  })
  const bAxis = useAxis({ label: 'multiply-accumulates per image (millions)', hold: 'union', key: `${C}-${k}-${side}` })
  const cAxis = useAxis({ categories: BARS, inverse: true })
  return (
    <Figure
      title="Grouped and depthwise convolution"
      state={state}
      caption="Left: which input channels each output channel reads, for 16 channels split into g groups. A filled cell means the output channel has a k × k kernel on that input channel. Drag the horizontal guide to pick an output channel. g = 1 is a standard convolution, g = 16 is depthwise. The readout checks the grouped layer against a full convolution whose kernel is zero outside the diagonal blocks, on a seeded random input: the two agree to rounding. Right: multiply-accumulates of four layers at the typed channel count, kernel and map size."
      readouts={
        <>
          <Readout
            label={`output channel ${o} reads`}
            value={`input channels ${first} to ${first + per - 1} (${per} of ${SHOWN})`}
          />
          <Readout label="max |grouped − block-diagonal full|" value={check === 0 ? '0' : check.toExponential(1)} />
          <Readout label="standard weights" value={cost.standard.weights.toLocaleString('en-GB')} />
          <Readout
            label={`grouped weights (g = ${g})`}
            value={cost.grouped ? cost.grouped.weights.toLocaleString('en-GB') : `C = ${C} is not divisible by ${g}`}
          />
          <Readout
            label="separable weights"
            value={`${separable.toLocaleString('en-GB')} (${formatNumber(cost.standard.weights / separable)} × fewer)`}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} ariaLabel="Connections between input and output channels">
          <Raster
            x={z.map((_, i) => i)}
            y={z.map((_, i) => i)}
            z={z}
            range={[0, 1]}
            valueLabel="connected"
            colorBar={false}
          />
          <Segments segments={edges} />
          <Segments segments={[{ from: [first - 0.5, o], to: [first + per - 0.5, o] }]} emphasis width={3} />
          <Handle kind="y" at={o} onDrag={(y) => state.set('output', y)} label="output channel" />
        </Plot>
        <Plot x={bAxis} y={cAxis} ariaLabel="Multiply-accumulates of four layers">
          <Bars x={[0, 1, 2, 3]} y={macs} orient="y" slot={0} name="multiply-accumulates" />
        </Plot>
      </div>
    </Figure>
  )
}
