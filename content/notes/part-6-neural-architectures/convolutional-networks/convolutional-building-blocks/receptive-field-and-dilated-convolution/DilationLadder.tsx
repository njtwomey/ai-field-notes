import { useMemo } from 'react'
import {
  Annotation,
  Bars,
  choice,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { conv1d, effectiveReceptiveField, receptiveField } from 'aifn-compute/nn/functional'
import { normals, stream } from 'aifn-compute/foundation/random'
import { full, toFlat, zeros, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { FixedHeight } from '../_shared/FixedHeight'

/** Input positions 0 … N − 1; the largest field shown, 1 + 4 · 63 = 253, fits. */
const N = 257

type Pattern = 'plain' | 'strided' | 'dilated' | 'gridding' | 'hybrid'
type Layer = { kernel: number; stride: number; dilation: number; padding: number }

/** The stack: L layers of one kernel size, with the strides and dilations of the chosen pattern and same padding. */
function stack(pattern: Pattern, L: number, k: number): Layer[] {
  return Array.from({ length: L }, (_, l) => {
    const stride = pattern === 'strided' ? 2 : 1
    const dilation =
      pattern === 'dilated' ? 2 ** l : pattern === 'gridding' ? 2 : pattern === 'hybrid' ? [1, 2, 3][l % 3] : 1
    return { kernel: k, stride, dilation, padding: (dilation * (k - 1)) / 2 }
  })
}

/**
 * A stack of one-dimensional convolutions drawn as a tree of connections from one output unit down to the input,
 * with the unit's theoretical receptive field and its effective field, the gradient of the unit with respect to each
 * input position.
 */
export function DilationLadder() {
  const state = useFigureState({
    pattern: choice<Pattern>(
      [
        { value: 'plain', label: 'plain: stride 1, dilation 1' },
        { value: 'strided', label: 'stride 2 in every layer' },
        { value: 'dilated', label: 'dilation 1, 2, 4, 8, …' },
        { value: 'gridding', label: 'dilation 2 in every layer' },
        { value: 'hybrid', label: 'dilation 1, 2, 3, 1, 2, 3' },
      ],
      'dilated',
      { label: 'stack' },
    ),
    layers: int(4, { ge: 1, le: 6, label: 'layers L' }),
    kernel: choice<'3' | '5'>(
      [
        { value: '3', label: '3 taps' },
        { value: '5', label: '5 taps' },
      ],
      '3',
      { label: 'kernel k' },
    ),
    weights: choice<'uniform' | 'random'>(
      [
        { value: 'uniform', label: 'uniform, 1/k each' },
        { value: 'random', label: 'random (seeded)' },
      ],
      'uniform',
      { label: 'weights' },
    ),
    unit: slider(0, N - 1, 128, { step: 1, onChart: true }),
  })
  const k = Number(state.kernel)
  const L = state.layers

  const net = useMemo(() => {
    const layers = stack(state.pattern, L, k)
    const fields = receptiveField(layers)
    const kernels = layers.map((_, l) =>
      state.weights === 'uniform'
        ? full([1, 1, k], 1 / k)
        : normals(stream(`receptive-field/weights/${l}`), [1, 1, k], 0, 1 / Math.sqrt(k)),
    )
    const f = (x: Value) =>
      layers.reduce<Value>(
        (h, layer, l) =>
          conv1d(h, kernels[l], { stride: layer.stride, dilation: layer.dilation, padding: layer.padding }),
        x,
      )
    // Output length of the stack, from the same arithmetic as the layers.
    let m = N
    for (const layer of layers) {
      m = Math.floor((m + 2 * layer.padding - layer.dilation * (layer.kernel - 1) - 1) / layer.stride) + 1
    }
    return { layers, fields, f, outputs: m }
  }, [state.pattern, L, k, state.weights])

  const top = net.fields[L]
  // The output unit nearest the handle; with same padding, unit i of the top layer is centred on input i · jump.
  const index = Math.min(Math.max(Math.round(state.unit / top.jump), 0), net.outputs - 1)
  const centre = index * top.jump

  const tree = useMemo(() => {
    const segments: { from: [number, number]; to: [number, number] }[] = []
    const nodes: { x: number[]; y: number[] } = { x: [centre], y: [L] }
    let level = new Set([centre])
    for (let l = L; l >= 1; l--) {
      const { kernel, dilation } = net.layers[l - 1]
      const step = dilation * net.fields[l - 1].jump
      const next = new Set<number>()
      for (const q of level) {
        for (let t = 0; t < kernel; t++) {
          const c = q + (t - (kernel - 1) / 2) * step
          if (c < 0 || c > N - 1) continue
          segments.push({ from: [q, l], to: [c, l - 1] })
          next.add(c)
        }
      }
      for (const c of next) {
        nodes.x.push(c)
        nodes.y.push(l - 1)
      }
      level = next
    }
    return { segments, nodes, touched: level.size }
  }, [net, centre, L])

  const gradient = useMemo(
    () => toFlat(effectiveReceptiveField(net.f, zeros([1, 1, N]), [0, 0, index]) as Tensor),
    [net, index],
  )
  const positions = useMemo(() => Array.from({ length: N }, (_, i) => i), [])

  // Spread of the effective field: the standard deviation of |gradient| as a distribution over positions.
  const total = gradient.reduce((a, g) => a + Math.abs(g), 0)
  const mu = gradient.reduce((a, g, i) => a + Math.abs(g) * i, 0) / total
  const sd = Math.sqrt(gradient.reduce((a, g, i) => a + Math.abs(g) * (i - mu) ** 2, 0) / total)
  const lo = centre - (top.size - 1) / 2
  const hi = centre + (top.size - 1) / 2

  const xAxis = useAxis({ label: 'input position', range: [-0.5, N - 0.5] })
  const yAxis = useAxis({ label: 'layer', range: [-0.4, L + 0.4], integer: true, zoom: false })
  const yAxis2 = useAxis({
    label: '∂ output / ∂ input',
    hold: 'union',
    key: `${state.pattern}-${L}-${k}-${state.weights}`,
  })
  return (
    <Figure
      title="Receptive field of a stack of convolutions"
      state={state}
      caption="Top: every connection from one output unit (layer L) down to the input (layer 0) in a stack of one-dimensional convolutions with same padding. Bottom: the unit's effective receptive field, the gradient of the unit with respect to each input value, with the theoretical field marked by the two dashed lines. Drag the output unit along the top row. Compare the stacks at four layers: plain layers see 9 inputs; stride 2 and dilations 1, 2, 4, 8 both see 31, but the dilated stack keeps one output per input position. Dilation 2 in every layer reaches only every other input, and dilations 1, 2, 3 fill the gaps. With uniform weights the effective field is a bell much narrower than the theoretical one."
      readouts={
        <>
          <Readout label="theoretical field r" value={`${top.size} inputs`} />
          <Readout label="inputs reached" value={`${tree.touched} of ${top.size}`} />
          <Readout label="effective field, ±2 standard deviations" value={`${formatNumber(4 * sd)} inputs`} />
          <Readout label="weights per channel pair" value={`${L} × ${k} = ${L * k}`} />
          <Readout label="output length" value={`${net.outputs} (input ${N}, jump ${top.jump})`} />
        </>
      }
    >
      <FixedHeight height={230}>
        <Plot x={xAxis} y={yAxis} ariaLabel="Connections from one output unit down to the input">
          <Segments segments={tree.segments} />
          <Points x={tree.nodes.x} y={tree.nodes.y} size={4} slot={0} name="units the output depends on" />
          <Handle kind="x" at={centre} onDrag={(x) => state.set('unit', x)} label="output unit" />
        </Plot>
      </FixedHeight>
      <FixedHeight height={200}>
        <div className="mt-3">
          <Plot x={xAxis} y={yAxis2} ariaLabel="Effective receptive field">
            <Bars name="gradient" x={positions} y={gradient} width={1} slot={0} />
            <Annotation x={lo} text="field start" dashed />
            <Annotation x={hi} text="field end" dashed />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
