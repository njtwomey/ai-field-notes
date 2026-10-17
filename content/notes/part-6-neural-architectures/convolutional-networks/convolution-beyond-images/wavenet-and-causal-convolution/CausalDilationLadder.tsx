import { useMemo } from 'react'
import {
  Annotation,
  Bars,
  choice,
  Figure,
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
import { conv } from 'aifn-compute/foundation/convolution'
import { fromData, full, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { FixedHeight } from '../_shared/FixedHeight'
import { receptiveField } from 'aifn-compute/nn/functional'

/** Sequence length: positions 0 … N − 1. */
const N = 128

type Schedule = 'doubling' | 'plain' | 'constant' | 'blocks' | 'quadrupling'
type Padding = 'causal' | 'centred'
type Layer = { kernel: number; dilation: number; lo: number; hi: number }

const SCHEDULES: { value: Schedule; label: string }[] = [
  { value: 'doubling', label: 'dilation 1, 2, 4, 8, …' },
  { value: 'plain', label: 'dilation 1 in every layer' },
  { value: 'constant', label: 'dilation 2 in every layer' },
  { value: 'blocks', label: 'dilation 1, 2, 4, repeated' },
  { value: 'quadrupling', label: 'dilation 1, 4, 16, …' },
]

const dilationOf = (s: Schedule, l: number) =>
  s === 'doubling' ? 2 ** l : s === 'constant' ? 2 : s === 'blocks' ? 2 ** (l % 3) : s === 'quadrupling' ? 4 ** l : 1

/**
 * The stack: L layers of k taps. Causal padding puts all (k − 1)d zeros before the sequence, so output t reads inputs
 * t − (k − 1)d … t; centred padding splits them, ⌊(k − 1)d/2⌋ before and the rest after, as PyTorch's "same" padding does.
 */
function stack(schedule: Schedule, padding: Padding, L: number, k: number): Layer[] {
  return Array.from({ length: L }, (_, l) => {
    const dilation = dilationOf(schedule, l)
    const span = (k - 1) * dilation
    const lo = padding === 'causal' ? span : Math.floor(span / 2)
    return { kernel: k, dilation, lo, hi: span - lo }
  })
}

/**
 * A stack of dilated one-dimensional convolutions drawn as the tree of connections from one output unit down to the
 * input, with a causality check: one input is perturbed and the change in every output is shown.
 */
export function CausalDilationLadder() {
  const state = useFigureState({
    schedule: choice<Schedule>(SCHEDULES, 'doubling', { label: 'dilations' }),
    layers: int(4, { ge: 1, le: 7, label: 'layers L' }),
    kernel: choice<'2' | '3'>(
      [
        { value: '2', label: '2 taps (WaveNet)' },
        { value: '3', label: '3 taps' },
      ],
      '2',
      { label: 'kernel k' },
    ),
    padding: choice<Padding>(
      [
        { value: 'causal', label: 'causal: all padding on the left' },
        { value: 'centred', label: 'centred ("same")' },
      ],
      'causal',
      { label: 'padding' },
    ),
    unit: slider(0, N - 1, 100, { step: 1, onChart: true }),
    probe: slider(0, N - 1, 70, { step: 1, onChart: true }),
  })
  const k = Number(state.kernel)
  const L = state.layers
  const t = Math.round(state.unit)
  const q = Math.round(state.probe)

  const net = useMemo(() => {
    const layers = stack(state.schedule, state.padding, L, k)
    const size = receptiveField(layers.map((l) => ({ kernel: l.kernel, dilation: l.dilation })))[L].size
    const w = full([1, 1, k], 1 / k)
    const f = (x: Value) =>
      layers.reduce<Value>(
        (h, l) => conv(h, w, { padding: [[l.lo, l.hi]], dilation: l.dilation, flip: false, layout: 'ncw' }),
        x,
      )
    return { layers, size, f }
  }, [state.schedule, state.padding, L, k])

  // Every connection from output unit t down to the input: layer l reads layer l − 1 at t − lo + i d.
  const tree = useMemo(() => {
    const segments: { from: [number, number]; to: [number, number] }[] = []
    const nodes = { x: [t], y: [L] }
    let level = new Set([t])
    for (let l = L; l >= 1; l--) {
      const { kernel, dilation, lo } = net.layers[l - 1]
      const next = new Set<number>()
      for (const p of level) {
        for (let i = 0; i < kernel; i++) {
          const c = p - lo + i * dilation
          if (c < 0 || c > N - 1) continue
          segments.push({ from: [p, l], to: [c, l - 1] })
          next.add(c)
        }
      }
      for (const c of next) {
        nodes.x.push(c)
        nodes.y.push(l - 1)
      }
      level = next
    }
    const reached = [...level]
    return { segments, nodes, reached: reached.length, latest: Math.max(...reached), earliest: Math.min(...reached) }
  }, [net, t, L])

  // The causality check: the change in every output when input q is raised by 1 (the stack is linear).
  const change = useMemo(() => {
    const e = new Float64Array(N)
    e[q] = 1
    return Array.from(toFlat(net.f(fromData(e, [1, 1, N])) as Tensor))
  }, [net, q])
  const changed = change.map((v, i) => (Math.abs(v) > 1e-12 ? i : -1)).filter((i) => i >= 0)
  const leaks = changed.filter((i) => i < q).length
  const positions = useMemo(() => Array.from({ length: N }, (_, i) => i), [])
  const cache = net.layers.reduce((a, l) => a + (l.kernel - 1) * l.dilation, 0)

  const xAxis = useAxis({ label: 'time step', range: [-0.5, N - 0.5] })
  const yAxis = useAxis({ label: 'layer', range: [-0.4, L + 0.4], integer: true, zoom: false })
  const dAxis = useAxis({ label: 'change in output', hold: 'union', key: `${state.schedule}-${L}-${k}` })
  return (
    <Figure
      title="Dilated causal convolutions"
      state={state}
      caption="Top: every connection from one output unit (layer L) down to the input (layer 0) in a stack of one-dimensional convolutions. Drag the output unit along the top row. With 2 taps and dilations 1, 2, 4, 8 the unit reads exactly the 16 inputs ending at its own time step, each through one path; the receptive field doubles with every layer added. Dilation 1 in every layer grows the field by one step per layer, dilation 2 in every layer skips every other input, and dilations 1, 4, 16 with 2 taps leave gaps. Bottom: the change in every output when the input at the dashed line is raised by 1, with every tap weight 1/k. Drag the perturbed input along the bottom row. With causal padding only outputs at or after the perturbed step change. Switch to centred padding, which splits the zeros between the two ends as PyTorch's 'same' padding does: outputs before the perturbed step change too, so a model trained this way reads the future."
      readouts={
        <>
          <Readout label="receptive field" value={`${net.size} steps`} />
          <Readout label="inputs the unit reads" value={`${tree.reached} of ${net.size}`} />
          <Readout
            label="latest input read"
            value={tree.latest > t ? `${tree.latest - t} steps after the unit` : 'the unit’s own step'}
          />
          <Readout label="earlier outputs changed by the perturbation" value={String(leaks)} />
          <Readout label="weights per channel pair" value={`${L} × ${k} = ${L * k}`} />
          <Readout label="activations cached per channel for generation" value={String(cache)} />
        </>
      }
    >
      <FixedHeight height={230}>
        <Plot x={xAxis} y={yAxis} ariaLabel="Connections from one output unit down to the input">
          <Segments segments={tree.segments} />
          <Points x={tree.nodes.x} y={tree.nodes.y} size={4} slot={0} name="units the output reads" />
          <Handle kind="x" at={t} onDrag={(x) => state.set('unit', x)} label="output unit" />
        </Plot>
      </FixedHeight>
      <FixedHeight height={180}>
        <div className="mt-3">
          <Plot x={xAxis} y={dAxis} ariaLabel="Change in every output when one input is perturbed">
            <Bars name="change in output" x={positions} y={change} width={1} slot={1} />
            <Annotation x={q} text="perturbed input" dashed />
            <Handle kind="x" at={q} onDrag={(x) => state.set('probe', x)} label="perturbed input" />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
