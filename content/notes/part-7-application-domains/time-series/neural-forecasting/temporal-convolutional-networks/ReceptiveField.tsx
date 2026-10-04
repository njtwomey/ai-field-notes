import { useMemo } from 'react'
import {
  choice,
  Figure,
  Handle,
  int,
  Plot,
  Readout,
  type Segment,
  Segments,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

/** Number of input time steps drawn. */
const N = 48
const TIMES = Array.from({ length: N }, (_, t) => t)

type Schedule = 'plain' | 'doubling' | 'repeat' | 'quadrupling'
type Kernel = '2' | '3'

/** Dilation of layer l = 1..L under each schedule. */
function dilations(schedule: Schedule, depth: number): number[] {
  return Array.from({ length: depth }, (_, i) =>
    schedule === 'plain' ? 1 : schedule === 'doubling' ? 2 ** i : schedule === 'repeat' ? 2 ** (i % 3) : 4 ** i,
  )
}

/**
 * Walk backwards from one output: the units of layer l-1 that feed a reached unit s of layer l are
 * s, s - d_l, ..., s - (k-1) d_l. Positions before the first input are zero padding and are dropped.
 */
function trace(output: number, k: number, ds: number[]) {
  const depth = ds.length
  const reached: Set<number>[] = Array.from({ length: depth + 1 }, () => new Set<number>())
  reached[depth].add(output)
  const edges: Segment[] = []
  for (let l = depth; l >= 1; l--) {
    const d = ds[l - 1]
    for (const s of reached[l]) {
      for (let i = 0; i < k; i++) {
        const u = s - i * d
        if (u < 0) continue
        reached[l - 1].add(u)
        edges.push({ from: [u, l - 1], to: [s, l] })
      }
    }
  }
  return { reached, edges }
}

/** Which inputs reach one output of a stack of causal convolutions, for plain and dilated schedules. */
export function ReceptiveField() {
  const state = useFigureState({
    kernel: choice<Kernel>(
      [
        { value: '2', label: '2' },
        { value: '3', label: '3' },
      ],
      '2',
      { label: 'kernel size k' },
    ),
    schedule: choice<Schedule>(
      [
        { value: 'plain', label: 'd = 1' },
        { value: 'doubling', label: '1, 2, 4, 8, …' },
        { value: 'repeat', label: '1, 2, 4, 1, 2, 4' },
        { value: 'quadrupling', label: '1, 4, 16, …' },
      ],
      'doubling',
      { label: 'dilation schedule' },
    ),
    depth: int(4, { min: 1, max: 6, step: 1, label: 'layers L', format: (v) => String(v) }),
    output: slider(0, N - 1, N - 1, { step: 1, label: 'output time t', format: (v) => String(v) }),
  })

  const k = Number(state.kernel)
  const r = useMemo(() => {
    const ds = dilations(state.schedule, state.depth)
    const { reached, edges } = trace(state.output, k, ds)
    const R = 1 + (k - 1) * ds.reduce((a, b) => a + b, 0)
    const grid: SeriesSpec = {
      name: 'unit',
      type: 'scatter',
      x: Array.from({ length: state.depth + 1 }, () => TIMES).flat(),
      y: Array.from({ length: state.depth + 1 }, (_, l) => TIMES.map(() => l)).flat(),
      muted: true,
    }
    const hidden = reached.slice(1, state.depth).flatMap((set, i) => [...set].map((s) => [s, i + 1]))
    const inputs = [...reached[0]].sort((a, b) => a - b)
    const series: SeriesSpec[] = [
      grid,
      {
        name: 'hidden unit on a path',
        type: 'scatter',
        x: hidden.map((p) => p[0]),
        y: hidden.map((p) => p[1]),
        slot: 1,
      },
      { name: 'input in the receptive field', type: 'scatter', x: inputs, y: inputs.map(() => 0), slot: 0 },
      { name: 'output', type: 'scatter', x: [state.output], y: [state.depth], emphasis: true },
    ]
    // Inputs inside the span [t - R + 1, t] that no path reaches: holes in the receptive field.
    const lo = Math.max(0, state.output - R + 1)
    const holes = state.output - lo + 1 - inputs.length
    return { ds, edges, series, R, reachedCount: inputs.length, holes }
  }, [state.schedule, state.depth, state.output, k])

  const xAxis = useAxis({ label: 'time step', range: [-1, N] })
  const yAxis = useAxis({ label: 'layer', range: [-0.5, state.depth + 0.5] })
  return (
    <Figure
      title="Receptive field of a causal convolution stack"
      state={state}
      caption={`Each row is a layer; row 0 is the input sequence and the top row is the output. Every unit at layer l reads k units of layer l − 1 spaced d_l apart, all at or before its own time. Lines trace every path from the chosen output back to the inputs. With d = 1 the field grows by k − 1 per layer. With doubling dilations it doubles per layer and, for k = 2, each input in the field is reached by exactly one path. Repeating 1, 2, 4 is WaveNet's pattern on a small scale. Quadrupling with k = 2 grows the span faster but leaves holes: inputs inside the span that no path reaches. Drag the output along the time axis, or click anywhere on the plot.`}

      readouts={
        <>
          <Readout label="dilations" value={r.ds.join(', ')} />
          <Readout label="R = 1 + (k − 1) Σ d" value={String(r.R)} />
          <Readout label="inputs reached" value={String(r.reachedCount)} />
          <Readout label="holes in the span" value={String(r.holes)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(r.series)}
        <Segments segments={r.edges} />
        <Handle kind="x" at={state.output} label="output t" onDrag={(x) => state.set('output', Math.round(x))} />
      </Plot>
    </Figure>
  )
}
