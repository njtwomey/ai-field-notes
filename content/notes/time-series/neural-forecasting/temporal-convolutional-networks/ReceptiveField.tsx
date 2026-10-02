import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  useParam,
  type Handle,
  type Segment,
  type XYSeries,
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
  const [kernel, setKernel] = useState<Kernel>('2')
  const [schedule, setSchedule] = useState<Schedule>('doubling')
  const depth = useParam(4, { min: 1, max: 6, step: 1 })
  const output = useParam(N - 1, { min: 0, max: N - 1, step: 1 })

  const k = Number(kernel)
  const r = useMemo(() => {
    const ds = dilations(schedule, depth.value)
    const { reached, edges } = trace(output.value, k, ds)
    const R = 1 + (k - 1) * ds.reduce((a, b) => a + b, 0)
    const grid: XYSeries = {
      name: 'unit',
      type: 'scatter',
      x: Array.from({ length: depth.value + 1 }, () => TIMES).flat(),
      y: Array.from({ length: depth.value + 1 }, (_, l) => TIMES.map(() => l)).flat(),
      muted: true,
    }
    const hidden = reached.slice(1, depth.value).flatMap((set, i) => [...set].map((s) => [s, i + 1]))
    const inputs = [...reached[0]].sort((a, b) => a - b)
    const series: XYSeries[] = [
      grid,
      {
        name: 'hidden unit on a path',
        type: 'scatter',
        x: hidden.map((p) => p[0]),
        y: hidden.map((p) => p[1]),
        slot: 1,
      },
      { name: 'input in the receptive field', type: 'scatter', x: inputs, y: inputs.map(() => 0), slot: 0 },
      { name: 'output', type: 'scatter', x: [output.value], y: [depth.value], emphasis: true },
    ]
    // Inputs inside the span [t - R + 1, t] that no path reaches: holes in the receptive field.
    const lo = Math.max(0, output.value - R + 1)
    const holes = output.value - lo + 1 - inputs.length
    return { ds, edges, series, R, reachedCount: inputs.length, holes }
  }, [schedule, depth.value, output.value, k])

  const handles: Handle[] = [
    { kind: 'x', at: output.value, label: 'output t', onDrag: (x) => output.set(Math.round(x)) },
  ]

  return (
    <Interactive
      title="Receptive field of a causal convolution stack"
      caption={`Each row is a layer; row 0 is the input sequence and the top row is the output. Every unit at layer l reads k units of layer l − 1 spaced d_l apart, all at or before its own time. Lines trace every path from the chosen output back to the inputs. With d = 1 the field grows by k − 1 per layer. With doubling dilations it doubles per layer and, for k = 2, each input in the field is reached by exactly one path. Repeating 1, 2, 4 is WaveNet's pattern on a small scale. Quadrupling with k = 2 grows the span faster but leaves holes: inputs inside the span that no path reaches. Drag the output along the time axis, or click anywhere on the plot.`}
      controls={
        <>
          <ParamChoice
            label="kernel size k"
            value={kernel}
            onChange={setKernel}
            options={[
              { value: '2', label: '2' },
              { value: '3', label: '3' },
            ]}
          />
          <ParamChoice
            label="dilation schedule"
            value={schedule}
            onChange={setSchedule}
            options={[
              { value: 'plain', label: 'd = 1' },
              { value: 'doubling', label: '1, 2, 4, 8, …' },
              { value: 'repeat', label: '1, 2, 4, 1, 2, 4' },
              { value: 'quadrupling', label: '1, 4, 16, …' },
            ]}
          />
          <ParamSlider label="layers L" param={depth} format={(v) => String(v)} withArrows />
          <ParamSlider label="output time t" param={output} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="dilations" value={r.ds.join(', ')} />
          <Readout label="R = 1 + (k − 1) Σ d" value={String(r.R)} />
          <Readout label="inputs reached" value={String(r.reachedCount)} />
          <Readout label="holes in the span" value={String(r.holes)} />
        </>
      }
    >
      <XYChart
        series={r.series}
        segments={r.edges}
        handles={handles}
        xLabel="time step"
        yLabel="layer"
        xRange={[-1, N]}
        yRange={[-0.5, depth.value + 0.5]}
        height={300}
      />
    </Interactive>
  )
}
