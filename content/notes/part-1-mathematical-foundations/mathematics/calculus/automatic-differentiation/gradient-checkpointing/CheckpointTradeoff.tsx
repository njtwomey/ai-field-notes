import { useMemo } from 'react'
import { Curve, Figure, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'

/**
 * Peak memory of reverse mode on a chain of n layers when a checkpoint is stored every s layers: ⌈n/s⌉ checkpoints plus
 * the s activations of the segment being recomputed. Storing everything (s = 1) and storing almost nothing (s = n)
 * both cost n; the minimum, about 2√n, is at s ≈ √n.
 */
export function CheckpointTradeoff() {
  const state = useFigureState({
    n: int(100, { min: 10, max: 400, step: 10, label: 'layers n' }),
    s: int(10, { min: 1, max: 400, step: 1, label: 'segment length s' }),
  })
  const segment = Math.min(state.s, state.n)

  const memory = (k: number) => Math.ceil(state.n / k) + k
  const series = useMemo(() => {
    const xs = Array.from({ length: state.n }, (_, i) => i + 1)
    return [
      { name: 'peak memory (activations stored)', x: xs, y: xs.map(memory), slot: 0 },
      { name: 'store everything', x: [1, state.n], y: [state.n, state.n], slot: 1, dashed: true },
    ] as const
    // memory depends only on n
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.n])

  // The segment length is a position on the x-axis, so it is dragged directly.

  const xAxis = useAxis({ label: 'segment length s (layers between checkpoints)', range: [1, state.n] })
  const yAxis = useAxis({ label: 'activations held at once', range: [0, state.n + 2] })
  return (
    <Figure
      title="Memory against recomputation"
      state={state}
      caption="A chain of n layers, with a checkpoint every s layers. The backward pass recomputes one segment at a time from its checkpoint, so peak memory is the ⌈n/s⌉ checkpoints plus the s activations of one segment. Drag s: storing every activation (s = 1) costs n, and so does checkpointing only the input (s = n). The minimum, about 2√n, is at s ≈ √n, for the price of one extra forward pass."

      readouts={
        <>
          <Readout label="checkpoints ⌈n/s⌉" value={Math.ceil(state.n / segment)} />
          <Readout label="peak memory" value={memory(segment)} />
          <Readout label="store everything" value={state.n} />
          <Readout label="2√n" value={(2 * Math.sqrt(state.n)).toFixed(1)} />
          <Readout label="extra forward passes" value={segment === 1 ? 0 : 1} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Handle kind="x" at={segment} label="s" onDrag={(x) => state.set('s', Math.round(x))} />
      </Plot>
    </Figure>
  )
}
