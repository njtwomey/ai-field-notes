import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, useParam, type Handle, type XYSeries } from 'aifn-render'

/**
 * Peak memory of reverse mode on a chain of n layers when a checkpoint is stored every s layers: ⌈n/s⌉ checkpoints plus
 * the s activations of the segment being recomputed. Storing everything (s = 1) and storing almost nothing (s = n)
 * both cost n; the minimum, about 2√n, is at s ≈ √n.
 */
export function CheckpointTradeoff() {
  const n = useParam(100, { min: 10, max: 400, step: 10 })
  const s = useParam(10, { min: 1, max: 400, step: 1 })
  const segment = Math.min(s.value, n.value)

  const memory = (k: number) => Math.ceil(n.value / k) + k
  const series = useMemo((): XYSeries[] => {
    const xs = Array.from({ length: n.value }, (_, i) => i + 1)
    return [
      { name: 'peak memory (activations stored)', type: 'line', x: xs, y: xs.map(memory), slot: 0 },
      { name: 'store everything', type: 'line', x: [1, n.value], y: [n.value, n.value], slot: 1, dashed: true },
    ]
    // memory depends only on n
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [n.value])

  // The segment length is a position on the x-axis, so it is dragged directly.
  const handles: Handle[] = [{ kind: 'x', at: segment, label: 's', onDrag: (x) => s.set(Math.round(x)) }]

  return (
    <Interactive
      title="Memory against recomputation"
      caption="A chain of n layers, with a checkpoint every s layers. The backward pass recomputes one segment at a time from its checkpoint, so peak memory is the ⌈n/s⌉ checkpoints plus the s activations of one segment. Drag s: storing every activation (s = 1) costs n, and so does checkpointing only the input (s = n). The minimum, about 2√n, is at s ≈ √n, for the price of one extra forward pass."
      controls={
        <>
          <ParamSlider label="layers n" param={n} />
          <ParamSlider label="segment length s" param={s} />
        </>
      }
      readout={
        <>
          <Readout label="checkpoints ⌈n/s⌉" value={Math.ceil(n.value / segment)} />
          <Readout label="peak memory" value={memory(segment)} />
          <Readout label="store everything" value={n.value} />
          <Readout label="2√n" value={(2 * Math.sqrt(n.value)).toFixed(1)} />
          <Readout label="extra forward passes" value={segment === 1 ? 0 : 1} />
        </>
      }
    >
      <XYChart
        series={series}
        handles={handles}
        xLabel="segment length s (layers between checkpoints)"
        yLabel="activations held at once"
        xRange={[1, n.value]}
        yRange={[0, n.value + 2]}
        height={320}
      />
    </Interactive>
  )
}
