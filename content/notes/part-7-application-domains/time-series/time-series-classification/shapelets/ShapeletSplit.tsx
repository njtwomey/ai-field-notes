import { useMemo, useState } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { bestSplit, informationGain, subsequenceDistance } from '../_shared/tsc'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const PER_CLASS = 8
const LENGTH = 60
const SHOWN = 3

/**
 * Two classes of noisy series. Class 1 contains a sharp spike-and-dip at a random position; class 0 contains a smooth
 * bump instead. The best shapelet is searched among all subsequences of the first two class-1 series.
 */
function dataset(seed: number, noise: number) {
  const g = stream(seed)
  const make = (label: number) => {
    const x: number[] = []
    let level = 0
    for (let t = 0; t < LENGTH; t++) {
      level = 0.8 * level + 0.3 * normal(g)
      x.push(level + noise * normal(g))
    }
    const at = 5 + Math.floor(uniform(g) * (LENGTH - 20))
    for (let k = 0; k < 10; k++) {
      x[at + k] +=
        label === 1
          ? k < 4
            ? 1.2 * k
            : k < 7
              ? 3.6 - 2.4 * (k - 3)
              : -3.6 + 1.2 * (k - 6)
          : 1.5 * Math.sin((Math.PI * k) / 9)
    }
    return x
  }
  const series: number[][] = []
  const labels: number[] = []
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < PER_CLASS; i++) {
      series.push(make(c))
      labels.push(c)
    }
  }
  return { series, labels }
}

export function ShapeletSplit() {
  const state = useFigureState({
    length: int(12, { min: 6, max: 24, step: 2, label: 'shapelet length', format: (v) => String(v) }),
    noise: float(0.2, { min: 0, max: 0.8, step: 0.05, label: 'noise' }),
    seed: int(2, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const { series, labels } = dataset(state.seed, state.noise)
    const m = state.length
    // Exhaustive search over candidates from two class-1 series, as in the original algorithm but on a small pool.
    let best = { gain: -1, threshold: 0, shapelet: [] as number[], source: 0, start: 0, distances: [] as number[] }
    for (const source of [PER_CLASS, PER_CLASS + 1]) {
      for (let start = 0; start + m <= LENGTH; start++) {
        const shapelet = series[source].slice(start, start + m)
        const distances = series.map((s) => subsequenceDistance(s, shapelet).distance)
        const split = bestSplit(distances, labels)
        if (split.gain > best.gain) best = { ...split, shapelet, source, start, distances }
      }
    }
    const positions = series.map((s) => subsequenceDistance(s, best.shapelet).position)
    return { series, labels, ...best, positions }
  }, [state.length, state.noise, state.seed])

  // A dragged threshold belongs to the data it was dragged on; new data starts at its own best split.
  const key = `${state.length}-${state.noise}-${state.seed}`
  const [dragged, setDragged] = useState<{ key: string; value: number } | null>(null)
  const t = dragged?.key === key ? dragged.value : r.threshold
  const gain = informationGain(r.distances, r.labels, t)
  const correct = r.distances.filter((d, i) => (d < t ? 1 : 0) === r.labels[i]).length

  const panel = (label: number): SeriesSpec[] => {
    const idx = r.labels
      .map((y, i) => (y === label ? i : -1))
      .filter((i) => i >= 0)
      .slice(0, SHOWN)
    return idx.flatMap((i, k) => {
      const offset = 4 * k
      const s = r.series[i]
      const p = r.positions[i]
      const span = Array.from({ length: state.length }, (_, j) => p + j)
      return [
        { name: 'series', type: 'line' as const, x: s.map((_, j) => j), y: s.map((v) => v + offset), muted: true },
        {
          name: 'best match to the shapelet',
          type: 'line' as const,
          x: span,
          y: span.map((j) => s[j] + offset),
          slot: label === 1 ? 1 : 0,
        },
      ]
    })
  }
  const orderline: SeriesSpec[] = [0, 1].map((c) => ({
    name: `class ${c}`,
    type: 'scatter' as const,
    x: r.distances.filter((_, i) => r.labels[i] === c),
    y: r.distances.filter((_, i) => r.labels[i] === c).map(() => c),
    slot: c,
  }))

  const xAxis = useAxis({ label: 't', hold: 'union' })
  const yAxis = useAxis({ hold: 'union' })
  const xAxis2 = useAxis({ label: 't', hold: 'union' })
  const yAxis2 = useAxis({ hold: 'union' })
  const xAxis3 = useAxis({ label: 'distance to the shapelet', hold: 'union' })
  const yAxis3 = useAxis({ label: 'class', range: [-0.5, 1.5] })
  return (
    <Figure
      title="A shapelet splits two classes"
      state={state}
      caption="Class 1 series contain a sharp spike-and-dip, class 0 a smooth bump, each at a random position in noise. The shapelet is the subsequence, searched among two class-1 series, whose distance to every series best separates the classes. Top: three series of each class with their best-matching window highlighted. Bottom: each series placed on the orderline by its distance to the shapelet; the split sends distances below the threshold to class 1. Drag the split to see accuracy and information gain change; change the shapelet length or the noise."

      readouts={
        <>
          <Readout label="best information gain" value={`${formatNumber(r.gain)} bits`} />
          <Readout label="gain at this split" value={`${formatNumber(gain)} bits`} />
          <Readout label="accuracy at this split" value={`${correct}/${r.labels.length}`} />
          <Readout label="shapelet from" value={`series ${r.source + 1}, t = ${r.start}`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">class 0 (smooth bump)</div>
          <Plot x={xAxis} y={yAxis} height={220} bare>
            {seriesLayers(panel(0))}
          </Plot>
        </div>
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">class 1 (spike and dip)</div>
          <Plot x={xAxis2} y={yAxis2} height={220} bare>
            {seriesLayers(panel(1))}
          </Plot>
        </div>
      </div>
      <Plot x={xAxis3} y={yAxis3} height={200}>
        {seriesLayers(orderline)}
        <Handle kind="x" at={t} label="split" onDrag={(x) => setDragged({ key, value: Math.max(0, x) })} />
      </Plot>
    </Figure>
  )
}
