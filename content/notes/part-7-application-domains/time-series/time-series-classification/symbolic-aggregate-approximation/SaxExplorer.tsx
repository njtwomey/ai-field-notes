import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { letters, minDist, paa, paaDistance, sax, saxBreakpoints, znorm } from '../_shared/tsc'
import { normal, stream } from 'aifn-compute/foundation/random'

const N = 64

function walk(seed: number) {
  const g = stream(seed)
  const x: number[] = []
  let v = 0
  for (let t = 0; t < N; t++) {
    v += normal(g)
    x.push(v)
  }
  return znorm(x)
}

/** A series step by step: z-normalise, average into w segments (PAA), and map each mean to a symbol by breakpoint. */
export function SaxExplorer() {
  const state = useFigureState({
    w: int(8, { min: 2, max: 16, step: 1, label: 'segments w', format: (v) => String(v) }),
    alphabet: int(4, { min: 2, max: 10, step: 1, label: 'alphabet size a', format: (v) => String(v) }),
    seed: int(3, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const x = walk(state.seed)
    const y = walk(state.seed + 100)
    const sx = sax(x, state.w, state.alphabet)
    const sy = sax(y, state.w, state.alphabet)
    const px = paa(x, state.w)
    const py = paa(y, state.w)
    const euclid = Math.sqrt(x.reduce((s, v, i) => s + (v - y[i]) ** 2, 0))
    return {
      x,
      sx,
      sy,
      mind: minDist(sx.symbols, sy.symbols, N, state.alphabet),
      paaDist: paaDistance(px, py, N),
      euclid,
      cuts: saxBreakpoints(state.alphabet),
    }
  }, [state.w, state.alphabet, state.seed])

  // PAA as a step function: each segment's mean held across its time span.
  const steps = r.sx.paa.flatMap((v, i) => {
    const start = Math.round((i * N) / state.w)
    const end = Math.round(((i + 1) * N) / state.w)
    return [
      [start, v],
      [end, v],
    ]
  })
  const series: SeriesSpec[] = [
    { name: 'z-normalised series', type: 'line', x: r.x.map((_, i) => i), y: r.x, muted: true },
    { name: 'PAA segment means', type: 'line', x: steps.map((p) => p[0]), y: steps.map((p) => p[1]), slot: 0 },
    ...r.cuts.map((c): SeriesSpec => ({
      name: 'breakpoints',
      type: 'line',
      x: [0, N],
      y: [c, c],
      dashed: true,
      slot: 2,
    })),
  ]

  const xAxis = useAxis({ label: 't', hold: 'union' })
  const yAxis = useAxis({ label: 'z', hold: 'union' })
  return (
    <Figure
      title="From a series to a word"
      state={state}
      caption="The series is z-normalised (grey), averaged over w segments of equal or near-equal length (PAA, blue steps), and each segment mean is mapped to a letter by the breakpoints (dashed) that cut the standard normal into a equally likely regions. The readouts compare a second random series with this one: the distance between their SAX words (MINDIST) never exceeds the distance between their PAA representations, which never exceeds the Euclidean distance between the series. That ordering is the lower-bounding property that makes SAX safe for search."

      readouts={
        <>
          <Readout label="SAX word" value={letters(r.sx.symbols)} />
          <Readout label="second series' word" value={letters(r.sy.symbols)} />
          <Readout label="MINDIST" value={formatNumber(r.mind)} />
          <Readout label="PAA distance" value={formatNumber(r.paaDist)} />
          <Readout label="Euclidean distance" value={formatNumber(r.euclid)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
