import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'
import { letters, minDist, paa, paaDistance, sax, saxBreakpoints, znorm } from '../_shared/tsc'

const N = 64

function walk(seed: number) {
  const g = rng(seed)
  const x: number[] = []
  let v = 0
  for (let t = 0; t < N; t++) {
    v += g.normal()
    x.push(v)
  }
  return znorm(x)
}

/** A series step by step: z-normalise, average into w segments (PAA), and map each mean to a symbol by breakpoint. */
export function SaxExplorer() {
  const w = useParam(8, { min: 2, max: 16, step: 1 })
  const alphabet = useParam(4, { min: 2, max: 10, step: 1 })
  const seed = useParam(3, { min: 1, max: 30, step: 1 })

  const r = useMemo(() => {
    const x = walk(seed.value)
    const y = walk(seed.value + 100)
    const sx = sax(x, w.value, alphabet.value)
    const sy = sax(y, w.value, alphabet.value)
    const px = paa(x, w.value)
    const py = paa(y, w.value)
    const euclid = Math.sqrt(x.reduce((s, v, i) => s + (v - y[i]) ** 2, 0))
    return {
      x,
      sx,
      sy,
      mind: minDist(sx.symbols, sy.symbols, N, alphabet.value),
      paaDist: paaDistance(px, py, N),
      euclid,
      cuts: saxBreakpoints(alphabet.value),
    }
  }, [w.value, alphabet.value, seed.value])

  // PAA as a step function: each segment's mean held across its time span.
  const steps = r.sx.paa.flatMap((v, i) => {
    const start = Math.round((i * N) / w.value)
    const end = Math.round(((i + 1) * N) / w.value)
    return [
      [start, v],
      [end, v],
    ]
  })
  const series: XYSeries[] = [
    { name: 'z-normalised series', type: 'line', x: r.x.map((_, i) => i), y: r.x, muted: true },
    { name: 'PAA segment means', type: 'line', x: steps.map((p) => p[0]), y: steps.map((p) => p[1]), slot: 0 },
    ...r.cuts.map((c): XYSeries => ({
      name: 'breakpoints',
      type: 'line',
      x: [0, N],
      y: [c, c],
      dashed: true,
      slot: 2,
    })),
  ]

  return (
    <Interactive
      title="From a series to a word"
      caption="The series is z-normalised (grey), averaged over w segments of equal or near-equal length (PAA, blue steps), and each segment mean is mapped to a letter by the breakpoints (dashed) that cut the standard normal into a equally likely regions. The readouts compare a second random series with this one: the distance between their SAX words (MINDIST) never exceeds the distance between their PAA representations, which never exceeds the Euclidean distance between the series. That ordering is the lower-bounding property that makes SAX safe for search."
      controls={
        <>
          <ParamSlider label="segments w" param={w} format={(v) => String(v)} withArrows />
          <ParamSlider label="alphabet size a" param={alphabet} format={(v) => String(v)} withArrows />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="SAX word" value={letters(r.sx.symbols)} />
          <Readout label="second series' word" value={letters(r.sy.symbols)} />
          <Readout label="MINDIST" value={formatNumber(r.mind)} />
          <Readout label="PAA distance" value={formatNumber(r.paaDist)} />
          <Readout label="Euclidean distance" value={formatNumber(r.euclid)} />
        </>
      }
    >
      <XYChart series={series} xLabel="t" yLabel="z" height={300} />
    </Interactive>
  )
}
