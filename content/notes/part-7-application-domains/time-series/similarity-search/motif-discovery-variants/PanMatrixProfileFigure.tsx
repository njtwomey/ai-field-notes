import { useMemo } from 'react'
import { Heatmap, Interactive, ParamSlider, Readout, XYChart, useParam, type XYSeries } from 'aifn-render'
import { selfJoin } from '../_shared/matrix-profile'
import { twoScales } from '../_shared/synthetic'

const N = 500
const LENGTHS = Array.from({ length: 23 }, (_, k) => 8 + 4 * k)
const CAP = 0.5

/**
 * A pan matrix profile: one matrix profile per subsequence length, each divided by 2√m (the largest possible
 * z-normalised distance) so rows of different lengths share one scale. Dark cells are good motifs at that length.
 */
export function PanMatrixProfileFigure() {
  const seed = useParam(3, { min: 1, max: 20, step: 1 })

  const r = useMemo(() => {
    const { x, short, long } = twoScales(N, seed.value)
    const z: number[][] = []
    const best: number[] = []
    for (const m of LENGTHS) {
      const p = selfJoin(x, m).profile
      const row = Array.from({ length: N }, (_, i) => (i < p.length ? Math.min(p[i] / (2 * Math.sqrt(m)), CAP) : CAP))
      let b = 0
      for (let i = 0; i < p.length; i++) if (p[i] < p[b]) b = i
      z.push(row)
      best.push(b)
    }
    return { x, short, long, z, best }
  }, [seed.value])

  const t = r.x.map((_, i) => i)
  const planted = (name: string, starts: number[], length: number, slot: number): XYSeries[] =>
    starts.map((s) => ({
      name,
      type: 'line',
      x: Array.from({ length }, (_, k) => s + k),
      y: Array.from({ length }, (_, k) => r.x[s + k]),
      slot,
    }))
  const top: XYSeries[] = [
    { name: 'series', type: 'line', x: t, y: r.x, muted: true },
    ...planted('short pattern (length 20)', r.short, 20, 1),
    ...planted('long pattern (length 72)', r.long, 72, 2),
  ]
  const overlay = useMemo(
    () => [{ name: 'best motif at each length', type: 'scatter' as const, x: r.best, y: LENGTHS, emphasis: true }],
    [r.best],
  )

  return (
    <Interactive
      title="Motifs at every length"
      caption="Top: smooth noise with a short sawtooth planted twice (length 20) and a long wave planted twice (length 72). Bottom: the pan matrix profile. Row m is the matrix profile for subsequence length m, divided by 2√m so that every row lies between 0 and 1 (values above 0.5 are drawn at 0.5, as are positions too late to start a subsequence of that length). Dark cells are subsequences with a close match at that length. Diamonds mark the top motif of each row. The sawtooth is darkest at short lengths; the long wave darkens only as m approaches its own length. A single fixed m would show only one of the two."
      controls={<ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />}
      readout={
        <>
          <Readout label="short pattern at" value={r.short.join(', ')} />
          <Readout label="long pattern at" value={r.long.join(', ')} />
          <Readout label={`top motif at m = ${LENGTHS[3]}`} value={r.best[3]} />
          <Readout label={`top motif at m = ${LENGTHS[16]}`} value={r.best[16]} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={top} xLabel="time" yLabel="x" height={180} />
        <Heatmap
          x={t}
          y={LENGTHS}
          z={r.z}
          range={[0, CAP]}
          overlay={overlay}
          xLabel="subsequence start i"
          yLabel="length m"
          valueLabel="P / 2√m"
          height={320}
        />
      </div>
    </Interactive>
  )
}
