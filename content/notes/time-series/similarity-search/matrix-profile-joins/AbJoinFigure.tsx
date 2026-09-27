import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { abJoin } from '../_shared/matrix-profile'
import { sharedAndNovel } from '../_shared/synthetic'

const N = 400

/**
 * The two AB-joins of two series. The minimum of P_AB finds the pattern the series share. The maximum of P_BA finds the
 * part of B with no counterpart anywhere in A.
 */
export function AbJoinFigure() {
  const m = useParam(40, { min: 16, max: 64, step: 4 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })

  const r = useMemo(() => {
    const s = sharedAndNovel(N, seed.value)
    const ab = abJoin(s.a, s.b, m.value)
    const ba = abJoin(s.b, s.a, m.value)
    let shared = 0
    ab.profile.forEach((v, i) => {
      if (v < ab.profile[shared]) shared = i
    })
    let novel = 0
    ba.profile.forEach((v, i) => {
      if (v > ba.profile[novel]) novel = i
    })
    return {
      a: s.a,
      b: s.b,
      planted: s.shared,
      novel: s.novel,
      ab,
      ba,
      shared,
      match: ab.index[shared],
      novelFound: novel,
    }
  }, [m.value, seed.value])

  const t = r.a.map((_, i) => i)
  const span = (s: number) => Array.from({ length: m.value }, (_, k) => s + k)
  const piece = (name: string, x: number[], s: number, slot: number): XYSeries => ({
    name,
    type: 'line',
    x: span(s),
    y: span(s).map((i) => x[i]),
    slot,
  })
  const seriesA: XYSeries[] = [
    { name: 'A', type: 'line', x: t, y: r.a, muted: true },
    piece('shared pattern', r.a, r.shared, 1),
  ]
  const seriesB: XYSeries[] = [
    { name: 'B', type: 'line', x: t, y: r.b, muted: true },
    piece('shared pattern', r.b, r.match, 1),
    piece('novel in B', r.b, r.novelFound, 2),
  ]
  const idx = Array.from(r.ab.profile, (_, i) => i)
  const profiles: XYSeries[] = [
    {
      name: 'P_AB: each A subsequence to its nearest in B',
      type: 'line',
      x: idx,
      y: Array.from(r.ab.profile),
      slot: 0,
    },
    {
      name: 'P_BA: each B subsequence to its nearest in A',
      type: 'line',
      x: idx,
      y: Array.from(r.ba.profile),
      slot: 3,
    },
    { name: 'shared (min of P_AB)', type: 'scatter', x: [r.shared], y: [r.ab.profile[r.shared]], slot: 1 },
    { name: 'novel (max of P_BA)', type: 'scatter', x: [r.novelFound], y: [r.ba.profile[r.novelFound]], slot: 2 },
  ]

  return (
    <Interactive
      title="Joining two series"
      caption="Two series of smooth noise. Both contain the same bump-and-dip pattern at different places; only B also contains a sawtooth. Bottom: the two AB-join profiles. P_AB gives, for each length-m subsequence of A, the z-normalised distance to its nearest subsequence of B; P_BA is the reverse. The minimum of P_AB is the pattern the two series share, and its index points to the copy in B. The maximum of P_BA is the subsequence of B least like anything in A: the novel sawtooth. The two joins are not symmetric: P_AB never sees the sawtooth, because no subsequence of A needs it as a neighbour."
      controls={
        <>
          <ParamSlider label="subsequence length m" param={m} format={(v) => String(v)} withArrows />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="shared pattern planted at" value={`A ${r.planted.a}, B ${r.planted.b}`} />
          <Readout label="min of P_AB at" value={`A ${r.shared} → B ${r.match}`} />
          <Readout label="its distance" value={formatNumber(r.ab.profile[r.shared])} />
          <Readout label="novel pattern planted at" value={`B ${r.novel}`} />
          <Readout label="max of P_BA at" value={`B ${r.novelFound}`} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={seriesA} xLabel="time" yLabel="A" height={150} />
        <XYChart series={seriesB} xLabel="time" yLabel="B" height={150} />
        <XYChart series={profiles} xLabel="subsequence start" yLabel="distance" height={200} yRange={[0, undefined]} />
      </div>
    </Interactive>
  )
}
