import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { abJoin } from '../_shared/matrix-profile'
import { sharedAndNovel } from '../_shared/synthetic'

const N = 400

/**
 * The two AB-joins of two series. The minimum of P_AB finds the pattern the series share. The maximum of P_BA finds the
 * part of B with no counterpart anywhere in A.
 */
export function AbJoinFigure() {
  const state = useFigureState({
    m: int(40, { min: 16, max: 64, step: 4, label: 'subsequence length m', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const s = sharedAndNovel(N, state.seed)
    const ab = abJoin(s.a, s.b, state.m)
    const ba = abJoin(s.b, s.a, state.m)
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
  }, [state.m, state.seed])

  const t = r.a.map((_, i) => i)
  const span = (s: number) => Array.from({ length: state.m }, (_, k) => s + k)
  const piece = (name: string, x: number[], s: number, slot: number): SeriesSpec => ({
    name,
    type: 'line',
    x: span(s),
    y: span(s).map((i) => x[i]),
    slot,
  })
  const seriesA: SeriesSpec[] = [
    { name: 'A', type: 'line', x: t, y: r.a, muted: true },
    piece('shared pattern', r.a, r.shared, 1),
  ]
  const seriesB: SeriesSpec[] = [
    { name: 'B', type: 'line', x: t, y: r.b, muted: true },
    piece('shared pattern', r.b, r.match, 1),
    piece('novel in B', r.b, r.novelFound, 2),
  ]
  const idx = Array.from(r.ab.profile, (_, i) => i)
  const profiles = [
    {
      name: 'P_AB: each A subsequence to its nearest in B',
      x: idx,
      y: Array.from(r.ab.profile),
      slot: 0,
    },
    {
      name: 'P_BA: each B subsequence to its nearest in A',
      x: idx,
      y: Array.from(r.ba.profile),
      slot: 3,
    },
    { name: 'shared (min of P_AB)', x: [r.shared], y: [r.ab.profile[r.shared]], slot: 1 },
    { name: 'novel (max of P_BA)', x: [r.novelFound], y: [r.ba.profile[r.novelFound]], slot: 2 },
  ] as const

  const xAxis = useAxis({ label: 'time', hold: 'union' })
  const yAxis = useAxis({ label: 'A', hold: 'union' })
  const xAxis2 = useAxis({ label: 'time', hold: 'union' })
  const yAxis2 = useAxis({ label: 'B', hold: 'union' })
  const xAxis3 = useAxis({ label: 'subsequence start', hold: 'union' })
  const yAxis3 = useAxis({ label: 'distance', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Joining two series"
      state={state}
      caption="Two series of smooth noise. Both contain the same bump-and-dip pattern at different places; only B also contains a sawtooth. Bottom: the two AB-join profiles. P_AB gives, for each length-m subsequence of A, the z-normalised distance to its nearest subsequence of B; P_BA is the reverse. The minimum of P_AB is the pattern the two series share, and its index points to the copy in B. The maximum of P_BA is the subsequence of B least like anything in A: the novel sawtooth. The two joins are not symmetric: P_AB never sees the sawtooth, because no subsequence of A needs it as a neighbour."

      readouts={
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
        <Plot x={xAxis} y={yAxis} height={150}>
          {seriesLayers(seriesA)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={150}>
          {seriesLayers(seriesB)}
        </Plot>
        <Plot x={xAxis3} y={yAxis3} height={200}>
          <Curve {...profiles[0]} />
          <Curve {...profiles[1]} />
          <Points {...profiles[2]} />
          <Points {...profiles[3]} />
        </Plot>
      </div>
    </Figure>
  )
}
