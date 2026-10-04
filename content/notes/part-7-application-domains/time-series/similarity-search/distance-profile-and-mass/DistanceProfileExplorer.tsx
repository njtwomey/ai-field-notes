import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { exclusionZone, mass, movingStats } from '../_shared/matrix-profile'
import { beats } from '../_shared/synthetic'

const N = 480

/**
 * One query against a whole series. The query is the length-m subsequence starting at q; MASS computes its distance to
 * every subsequence with one FFT convolution. Minima are its matches; the exclusion zone around q hides trivial ones.
 */
export function DistanceProfileExplorer() {
  const state = useFigureState({
    m: int(40, { ge: 8, le: 80, step: 4, suggestions: [8, 20, 40, 80], label: 'subsequence length m' }),
    q: slider(0, N - 8, 130, { step: 1, label: 'query start q (at most N − m)' }),
    seed: int(2, { ge: 0, label: 'seed' }),
  })
  const { m, seed } = state

  const data = useMemo(() => beats(N, seed, 0.04, [7]), [seed])
  const r = useMemo(() => {
    const start = Math.min(state.q, N - m)
    const query = data.x.slice(start, start + m)
    const profile = mass(query, data.x, movingStats(data.x, m))
    const e = exclusionZone(m)
    let best = -1
    profile.forEach((d, i) => {
      if (Math.abs(i - start) > e && (best < 0 || d < profile[best])) best = i
    })
    return { start, profile, best, e }
  }, [data, m, state.q])

  const t = data.x.map((_, i) => i)
  const span = (s: number) => Array.from({ length: m }, (_, k) => s + k)
  const top: SeriesSpec[] = [
    { name: 'series', type: 'line', x: t, y: data.x, muted: true },
    { name: 'query', type: 'line', x: span(r.start), y: span(r.start).map((i) => data.x[i]), slot: 1 },
    { name: 'nearest match', type: 'line', x: span(r.best), y: span(r.best).map((i) => data.x[i]), slot: 2 },
  ]
  const profileX = Array.from(r.profile, (_, i) => i)
  const bottom: SeriesSpec[] = [
    { name: 'distance profile', type: 'line', x: profileX, y: Array.from(r.profile), slot: 0 },
    { name: 'nearest match', type: 'scatter', x: [r.best], y: [r.profile[r.best]], slot: 2 },
  ]
  const xAxis = useAxis({ label: 'time', range: [0, N - 1] })
  const yAxis = useAxis({ label: 'x', hold: 'union', key: seed })
  const xAxis2 = useAxis({ label: 'subsequence start i', range: [0, N - 1] })
  const yAxis2 = useAxis({ label: 'distance', range: [0, undefined], hold: 'union' })

  return (
    <Figure
      title="The distance profile of one query"
      state={state}
      caption="Top: a heartbeat-like series with one abnormal beat; the query is the highlighted subsequence starting at the dashed line (drag it). Bottom: the distance profile, the z-normalised distance from the query to every subsequence, computed by MASS with one FFT convolution. It dips to near 0 wherever another beat aligns with the query and rises between them. The profile is exactly 0 at the query itself; matches inside the exclusion zone around it are trivial and ignored. Drag the query onto the abnormal beat and every other position is far away."
      readouts={
        <>
          <Readout label="query at" value={r.start} />
          <Readout label="nearest non-trivial match at" value={r.best} />
          <Readout label="its distance" value={formatNumber(r.profile[r.best])} />
          <Readout label="largest possible, 2√m" value={formatNumber(2 * Math.sqrt(m))} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={200}>
          {seriesLayers(top)}
          <Handle
            kind="x"
            at={r.start}
            label="query"
            onDrag={(v) => state.set('q', Math.round(Math.min(Math.max(v, 0), N - m)))}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={200}>
          {seriesLayers(bottom)}
        </Plot>
      </div>
    </Figure>
  )
}
