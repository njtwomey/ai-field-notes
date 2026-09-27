import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { exclusionZone, mass, movingStats } from '../_shared/matrix-profile'
import { beats } from '../_shared/synthetic'

const N = 480

/**
 * One query against a whole series. The query is the length-m subsequence starting at q; MASS computes its distance to
 * every subsequence with one FFT convolution. Minima are its matches; the exclusion zone around q hides trivial ones.
 */
export function DistanceProfileExplorer() {
  const m = useParam(40, { min: 8, max: 80, step: 4 })
  const q = useParam(130, { min: 0, max: N - 80, step: 1 })
  const seed = useParam(2, { min: 1, max: 20, step: 1 })

  const data = useMemo(() => beats(N, seed.value, 0.04, [7]), [seed.value])
  const r = useMemo(() => {
    const start = Math.min(q.value, N - m.value)
    const query = data.x.slice(start, start + m.value)
    const profile = mass(query, data.x, movingStats(data.x, m.value))
    const e = exclusionZone(m.value)
    let best = -1
    profile.forEach((d, i) => {
      if (Math.abs(i - start) > e && (best < 0 || d < profile[best])) best = i
    })
    return { start, profile, best, e }
  }, [data, m.value, q.value])

  const t = data.x.map((_, i) => i)
  const span = (s: number) => Array.from({ length: m.value }, (_, k) => s + k)
  const top: XYSeries[] = [
    { name: 'series', type: 'line', x: t, y: data.x, muted: true },
    { name: 'query', type: 'line', x: span(r.start), y: span(r.start).map((i) => data.x[i]), slot: 1 },
    { name: 'nearest match', type: 'line', x: span(r.best), y: span(r.best).map((i) => data.x[i]), slot: 2 },
  ]
  const profileX = Array.from(r.profile, (_, i) => i)
  const bottom: XYSeries[] = [
    { name: 'distance profile', type: 'line', x: profileX, y: Array.from(r.profile), slot: 0 },
    { name: 'nearest match', type: 'scatter', x: [r.best], y: [r.profile[r.best]], slot: 2 },
  ]
  const handles: Handle[] = [
    { kind: 'x', at: r.start, label: 'query', onDrag: (v) => q.set(Math.round(Math.min(Math.max(v, 0), N - m.value))) },
  ]

  return (
    <Interactive
      title="The distance profile of one query"
      caption="Top: a heartbeat-like series with one abnormal beat; the query is the highlighted subsequence starting at the dashed line (drag it). Bottom: the distance profile, the z-normalised distance from the query to every subsequence, computed by MASS with one FFT convolution. It dips to near 0 wherever another beat aligns with the query and rises between them. The profile is exactly 0 at the query itself; matches inside the exclusion zone around it are trivial and ignored. Drag the query onto the abnormal beat and every other position is far away."
      controls={
        <>
          <ParamSlider label="subsequence length m" param={m} format={(v) => String(v)} withArrows />
          <ParamSlider label="query start q" param={q} format={(v) => String(v)} withArrows />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="query at" value={r.start} />
          <Readout label="nearest non-trivial match at" value={r.best} />
          <Readout label="its distance" value={formatNumber(r.profile[r.best])} />
          <Readout label="largest possible, 2√m" value={formatNumber(2 * Math.sqrt(m.value))} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={top} xLabel="time" yLabel="x" height={200} handles={handles} />
        <XYChart series={bottom} xLabel="subsequence start i" yLabel="distance" height={200} yRange={[0, undefined]} />
      </div>
    </Interactive>
  )
}
