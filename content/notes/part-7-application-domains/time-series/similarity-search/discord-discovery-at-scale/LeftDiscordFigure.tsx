import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { selfJoin } from '../_shared/matrix-profile'
import { beats } from '../_shared/synthetic'

const N = 480
const M = 40
/** The first three beats serve as a training prefix: with few earlier subsequences, every left distance is large. */
const PREFIX = 120

const argmax = (a: ArrayLike<number>, from: number) => {
  let best = from
  for (let i = from; i < a.length; i++) if (a[i] > a[best]) best = i
  return best
}

/**
 * Twin anomalies. Two identical abnormal beats are each other's nearest neighbour, so the full matrix profile is low at
 * both. The left matrix profile compares each subsequence only with the past, so the first twin has no match and stands
 * out: the left discord that DAMP searches for.
 */
export function LeftDiscordFigure() {
  const seed = useParam(1, { min: 1, max: 20, step: 1 })

  const r = useMemo(() => {
    const { x, starts } = beats(N, seed.value, 0.03, [5, 10])
    const p = selfJoin(x, M)
    return { x, twins: [starts[5], starts[10]], p, full: argmax(p.profile, PREFIX), left: argmax(p.left, PREFIX) }
  }, [seed.value])

  const t = r.x.map((_, i) => i)
  const span = (s: number) => Array.from({ length: M }, (_, k) => s + k)
  const top: XYSeries[] = [
    { name: 'series', type: 'line', x: t, y: r.x, muted: true },
    ...r.twins.map((s) => ({
      name: 'abnormal beat',
      type: 'line' as const,
      x: span(s),
      y: span(s).map((i) => r.x[i]),
      slot: 2,
    })),
  ]
  const idx = Array.from(r.p.profile, (_, i) => i).slice(PREFIX)
  const keep = (a: Float64Array) => Array.from(a).slice(PREFIX)
  const bottom: XYSeries[] = [
    { name: 'matrix profile', type: 'line', x: idx, y: keep(r.p.profile), slot: 0 },
    { name: 'left matrix profile', type: 'line', x: idx, y: keep(r.p.left), slot: 1 },
    {
      name: 'maximum of each',
      type: 'scatter',
      x: [r.full, r.left],
      y: [r.p.profile[r.full], r.p.left[r.left]],
      emphasis: true,
    },
  ]

  return (
    <Interactive
      title="The left matrix profile catches twin anomalies"
      caption={`Top: heartbeat-like beats in which the same abnormal beat occurs twice. Bottom: the matrix profile and the left matrix profile for m = ${M}, after a training prefix of ${PREFIX} samples. The two abnormal beats match each other, so the full matrix profile is low at both and its maximum falls elsewhere. The left matrix profile compares each subsequence only with earlier ones. The first abnormal beat has no earlier match, so the left profile peaks there. The second one matches the first and stays low: once an anomaly has been seen, a repeat of it is no longer new.`}
      controls={<ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />}
      readout={
        <>
          <Readout label="abnormal beats start at" value={r.twins.join(', ')} />
          <Readout label="matrix profile maximum at" value={r.full} />
          <Readout label="left matrix profile maximum at" value={r.left} />
          <Readout label="left discord score" value={formatNumber(r.p.left[r.left])} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={top} xLabel="time" yLabel="x" height={200} />
        <XYChart series={bottom} xLabel="subsequence start i" yLabel="distance" height={220} yRange={[0, undefined]} />
      </div>
    </Interactive>
  )
}
