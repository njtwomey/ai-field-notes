import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type Segment } from 'aifn-render'
import { ACTIVITIES, DAYS, SENSORS, fit, segment, simulate, topicToActivity } from './smarthome'

const TOPIC_ROW = -1.5
const TRUTH_ROW = -3

/** A fortnight of simulated sensor events, segmented into documents and clustered into activity-topics. */
export function SmartHomeTopics() {
  const [threshold, setThreshold] = useState(5)
  const [K, setK] = useState(7)
  const [seed, setSeed] = useState(1)
  const day = useParam(3, { min: 1, max: DAYS, step: 1 })

  const events = useMemo(() => simulate(seed), [seed])
  const docs = useMemo(() => segment(events, threshold), [events, threshold])
  const model = useMemo(() => fit(events, docs, K, false, seed, 25), [events, docs, K, seed])
  const toActivity = useMemo(() => topicToActivity(events, docs, model.z, K), [events, docs, model, K])

  const docOf = useMemo(() => {
    const out = new Array<number>(events.length)
    docs.forEach(([s, e], d) => {
      for (let i = s; i <= e; i++) out[i] = d
    })
    return out
  }, [events, docs])

  const agreement = useMemo(() => {
    let ok = 0
    events.forEach((ev, i) => {
      if (toActivity[model.z[docOf[i]]] === ev.activity) ok++
    })
    return ok / events.length
  }, [events, docOf, model, toActivity])
  const occurrences = useMemo(
    () => events.reduce((n, ev, i) => n + (i === 0 || events[i - 1].activity !== ev.activity ? 1 : 0), 0),
    [events],
  )

  const lo = (day.value - 1) * 1440
  const idx = events.flatMap((ev, i) => (ev.t >= lo && ev.t < lo + 1440 && ev.word % 2 === 0 ? [i] : []))
  const hour = (i: number) => (events[i].t - lo) / 60
  const bounds: Segment[] = docs
    .filter(([s]) => events[s].t >= lo && events[s].t < lo + 1440)
    .map(([s]) => ({ from: [hour(s), TRUTH_ROW - 0.7], to: [hour(s), SENSORS.length + 0.5] }))
  const docsToday = bounds.length

  const topicRows = model.unigram.map((row, k) => {
    const bySensor = SENSORS.map((_, s) => row[2 * s] + row[2 * s + 1])
    const top = bySensor
      .map((p, s) => [p, s] as const)
      .sort((a, b) => b[0] - a[0])
      .slice(0, 3)
      .map(([p, s]) => `${SENSORS[s].name} ${Math.round(100 * p)}%`)
    return { k, activity: ACTIVITIES[toActivity[k]], docs: model.docCount[k], top }
  })

  return (
    <Interactive
      title="Activity discovery from a fortnight of smart-home sensor events"
      caption="A simulated resident's 14 days in a five-room home with 14 binary sensors. Top rows: the ON events of one day, one row per sensor (numbered from the bed at 1 to the front door at 14). Vertical lines are document boundaries from the segmentation algorithm: a new document starts when the location changes and the current document has lasted longer than the threshold. The model gives each document one topic, fitted to all 14 days by collapsed Gibbs sampling on sensor-event unigrams. Row −1.5 colours each event by its document's topic, named after the activity that produced most of that topic's events; row −3 shows the true activity. A threshold of zero gives one document per room visit and the best agreement; longer thresholds merge short visits, which lowers the number of fragments per activity but mixes activities within documents. Step through the days with the arrows."
      controls={
        <>
          <ParamSlider label="day" param={day} withArrows />
          <ParamSlider
            label="minimum document duration t_th (min)"
            value={threshold}
            onChange={setThreshold}
            min={0}
            max={60}
            step={1}
          />
          <ParamSlider label="topics K" value={K} onChange={setK} min={3} max={10} step={1} />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={1} max={10} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="documents (14 days)" value={docs.length} />
          <Readout label="documents today" value={docsToday} />
          <Readout label="documents per activity occurrence" value={formatNumber(docs.length / occurrences)} />
          <Readout label="events whose topic matches their activity" value={`${formatNumber(100 * agreement)}%`} />
        </>
      }
    >
      <XYChart
        height={380}
        xLabel="hour of day"
        yLabel="sensor · topic · truth"
        xRange={[0, 24]}
        yRange={[-4, 15]}
        segments={bounds}
        series={[
          {
            name: 'sensor ON events',
            type: 'scatter',
            x: idx.map(hour),
            y: idx.map((i) => events[i].sensor + 1),
            muted: true,
          },
          {
            name: 'topic',
            type: 'scatter',
            x: idx.map(hour),
            y: idx.map(() => TOPIC_ROW),
            group: idx.map((i) => toActivity[model.z[docOf[i]]]),
            groupNames: [...ACTIVITIES],
          },
          {
            name: 'truth',
            type: 'scatter',
            x: idx.map(hour),
            y: idx.map(() => TRUTH_ROW),
            group: idx.map((i) => events[i].activity),
            groupNames: [...ACTIVITIES],
          },
        ]}
      />
      <table className="mt-3 w-full text-xs">
        <thead className="text-muted-foreground">
          <tr className="text-left">
            <th className="py-1 pr-3 font-normal">topic</th>
            <th className="py-1 pr-3 font-normal">mostly</th>
            <th className="py-1 pr-3 font-normal">documents</th>
            <th className="py-1 font-normal">most probable sensors</th>
          </tr>
        </thead>
        <tbody>
          {topicRows.map((t) => (
            <tr key={t.k} className="border-t border-border">
              <td className="py-1 pr-3 tabular-nums">{t.k + 1}</td>
              <td className="py-1 pr-3">{t.activity}</td>
              <td className="py-1 pr-3 tabular-nums">{t.docs}</td>
              <td className="py-1">{t.top.join(', ')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Interactive>
  )
}
