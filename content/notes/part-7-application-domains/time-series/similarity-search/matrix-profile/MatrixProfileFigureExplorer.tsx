import { useMemo, useState } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import {
  discords,
  distanceProfile,
  matrixProfile,
  motifs,
  scrimpProfile,
  scrimpSteps,
} from 'aifn-compute/signal/similarity'
import { motifSeries } from 'aifn-methods/data/synthetic'
import {
  Figure,
  ControlGroup,
  NumberSelector,
  Player,
  Plots,
  Plot,
  Curve,
  Points,
  Annotation,
  Handle,
  Readout,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const range = (a: number, b: number) => Array.from({ length: b - a }, (_, i) => a + i)

export function MatrixProfileFigureExplorer() {
  const [n, setN] = useState(600)
  const [mRaw, setMRaw] = useState(40)
  const [probe, setProbe] = useState(90)
  const [step, setStep] = useState(0)

  const m = Math.min(mRaw, Math.floor(n / 4))
  const series = useMemo(() => motifSeries(stream('content/matrix-profile/seed-1'), { n, m }), [n, m])
  const y = useMemo(() => Array.from(toFlat(series.y)), [series])
  const t = useMemo(() => range(0, n), [n])

  const run = useMemo(
    () =>
      trace(scrimpSteps(series.y, m, { diagonalsPerStep: 12, prescrimp: true }), undefined, 100000, {
        keep: 'all',
        stream: stream('content/scrimp'),
      }),
    [series, m],
  )

  const exact = useMemo(() => matrixProfile(series.y, m), [series, m])

  const at = Math.min(step, run.steps.length - 1)
  const s = run.steps[at]
  const mp = scrimpProfile(s, m)
  const profile = Array.from(toFlat(mp.profile)).map((v) => (Number.isFinite(v) ? v : NaN))
  const exactProfile = Array.from(toFlat(exact.profile))
  const top = motifs(mp, { count: 1 })[0]
  const odd = discords(mp, { count: 1 })[0]

  const p = Math.max(0, Math.min(n - m, Math.round(probe)))
  const dp = useMemo(() => Array.from(toFlat(distanceProfile(y.slice(p, p + m), series.y))), [y, p, m, series])
  const best = dp.reduce((b, v, i) => (Math.abs(i - p) > Math.ceil(m / 4) && v < dp[b] ? i : b), p === 0 ? m : 0)

  const windowAt = (start: number) => ({ x: t.slice(start, start + m), y: y.slice(start, start + m) })

  const tx = useAxis({ label: 'time index', range: [0, n], key: n })
  const vy = useAxis({ label: 'amplitude', hold: 'union', key: `${n}-${m}` })
  const py = useAxis({ label: 'profile (NN dist)', range: [0, Math.sqrt(4 * m)], key: m })
  const dy = useAxis({ label: `distance to [${p}, ${p + m})`, range: [0, Math.sqrt(4 * m)], key: m })
  const k = t.slice(0, n - m + 1)

  return (
    <Figure
      title="The matrix profile: motif discovery, discords, and distance profiles"
      purpose="Every subsequence records the z-normalised Euclidean distance to its nearest non-overlapping neighbour. Minima in the profile pinpoint repeated shapes (motifs); maxima locate unique, anomalous behaviour (discords)."
      defaultSize="XL"
      controls={
        <ControlGroup>
          <NumberSelector
            label="Series length n"
            value={n}
            onChange={(val) => {
              setN(val)
              setStep(0)
            }}
            min={200}
            max={1000}
            step={50}
            suggestions={[400, 600, 800]}
          />
          <NumberSelector
            label="Subsequence length m"
            value={mRaw}
            onChange={(val) => {
              setMRaw(val)
              setStep(0)
            }}
            min={10}
            max={80}
            step={5}
            suggestions={[20, 40, 60]}
          />
          <NumberSelector
            label="Probe window start"
            value={p}
            onChange={setProbe}
            min={0}
            max={n - m}
            step={5}
            suggestions={[30, 90, 200, 400]}
          />
          <Player label="SCRIMP++ diagonals" value={at} onChange={setStep} count={run.steps.length} />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout label="Top motif pair" value={top ? `[${top.a}, ${top.b}] (dist: ${fmt(top.distance)})` : '—'} />
          <Readout label="Top discord" value={odd ? `[${odd.at}] (dist: ${fmt(odd.distance)})` : '—'} />
          <Readout label="SCRIMP++ progress" value={`${Math.round((100 * at) / (run.steps.length - 1))}%`} />
        </>
      }
      caption="Top: synthetic time series with planted motifs (highlighted in color) and anomaly discord, plus dashed movable probe subsequence. Middle: exact STOMP matrix profile (dashed) against anytime SCRIMP++ profile. Bottom: MASS distance profile showing distances from the probe window across all time offsets."
    >
      <Plots rows={3} heights={[1.2, 1, 1]}>
        <Plot x={tx} y={vy}>
          <Curve name="series" x={t} y={y} muted width={1} />
          {top && <Curve name="motif instance A" x={windowAt(top.a).x} y={windowAt(top.a).y} slot={0} width={2.5} />}
          {top && <Curve name="motif instance B" x={windowAt(top.b).x} y={windowAt(top.b).y} slot={1} width={2.5} />}
          {odd && <Curve name="discord" x={windowAt(odd.at).x} y={windowAt(odd.at).y} tone="destructive" width={2.5} />}
          <Curve name="probe window" x={windowAt(p).x} y={windowAt(p).y} emphasis width={1.5} dashed />
          <Handle kind="x" at={p} onDrag={setProbe} label="window" />
        </Plot>
        <Plot x={tx} y={py}>
          <Curve name="exact (STOMP)" x={k} y={exactProfile} emphasis dashed width={1} />
          <Curve name="SCRIMP++ so far" x={k} y={profile} slot={2} />
          {top && <Points name="motif points" x={[top.a, top.b]} y={[top.distance, top.distance]} slot={0} size={8} />}
          {odd && <Points name="discord point" x={[odd.at]} y={[odd.distance]} tone="destructive" size={8} />}
        </Plot>
        <Plot x={tx} y={dy}>
          <Curve name="distance profile" x={k} y={dp} slot={3} />
          <Points name="nearest match" x={[best]} y={[dp[best]]} emphasis size={8} />
          <Annotation x={p} text="query" dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
