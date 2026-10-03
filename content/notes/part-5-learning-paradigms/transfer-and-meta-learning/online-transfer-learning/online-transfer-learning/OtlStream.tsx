import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'

// Homogeneous online transfer learning (Zhao & Hoi, 2010) on a stream of 2-D points. The source classifier h is a
// fixed line; the target learner f is a linear passive-aggressive classifier; the ensemble mixes the two with
// exponential weights on the squared loss of their predictions mapped to [0, 1].
const T = 400
const SOURCE_ANGLE = (20 * Math.PI) / 180
const NOISE = 0.05
const C = 0.5
const ETA = 0.5
const SOURCE_SCALE = 3

const STREAM = (() => {
  const r = rng(2)
  return Array.from({ length: T }, () => ({ x: [2 * r.uniform() - 1, 2 * r.uniform() - 1], flip: r.uniform() < NOISE }))
})()

const clip01 = (z: number) => Math.min(1, Math.max(0, (z + 1) / 2))
const sign = (z: number) => (z >= 0 ? 1 : -1)

function simulate(shiftDeg: number, revert: boolean) {
  const us = [Math.cos(SOURCE_ANGLE), Math.sin(SOURCE_ANGLE)]
  const a = SOURCE_ANGLE + (shiftDeg * Math.PI) / 180
  const ut = [Math.cos(a), Math.sin(a)]
  let w1 = 0.5
  let w2 = 0.5
  let f = [0, 0, 0]
  let M = 0
  let Mh = 0
  let Mf = 0
  let Sh = 0
  let Sf = 0
  const rec = { t: [0], M: [0], Mh: [0], Mf: [0], wh: [0.5], bound: [8 * Math.log(2)], f: [f] }
  const labels: number[] = []
  STREAM.forEach(({ x, flip }, i) => {
    const u = revert && i >= T / 2 ? us : ut
    const clean = sign(u[0] * x[0] + u[1] * x[1])
    const y = flip ? -clean : clean
    labels.push(y)
    const hs = SOURCE_SCALE * (us[0] * x[0] + us[1] * x[1])
    const fs = f[0] * x[0] + f[1] * x[1] + f[2]
    const p = w1 * clip01(hs) + w2 * clip01(fs)
    if ((p >= 0.5 ? 1 : -1) !== y) M++
    if (sign(hs) !== y) Mh++
    if (sign(fs) !== y) Mf++
    const target = clip01(y)
    const lh = (clip01(hs) - target) ** 2
    const lf = (clip01(fs) - target) ** 2
    Sh += lh
    Sf += lf
    const a1 = w1 * Math.exp(-ETA * lh)
    const a2 = w2 * Math.exp(-ETA * lf)
    w1 = a1 / (a1 + a2)
    w2 = a2 / (a1 + a2)
    // Passive-aggressive update (PA-I) of the target learner on the hinge loss.
    const loss = Math.max(0, 1 - y * fs)
    if (loss > 0) {
      const tau = Math.min(C, loss / (x[0] * x[0] + x[1] * x[1] + 1))
      f = [f[0] + tau * y * x[0], f[1] + tau * y * x[1], f[2] + tau * y]
    }
    rec.t.push(i + 1)
    rec.M.push(M)
    rec.Mh.push(Mh)
    rec.Mf.push(Mf)
    rec.wh.push(w1)
    rec.bound.push(4 * Math.min(Sh, Sf) + 8 * Math.log(2))
    rec.f.push(f)
  })
  return { rec, labels, us, ut }
}

/** The line θ₀x₁ + θ₁x₂ + θ₂ = 0 between x₁ = −1 and x₁ = 1; empty while the line is undefined. */
const line = (th: number[]) =>
  Math.abs(th[1]) < 1e-9 ? { x: [], y: [] } : { x: [-1, 1], y: [-1, 1].map((x) => -(th[0] * x + th[2]) / th[1]) }

export function OtlStream() {
  const [shift, setShift] = useState(30)
  const [revert, setRevert] = useState(false)
  const t = useParam(100, { min: 0, max: T, step: 10 })

  const sim = useMemo(() => simulate(shift, revert), [shift, revert])

  const scatter = useMemo<XYSeries[]>(() => {
    const lo = Math.max(0, t.value - 80)
    const pts = STREAM.slice(lo, t.value)
    const f = sim.rec.f[t.value]
    const current = revert && t.value > T / 2 ? sim.us : sim.ut
    return [
      {
        name: 'recent points',
        type: 'scatter',
        x: pts.map((p) => p.x[0]),
        y: pts.map((p) => p.x[1]),
        group: sim.labels.slice(lo, t.value).map((y) => (y > 0 ? 1 : 0)),
        groupNames: ['y = −1', 'y = +1'],
      },
      { name: 'target concept', type: 'line', ...line([current[0], current[1], 0]), emphasis: true },
      { name: 'source classifier h', type: 'line', ...line([sim.us[0], sim.us[1], 0]), slot: 2 },
      { name: 'target learner f', type: 'line', ...line(f), slot: 3, dashed: true },
    ]
  }, [sim, t.value, revert])

  const mistakes = useMemo<XYSeries[]>(
    () => [
      { name: 'source classifier h', type: 'line', x: sim.rec.t, y: sim.rec.Mh, slot: 2 },
      { name: 'target learner f', type: 'line', x: sim.rec.t, y: sim.rec.Mf, slot: 3 },
      { name: 'OTL ensemble', type: 'line', x: sim.rec.t, y: sim.rec.M, slot: 4 },
      { name: 'mistake bound', type: 'line', x: sim.rec.t, y: sim.rec.bound, dashed: true, muted: true },
    ],
    [sim],
  )
  const weights = useMemo<XYSeries[]>(
    () => [
      { name: 'weight on h', type: 'line', x: sim.rec.t, y: sim.rec.wh, slot: 2 },
      { name: 'weight on f', type: 'line', x: sim.rec.t, y: sim.rec.wh.map((w) => 1 - w), slot: 3 },
    ],
    [sim],
  )
  const handle = [{ kind: 'x' as const, at: t.value, label: 'time', onDrag: t.set }]
  const i = t.value

  return (
    <Interactive
      title="Online transfer learning on a shifted stream"
      caption="The target concept is the source concept rotated by the shift angle, with 5% label noise. The ensemble puts its weight on whichever of h and f has had the smaller squared loss so far, so it follows h while f is still learning and switches to f once f is better. Its mistakes stay under 4 min(Σh, Σf) + 8 ln 2 (dashed). Step through time with the arrows or drag the time line."
      controls={
        <>
          <ParamSlider label="shift angle (degrees)" value={shift} onChange={setShift} min={0} max={90} step={5} />
          <ParamSlider label="time t" param={t} withArrows format={(v) => String(v)} />
          <ParamSwitch label="target reverts to the source concept at t = 200" checked={revert} onChange={setRevert} />
        </>
      }
      readout={
        <>
          <Readout label="mistakes of h" value={sim.rec.Mh[i]} />
          <Readout label="mistakes of f" value={sim.rec.Mf[i]} />
          <Readout label="mistakes of OTL" value={sim.rec.M[i]} />
          <Readout label="bound" value={formatNumber(sim.rec.bound[i])} />
          <Readout label="weight on h" value={formatNumber(sim.rec.wh[i])} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <XYChart series={scatter} xLabel="x₁" yLabel="x₂" xRange={[-1, 1]} yRange={[-1, 1]} equalAspect />
        <div className="flex flex-col gap-2">
          <XYChart
            series={mistakes}
            xLabel="t"
            yLabel="cumulative mistakes"
            xRange={[0, T]}
            height={200}
            handles={handle}
          />
          <XYChart
            series={weights}
            xLabel="t"
            yLabel="ensemble weight"
            xRange={[0, T]}
            yRange={[0, 1]}
            height={170}
            handles={handle}
          />
        </div>
      </div>
    </Interactive>
  )
}
