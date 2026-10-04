import { useMemo, useState } from 'react'
import {
  alphaBarAt,
  cosineSchedule,
  forwardNoise,
  gaussianMixtureData,
  linearSchedule,
  sampleMixture,
} from 'aifn-applied/generative/diffusion'
import { stream } from 'aifn/foundation/random'
import { toRows, type Tensor } from 'aifn/foundation/tensor'
import {
  Figure,
  ControlGroup,
  Select,
  NumberSelector,
  Player,
  Plots,
  Plot,
  Points,
  Curve,
  Handle,
  Readout,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(v) : '—')

/** A 2-D mixture of three components: two elongated blobs and a small round one. */
const MIX = gaussianMixtureData(
  [0.4, 0.35, 0.25],
  [
    [-1.5, -0.5],
    [1.2, 1.2],
    [1.5, -1.3],
  ],
  [
    [
      [0.35, 0.2],
      [0.2, 0.2],
    ],
    [0.25, 0.45],
    0.2,
  ],
)

const DATA = sampleMixture(stream('diffusion-data'), MIX, 600)

const columns = (x: Tensor) => {
  const rows = toRows(x)
  return { x: rows.map((r) => r[0]), y: rows.map((r) => r[1]) }
}

const RANGE: [number, number] = [-3.5, 3.5]
const SCHEDULES = { linear: linearSchedule(), cosine: cosineSchedule() }
type ScheduleId = keyof typeof SCHEDULES
const TICKS = Array.from({ length: 101 }, (_, k) => k * 10)

const SCHEDULE_OPTIONS = [
  { value: 'linear', label: 'linear (Ho et al.)' },
  { value: 'cosine', label: 'cosine (Nichol & Dhariwal)' },
]

const SCALES = (Object.keys(SCHEDULES) as ScheduleId[]).map((name) => ({
  name,
  signal: TICKS.map((s) => Math.sqrt(alphaBarAt(SCHEDULES[name], s))),
  noise: TICKS.map((s) => Math.sqrt(1 - alphaBarAt(SCHEDULES[name], s))),
}))

export function ForwardNoisingExplorer() {
  const [scheduleId, setScheduleId] = useState<ScheduleId>('linear')
  const [k, setK] = useState(0)

  const schedule = SCHEDULES[scheduleId]
  const t = TICKS[k]
  const ab = alphaBarAt(schedule, t)

  // The same noise draw at every t, so each point moves smoothly from its data position towards N(0, I).
  const xt = useMemo(() => columns(forwardNoise(stream('diffusion-eps'), DATA, ab).x), [ab])

  const x1 = useAxis({ label: 'x₁', range: RANGE })
  const x2 = useAxis({ label: 'x₂', range: RANGE, equal: x1 })
  const stepsAxis = useAxis({ label: 'step t', range: [0, 1000] })
  const scaleAxis = useAxis({ label: 'scale', range: [0, 1] })

  const snr = ab / Math.max(1e-9, 1 - ab)

  return (
    <Figure
      title="Forward noising of a 2-D mixture"
      purpose="The forward process shrinks data by √ᾱ_t and adds noise with standard deviation √(1 − ᾱ_t), blurring multimodal structures into standard normal noise. The cosine schedule preserves signal contrast for longer in early steps."
      defaultSize="L"
      controls={
        <ControlGroup>
          <Select
            label="Schedule"
            options={SCHEDULE_OPTIONS}
            value={scheduleId}
            onChange={(v) => setScheduleId(v as ScheduleId)}
          />
          <NumberSelector
            label="Step t"
            value={t}
            onChange={(val) => {
              const clamped = Math.max(0, Math.min(1000, val))
              setK(Math.round(clamped / 10))
            }}
            min={0}
            max={1000}
            step={10}
            suggestions={[0, 100, 250, 500, 750, 1000]}
          />
          <Player
            value={k}
            onChange={setK}
            count={TICKS.length}
            format={(i) => String(TICKS[i])}
            label="Play t"
          />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout label="t" value={String(t)} />
          <Readout label="ᾱ_t" value={fmt(ab)} />
          <Readout label="√ᾱ_t (signal)" value={fmt(Math.sqrt(ab))} />
          <Readout label="√(1 − ᾱ_t) (noise)" value={fmt(Math.sqrt(1 - ab))} />
          <Readout label="SNR" value={fmt(snr)} />
        </>
      }
      caption="600 samples from a 2-D Gaussian mixture noised under x_t = √ᾱ_t x₀ + √(1 − ᾱ_t) ε. Right: signal (solid) and noise (dashed) amplitudes for linear and cosine schedules across 1,000 diffusion steps."
    >
      <Plots cols={2} widths={[1, 1.2]}>
        <Plot x={x1} y={x2}>
          <Points name="x_t" x={xt.x} y={xt.y} slot={0} />
        </Plot>
        <Plot x={stepsAxis} y={scaleAxis}>
          {SCALES.flatMap((c, slot) => [
            <Curve key={`${c.name}-s`} name={`√ᾱ (${c.name})`} x={TICKS} y={c.signal} slot={slot} />,
            <Curve key={`${c.name}-n`} name={`√(1 − ᾱ) (${c.name})`} x={TICKS} y={c.noise} slot={slot} dashed />,
          ])}
          <Handle kind="x" at={t} label="t" onDrag={(x) => setK(Math.max(0, Math.min(100, Math.round(x / 10))))} />
        </Plot>
      </Plots>
    </Figure>
  )
}
