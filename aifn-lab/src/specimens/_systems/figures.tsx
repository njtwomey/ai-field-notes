import { bode, margins, transferFunction } from 'aifn/systems'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { useMemo, useState } from 'react'
import { Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, formatNumber, type XYSeries } from '@lab/viz'

const flat = (t: Tensor) => toFlat(t)
const fmt = (v: number) => formatNumber(v)

// ── Bode plot and stability margins ──────────────────────────────────────────────────────────────────────────────────

const LOOPS = [
  { value: 'third', label: 'K / (s(s + 1)(s + 2))', num: [1], den: [1, 3, 2, 0], delay: 0, k: 2 },
  { value: 'lag', label: 'K / (s + 1)³', num: [1], den: [1, 3, 3, 1], delay: 0, k: 4 },
  { value: 'delay', label: 'K e^{−0.5s} / (s + 1)', num: [1], den: [1, 1], delay: 0.5, k: 1.5 },
] as const
type LoopName = (typeof LOOPS)[number]['value']
const W = Array.from({ length: 600 }, (_, i) => 10 ** (-2 + (4 * i) / 599))

export function BodeMarginsSpecimen() {
  const [which, setWhich] = useState<LoopName>('third')
  const loop = LOOPS.find((l) => l.value === which)!
  const [logK, setLogK] = useState(Math.log10(loop.k))
  const L = useMemo(
    () =>
      transferFunction(
        loop.num.map((v) => v * 10 ** logK),
        loop.den,
        { delay: loop.delay },
      ),
    [loop, logK],
  )
  const b = useMemo(() => bode(L, W), [L])
  const m = useMemo(() => margins(L), [L])
  const phase = flat(b.phase)
  const db = flat(b.magnitudeDb)
  const gmDb = m.gainMarginDb
  const mag: XYSeries[] = [
    { name: '|L(jω)| (dB)', type: 'line', x: W, y: db, slot: 0 },
    { name: '0 dB', type: 'line', x: [W[0], W[W.length - 1]], y: [0, 0], muted: true },
  ]
  const ph: XYSeries[] = [
    { name: '∠L(jω) (°)', type: 'line', x: W, y: phase, slot: 1 },
    { name: '−180°', type: 'line', x: [W[0], W[W.length - 1]], y: [-180, -180], muted: true },
  ]
  if (Number.isFinite(m.phaseCrossover)) {
    mag.push({
      name: 'gain margin',
      type: 'line',
      x: [m.phaseCrossover, m.phaseCrossover],
      y: [-gmDb, 0],
      slot: 3,
      showPoints: true,
    })
  }
  if (Number.isFinite(m.gainCrossover)) {
    ph.push({
      name: 'phase margin',
      type: 'line',
      x: [m.gainCrossover, m.gainCrossover],
      y: [-180, m.phaseMargin - 180],
      slot: 4,
      showPoints: true,
    })
  }
  return (
    <Figure
      title="Gain and phase margins on the Bode plot"
      description="The gain margin is how far |L| sits below 0 dB where the phase reaches −180°, and the phase margin how far the phase sits above −180° where |L| = 0 dB; raising the gain K shrinks both."
      defaultSize="L"
      controls={
        <ControlRow label="loop">
          <Select
            label="open loop L(s)"
            value={which}
            onChange={(v) => {
              setWhich(v)
              setLogK(Math.log10(LOOPS.find((l) => l.value === v)!.k))
            }}
            options={LOOPS.map((l) => ({ value: l.value, label: l.label }))}
          />
          <Slider label="log₁₀ gain K" value={logK} min={-1} max={1.5} step={0.02} onChange={setLogK} />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="K" value={fmt(10 ** logK)} />
          <Readout
            label="gain margin"
            value={Number.isFinite(gmDb) ? `${fmt(gmDb)} dB at ω = ${fmt(m.phaseCrossover)}` : '∞'}
          />
          <Readout
            label="phase margin"
            value={Number.isFinite(m.phaseMargin) ? `${fmt(m.phaseMargin)}° at ω = ${fmt(m.gainCrossover)}` : '∞'}
          />
          <Readout label="delay margin" value={Number.isFinite(m.delayMargin) ? `${fmt(m.delayMargin)} s` : '∞'} />
          <Readout label="closed loop" value={m.gainMargin > 1 && m.phaseMargin > 0 ? 'stable' : 'unstable'} />
        </>
      }
      caption="The coloured vertical segments are the margins: on the magnitude panel from |L| up to 0 dB at the phase crossover, on the phase panel from −180° up to ∠L at the gain crossover. A negative margin (segment pointing the other way) means the unity-feedback loop is unstable. The delay adds −ωτ to the phase without changing the magnitude."
    >
      <Subplots rows={2} sharex xLog hoverGroup axisKey={which} rescaleOnChange={false}>
        <Panel>
          <XYChart series={mag} xLog xLabel="ω (rad/s)" yLabel="dB" />
        </Panel>
        <Panel>
          <XYChart series={ph} xLog xLabel="ω (rad/s)" yLabel="degrees" />
        </Panel>
      </Subplots>
    </Figure>
  )
}
