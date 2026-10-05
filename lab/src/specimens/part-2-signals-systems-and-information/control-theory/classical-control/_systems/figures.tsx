import { bode, margins, transferFunction } from 'aifn-compute/systems'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { useMemo } from 'react'
import { Equation, Figure, live, tex } from 'aifn-render/layout'
import { slider, useFigureState, variants } from 'aifn-render/state'
import { Annotation, Curve, Plot, Plots, Readout, formatNumber, useAxis } from 'aifn-render/viz'

const fmt = (v: number) => formatNumber(v)

// ── Bode plot and stability margins ──────────────────────────────────────────────────────────────────────────────────

const LOOPS = {
  third: { label: 'K / (s(s + 1)(s + 2))', num: [1], den: [1, 3, 2, 0], delay: 0, k: 2 },
  lag: { label: 'K / (s + 1)³', num: [1], den: [1, 3, 3, 1], delay: 0, k: 4 },
  delay: { label: 'K e^{−0.5s} / (s + 1)', num: [1], den: [1, 1], delay: 0.5, k: 1.5 },
} as const
const gain = (k: number) => ({ logK: slider(-1, 1.5, Math.log10(k), { label: 'log₁₀ gain K', step: 0.02 }) })
const LOOP = variants(
  {
    third: { label: LOOPS.third.label, params: gain(LOOPS.third.k) },
    lag: { label: LOOPS.lag.label, params: gain(LOOPS.lag.k) },
    delay: { label: LOOPS.delay.label, params: gain(LOOPS.delay.k) },
  },
  { label: 'loop', choiceLabel: 'open loop L(s)' },
)
const W = Array.from({ length: 600 }, (_, i) => 10 ** (-2 + (4 * i) / 599))

export function BodeMarginsSpecimen() {
  const state = useFigureState({ loop: LOOP })
  const which = state.loop.key
  const logK = state.loop.values.logK
  const L = useMemo(() => {
    const loop = LOOPS[which]
    return transferFunction(
      loop.num.map((v) => v * 10 ** logK),
      loop.den,
      { delay: loop.delay },
    )
  }, [which, logK])
  const b = useMemo(() => bode(L, W), [L])
  const m = useMemo(() => margins(L), [L])
  const phase = useMemo(() => toFlat(b.phase), [b])
  const db = useMemo(() => toFlat(b.magnitudeDb), [b])
  const gmDb = m.gainMarginDb
  const x = useAxis({ label: 'ω (rad/s)', log: true, range: [W[0], W[W.length - 1]] })
  const yMag = useAxis({ label: 'dB', hold: 'union', key: which })
  const yPh = useAxis({ label: 'degrees', hold: 'union', key: which })
  const hasGm = Number.isFinite(m.phaseCrossover)
  const hasPm = Number.isFinite(m.gainCrossover)
  return (
    <Figure
      title="Gain and phase margins on the Bode plot"
      purpose="The gain margin is how far |L| sits below 0 dB where the phase reaches −180°, and the phase margin how far the phase sits above −180° where |L| = 0 dB; raising the gain K shrinks both."
      defaultSize="L"
      state={state}
      equation={
        <Equation>
          {tex`\text{GM} = -20\log_{10}|L(j\omega_{pc})| = ${live(hasGm ? fmt(gmDb) : '\\infty', { strong: true })}\,\text{dB}, \qquad \text{PM} = 180^\circ + \angle L(j\omega_{gc}) = ${live(Number.isFinite(m.phaseMargin) ? fmt(m.phaseMargin) : '\\infty', { strong: true })}^\circ`}
        </Equation>
      }
      readouts={{
        margins: (
          <>
            <Readout label="K" value={fmt(10 ** logK)} />
            <Readout label="phase crossover ω_pc" value={hasGm ? fmt(m.phaseCrossover) : 'none'} />
            <Readout label="gain crossover ω_gc" value={hasPm ? fmt(m.gainCrossover) : 'none'} />
            <Readout label="delay margin" value={Number.isFinite(m.delayMargin) ? `${fmt(m.delayMargin)} s` : '∞'} />
            <Readout label="closed loop" value={m.gainMargin > 1 && m.phaseMargin > 0 ? 'stable' : 'unstable'} />
          </>
        ),
      }}
      caption="The coloured vertical segments are the margins: on the magnitude panel from |L| up to 0 dB at the phase crossover, on the phase panel from −180° up to ∠L at the gain crossover. A negative margin (segment pointing the other way) means the unity-feedback loop is unstable. The delay adds −ωτ to the phase without changing the magnitude."
    >
      <Plots rows={2} hoverGroup>
        <Plot x={x} y={yMag}>
          <Curve name="|L(jω)| (dB)" x={W} y={db} slot={0} />
          <Annotation y={0} />
          {hasGm && (
            <Curve
              name="gain margin"
              x={[m.phaseCrossover, m.phaseCrossover]}
              y={[-gmDb, 0]}
              slot={3}
              showPoints
              width={3}
            />
          )}
        </Plot>
        <Plot x={x} y={yPh}>
          <Curve name="∠L(jω) (°)" x={W} y={phase} slot={1} />
          <Annotation y={-180} />
          {hasPm && (
            <Curve
              name="phase margin"
              x={[m.gainCrossover, m.gainCrossover]}
              y={[-180, m.phaseMargin - 180]}
              slot={4}
              showPoints
              width={3}
            />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
