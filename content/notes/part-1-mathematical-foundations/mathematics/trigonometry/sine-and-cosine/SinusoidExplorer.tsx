import { useMemo } from 'react'
import {
  Annotation,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { grid, TAU, wrapPositive, wrapSigned } from '../_shared/geometry'

const SPAN = 2 * TAU
const T = grid(0, SPAN, 481)
const REF = T.map(Math.sin)
const A_MAX = 2

/**
 * The sinusoid A sin(ωt + φ) against the reference sin t. Its first crest after t = 0 is draggable: height sets the
 * amplitude A and position sets the phase φ for the current angular frequency ω.
 */
export function SinusoidExplorer() {
  const state = useFigureState({
    amp: slider(0.1, A_MAX, 1.5, { step: 0.05, label: 'amplitude A' }),
    omega: slider(0.25, 4, 2, { step: 0.05, label: 'angular frequency ω (rad per unit t)' }),
    phi: slider(-Math.PI, Math.PI, 0.6, { step: 0.01, label: 'phase φ (rad)' }),
  })
  const { amp, omega, phi } = state

  const wave = useMemo(() => T.map((t) => amp * Math.sin(omega * t + phi)), [amp, omega, phi])
  // The first crest at t ≥ 0 solves ωt + φ = π/2 + 2πk.
  const crest = wrapPositive(Math.PI / 2 - phi) / omega
  const period = TAU / omega

  const x = useAxis({ label: 't', range: [0, SPAN] })
  const y = useAxis({ label: 'y', range: [-A_MAX - 0.2, A_MAX + 0.2] })
  return (
    <Figure
      title="Amplitude, frequency and phase"
      state={state}
      caption="The blue curve is y = A sin(ωt + φ); the dashed grey curve is the reference sin t, with A = 1, ω = 1 and φ = 0. Drag the first crest: its height is the amplitude A and its position sets the phase φ. The angular frequency ω squeezes the curve horizontally, so one cycle lasts T = 2π/ω. The phase moves the curve left by φ/ω; the vertical line marks the start of the first full cycle after t = 0, where the argument ωt + φ is a multiple of 2π."
      readouts={
        <>
          <Readout label="period T = 2π/ω" value={formatNumber(period)} />
          <Readout label="frequency f = ω/2π" value={formatNumber(omega / TAU)} />
          <Readout label="shift left φ/ω" value={formatNumber(phi / omega)} />
          <Readout label="first crest at t" value={formatNumber(crest)} />
        </>
      }
    >
      <Plot x={x} y={y}>
        <Curve name="sin t" x={T} y={REF} muted dashed />
        <Curve name="A sin(ωt + φ)" x={T} y={wave} slot={0} />
        <Annotation x={wrapPositive(-phi) / omega} text="cycle start" dashed muted />
        <Handle
          kind="point"
          at={[crest, amp]}
          label="crest"
          onDrag={([u, v]) => {
            state.set('amp', Math.min(A_MAX, Math.max(0.1, v)))
            state.set('phi', wrapSigned(Math.PI / 2 - omega * u))
          }}
        />
      </Plot>
    </Figure>
  )
}
