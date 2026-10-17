import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { onesUntilZero, parityAccuracyByLength, parityStates, type ParityReadout } from 'aifn-methods/neural/sequence'
import { parityStrings } from 'aifn-methods/data/synthetic'

const SHOWN = 24
const READOUTS = [
  { value: 'threshold', label: 'one threshold on the state' },
  { value: 'any', label: 'any function of the state' },
] as const

/** Parity tracked by the one-state recurrence h_t = a(x_t) h_{t-1}, a(0) = 1, a(1) = λ, at finite precision. */
export function ParityTracking() {
  const state = useFigureState({
    lambda: slider(-1, 1, 0.9, { step: 0.01, label: 'transition λ for a one bit' }),
    readout: choice<ParityReadout>(READOUTS, 'threshold', { label: 'readout' }),
    precision: int(8, { min: 2, max: 30, suggestions: [4, 8, 16], label: 'binary digits kept b' }),
    maxLength: int(256, { min: 8, max: 1024, suggestions: [64, 256, 512], label: 'longest string T' }),
    seed: int(3, { min: 0, max: 999, suggestions: [1, 2, 3], label: 'seed of the example string' }),
  })
  const { lambda, readout, precision, maxLength, seed } = state

  const lengths = useMemo(() => Array.from({ length: maxLength }, (_, i) => i + 1), [maxLength])
  const curves = useMemo(() => {
    const at = (l: number) => parityAccuracyByLength({ lambda: l, precision, maxLength, readout }).accuracy
    return { chosen: at(lambda), opposite: at(-lambda), flip: at(-1) }
  }, [lambda, precision, maxLength, readout])

  const example = useMemo(() => {
    const bits = parityStrings(stream(seed), { length: SHOWN, n: 1 }).bits[0]
    const t = Array.from({ length: SHOWN + 1 }, (_, i) => i)
    const ones = bits.flatMap((b, i) => (b === 1 ? [i + 1] : []))
    return { bits, t, ones }
  }, [seed])
  const hChosen = useMemo(() => [1, ...parityStates(example.bits, lambda, precision)], [example, lambda, precision])
  const hFlip = useMemo(() => [1, ...parityStates(example.bits, -1, precision)], [example, precision])

  const tAxis = useAxis({ label: 'position t in the example string', range: [0, SHOWN], integer: true })
  const hAxis = useAxis({ label: 'state h_t', range: [-1.05, 1.05] })
  const lAxis = useAxis({ label: 'string length T', range: [1, maxLength], integer: true })
  const aAxis = useAxis({ label: 'parity accuracy, all strings of length T', range: [0.45, 1.02] })
  const at = (T: number) => curves.chosen[Math.min(T, maxLength) - 1]

  return (
    <Figure
      title="Tracking parity with a one-state linear recurrence"
      state={state}
      caption="The state starts at 1 and is multiplied by λ at every one bit; zero bits leave it alone. After the rounding to b binary digits (toward zero), the state depends only on the number of ones. Top: the state over an example string, with the one bits marked, for the chosen λ and for λ = −1. Bottom: the best parity accuracy over all strings of each length, for the chosen λ, for −λ and for λ = −1, read by one threshold on the final state (what a linear output layer can do with one number) or by any function of it. With λ = −1 the sign is the parity at every length. A positive λ never changes the sign, so one threshold falls towards chance as T grows (0.60 at T = 16 for λ = 0.9); any function of the state helps only until λ^c falls below the last digit and every longer count collapses to 0. A negative λ with |λ| < 1 carries the parity in its sign, but loses it at the same count. Slide λ and raise b or T."
      readouts={
        <>
          <Readout label="accuracy at T = 16" value={formatNumber(at(16))} />
          <Readout label="accuracy at T = 64" value={formatNumber(at(64))} />
          <Readout label={`accuracy at T = ${maxLength}`} value={formatNumber(at(maxLength))} />
          <Readout
            label="ones before the state reaches 0"
            value={Math.abs(lambda) >= 1 ? 'never' : String(onesUntilZero(lambda, precision))}
          />
        </>
      }
    >
      <Plots rows={2}>
        <Plot x={tAxis} y={hAxis}>
          <Curve name={`h_t, λ = ${formatNumber(lambda)}`} x={example.t} y={hChosen} slot={0} showPoints />
          <Curve name="h_t, λ = −1" x={example.t} y={hFlip} slot={1} dashed />
          <Points name="one bits" x={example.ones} y={example.ones.map((t) => hChosen[t])} emphasis size={7} />
        </Plot>
        <Plot x={lAxis} y={aAxis}>
          <Curve name={`λ = ${formatNumber(lambda)}`} x={lengths} y={curves.chosen} slot={0} />
          <Curve name={`λ = ${formatNumber(-lambda)}`} x={lengths} y={curves.opposite} slot={2} />
          <Curve name="λ = −1" x={lengths} y={curves.flip} slot={1} dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
