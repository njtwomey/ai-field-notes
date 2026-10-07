import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const P = Array.from({ length: 197 }, (_, i) => 0.01 + i * 0.005)

type Rule = 'log' | 'brier' | 'wrong'

/** The average penalty per message when k of 10 look-alike messages are billing and the model says billing with p. */
function penalty(rule: Rule, k: number, p: number): number {
  const a = k / 10
  if (rule === 'log') return -(a * Math.log(p) + (1 - a) * Math.log(1 - p))
  if (rule === 'brier') return a * (1 - p) ** 2 + (1 - a) * p ** 2
  // Right or wrong only: the model's answer is drawn with these probabilities and only mistakes are counted.
  return a * (1 - p) + (1 - a) * p
}

export function HonestPenalty() {
  const state = useFigureState({
    k: int(7, { min: 0, max: 10, label: 'billing messages, k of 10' }),
    rule: choice(
      [
        { value: 'log', label: 'log loss (the usual training penalty)' },
        { value: 'brier', label: 'squared error (Brier)' },
        { value: 'wrong', label: 'right or wrong only' },
      ],
      'log',
      { label: 'penalty' },
    ),
    p: slider(0.01, 0.99, 0.5, { step: 0.01, onChart: true }),
  })
  const rule = state.rule as Rule
  const { k, p } = state
  const curve = useMemo(() => P.map((x) => penalty(rule, k, x)), [rule, k])
  const best = P[curve.indexOf(Math.min(...curve))]

  const xAxis = useAxis({ label: 'probability the model gives to "billing"', range: [0, 1] })
  const yAxis = useAxis({ label: 'average penalty per message', range: [0, undefined], hold: 'union', key: rule })
  return (
    <Figure
      title="Why the training penalty makes probabilities honest"
      state={state}
      caption="Ten messages look the same to the model, and k of them really are billing. Whatever the model says for one, it says for all ten. Drag the marker to choose what it says. With log loss or squared error the penalty is lowest exactly at k/10: the honest answer wins, not certainty. With a right-or-wrong penalty the lowest point is always at 0 or 1, so a model trained that way learns to sound certain."
      readouts={
        <>
          <Readout label="model says" value={formatNumber(p)} />
          <Readout label="penalty" value={formatNumber(penalty(rule, k, p))} />
          <Readout label="best thing to say" value={formatNumber(best)} />
          <Readout label="honest answer k/10" value={formatNumber(k / 10)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve name="penalty" x={P} y={curve} slot={0} />
          <Points name="lowest penalty" x={[best]} y={[Math.min(...curve)]} emphasis size={9} />
          <Handle {...state.handle('p', { label: 'model says' })} />
        </Plot>
      </div>
    </Figure>
  )
}
