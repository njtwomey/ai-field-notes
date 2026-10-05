import { useMemo } from 'react'
import { choice, Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { stream, uniform } from 'aifn-compute/foundation/random'

const T = 100
type Sequence = 'alternating' | 'random'

/** Gradients z_t of the linear losses f_t(w) = z_t w on [-1, 1]. */
function gradients(sequence: Sequence, seed: number): number[] {
  if (sequence === 'alternating') return Array.from({ length: T }, (_, t) => (t === 0 ? 0.5 : t % 2 === 1 ? -1 : 1))
  const r = stream(seed)
  return Array.from({ length: T }, () => (uniform(r) < 0.5 ? -1 : 1))
}

/** FTL and FTRL with ψ(w) = w²/2 on [-1, 1]: both are functions of the cumulative gradient S_t alone. */
function play(z: number[], eta: number) {
  const ftl: number[] = []
  const ftrl: number[] = []
  const regretFtl: number[] = []
  const regretFtrl: number[] = []
  let s = 0
  let lossFtl = 0
  let lossFtrl = 0
  for (let t = 0; t < T; t++) {
    const wFtl = s === 0 ? 0 : -Math.sign(s)
    const wFtrl = Math.max(-1, Math.min(1, -eta * s))
    ftl.push(wFtl)
    ftrl.push(wFtrl)
    lossFtl += z[t] * wFtl
    lossFtrl += z[t] * wFtrl
    s += z[t]
    // The best fixed decision in hindsight sits at the end of [-1, 1] opposite the sign of S_t.
    const best = -Math.abs(s)
    regretFtl.push(lossFtl - best)
    regretFtrl.push(lossFtrl - best)
  }
  return { ftl, ftrl, regretFtl, regretFtrl }
}

const ROUNDS = Array.from({ length: T }, (_, t) => t + 1)

/** Follow the leader against follow the regularised leader on one-dimensional linear losses. */
export function FtlFigure() {
  const state = useFigureState({
    sequence: choice<Sequence>(
      [
        { value: 'alternating', label: 'alternating' },
        { value: 'random', label: 'random ±1' },
      ],
      'alternating',
      { label: 'gradients z_t' },
    ),
    logEta: float(Math.log10(1 / Math.sqrt(2 * T)), {
      min: -3,
      max: 1,
      step: 0.05,
      label: 'learning rate η',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    seed: int(1, {
      min: 1,
      max: 20,
      step: 1,
      label: 'seed',
      format: (v) => String(v),
      when: (v) => v.sequence === 'random',
    }),
  })
  const eta = 10 ** state.logEta
  const z = useMemo(() => gradients(state.sequence, state.seed), [state.sequence, state.seed])
  const result = useMemo(() => play(z, eta), [z, eta])

  const decisions = [
    { name: 'FTL', x: ROUNDS, y: result.ftl, slot: 1 },
    { name: 'FTRL', x: ROUNDS, y: result.ftrl, slot: 0 },
  ] as const
  const regret = [
    { name: 'FTL', x: ROUNDS, y: result.regretFtl, slot: 1 },
    { name: 'FTRL', x: ROUNDS, y: result.regretFtrl, slot: 0 },
    {
      name: 'FTRL bound 1/(2η) + ηt',
      x: ROUNDS,
      y: ROUNDS.map((t) => 1 / (2 * eta) + eta * t),
      emphasis: true,
      dashed: true,
    },
  ] as const

  const xAxis = useAxis({ label: 'round t', range: [1, T] })
  const yAxis = useAxis({ label: 'decision w_t', range: [-1.05, 1.05] })
  const xAxis2 = useAxis({ label: 'round t', range: [1, T] })
  const yAxis2 = useAxis({ label: 'regret against best fixed w', hold: 'union' })
  return (
    <Figure
      title="Following the leader, with and without a regulariser"
      state={state}
      caption="Linear losses f_t(w) = z_t w on the interval [−1, 1]. With alternating gradients (z₁ = ½, then −1, +1, …) the leader flips every round to the end that the next loss punishes, and FTL's regret grows by 1 per round. FTRL with ψ(w) = w²/2 plays −ηS_t, a damped version of the same decision, and its regret stays under the bound 1/(2η) + ηt. A large η makes FTRL behave like FTL; a tiny η makes it sit at 0. On random signs the leader rarely switches and FTL does well."

      readouts={
        <>
          <Readout label="FTL regret" value={formatNumber(result.regretFtl[T - 1])} />
          <Readout label="FTRL regret" value={formatNumber(result.regretFtrl[T - 1])} />
          <Readout label="bound at T = 100" value={formatNumber(1 / (2 * eta) + eta * T)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          <Curve {...decisions[0]} />
          <Curve {...decisions[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          <Curve {...regret[0]} />
          <Curve {...regret[1]} />
          <Curve {...regret[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
