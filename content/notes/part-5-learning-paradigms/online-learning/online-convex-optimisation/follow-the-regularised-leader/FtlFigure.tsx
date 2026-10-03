import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'

const T = 100
type Sequence = 'alternating' | 'random'

/** Gradients z_t of the linear losses f_t(w) = z_t w on [-1, 1]. */
function gradients(sequence: Sequence, seed: number): number[] {
  if (sequence === 'alternating') return Array.from({ length: T }, (_, t) => (t === 0 ? 0.5 : t % 2 === 1 ? -1 : 1))
  const r = rng(seed)
  return Array.from({ length: T }, () => (r.uniform() < 0.5 ? -1 : 1))
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
  const [sequence, setSequence] = useState<Sequence>('alternating')
  const logEta = useParam(Math.log10(1 / Math.sqrt(2 * T)), { min: -3, max: 1, step: 0.05 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const eta = 10 ** logEta.value
  const z = useMemo(() => gradients(sequence, seed.value), [sequence, seed.value])
  const result = useMemo(() => play(z, eta), [z, eta])

  const decisions: XYSeries[] = [
    { name: 'FTL', type: 'line', x: ROUNDS, y: result.ftl, slot: 1 },
    { name: 'FTRL', type: 'line', x: ROUNDS, y: result.ftrl, slot: 0 },
  ]
  const regret: XYSeries[] = [
    { name: 'FTL', type: 'line', x: ROUNDS, y: result.regretFtl, slot: 1 },
    { name: 'FTRL', type: 'line', x: ROUNDS, y: result.regretFtrl, slot: 0 },
    {
      name: 'FTRL bound 1/(2η) + ηt',
      type: 'line',
      x: ROUNDS,
      y: ROUNDS.map((t) => 1 / (2 * eta) + eta * t),
      emphasis: true,
      dashed: true,
    },
  ]

  return (
    <Interactive
      title="Following the leader, with and without a regulariser"
      caption="Linear losses f_t(w) = z_t w on the interval [−1, 1]. With alternating gradients (z₁ = ½, then −1, +1, …) the leader flips every round to the end that the next loss punishes, and FTL's regret grows by 1 per round. FTRL with ψ(w) = w²/2 plays −ηS_t, a damped version of the same decision, and its regret stays under the bound 1/(2η) + ηt. A large η makes FTRL behave like FTL; a tiny η makes it sit at 0. On random signs the leader rarely switches and FTL does well."
      controls={
        <>
          <ParamChoice
            label="gradients z_t"
            value={sequence}
            onChange={setSequence}
            options={[
              { value: 'alternating', label: 'alternating' },
              { value: 'random', label: 'random ±1' },
            ]}
          />
          <ParamSlider label="learning rate η" param={logEta} format={(v) => formatNumber(10 ** v)} />
          {sequence === 'random' && <ParamSlider label="seed" param={seed} format={(v) => String(v)} />}
        </>
      }
      readout={
        <>
          <Readout label="FTL regret" value={formatNumber(result.regretFtl[T - 1])} />
          <Readout label="FTRL regret" value={formatNumber(result.regretFtrl[T - 1])} />
          <Readout label="bound at T = 100" value={formatNumber(1 / (2 * eta) + eta * T)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={decisions} xLabel="round t" yLabel="decision w_t" xRange={[1, T]} yRange={[-1.05, 1.05]} />
        <XYChart series={regret} xLabel="round t" yLabel="regret against best fixed w" xRange={[1, T]} />
      </div>
    </Interactive>
  )
}
