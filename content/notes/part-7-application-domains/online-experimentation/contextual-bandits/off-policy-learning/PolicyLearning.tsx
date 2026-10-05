import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { BEST_VALUE, logData, thresholdValue } from '../_shared/ope'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const THETAS = toFlat(linspace(0, 1, 201))
const SIZES = ['200', '1000', '5000'] as const
type Size = (typeof SIZES)[number]

const argmaxOf = (ys: number[]) => ys.reduce((best, y, i) => (y > ys[best] ? i : best), 0)

/**
 * Learning a threshold policy (arm 1 below θ, arm 2 above) from logged bandit data by maximising an estimate of its
 * value over θ: IPS with rewards shifted by c, SNIPS, and IPS minus one standard error (the counterfactual risk
 * minimisation penalty). The true value curve is known in closed form.
 */
export function PolicyLearning() {
  const state = useFigureState({
    size: choice<Size>(
      SIZES.map((v) => ({ value: v, label: Number(v).toLocaleString() })),
      '1000',
      { label: 'logged rows n' },
    ),
    eps: float(0.2, { min: 0.02, max: 1, step: 0.01, label: 'logging exploration ε' }),
    shift: float(0, { min: -1, max: 1, step: 0.05, label: 'reward shift c' }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })
  const n = Number(state.size)

  const rows = useMemo(() => logData(n, state.eps, stream(state.seed * 7919 + 5)), [n, state.eps, state.seed])
  const c = state.shift
  const curves = useMemo(() => {
    const ips: number[] = []
    const snips: number[] = []
    const pessimistic: number[] = []
    for (const theta of THETAS) {
      let sw = 0
      let swr = 0
      let swr2 = 0
      for (const row of rows) {
        const target = row.x < theta ? 0 : 1
        if (row.a !== target) continue
        const w = 1 / row.p
        sw += w
        swr += w * (row.r + c)
        swr2 += (w * (row.r + c)) ** 2
      }
      const mean = swr / n
      const sd = Math.sqrt(Math.max(swr2 / n - mean * mean, 0))
      // Shift back, so every curve is in the original reward units and would coincide without noise.
      ips.push(mean - c)
      pessimistic.push(mean - sd / Math.sqrt(n) - c)
      snips.push(sw > 0 ? swr / sw - c : -c)
    }
    return { ips, snips, pessimistic }
  }, [rows, c, n])

  const truth = useMemo(() => THETAS.map(thresholdValue), [])
  const picks = {
    ips: THETAS[argmaxOf(curves.ips)],
    snips: THETAS[argmaxOf(curves.snips)],
    pessimistic: THETAS[argmaxOf(curves.pessimistic)],
  }
  const series = [
    { name: 'true value V(π_θ)', x: THETAS, y: truth, emphasis: true, dashed: true },
    { name: 'IPS', x: THETAS, y: curves.ips, slot: 0 },
    { name: 'SNIPS', x: THETAS, y: curves.snips, slot: 1 },
    { name: 'IPS − 1 standard error', x: THETAS, y: curves.pessimistic, slot: 2 },
    {
      name: 'learned θ',
      x: [picks.ips, picks.snips, picks.pessimistic],
      y: [picks.ips, picks.snips, picks.pessimistic].map(thresholdValue),
      emphasis: true,
    },
  ] as const

  const xAxis = useAxis({ label: 'threshold θ', range: [0, 1] })
  const yAxis = useAxis({ label: 'value', hold: 'union' })
  return (
    <Figure
      title="Learning a policy from logs"
      state={state}
      caption="Policies π_θ play arm 1 for contexts x below θ and arm 2 above; the best threshold is 0.4375. Logs come from a policy that mostly plays poor arms and explores with probability ε. Each curve estimates V(π_θ) from the same log, and learning picks its maximiser (diamonds on the true curve, dashed). With few rows or small ε the IPS curve is jagged and its peak is often a spike of luck. Shift the rewards by c, which changes nothing about which policy is best: the IPS curve tilts and its maximiser moves, because the average weight differs between policies. SNIPS is unchanged by the shift. The pessimistic curve subtracts one standard error, which steers away from policies whose estimate rests on few heavily weighted rows."

      readouts={
        <>
          <Readout label="best θ, value" value={`0.4375, ${formatNumber(BEST_VALUE)}`} />
          <Readout
            label="IPS: θ, true value"
            value={`${formatNumber(picks.ips)}, ${formatNumber(thresholdValue(picks.ips))}`}
          />
          <Readout
            label="SNIPS: θ, true value"
            value={`${formatNumber(picks.snips)}, ${formatNumber(thresholdValue(picks.snips))}`}
          />
          <Readout
            label="pessimistic: θ, true value"
            value={`${formatNumber(picks.pessimistic)}, ${formatNumber(thresholdValue(picks.pessimistic))}`}
          />
          <Readout label="IPS estimate at its maximiser" value={formatNumber(Math.max(...curves.ips))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
        <Points {...series[4]} />
      </Plot>
    </Figure>
  )
}
