import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { stream, uniform as drawUniform } from 'aifn/foundation/random'

/** Success rates by stone size, from Charig et al. (1986). A is open surgery, B is percutaneous nephrolithotomy. */
const RATES = {
  A: { small: 81 / 87, large: 192 / 263 },
  B: { small: 234 / 270, large: 55 / 80 },
}

const overall = (t: keyof typeof RATES, largeShare: number) =>
  (1 - largeShare) * RATES[t].small + largeShare * RATES[t].large

/**
 * Each treatment's overall success rate is a weighted average of its two stratum rates, so it lies on the segment
 * between them, at the treatment's share of large stones. Dragging a point changes that case mix.
 */
export function CaseMix() {
  const state = useFigureState({
    mixA: float(263 / 350, { min: 0, max: 1, step: 0.001, label: 'A: share of large stones' }),
    mixB: float(80 / 350, { min: 0, max: 1, step: 0.001, label: 'B: share of large stones' }),
    patients: int(350, { min: 50, max: 5000, step: 50, label: 'simulated patients per arm' }),
    seed: int(1, { min: 0, max: 30, step: 1, label: 'seed' }),
  })

  const series = useMemo(
    () =>
      [
        { name: 'A: open surgery', x: [0, 1], y: [RATES.A.small, RATES.A.large], slot: 0 },
        { name: 'B: nephrolithotomy', x: [0, 1], y: [RATES.B.small, RATES.B.large], slot: 1 },
      ] as const,
    [],
  )

  const simulated = useMemo(() => {
    const draws = stream(state.seed)
    const uniform = () => drawUniform(draws)
    const arm = (t: keyof typeof RATES, share: number) => {
      let successes = 0
      for (let k = 0; k < state.patients; k++) {
        const rate = uniform() < share ? RATES[t].large : RATES[t].small
        if (uniform() < rate) successes++
      }
      return successes / state.patients
    }
    return { A: arm('A', state.mixA), B: arm('B', state.mixB) }
  }, [state.mixA, state.mixB, state.patients, state.seed])

  const a = overall('A', state.mixA)
  const b = overall('B', state.mixB)

  const xAxis = useAxis({ label: 'share of patients with large stones', range: [0, 1] })
  const yAxis = useAxis({ label: 'success rate', range: [0.6, 1] })
  return (
    <Figure
      title="Success rate against case mix"
      state={state}
      caption="Each line joins a treatment's success rate for small stones (left) and for large stones (right). A is better in both groups, so its line is higher everywhere. Each dot is a treatment's overall success rate, placed at its share of large stones. In the study, A treated mostly large stones, so its dot sits low on a higher line. Drag the dots to change the case mix: with equal mixes, A wins overall too."

      readouts={
        <>
          <Readout label="A overall, exact" value={formatNumber(a)} />
          <Readout label="B overall, exact" value={formatNumber(b)} />
          <Readout label="A overall, simulated" value={formatNumber(simulated.A)} />
          <Readout label="B overall, simulated" value={formatNumber(simulated.B)} />
          <Readout label="better overall" value={a > b ? 'A' : a < b ? 'B' : 'tie'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Handle kind="point" at={[state.mixA, a]} label="A overall" onDrag={([x]) => state.set('mixA', x)} />
        <Handle kind="point" at={[state.mixB, b]} label="B overall" onDrag={([x]) => state.set('mixB', x)} />
      </Plot>
    </Figure>
  )
}
