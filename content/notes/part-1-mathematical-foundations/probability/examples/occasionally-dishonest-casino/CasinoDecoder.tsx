import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { forwardBackward, simulate, viterbi, type Casino } from './hmm'

/** A 0/1 sequence as a step line, so each roll's state covers the width of that roll. */
function steps(values: number[]): { x: number[]; y: number[] } {
  return {
    x: values.flatMap((_, t) => [t + 0.5, t + 1.5]),
    y: values.flatMap((v) => [v, v]),
  }
}

/** Simulates the casino, then decodes the hidden die from the rolls alone with forward–backward and Viterbi. */
export function CasinoDecoder() {
  const state = useFigureState({
    toLoaded: float(0.05, { min: 0.01, max: 0.3, step: 0.01, label: 'P(fair → loaded)' }),
    toFair: float(0.1, { min: 0.01, max: 0.5, step: 0.01, label: 'P(loaded → fair)' }),
    loadedSix: float(0.5, { min: 0.2, max: 0.9, step: 0.05, label: 'P(six | loaded)' }),
    rolls: int(300, { min: 50, max: 600, step: 10, label: 'rolls' }),
    seed: int(1, { min: 0, max: 30, step: 1, label: 'seed' }),
  })

  const result = useMemo(() => {
    const model: Casino = { toLoaded: state.toLoaded, toFair: state.toFair, loadedSix: state.loadedSix }
    const sim = simulate(model, state.rolls, state.seed)
    const { posterior, logLikelihood } = forwardBackward(model, sim.rolls)
    const path = viterbi(model, sim.rolls)
    const agree = (guess: number[]) => guess.filter((g, t) => g === sim.states[t]).length / state.rolls
    return {
      sim,
      posterior,
      path,
      logLikelihood,
      posteriorAccuracy: agree(posterior.map((p) => (p > 0.5 ? 1 : 0))),
      viterbiAccuracy: agree(path),
      sixes: sim.rolls.filter((r) => r === 6).length / state.rolls,
      loaded: sim.states.filter((s) => s === 1).length / state.rolls,
    }
  }, [state.toLoaded, state.toFair, state.loadedSix, state.rolls, state.seed])

  const decoded = useMemo((): SeriesSpec[] => {
    const ts = result.posterior.map((_, t) => t + 1)
    return [
      { name: 'true die (loaded = 1)', type: 'line', ...steps(result.sim.states), area: true, slot: 2 },
      { name: 'P(loaded | all rolls)', type: 'line', x: ts, y: result.posterior, slot: 0 },
      { name: 'Viterbi path', type: 'line', ...steps(result.path), dashed: true, slot: 1 },
    ]
  }, [result])

  const faces = useMemo(
    () =>
      [
        {
          name: 'rolls',
          x: result.sim.rolls.map((_, t) => t + 1),
          y: result.sim.rolls,
          group: result.sim.states,
          groupNames: ['fair die', 'loaded die'],
        },
      ] as const,
    [result],
  )

  const xAxis = useAxis({ label: 'roll', range: [0, state.rolls + 1] })
  const yAxis = useAxis({ label: 'face', range: [0.5, 6.5] })
  const xAxis2 = useAxis({ label: 'roll', range: [0, state.rolls + 1] })
  const yAxis2 = useAxis({ label: 'loaded', range: [-0.05, 1.05] })
  return (
    <Figure
      title="Decoding the dishonest casino"
      state={state}
      caption="The top chart shows the simulated rolls, marked by the die that produced them; the decoder never sees the marks. The bottom chart shows the true die (shaded), the forward–backward posterior probability that each roll came from the loaded die, and the Viterbi path. Both decoders use the true model parameters. Short loaded runs are hard to detect: a few sixes in a row are also common with a fair die."

      readouts={
        <>
          <Readout label="share of sixes" value={formatNumber(result.sixes)} />
          <Readout label="time on loaded die" value={formatNumber(result.loaded)} />
          <Readout label="posterior decoding correct" value={formatNumber(result.posteriorAccuracy)} />
          <Readout label="Viterbi correct" value={formatNumber(result.viterbiAccuracy)} />
          <Readout label="log P(rolls)" value={formatNumber(result.logLikelihood)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={170}>
        <Points {...faces[0]} />
      </Plot>
      <Plot x={xAxis2} y={yAxis2} height={240}>
        {seriesLayers(decoded)}
      </Plot>
    </Figure>
  )
}
