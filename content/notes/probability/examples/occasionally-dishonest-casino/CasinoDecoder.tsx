import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
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
  const [toLoaded, setToLoaded] = useState(0.05)
  const [toFair, setToFair] = useState(0.1)
  const [loadedSix, setLoadedSix] = useState(0.5)
  const [rolls, setRolls] = useState(300)
  const [seed, setSeed] = useState(1)

  const result = useMemo(() => {
    const model: Casino = { toLoaded, toFair, loadedSix }
    const sim = simulate(model, rolls, seed)
    const { posterior, logLikelihood } = forwardBackward(model, sim.rolls)
    const path = viterbi(model, sim.rolls)
    const agree = (guess: number[]) => guess.filter((g, t) => g === sim.states[t]).length / rolls
    return {
      sim,
      posterior,
      path,
      logLikelihood,
      posteriorAccuracy: agree(posterior.map((p) => (p > 0.5 ? 1 : 0))),
      viterbiAccuracy: agree(path),
      sixes: sim.rolls.filter((r) => r === 6).length / rolls,
      loaded: sim.states.filter((s) => s === 1).length / rolls,
    }
  }, [toLoaded, toFair, loadedSix, rolls, seed])

  const decoded = useMemo((): XYSeries[] => {
    const ts = result.posterior.map((_, t) => t + 1)
    return [
      { name: 'true die (loaded = 1)', type: 'line', ...steps(result.sim.states), area: true, slot: 2 },
      { name: 'P(loaded | all rolls)', type: 'line', x: ts, y: result.posterior, slot: 0 },
      { name: 'Viterbi path', type: 'line', ...steps(result.path), dashed: true, slot: 1 },
    ]
  }, [result])

  const faces = useMemo(
    (): XYSeries[] => [
      {
        name: 'rolls',
        type: 'scatter',
        x: result.sim.rolls.map((_, t) => t + 1),
        y: result.sim.rolls,
        group: result.sim.states,
        groupNames: ['fair die', 'loaded die'],
      },
    ],
    [result],
  )

  return (
    <Interactive
      title="Decoding the dishonest casino"
      caption="The top chart shows the simulated rolls, marked by the die that produced them; the decoder never sees the marks. The bottom chart shows the true die (shaded), the forward–backward posterior probability that each roll came from the loaded die, and the Viterbi path. Both decoders use the true model parameters. Short loaded runs are hard to detect: a few sixes in a row are also common with a fair die."
      controls={
        <>
          <ParamSlider
            label="P(fair → loaded)"
            value={toLoaded}
            onChange={setToLoaded}
            min={0.01}
            max={0.3}
            step={0.01}
          />
          <ParamSlider label="P(loaded → fair)" value={toFair} onChange={setToFair} min={0.01} max={0.5} step={0.01} />
          <ParamSlider
            label="P(six | loaded)"
            value={loadedSix}
            onChange={setLoadedSix}
            min={0.2}
            max={0.9}
            step={0.05}
          />
          <ParamSlider label="rolls" value={rolls} onChange={setRolls} min={50} max={600} step={10} />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={0} max={30} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="share of sixes" value={formatNumber(result.sixes)} />
          <Readout label="time on loaded die" value={formatNumber(result.loaded)} />
          <Readout label="posterior decoding correct" value={formatNumber(result.posteriorAccuracy)} />
          <Readout label="Viterbi correct" value={formatNumber(result.viterbiAccuracy)} />
          <Readout label="log P(rolls)" value={formatNumber(result.logLikelihood)} />
        </>
      }
    >
      <XYChart height={170} xLabel="roll" yLabel="face" series={faces} xRange={[0, rolls + 1]} yRange={[0.5, 6.5]} />
      <XYChart
        height={240}
        xLabel="roll"
        yLabel="loaded"
        series={decoded}
        xRange={[0, rolls + 1]}
        yRange={[-0.05, 1.05]}
      />
    </Interactive>
  )
}
