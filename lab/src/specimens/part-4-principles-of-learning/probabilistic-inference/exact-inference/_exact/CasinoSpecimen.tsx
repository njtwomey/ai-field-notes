import { dishonestCasino, hmmChain, sampleHmm } from 'aifn-methods/inference/sequence-models'
import { forwardBackward, viterbi } from 'aifn-compute/inference/exact'
import { child, stream } from 'aifn-compute/foundation/random'
import { toRows } from 'aifn-compute/foundation/tensor'
import { useMemo } from 'react'
import { Figure } from 'aifn-render/layout'
import { row, slider, toggle, useFigureState } from 'aifn-render/state'
import { Area, Curve, Plot, Plots, Points, Readout, formatNumber, useAxis } from 'aifn-render/viz'

const CASINO = dishonestCasino()

/** Forward–backward and Viterbi on rolls of the occasionally dishonest casino. */
export function CasinoSpecimen() {
  const state = useFigureState({
    data: row('1 · rolls', {
      n: slider(20, 400, 200, { label: 'rolls', step: 10 }),
      seed: slider(1, 50, 1, { label: 'seed', step: 1 }),
    }),
    reveal: row('2 · reveal', { showFiltered: toggle(false, 'filtered probability') }),
  })
  const { n, seed } = state.data
  const { showFiltered } = state.reveal
  const data = useMemo(() => sampleHmm(child(stream('casino'), seed), CASINO, n), [n, seed])
  const obs = useMemo(() => Array.from(data.observations.data), [data])
  // The HMM's chain potentials for these rolls; forward–backward and Viterbi run on any chain.
  const chain = useMemo(() => hmmChain(CASINO, obs), [obs])
  const fb = useMemo(() => forwardBackward(chain), [chain])
  const vit = useMemo(() => viterbi(chain), [chain])
  const view = useMemo(() => {
    const truth = Array.from(data.states.data)
    const path = Array.from(vit.path.data)
    return {
      t: obs.map((_, i) => i),
      faces: obs.map((o) => o + 1),
      truth,
      smoothed: toRows(fb.marginals).map((r) => r[1]),
      filtered: toRows(fb.filtered).map((r) => r[1]),
      path,
      // Lifted slightly so the path stays visible where it runs along 0 or 1.
      lifted: path.map((k) => k * 1.04),
    }
  }, [data, obs, fb, vit])
  const accuracy = (xs: number[]) => xs.filter((x, i) => x === view.truth[i]).length / xs.length
  const roll = useAxis({ label: 'roll', hold: 'initial', key: n })
  const face = useAxis({ label: 'face', range: [0.5, 6.5] })
  const prob = useAxis({ label: 'p(loaded)', range: [0, 1.1] })
  return (
    <Figure
      title="Forward–backward and Viterbi on the dishonest casino"
      purpose="Forward–backward gives each roll's posterior probability of the loaded die from all the rolls; Viterbi gives the single most probable sequence of dice; both recover the switches the rolls alone hide."
      defaultSize="L"
      state={state}
      readouts={{
        decoding: (
          <>
            <Readout label="log p(rolls)" value={formatNumber(fb.logLikelihood)} />
            <Readout
              label="posterior decoding accuracy"
              value={formatNumber(accuracy(view.smoothed.map((p) => (p > 0.5 ? 1 : 0))))}
            />
            <Readout label="Viterbi accuracy" value={formatNumber(accuracy(view.path))} />
          </>
        ),
      }}
      caption="The casino switches to a loaded die (a six half the time) with probability 0.05 per roll and back with 0.1 (Durbin et al. 1998). Top: the rolls, coloured by the die that produced them. Bottom: the grey band is the true die; the smoothed posterior uses every roll, forward and backward; the Viterbi path is the single best sequence, lifted slightly to stay visible. The filtered probability (a toggle) uses only the rolls so far, so it lags behind each switch."
    >
      <Plots rows={2} heights={[1, 1.4]} hoverGroup>
        <Plot x={roll} y={face}>
          <Points
            name="roll"
            x={view.t}
            y={view.faces}
            group={view.truth}
            groupNames={['fair die', 'loaded die']}
            thin
          />
        </Plot>
        <Plot x={roll} y={prob}>
          <Area name="true state (loaded = 1)" x={view.t} y={view.truth} muted />
          <Curve name="p(loaded | all rolls)" x={view.t} y={view.smoothed} slot={0} />
          {showFiltered && <Curve name="p(loaded | rolls so far)" x={view.t} y={view.filtered} slot={2} dashed />}
          <Curve name="Viterbi path" x={view.t} y={view.lifted} slot={1} />
        </Plot>
      </Plots>
    </Figure>
  )
}
