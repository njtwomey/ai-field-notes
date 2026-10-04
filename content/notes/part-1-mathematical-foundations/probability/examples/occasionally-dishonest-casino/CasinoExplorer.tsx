import { useMemo, useState } from 'react'
import { dishonestCasino, hmmChain, sampleHmm } from 'aifn-applied/inference/sequence-models'
import { forwardBackward, viterbi } from 'aifn/inference/exact'
import { child, stream } from 'aifn/foundation/random'
import { toRows } from 'aifn/foundation/tensor'
import {
  Figure,
  ControlGroup,
  Select,
  NumberSelector,
  Plots,
  Plot,
  Points,
  Area,
  Curve,
  Readout,
  formatNumber,
  useAxis,
} from 'aifn-render'

const CASINO = dishonestCasino()

export function CasinoExplorer() {
  const [n, setN] = useState(200)
  const [seed, setSeed] = useState(1)
  const [showFiltered, setShowFiltered] = useState(false)

  const data = useMemo(() => sampleHmm(child(stream('content/casino'), seed), CASINO, n), [n, seed])
  const obs = useMemo(() => Array.from(data.observations.data), [data])

  const chain = useMemo(() => hmmChain(CASINO, obs), [obs])
  const fb = useMemo(() => forwardBackward(chain), [chain])
  const vit = useMemo(() => viterbi(chain), [chain])

  const view = useMemo(() => {
    const truth = Array.from(data.states.data)
    const path = Array.from(vit.path.data)
    return {
      t: obs.map((_, i) => i + 1),
      faces: obs.map((o) => o + 1),
      truth,
      smoothed: toRows(fb.marginals).map((r) => r[1]),
      filtered: toRows(fb.filtered).map((r) => r[1]),
      path,
      lifted: path.map((k) => k * 1.02),
    }
  }, [data, obs, fb, vit])

  const accuracy = (xs: number[]) => xs.filter((x, i) => x === view.truth[i]).length / xs.length

  const rollAxis = useAxis({ label: 'roll number', hold: 'initial', key: n })
  const faceAxis = useAxis({ label: 'observed face', range: [0.5, 6.5] })
  const probAxis = useAxis({ label: 'p(loaded die)', range: [0, 1.1] })

  const sixesCount = obs.filter((o) => o === 5).length / obs.length
  const loadedCount = view.truth.filter((s) => s === 1).length / view.truth.length

  return (
    <Figure
      title="Forward–backward smoothing and Viterbi decoding on the dishonest casino"
      purpose="Forward–backward gives the full posterior probability of the loaded die given all rolls, while Viterbi recovers the globally most probable state trajectory. Both reveal secret die switches hidden by noisy outcomes."
      defaultSize="L"
      controls={
        <ControlGroup>
          <NumberSelector
            label="Roll count n"
            value={n}
            onChange={setN}
            min={50}
            max={400}
            step={25}
            suggestions={[100, 200, 300]}
          />
          <NumberSelector
            label="Random seed"
            value={seed}
            onChange={setSeed}
            min={1}
            max={50}
            step={1}
            suggestions={[1, 2, 7, 13]}
          />
          <Select
            label="Inference"
            options={[
              { value: 'smoothed', label: 'Smoothed posterior (all rolls)' },
              { value: 'both', label: 'Smoothed + filtered (online)' },
            ]}
            value={showFiltered ? 'both' : 'smoothed'}
            onChange={(v) => setShowFiltered(v === 'both')}
          />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout label="log p(rolls)" value={formatNumber(fb.logLikelihood)} />
          <Readout
            label="Posterior accuracy"
            value={formatNumber(accuracy(view.smoothed.map((p) => (p > 0.5 ? 1 : 0))))}
          />
          <Readout label="Viterbi accuracy" value={formatNumber(accuracy(view.path))} />
          <Readout label="Fraction sixes" value={formatNumber(sixesCount)} />
          <Readout label="Fraction loaded" value={formatNumber(loadedCount)} />
        </>
      }
      caption="Top: observed dice faces (1 to 6). Bottom: shaded grey area indicates true intervals where the loaded die was used; solid curve is the smoothed posterior P(loaded | all rolls); dashed line is the Viterbi maximum-probability path."
    >
      <Plots rows={2} heights={[1, 1.4]}>
        <Plot x={rollAxis} y={faceAxis}>
          <Points name="roll" x={view.t} y={view.faces} thin />
        </Plot>
        <Plot x={rollAxis} y={probAxis}>
          <Area name="true loaded state" x={view.t} y={view.truth} base={0} opacity={0.15} />
          <Curve name="smoothed P(loaded | all rolls)" x={view.t} y={view.smoothed} slot={0} />
          {showFiltered && <Curve name="filtered P(loaded | past)" x={view.t} y={view.filtered} slot={2} dashed />}
          <Curve name="Viterbi path" x={view.t} y={view.lifted} slot={1} dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
