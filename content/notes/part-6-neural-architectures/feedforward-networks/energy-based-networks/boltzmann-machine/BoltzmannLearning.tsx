import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Bars,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform } from 'aifn/foundation/random'
import { train, visibleMarginal, type Negative } from './bm'

/** Seeded uniform draws. */
function draws(seed: number) {
  const g = stream(seed)
  return () => uniform(g)
}

const V = 4
const EPOCHS = 2000
const STATES = Array.from({ length: 1 << V }, (_, c) => c)
/** Visible vector of a state index, v1 first: bit k of the index is unit k + 1. */
const bits = (c: number) => Array.from({ length: V }, (_, k) => (c >> k) & 1).join('')
const indexOf = (s: string) => [...s].reduce((c, b, k) => c | (b === '1' ? 1 << k : 0), 0)

type Dataset = 'bars' | 'parity'
const DATASETS: { value: Dataset; label: string }[] = [
  { value: 'bars', label: 'bars on a 2 × 2 grid' },
  { value: 'parity', label: 'even parity' },
]

function distribution(name: Dataset): Float64Array {
  const p = new Float64Array(1 << V)
  if (name === 'bars') for (const s of ['1100', '0011', '1010', '0101']) p[indexOf(s)] = 0.25
  else
    for (const c of STATES) {
      const ones = [...bits(c)].filter((b) => b === '1').length
      if (ones % 2 === 0) p[c] = 1 / 8
    }
  return p
}

/**
 * A Boltzmann machine with four visible and up to four hidden units, every pair connected, trained by gradient ascent on
 * the exact log-likelihood of a distribution over the 16 visible vectors. Every state is enumerated, so the model's
 * distribution and the KL divergence to the data are exact at every epoch.
 */
export function BoltzmannLearning() {
  const state = useFigureState({
    dataset: choice<Dataset>(DATASETS, 'parity', { label: 'data' }),
    H: int(2, { min: 0, max: 4, step: 1, label: 'hidden units' }),
    negative: choice<Negative>(
      [
        { value: 'exact', label: 'exact' },
        { value: 'gibbs', label: 'Gibbs chains' },
      ],
      'exact',
      { label: 'model averages' },
    ),
    rate: float(0.5, { gt: 0, max: 1, scale: 'log10', suggestions: [0.05, 0.1, 0.25, 0.5, 1], label: 'learning rate' }),
    temperature: float(1, { min: 0.25, max: 3, step: 0.05, label: 'display temperature T' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  const data = useMemo(() => distribution(state.dataset), [state.dataset])
  const snapshots = useMemo(
    () =>
      train(data, V, state.H, {
        epochs: EPOCHS,
        rate: state.rate,
        negative: state.negative,
        uniform: draws(state.seed),
        init: 2,
        chains: 50,
      }),
    [data, state.H, state.rate, state.negative, state.seed],
  )
  // The walk through the epochs restarts at 0 for a new training run.
  const [position, setPosition] = useState({ snapshots, epoch: 0 })
  const epoch = position.snapshots === snapshots ? position.epoch : 0
  const go = (v: number) => setPosition({ snapshots, epoch: Math.max(0, Math.min(EPOCHS, Math.round(v))) })
  const snap = snapshots[epoch]
  const n = V + state.H
  const marginal = useMemo(
    () =>
      state.temperature === 1
        ? snap.marginal
        : visibleMarginal({ V, H: state.H, w: snap.w, b: snap.b }, state.temperature),
    [snap, state.H, state.temperature],
  )

  const modelBars = useMemo(() => Array.from(marginal), [marginal])
  const dataPoints = useMemo(() => Array.from(data), [data])
  const epochs = useMemo(() => snapshots.map((_, e) => e), [snapshots])
  const trace = useMemo(
    () =>
      [
        { name: 'KL(data ‖ model)', x: epochs, y: snapshots.map((s) => s.kl), slot: 0 },
        { name: 'now', x: [epoch], y: [snap.kl], emphasis: true },
      ] as const,
    [epochs, snapshots, epoch, snap],
  )
  const units = useMemo(() => Array.from({ length: n }, (_, i) => i + 1), [n])
  // Symmetric weights with the biases on the diagonal.
  const weights = useMemo(
    () =>
      units.map((_, i) => units.map((__, j) => (i === j ? snap.b[i] : snap.w[Math.min(i, j) * n + Math.max(i, j)]))),
    [units, snap, n],
  )
  const bound = useMemo(() => Math.max(1, ...weights.flat().map(Math.abs)), [weights])
  const top = [...STATES].sort((a, b) => marginal[b] - marginal[a]).slice(0, 4)

  const xAxis = useAxis({ label: 'unit (1–4 visible)' })
  const yAxis = useAxis({ label: 'unit' })
  const xAxis2 = useAxis({ label: 'epoch', hold: 'union' })
  const yAxis2 = useAxis({ label: 'KL (nats)', range: [0, undefined], hold: 'union' })
  const xAxis3 = useAxis({ label: 'visible state index', range: [-0.5, 15.5], integer: true, nice: false })
  const yAxis3 = useAxis({ label: 'probability', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="A Boltzmann machine learns a distribution"
      state={state}
      caption={
        <>
          The data are a distribution over the 16 binary vectors v = (v1, v2, v3, v4); the state index on the x-axis is
          v1 + 2v2 + 4v3 + 8v4. Bars on a 2 × 2 grid are four vectors (1100, 0011, 1010, 0101); even parity is the eight
          vectors with an even number of ones. Step through the epochs: each moves every weight by the rate times the
          difference between how often its two units are on together with the data clamped and with the model running
          freely. Even parity has no pairwise correlations, so with no hidden units the model stays uniform over all 16
          vectors; with two hidden units it learns after a long plateau. Gibbs chains replace the exact model averages
          with 50 persistent chains, and a large rate then makes learning noisy. The temperature only changes how the
          trained model is displayed, as p(v) at temperature T. Drag the epoch line on the trace.
        </>
      }
      controls={<Player value={epoch} onChange={go} count={EPOCHS + 1} label="epoch" format={(v) => `epoch ${v}`} />}
      readouts={
        <>
          <Readout label="epoch" value={epoch} />
          <Readout label="KL(data ‖ model)" value={formatNumber(snap.kl)} />
          <Readout
            label="most probable v"
            value={top.map((c) => `${bits(c)} ${formatNumber(marginal[c])}`).join(', ')}
          />
        </>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <Plot x={xAxis3} y={yAxis3} height={260} ariaLabel="Model and data probabilities of the 16 visible vectors">
          <Bars name={`model p(v), T = ${formatNumber(state.temperature)}`} x={STATES} y={modelBars} slot={0} />
          <Points name="data" x={STATES} y={dataPoints} emphasis />
        </Plot>
        <Plot x={xAxis} y={yAxis} height={260} ariaLabel={'Weights between units, with biases on the diagonal'}>
          <Raster x={units} y={units} z={weights} scale={'diverging'} range={[-bound, bound]} valueLabel={'w'} />
        </Plot>
      </div>
      <Plot x={xAxis2} y={yAxis2} height={200} ariaLabel={'KL divergence from the data to the model over training'}>
        <Curve {...trace[0]} />
        <Points {...trace[1]} />
        <Handle kind="x" at={epoch} label="epoch" onDrag={go} />
      </Plot>
    </Figure>
  )
}
