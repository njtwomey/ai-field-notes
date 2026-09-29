import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { PlayButton } from '../_shared/Playback'
import { usePlayLoop } from '../_shared/usePlayLoop'
import { train, visibleMarginal, type Negative } from './bm'

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
  const [dataset, setDataset] = useState<Dataset>('parity')
  const [H, setH] = useState(2)
  const [negative, setNegative] = useState<Negative>('exact')
  const [rate, setRate] = useState(0.5)
  const [seed, setSeed] = useState(1)
  const [epoch, setEpoch] = useState(EPOCHS)
  const [temperature, setTemperature] = useState(1)
  const [playing, setPlaying] = useState(false)

  const data = useMemo(() => distribution(dataset), [dataset])
  const snapshots = useMemo(
    () => train(data, V, H, { epochs: EPOCHS, rate, negative, uniform: rng(seed).uniform, init: 2, chains: 50 }),
    [data, H, rate, negative, seed],
  )
  const snap = snapshots[epoch]
  const n = V + H
  const marginal = useMemo(
    () => (temperature === 1 ? snap.marginal : visibleMarginal({ V, H, w: snap.w, b: snap.b }, temperature)),
    [snap, H, temperature],
  )

  usePlayLoop(playing, 150, (k) => {
    const next = Math.min(EPOCHS, epoch + k)
    setEpoch(next)
    if (next >= EPOCHS) setPlaying(false)
    return next < EPOCHS
  })

  const bars = useMemo(
    (): XYSeries[] => [
      {
        name: `model p(v), T = ${formatNumber(temperature)}`,
        type: 'bar',
        x: STATES,
        y: Array.from(marginal),
        slot: 0,
      },
      { name: 'data', type: 'scatter', x: STATES, y: Array.from(data), emphasis: true },
    ],
    [marginal, data, temperature],
  )
  const epochs = useMemo(() => snapshots.map((_, e) => e), [snapshots])
  const trace = useMemo(
    (): XYSeries[] => [
      { name: 'KL(data ‖ model)', type: 'line', x: epochs, y: snapshots.map((s) => s.kl), slot: 0 },
      { name: 'now', type: 'scatter', x: [epoch], y: [snap.kl], emphasis: true },
    ],
    [epochs, snapshots, epoch, snap],
  )
  const traceHandles = useMemo(
    (): Handle[] => [
      {
        kind: 'x',
        at: epoch,
        label: 'epoch',
        onDrag: (x) => {
          setPlaying(false)
          setEpoch(Math.max(0, Math.min(EPOCHS, Math.round(x))))
        },
      },
    ],
    [epoch],
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

  return (
    <Interactive
      title="A Boltzmann machine learns a distribution"
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
      controls={
        <>
          <ParamChoice label="data" value={dataset} onChange={setDataset} options={DATASETS} />
          <ParamSlider label="hidden units" value={H} onChange={setH} min={0} max={4} step={1} />
          <ParamChoice
            label="model averages"
            value={negative}
            onChange={setNegative}
            options={[
              { value: 'exact', label: 'exact' },
              { value: 'gibbs', label: 'Gibbs chains' },
            ]}
          />
          <ParamSlider label="learning rate" value={rate} onChange={setRate} min={0.05} max={1} step={0.05} />
          <ParamSlider
            label="epoch"
            value={epoch}
            onChange={(v) => {
              setPlaying(false)
              setEpoch(v)
            }}
            min={0}
            max={EPOCHS}
            step={1}
            withArrows
          />
          <ParamSlider
            label="display temperature T"
            value={temperature}
            onChange={setTemperature}
            min={0.25}
            max={3}
            step={0.05}
          />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={1} max={10} step={1} />
          <div className="flex gap-2 self-end">
            <PlayButton
              playing={playing}
              onToggle={() => {
                if (!playing && epoch >= EPOCHS) setEpoch(0)
                setPlaying((p) => !p)
              }}
            />
          </div>
        </>
      }
      readout={
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
        <XYChart
          height={260}
          xLabel="visible state index"
          yLabel="probability"
          xRange={[-0.5, 15.5]}
          yRange={[0, undefined]}
          integerX
          series={bars}
          ariaLabel="Model and data probabilities of the 16 visible vectors"
        />
        <Heatmap
          x={units}
          y={units}
          z={weights}
          xLabel="unit (1–4 visible)"
          yLabel="unit"
          scale="diverging"
          range={[-bound, bound]}
          valueLabel="w"
          height={260}
          ariaLabel="Weights between units, with biases on the diagonal"
        />
      </div>
      <XYChart
        height={200}
        xLabel="epoch"
        yLabel="KL (nats)"
        yRange={[0, undefined]}
        series={trace}
        handles={traceHandles}
        ariaLabel="KL divergence from the data to the model over training"
      />
    </Interactive>
  )
}
