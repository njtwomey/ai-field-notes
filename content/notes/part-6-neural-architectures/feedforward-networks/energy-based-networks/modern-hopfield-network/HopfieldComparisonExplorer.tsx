import { useMemo, useState } from 'react'
import { digitGlyphs } from 'aifn-applied/data/synthetic'
import {
  capacityCurve,
  corruptPattern,
  hebbianWeights,
  hopfieldRecall,
  modernHopfieldEnergy,
  modernHopfieldUpdate,
  overlaps,
} from 'aifn-applied/generative/boltzmann'
import { child, stream } from 'aifn/foundation/random'
import { fromData, toFlat } from 'aifn/foundation/tensor'
import {
  Figure,
  ControlGroup,
  NumberSelector,
  Player,
  Plots,
  Plot,
  Raster,
  Bars,
  Curve,
  Readout,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

/** Images of h × w pixels laid out horizontally with a 1-pixel gap. */
function tiles(images: ArrayLike<number>, count: number, h: number, w: number, cols: number, map = (v: number) => v) {
  const r = Math.ceil(count / cols)
  const H = r * (h + 1) - 1
  const W = cols * (w + 1) - 1
  const out = Array.from({ length: H }, () => new Array<number>(W).fill(NaN))
  for (let k = 0; k < count; k++) {
    const R = Math.floor(k / cols)
    const C = k % cols
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        out[H - 1 - (R * (h + 1) + i)][C * (w + 1) + j] = map(images[k * h * w + i * w + j])
      }
    }
  }
  return { z: out, x: Array.from({ length: W }, (_, j) => j), y: Array.from({ length: H }, (_, i) => i) }
}

export function HopfieldComparisonExplorer() {
  const [stored, setStored] = useState(4)
  const [cueDigit, setCueDigit] = useState(1)
  const [flip, setFlip] = useState(0.15)
  const [beta, setBeta] = useState(0.5)
  const [sweep, setSweep] = useState(0)

  const seed = 1
  const glyphs = useMemo(() => Float64Array.from(toFlat(digitGlyphs()), (v) => 2 * v - 1), [])
  const n = Math.max(stored, Math.min(cueDigit + 1, 10))
  const X = useMemo(() => fromData(glyphs.slice(0, n * 35), [n, 35]), [glyphs, n])

  const recall = useMemo(() => {
    const W = hebbianWeights(X)
    const cue = corruptPattern(glyphs.subarray(cueDigit * 35, (cueDigit + 1) * 35), stream(`cue/${seed}`), flip)
    const classical = hopfieldRecall(W, cue, stream(`recall/${seed}`))
    const modern = modernHopfieldUpdate(X, cue, beta)
    return {
      cue,
      classical,
      modern,
      modernEnergy: [modernHopfieldEnergy(X, cue, beta), modernHopfieldEnergy(X, modern.state, beta)],
    }
  }, [X, glyphs, cueDigit, flip, seed, beta])

  const currentSweep = Math.min(sweep, recall.classical.states.length - 1)
  const state0 = recall.classical.states[currentSweep]
  const ov = overlaps(X, state0)

  const curve = useMemo(
    () => capacityCurve(child(stream('capacity'), seed), { units: 70, maxPatterns: 20, trials: 5, flip, beta }),
    [seed, flip, beta],
  )

  const pictures = useMemo(
    () =>
      tiles(
        Float64Array.from([...recall.cue, ...state0, ...recall.modern.state]),
        3,
        7,
        5,
        3,
        (v) => (v + 1) / 2,
      ),
    [recall, state0],
  )

  const px = useAxis({ label: 'cue · classical · modern', range: [-0.5, pictures.x.length - 0.5] })
  const py = useAxis({ label: '', range: [-0.5, pictures.y.length - 0.5], equal: px })
  const digitAxis = useAxis({ label: 'stored digit', range: [-0.5, n - 0.5] })
  const wAxis = useAxis({ label: 'overlap / softmax weight', range: [-1, 1] })
  const loadAxis = useAxis({ label: 'random patterns stored (70 units)', range: [1, 20] })
  const qAxis = useAxis({ label: 'final overlap with cue', range: [0, 1] })

  return (
    <Figure
      title="Classical vs modern Hopfield network recall and capacity"
      purpose="A classical Hopfield network stores patterns in Hebbian weights and recalls through asynchronous sign updates, but fails past ~0.138N patterns per unit. Modern continuous Hopfield networks update via softmax-weighted attention, recalling far past that threshold in a single step."
      defaultSize="XL"
      controls={
        <ControlGroup>
          <NumberSelector
            label="Stored digits"
            value={stored}
            onChange={setStored}
            min={2}
            max={10}
            step={1}
            suggestions={[2, 4, 7, 10]}
          />
          <NumberSelector
            label="Cued digit"
            value={cueDigit}
            onChange={(d) => setCueDigit(Math.max(0, Math.min(9, d)))}
            min={0}
            max={9}
            step={1}
            suggestions={[0, 1, 4, 7]}
          />
          <NumberSelector
            label="Noise flip prob"
            value={flip}
            onChange={setFlip}
            min={0}
            max={0.5}
            step={0.05}
            suggestions={[0.05, 0.15, 0.3]}
          />
          <NumberSelector
            label="Inverse temp β"
            value={beta}
            onChange={setBeta}
            min={0.05}
            max={5}
            step={0.1}
            suggestions={[0.1, 0.5, 1.0, 2.0]}
          />
          <Player
            label="Classical sweeps"
            value={currentSweep}
            onChange={setSweep}
            count={recall.classical.states.length}
          />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout
            label="Classical energy"
            value={fmt(recall.classical.energies[currentSweep])}
          />
          <Readout label="Classical overlap" value={fmt(ov[cueDigit] ?? NaN)} />
          <Readout label="Classical converged" value={recall.classical.converged ? 'yes' : 'no'} />
          <Readout label="Modern attention weight" value={fmt(recall.modern.weights[cueDigit] ?? NaN)} />
          <Readout
            label="Modern energy before → after"
            value={`${fmt(recall.modernEnergy[0])} → ${fmt(recall.modernEnergy[1])}`}
          />
        </>
      }
      caption="Left: corrupted cue, classical network state after asynchronous sweeps, and modern network result after one softmax attention update. Middle: overlap with stored digits vs modern attention weights. Right: capacity on random patterns of 70 units showing the classical 0.138N limit vs modern robust recall."
    >
      <Plots cols={3}>
        <Plot x={px} y={py} title="cue · classical recall · modern recall">
          <Raster x={pictures.x} y={pictures.y} z={pictures.z} scale="sequential" range={[0, 1]} valueLabel="pixel" />
        </Plot>
        <Plot x={digitAxis} y={wAxis} title="memory overlap & attention">
          <Bars
            name="classical overlap"
            slot={0}
            x={Array.from({ length: n }, (_, i) => i - 0.2)}
            y={Array.from(ov)}
            width={0.35}
          />
          <Bars
            name="modern attention weight"
            slot={1}
            x={Array.from({ length: n }, (_, i) => i + 0.2)}
            y={Array.from(recall.modern.weights)}
            width={0.35}
          />
        </Plot>
        <Plot x={loadAxis} y={qAxis} title="capacity breakdown">
          <Curve
            name="classical"
            slot={0}
            x={curve.patterns}
            y={Array.from(curve.classical, (v) => Math.max(0, v))}
            showPoints
          />
          <Curve
            name="modern"
            slot={1}
            x={curve.patterns}
            y={Array.from(curve.modern, (v) => Math.max(0, v))}
            showPoints
          />
          <Curve name="0.138 N" x={[0.138 * 70, 0.138 * 70]} y={[0, 1]} emphasis dashed thin />
        </Plot>
      </Plots>
    </Figure>
  )
}
