import { useMemo, useState } from 'react'
import { digits } from 'aifn-methods/data/synthetic'
import { dbnRun, type DbnRun } from 'aifn-methods/generative/boltzmann'
import { stream } from 'aifn-compute/foundation/random'
import {
  ControlRow,
  Curve,
  Figure,
  NumberSelector,
  Player,
  Plot,
  Plots,
  Raster,
  Readout,
  Select,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

/** Images of h × w pixels laid out in cols columns with a 1-pixel gap. */
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

const ARCHITECTURES = [
  { value: '24-12', label: '24 → 12', layers: [24, 12] },
  { value: '32-16', label: '32 → 16', layers: [32, 16] },
  { value: '24', label: '24 (single RBM)', layers: [24] },
]

export function DbnExplorer() {
  const [arch, setArch] = useState('24-12')
  const [epochs, setEpochs] = useState(30)
  const [flip, setFlip] = useState(0.1)
  const [labelsPerClass, setLabelsPerClass] = useState(2)
  const [picked, setPicked] = useState(0)

  const selectedArch = ARCHITECTURES.find((a) => a.value === arch) ?? ARCHITECTURES[0]

  const run: DbnRun = useMemo(() => {
    const d = digits(stream('dbn/data'), { perClass: 15, flip })
    let last: DbnRun | null = null
    for (const step of dbnRun(d, {
      layers: selectedArch.layers,
      epochs,
      learningRate: 0.1,
      k: 1,
      labelsPerClass,
      fineTuneEpochs: 30,
      sampleSteps: 30,
      samples: 8,
      seed: 1,
    })) {
      last = step
    }
    return last!
  }, [selectedArch, epochs, flip, labelsPerClass])

  const shots = run.checkpoints ?? []
  const index = Math.min(picked, Math.max(0, shots.length - 1))
  const shot = shots[index]
  const sizes = run.sizes ?? selectedArch.layers
  const H1 = sizes[0]

  const fields = useMemo(() => {
    if (!shot) return null
    const img = new Float64Array(H1 * 35)
    for (let j = 0; j < H1; j++) {
      for (let i = 0; i < 35; i++) {
        img[j * 35 + i] = shot.weights[i * H1 + j]
      }
    }
    return tiles(img, H1, 7, 5, Math.min(8, H1))
  }, [shot, H1])

  const wMax = useMemo(() => (shot ? Math.max(...Array.from(shot.weights, Math.abs)) || 1 : 1), [shot])
  const samples = useMemo(() => (shot ? tiles(shot.samples, shot.samples.length / 35, 7, 5, 8) : null), [shot])

  const errAxis = useAxis({ label: 'CD epoch of layer', range: [0, epochs] })
  const eAxis = useAxis({ label: 'squared reconstruction error', hold: 'union', key: `${arch}/${epochs}` })
  const fx = useAxis({ label: '', range: [-0.5, (fields?.x.length ?? 1) - 0.5], key: fields?.x.length })
  const fy = useAxis({ label: '', range: [-0.5, (fields?.y.length ?? 1) - 0.5], key: fields?.y.length, equal: fx })
  const sx = useAxis({ label: '', range: [-0.5, (samples?.x.length ?? 1) - 0.5], key: samples?.x.length })
  const sy = useAxis({ label: '', range: [-0.5, (samples?.y.length ?? 1) - 0.5], key: samples?.y.length, equal: sx })
  const tAxis = useAxis({ label: 'fine-tune epoch', range: [0, 30] })
  const accAxis = useAxis({ label: 'test accuracy', range: [0, 1] })

  const ft = run.fineTune ?? null

  return (
    <Figure
      title="A deep belief network, stacked one RBM at a time"
      purpose="Greedy layer-wise training fits an RBM to the data, then another RBM to the hidden activities of the first. The stack generates by Gibbs sampling in the top layer and one ancestral pass down, providing pre-trained feature weights that allow a classifier to learn from few labels."
      controls={
        <>
          <ControlRow label="Architecture & data">
            <Select
              label="Stack shape"
              value={arch}
              onChange={setArch}
              options={ARCHITECTURES.map((a) => ({ value: a.value, label: a.label }))}
            />
            <NumberSelector
              label="CD epochs per layer"
              value={epochs}
              onChange={setEpochs}
              min={10}
              max={60}
              step={10}
              suggestions={[20, 30, 50]}
            />
            <NumberSelector
              label="Pixel flip noise"
              value={flip}
              onChange={setFlip}
              min={0}
              max={0.25}
              step={0.05}
              suggestions={[0, 0.1, 0.2]}
            />
          </ControlRow>
          <ControlRow label="Fine-tune & inspection">
            <NumberSelector
              label="Labels per digit"
              value={labelsPerClass}
              onChange={setLabelsPerClass}
              min={1}
              max={5}
              step={1}
              suggestions={[1, 2, 5]}
            />
            <Player
              value={index}
              onChange={setPicked}
              count={Math.max(1, shots.length)}
              label="Checkpoint"
              format={(i) => (shots[i] ? `L${shots[i].layer + 1} ep ${shots[i].epoch}` : 'L1 ep 0')}
            />
          </ControlRow>
        </>
      }
      readouts={{
        checkpoint: (
          <>
            <Readout
              label="Checkpoint"
              value={shot ? `Layer ${shot.layer + 1} of ${sizes.length}, epoch ${shot.epoch}` : '—'}
            />
            <Readout
              label="Test accuracy (DBN init · Random init)"
              value={ft ? `${fmt(ft.pretrained.at(-1) ?? NaN)} · ${fmt(ft.random.at(-1) ?? NaN)}` : '—'}
            />
            <Readout label="Labelled · test images" value={ft ? `${ft.labelled} · ${ft.test}` : '—'} />
          </>
        ),
      }}
      caption="Synthetic 5 × 7 digit glyphs with noise train each layer greedily with contrastive divergence (CD-1) on the hidden representation of the layer below. Top left: reconstruction error per layer. Top right: layer 1 receptive fields. Bottom left: ancestral samples from top-level Gibbs sampling down the stack. Bottom right: supervised fine-tuning comparing pre-trained DBN weights against random initialization."
    >
      <Plots cols={2}>
        <Plot x={errAxis} y={eAxis} title="Layer reconstruction error">
          {run.reconstructionError.map((e, l) => (
            <Curve key={l} name={`layer ${l + 1}`} slot={l} x={e.map((_, i) => i).slice(1)} y={e.slice(1)} />
          ))}
        </Plot>
        <Plot x={fx} y={fy} title="Layer 1 receptive fields">
          {fields && (
            <Raster
              x={fields.x}
              y={fields.y}
              z={fields.z}
              scale="diverging"
              range={[-wMax, wMax]}
              valueLabel="weight"
            />
          )}
        </Plot>
        <Plot x={sx} y={sy} title="Samples (ancestral top-down)">
          {samples && (
            <Raster x={samples.x} y={samples.y} z={samples.z} scale="sequential" range={[0, 1]} valueLabel="pixel" />
          )}
        </Plot>
        <Plot x={tAxis} y={accAxis} title="Fine-tune from few labels">
          {ft && <Curve name="DBN pre-trained weights" slot={0} x={ft.pretrained.map((_, i) => i)} y={ft.pretrained} />}
          {ft && <Curve name="Random initialization" slot={1} x={ft.random.map((_, i) => i)} y={ft.random} />}
        </Plot>
      </Plots>
    </Figure>
  )
}
