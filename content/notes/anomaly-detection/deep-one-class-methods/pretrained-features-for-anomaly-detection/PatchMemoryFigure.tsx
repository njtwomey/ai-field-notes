import { useDeferredValue, useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamChoice, ParamSwitch, Readout, type Handle } from '@/components/viz'
import { rng } from '@/lib/math'
import {
  CENTRES,
  SIZE,
  anomalyMap,
  greedyCoreset,
  patches,
  randomSubset,
  texture,
  type Spot,
} from '../_shared/patchMemory'

const PIXELS = Array.from({ length: SIZE }, (_, i) => i)
const IMAGE_RANGE: [number, number] = [-0.3, 1.7]
const MAP_RANGE: [number, number] = [0, 1.5]

/** Eight normal training images: stripes of random phase, each with one bright rivet at a random place. */
const BANK = (() => {
  const r = rng(3)
  return Array.from({ length: 8 }, () => {
    const phase = 2 * Math.PI * r.uniform()
    const rivet: Spot = [4 + Math.floor(20 * r.uniform()), 4 + Math.floor(20 * r.uniform())]
    return patches(texture(phase, rivet, null, r))
  }).flat()
})()

type Sampling = 'coreset' | 'random'
const SAMPLINGS = [
  { value: 'coreset' as const, label: 'greedy coreset' },
  { value: 'random' as const, label: 'random subset' },
]
const FRACTIONS = [
  { value: '1', label: '100 %' },
  { value: '0.3', label: '30 %' },
  { value: '0.1', label: '10 %' },
  { value: '0.03', label: '3 %' },
  { value: '0.01', label: '1 %' },
]

const clampSpot = ([c, r]: [number, number]): Spot => [
  Math.round(Math.min(CENTRES[CENTRES.length - 1], Math.max(CENTRES[0], c))),
  Math.round(Math.min(CENTRES[CENTRES.length - 1], Math.max(CENTRES[0], r))),
]

/** Nearest-neighbour patch scores against a memory bank, with coreset and random subsampling of the bank. */
export function PatchMemoryFigure() {
  const [sampling, setSampling] = useState<Sampling>('coreset')
  const [fraction, setFraction] = useState('0.1')
  const [withDent, setWithDent] = useState(true)
  const [rivet, setRivet] = useState<Spot>([20, 8])
  const [dent, setDent] = useState<Spot>([10, 18])

  const memory = useMemo(() => {
    const f = Number(fraction)
    const k = Math.max(1, Math.round(f * BANK.length))
    if (f === 1) return BANK
    const idx = sampling === 'coreset' ? greedyCoreset(BANK, k) : randomSubset(BANK.length, k, rng(5))
    return idx.map((i) => BANK[i])
  }, [sampling, fraction])

  const image = useMemo(() => texture(1, rivet, withDent ? dent : null, rng(99)), [rivet, dent, withDent])
  const deferred = useDeferredValue({ image, memory })
  const map = useMemo(() => anomalyMap(deferred.image, deferred.memory), [deferred])

  const z = useMemo(() => PIXELS.map((row) => PIXELS.map((col) => image[row * SIZE + col])), [image])
  const flat = map.flat()
  const top = Math.max(...flat)
  const at = (s: Spot) => map[CENTRES.indexOf(s[1])][CENTRES.indexOf(s[0])]
  const argmax = flat.indexOf(top)
  const peak: [number, number] = [CENTRES[argmax % CENTRES.length], CENTRES[Math.floor(argmax / CENTRES.length)]]

  const handles: Handle[] = [
    { kind: 'point', at: rivet, label: 'rivet', onDrag: (p) => setRivet(clampSpot(p)) },
    ...(withDent
      ? [{ kind: 'point' as const, at: dent, label: 'dent', onDrag: (p: [number, number]) => setDent(clampSpot(p)) }]
      : []),
  ]

  return (
    <Interactive
      title="A patch memory bank, and why PatchCore subsamples it with a coreset"
      caption="Left: a 28 × 28 test image of diagonal stripes with a bright rivet, which is normal, and a dark dent, which is a defect. Drag either. Right: each 5 × 5 patch's distance to its nearest patch in a memory bank built from eight normal training images (4,608 patches), each with a rivet in a random place. The image's anomaly score is the largest patch score, marked with a diamond. Shrink the bank. A random subset loses most of the rare rivet patches, so at 10 % the largest patch score in the image sits beside the normal rivet instead of the dent. A greedy coreset keeps the patches farthest from those already kept, which preserves rare normal patterns and keeps the rivet low. The features here are raw pixels; PatchCore uses a pretrained network's mid-level features."
      controls={
        <>
          <ParamChoice label="memory bank" value={sampling} onChange={setSampling} options={SAMPLINGS} />
          <ParamChoice label="bank size" value={fraction} onChange={setFraction} options={FRACTIONS} />
          <ParamSwitch label="dent in the test image" checked={withDent} onChange={setWithDent} />
        </>
      }
      readout={
        <>
          <Readout label="patches in bank" value={memory.length} />
          <Readout label="image score (max)" value={top.toFixed(2)} />
          <Readout label="rivet patch" value={at(rivet).toFixed(2)} />
          {withDent && <Readout label="dent patch" value={at(dent).toFixed(2)} />}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Heatmap
          x={PIXELS}
          y={PIXELS}
          z={z}
          range={IMAGE_RANGE}
          handles={handles}
          xLabel="column"
          yLabel="row"
          valueLabel="intensity"
          height={340}
        />
        <Heatmap
          x={CENTRES}
          y={CENTRES}
          z={map}
          range={MAP_RANGE}
          marker={peak}
          xLabel="column"
          yLabel="row"
          valueLabel="patch score"
          height={340}
        />
      </div>
    </Interactive>
  )
}
