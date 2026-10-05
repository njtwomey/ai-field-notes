import { useDeferredValue, useMemo, useState } from 'react'
import { choice, Figure, Handle, Plot, Points, Raster, Readout, setting, useAxis, useFigureState } from 'aifn-render'
import { normal as drawNormal, stream, uniform as drawUniform } from 'aifn-compute/foundation/random'
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

/** Uniform and normal draws from one aifn stream, in the shape the shared helpers take. */
const rand = (seed: number) => {
  const s = stream(seed)
  return { uniform: () => drawUniform(s), normal: () => drawNormal(s) }
}

const PIXELS = Array.from({ length: SIZE }, (_, i) => i)
const IMAGE_RANGE: [number, number] = [-0.3, 1.7]
const MAP_RANGE: [number, number] = [0, 1.5]

/** Eight normal training images: stripes of random phase, each with one bright rivet at a random place. */
const BANK = (() => {
  const r = rand(3)
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
  const state = useFigureState({
    sampling: choice<Sampling>(SAMPLINGS, 'coreset', { label: 'memory bank' }),
    fraction: choice(FRACTIONS, '0.1', { label: 'bank size' }),
    withDent: setting(true, 'dent in the test image'),
  })
  const [rivet, setRivet] = useState<Spot>([20, 8])
  const [dent, setDent] = useState<Spot>([10, 18])

  const memory = useMemo(() => {
    const f = Number(state.fraction)
    const k = Math.max(1, Math.round(f * BANK.length))
    if (f === 1) return BANK
    const idx = state.sampling === 'coreset' ? greedyCoreset(BANK, k) : randomSubset(BANK.length, k, rand(5))
    return idx.map((i) => BANK[i])
  }, [state.sampling, state.fraction])

  const image = useMemo(() => texture(1, rivet, state.withDent ? dent : null, rand(99)), [rivet, dent, state.withDent])
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
    ...(state.withDent
      ? [{ kind: 'point' as const, at: dent, label: 'dent', onDrag: (p: [number, number]) => setDent(clampSpot(p)) }]
      : []),
  ]

  const xAxis = useAxis({ label: 'column' })
  const yAxis = useAxis({ label: 'row' })
  const xAxis2 = useAxis({ label: 'column' })
  const yAxis2 = useAxis({ label: 'row' })
  return (
    <Figure
      title="A patch memory bank, and why PatchCore subsamples it with a coreset"
      state={state}
      caption="Left: a 28 × 28 test image of diagonal stripes with a bright rivet, which is normal, and a dark dent, which is a defect. Drag either. Right: each 5 × 5 patch's distance to its nearest patch in a memory bank built from eight normal training images (4,608 patches), each with a rivet in a random place. The image's anomaly score is the largest patch score, marked with a diamond. Shrink the bank. A random subset loses most of the rare rivet patches, so at 10 % the largest patch score in the image sits beside the normal rivet instead of the dent. A greedy coreset keeps the patches farthest from those already kept, which preserves rare normal patterns and keeps the rivet low. The features here are raw pixels; PatchCore uses a pretrained network's mid-level features."

      readouts={
        <>
          <Readout label="patches in bank" value={memory.length} />
          <Readout label="image score (max)" value={top.toFixed(2)} />
          <Readout label="rivet patch" value={at(rivet).toFixed(2)} />
          {state.withDent && <Readout label="dent patch" value={at(dent).toFixed(2)} />}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Raster x={PIXELS} y={PIXELS} z={z} range={IMAGE_RANGE} valueLabel={'intensity'} />
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          <Raster x={CENTRES} y={CENTRES} z={map} range={MAP_RANGE} valueLabel={'patch score'} />
          {peak && <Points x={[peak[0]]} y={[peak[1]]} emphasis live />}
        </Plot>
      </div>
    </Figure>
  )
}
