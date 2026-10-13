import { useContext, useMemo, type ReactNode } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  FrameContext,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  Segments,
  setting,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, normals, stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { rbf } from 'aifn-compute/learning/kernels'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import {
  buffered,
  crossValidate,
  groupKFold,
  kFold,
  knndm,
  leaveOneOut,
  nndmLeaveOneOut,
  spatialBlocks,
  type Split,
  type Splitter,
} from 'aifn-compute/learning/validate'
import { samplePrior } from 'aifn-methods/learning/gaussian-processes'
import { kNearestNeighboursRegression } from 'aifn-methods/learning/neighbours'

const PANEL = 300
/** Training points in the unit square, test points in the same square, test points in the square to its right. */
const N_TRAIN = 150
const N_TEST = 200
/** Blocks per side tried by blocked cross-validation; each needs at least five blocks for five folds. */
const BLOCKS = [3, 4, 5, 6, 8, 10]
const FOLDS = 5
/** Buffer radii tried by buffered leave-one-out; radius 0 is ordinary leave-one-out. */
const RADII = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
const MAPS = [
  { value: 'blocks', label: 'blocked fold' },
  { value: 'buffer', label: 'buffered leave-one-out' },
] as const

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/**
 * One replicate: locations, three smooth fields drawn jointly from a Gaussian process over all locations (two
 * covariates and a residual), and the target y = z₁ + z₂²/2 + r + ε with ε ~ N(0, 0.5²).
 */
function replicate(s: Stream, range: number, coords: boolean) {
  const total = N_TRAIN + 2 * N_TEST
  const loc = Float64Array.from(toFlat(uniform(child(s, 'loc'), 0, 1, { shape: [total, 2] })))
  for (let i = N_TRAIN + N_TEST; i < total; i++) loc[2 * i] += 1
  const fields = toFlat(
    samplePrior(child(s, 'fields'), rbf({ lengthscale: range, variance: 1 }), fromData(loc, [total, 2]), 3).draws,
  )
  const noise = toFlat(normals(child(s, 'noise'), [total]))
  const width = coords ? 4 : 2
  const rows = (from: number, count: number) => {
    const x = new Float64Array(count * width)
    const y = new Float64Array(count)
    for (let k = 0; k < count; k++) {
      const i = from + k
      const z1 = fields[i]
      const z2 = fields[total + i]
      const r = fields[2 * total + i]
      const f = coords ? [loc[2 * i], loc[2 * i + 1], z1, z2] : [z1, z2]
      x.set(f, k * width)
      y[k] = z1 + 0.5 * z2 * z2 + r + 0.5 * noise[i]
    }
    return dataset(fromData(x, [count, width]), fromData(y))
  }
  return {
    loc,
    train: rows(0, N_TRAIN),
    same: rows(N_TRAIN, N_TEST),
    fresh: rows(N_TRAIN + N_TEST, N_TEST),
  }
}

/** The value of `list` nearest to `x`. */
const nearestOf = (list: readonly number[], x: number) =>
  list.reduce((a, b) => (Math.abs(b - x) < Math.abs(a - x) ? b : a))

/**
 * Random, blocked, buffered and distance-matched cross-validation of k-nearest-neighbour regression on spatially
 * autocorrelated data, beside the errors in the sampled square and in a new region.
 */
export function SpatialBlocks() {
  const state = useFigureState({
    range: float(0.1, { ge: 0.02, le: 0.4, suggestions: [0.03, 0.1, 0.2, 0.3], label: 'correlation range ℓ' }),
    map: choice(MAPS, 'blocks', { label: 'map shows' }),
    blocks: int(4, { ge: 3, le: 10, suggestions: BLOCKS, label: 'blocks per side (map)' }),
    radius: float(0.2, { ge: 0, le: 0.5, suggestions: RADII, label: 'buffer radius r (map)' }),
    replicates: int(6, { ge: 1, le: 20, suggestions: [2, 6, 12], label: 'replicate data sets' }),
    coords: setting(true, { label: 'coordinates as features' }),
  })
  const { range, replicates, coords } = state
  const blocks = BLOCKS.includes(state.blocks) ? state.blocks : 4
  const radius = nearestOf(RADII, state.radius)
  const computed = useComputed(
    () => {
      const model = kNearestNeighboursRegression({ k: 5 })
      const sums = {
        random: 0,
        same: 0,
        fresh: 0,
        nndm: 0,
        knndm: 0,
        blocked: BLOCKS.map(() => 0),
        buffered: RADII.map(() => 0),
      }
      let map: { loc: Float64Array; blocked: Split[][]; buffered: Split[][] } | null = null
      for (let r = 0; r < replicates; r++) {
        const s = child(stream('spatial-blocks'), r)
        const d = replicate(s, range, coords)
        const at = (i: number) => [d.loc[2 * i], d.loc[2 * i + 1]]
        const xy = Array.from({ length: N_TRAIN }, (_, i) => at(i))
        const area = Array.from({ length: N_TEST }, (_, k) => at(N_TRAIN + N_TEST + k))
        const cv = (splitter: Splitter, data = d.train) =>
          crossValidate(model, data, splitter, [meanSquaredError], { stream: child(s, 'cv') })
        sums.random += cv(kFold({ k: FOLDS, shuffle: true })).mean.meanSquaredError / replicates
        const blocked = BLOCKS.map((b, j) => {
          const groups = spatialBlocks({ coords: xy, size: 1 / b, origin: [0, 0] })
          const run = cv(groupKFold({ k: FOLDS }), dataset(d.train.x, d.train.y, { groups }))
          sums.blocked[j] += run.mean.meanSquaredError / replicates
          return run.splits
        })
        const buffer = RADII.map((radius, j) => {
          const run = cv(buffered(leaveOneOut(), xy, { radius }))
          sums.buffered[j] += run.mean.meanSquaredError / replicates
          return run.splits
        })
        sums.nndm += cv(nndmLeaveOneOut(xy, area, { minTrain: 0.5 })).mean.meanSquaredError / replicates
        sums.knndm +=
          cv(knndm(xy, area, { k: FOLDS, maxp: 0.5, stream: child(s, 'knndm') })).mean.meanSquaredError / replicates
        const fitted = model.fit(d.train)
        sums.same += meanSquaredError(d.same.y, fitted.decide(d.same.x)) / replicates
        sums.fresh += meanSquaredError(d.fresh.y, fitted.decide(d.fresh.x)) / replicates
        if (r === 0) map = { loc: d.loc, blocked, buffered: buffer }
      }
      return { ...sums, map }
    },
    [range, replicates, coords],
    { mode: 'release' },
  )
  const result = computed.value

  const map = useMemo(() => {
    if (!result.map) return null
    const loc = result.map.loc
    const pick = (keep: (i: number) => boolean, from: number, to: number) => {
      const idx = Array.from({ length: to - from }, (_, k) => from + k).filter(keep)
      return { x: idx.map((i) => loc[2 * i]), y: idx.map((i) => loc[2 * i + 1]) }
    }
    const fresh = pick(() => true, N_TRAIN + N_TEST, N_TRAIN + 2 * N_TEST)
    if (state.map === 'buffer') {
      // The split of the radius shown whose buffer removes the most training points.
      const splits = result.map.buffered[RADII.indexOf(radius)]
      const split = splits.reduce((a, b) => (b.train.shape[0] < a.train.shape[0] ? b : a))
      const test = new Set(Array.from(toFlat(split.test)))
      const train = new Set(Array.from(toFlat(split.train)))
      return {
        train: pick((i) => train.has(i), 0, N_TRAIN),
        test: pick((i) => test.has(i), 0, N_TRAIN),
        dropped: pick((i) => !train.has(i) && !test.has(i), 0, N_TRAIN),
        fresh,
        lines: [],
      }
    }
    const test = new Set(Array.from(toFlat(result.map.blocked[BLOCKS.indexOf(blocks)][0].test)))
    const lines = Array.from({ length: blocks + 1 }, (_, k) => k / blocks).flatMap((t) => [
      { from: [t, 0] as const, to: [t, 1] as const },
      { from: [0, t] as const, to: [1, t] as const },
    ])
    return {
      train: pick((i) => !test.has(i), 0, N_TRAIN),
      test: pick((i) => test.has(i), 0, N_TRAIN),
      dropped: { x: [], y: [] },
      fresh,
      lines,
    }
  }, [result.map, blocks, radius, state.map])

  const side = BLOCKS.map((b) => 1 / b)
  const mx = useAxis({ label: 'east', range: [0, 2] })
  const my = useAxis({ label: 'north', range: [0, 1], equal: mx })
  const bx = useAxis({ label: 'block side b or buffer radius r (unit square = 1)', range: [0, 0.52] })
  const by = useAxis({ label: 'mean squared error', range: [0, undefined] })
  const buffer = state.map === 'buffer'
  return (
    <Figure
      title="Random folds against spatial blocks and buffers"
      state={state}
      hoverReadout={false}
      caption={`A target that depends on two covariates which vary smoothly in space, plus a spatially smooth residual and independent noise (variance 0.25); all three fields are Gaussian-process draws with correlation range ℓ. ${N_TRAIN} training points lie in the left unit square. Left: the map of one data set, showing either the block grid and the test fold of the first blocked split, or the buffered leave-one-out split at radius r whose buffer removes the most training points (grey inside the square). The grey points on the right are a new region the model has never seen. Right: the mean squared error of 5-nearest-neighbour regression, averaged over the replicate data sets, estimated by random 5-fold cross-validation, by blocked 5-fold cross-validation at each block side, and by buffered leave-one-out at each radius, beside the error measured on ${N_TEST} fresh points in the same square (interpolation) and in the new region (extrapolation). The readouts add NNDM leave-one-out and kNNDM 5-fold cross-validation, which match their test-to-training distances to the distances from the new region to the sample. Drag the vertical line to change the map's block side or buffer radius. Raise ℓ to widen the gap between the two truths.`}
      readouts={{
        estimates: (
          <>
            <Readout label="random 5-fold CV" value={formatNumber(result.random)} />
            <Readout
              label={`blocked CV, ${blocks}×${blocks} blocks`}
              value={formatNumber(result.blocked[BLOCKS.indexOf(blocks)])}
            />
            <Readout
              label={`buffered LOO, r = ${radius}`}
              value={formatNumber(result.buffered[RADII.indexOf(radius)])}
            />
          </>
        ),
        'matched to the new region': (
          <>
            <Readout label="NNDM leave-one-out" value={formatNumber(result.nndm)} />
            <Readout label="kNNDM 5-fold" value={formatNumber(result.knndm)} />
          </>
        ),
        truth: (
          <>
            <Readout label="error in the same square" value={formatNumber(result.same)} />
            <Readout label="error in the new region" value={formatNumber(result.fresh)} />
          </>
        ),
      }}
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot
            x={mx}
            y={my}
            title={buffer ? 'one buffered leave-one-out split' : 'training points, test fold and new region'}
          >
            {map && map.lines.length > 0 && <Segments segments={map.lines} muted />}
            {map && map.dropped.x.length > 0 && (
              <Points name="inside the buffer" x={map.dropped.x} y={map.dropped.y} muted size={5} />
            )}
            {map && map.train.x.length > 0 && (
              <Points name="training rows" x={map.train.x} y={map.train.y} slot={0} size={5} stale={computed.stale} />
            )}
            {map && map.test.x.length > 0 && (
              <Points
                name={buffer ? 'test point' : 'test fold'}
                x={map.test.x}
                y={map.test.y}
                slot={1}
                size={buffer ? 9 : 6}
                stale={computed.stale}
              />
            )}
            {map && <Points name="new region" x={map.fresh.x} y={map.fresh.y} muted thin stale={computed.stale} />}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={bx} y={by} title="estimates and truths">
            <Curve name="blocked CV" x={side} y={result.blocked} slot={1} showPoints stale={computed.stale} />
            <Curve name="buffered LOO" x={RADII} y={result.buffered} slot={2} showPoints stale={computed.stale} />
            <Annotation y={result.random} text="random CV" dashed slot={0} />
            <Annotation y={result.same} text="same square" emphasis />
            <Annotation y={result.fresh} text="new region" emphasis dashed />
            <Handle
              kind="x"
              at={buffer ? radius : 1 / blocks}
              label="map"
              onDrag={(x) => {
                if (buffer) state.set('radius', nearestOf(RADII, x))
                else state.set('blocks', BLOCKS[side.indexOf(nearestOf(side, x))])
              }}
            />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
