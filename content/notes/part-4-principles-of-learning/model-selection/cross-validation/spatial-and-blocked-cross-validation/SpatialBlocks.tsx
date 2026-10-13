import { useContext, useMemo, type ReactNode } from 'react'
import {
  Annotation,
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
import { crossValidate, groupKFold, kFold, type Split } from 'aifn-compute/learning/validate'
import { samplePrior } from 'aifn-methods/learning/gaussian-processes'
import { kNearestNeighboursRegression } from 'aifn-methods/learning/neighbours'

const PANEL = 300
/** Training points in the unit square, test points in the same square, test points in the square to its right. */
const N_TRAIN = 150
const N_TEST = 200
/** Blocks per side tried by blocked cross-validation; each needs at least five blocks for five folds. */
const BLOCKS = [3, 4, 5, 6, 8, 10]
const FOLDS = 5

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** The block (row-major on a b × b grid of the unit square) that a point falls in. */
function blockOf(u: number, v: number, b: number) {
  return Math.min(b - 1, Math.floor(u * b)) * b + Math.min(b - 1, Math.floor(v * b))
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

/** Random and blocked cross-validation of k-nearest-neighbour regression on spatially autocorrelated data. */
export function SpatialBlocks() {
  const state = useFigureState({
    range: float(0.1, { ge: 0.02, le: 0.4, suggestions: [0.03, 0.1, 0.2, 0.3], label: 'correlation range ℓ' }),
    blocks: int(4, { ge: 3, le: 10, suggestions: BLOCKS, label: 'blocks per side (map)' }),
    replicates: int(6, { ge: 1, le: 20, suggestions: [2, 6, 12], label: 'replicate data sets' }),
    coords: setting(true, { label: 'coordinates as features' }),
  })
  const { range, replicates, coords } = state
  const blocks = BLOCKS.includes(state.blocks) ? state.blocks : 4
  const computed = useComputed(
    () => {
      const model = kNearestNeighboursRegression({ k: 5 })
      const sums = { random: 0, same: 0, fresh: 0, blocked: BLOCKS.map(() => 0) }
      let map: { loc: Float64Array; splits: Split[][] } | null = null
      for (let r = 0; r < replicates; r++) {
        const s = child(stream('spatial-blocks'), r)
        const d = replicate(s, range, coords)
        const random = crossValidate(model, d.train, kFold({ k: FOLDS, shuffle: true }), [meanSquaredError], {
          stream: child(s, 'cv'),
        })
        sums.random += random.mean.meanSquaredError / replicates
        const splitsPerB: Split[][] = []
        BLOCKS.forEach((b, j) => {
          const groups = Array.from({ length: N_TRAIN }, (_, i) => blockOf(d.loc[2 * i], d.loc[2 * i + 1], b))
          const cv = crossValidate(model, dataset(d.train.x, d.train.y, { groups }), groupKFold({ k: FOLDS }), [
            meanSquaredError,
          ])
          sums.blocked[j] += cv.mean.meanSquaredError / replicates
          splitsPerB.push(cv.splits)
        })
        const fitted = model.fit(d.train)
        sums.same += meanSquaredError(d.same.y, fitted.decide(d.same.x)) / replicates
        sums.fresh += meanSquaredError(d.fresh.y, fitted.decide(d.fresh.x)) / replicates
        if (r === 0) map = { loc: d.loc, splits: splitsPerB }
      }
      return { ...sums, map }
    },
    [range, replicates, coords],
    { mode: 'release' },
  )
  const result = computed.value

  const map = useMemo(() => {
    if (!result.map) return null
    const j = BLOCKS.indexOf(blocks)
    const test = new Set(Array.from(toFlat(result.map.splits[j][0].test)))
    const loc = result.map.loc
    const pick = (keep: (i: number) => boolean, from: number, to: number) => {
      const idx = Array.from({ length: to - from }, (_, k) => from + k).filter(keep)
      return { x: idx.map((i) => loc[2 * i]), y: idx.map((i) => loc[2 * i + 1]) }
    }
    const lines = Array.from({ length: blocks + 1 }, (_, k) => k / blocks).flatMap((t) => [
      { from: [t, 0] as const, to: [t, 1] as const },
      { from: [0, t] as const, to: [1, t] as const },
    ])
    return {
      train: pick((i) => !test.has(i), 0, N_TRAIN),
      test: pick((i) => test.has(i), 0, N_TRAIN),
      fresh: pick(() => true, N_TRAIN + N_TEST, N_TRAIN + 2 * N_TEST),
      lines,
    }
  }, [result.map, blocks])

  const side = BLOCKS.map((b) => 1 / b)
  const mx = useAxis({ label: 'east', range: [0, 2] })
  const my = useAxis({ label: 'north', range: [0, 1], equal: mx })
  const bx = useAxis({ label: 'block side (unit square = 1)', range: [0.08, 0.36] })
  const by = useAxis({ label: 'mean squared error', range: [0, undefined] })
  return (
    <Figure
      title="Random folds against spatial blocks"
      state={state}
      hoverReadout={false}
      caption={`A target that depends on two covariates which vary smoothly in space, plus a spatially smooth residual and independent noise (variance 0.25); all three fields are Gaussian-process draws with correlation range ℓ. ${N_TRAIN} training points lie in the left unit square. Left: the map of one data set, with the block grid and the test fold of the first blocked split; the points on the right are a new region the model has never seen. Right: the mean squared error of 5-nearest-neighbour regression, averaged over the replicate data sets, estimated by random 5-fold cross-validation, by blocked 5-fold cross-validation at each block size, and measured on ${N_TEST} fresh points in the same square (interpolation) and in the new region (extrapolation). Drag the vertical line, or set the blocks per side, to change the map's grid. Raise ℓ to widen the gap between the two truths.`}
      readouts={{
        estimates: (
          <>
            <Readout label="random 5-fold CV" value={formatNumber(result.random)} />
            <Readout
              label={`blocked CV, ${blocks}×${blocks} blocks`}
              value={formatNumber(result.blocked[BLOCKS.indexOf(blocks)])}
            />
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
          <Plot x={mx} y={my} title="training points, test fold and new region">
            {map && <Segments segments={map.lines} muted />}
            {map && map.train.x.length > 0 && (
              <Points name="training fold" x={map.train.x} y={map.train.y} slot={0} size={5} stale={computed.stale} />
            )}
            {map && map.test.x.length > 0 && (
              <Points name="test fold" x={map.test.x} y={map.test.y} slot={1} size={6} stale={computed.stale} />
            )}
            {map && <Points name="new region" x={map.fresh.x} y={map.fresh.y} slot={2} thin stale={computed.stale} />}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={bx} y={by} title="estimates and truths">
            <Curve name="blocked CV" x={side} y={result.blocked} slot={1} showPoints stale={computed.stale} />
            <Annotation y={result.random} text="random CV" dashed slot={0} />
            <Annotation y={result.same} text="same square" emphasis />
            <Annotation y={result.fresh} text="new region" emphasis dashed />
            <Handle
              kind="x"
              at={1 / blocks}
              label="map"
              onDrag={(x) => {
                const nearest = BLOCKS.reduce((a, b) => (Math.abs(1 / b - x) < Math.abs(1 / a - x) ? b : a))
                state.set('blocks', nearest)
              }}
            />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
