import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'
import { loopData, LOOP_N } from './data'
import { fitGplvm, pcaScores, predictor, randomLatent, type Point } from './gplvm'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const ITERATIONS = 400
const EVERY = 10
const GRID = 31
const BACK_CONSTRAINT_GAMMA = 4

type Init = 'pca' | 'random'
type Mapping = 'free' | 'back'
const INIT_OPTIONS = [
  { value: 'pca' as const, label: 'PCA' },
  { value: 'random' as const, label: 'random' },
]
const MAPPING_OPTIONS = [
  { value: 'free' as const, label: 'free latent points' },
  { value: 'back' as const, label: 'back-constrained' },
]

/** Largest distance between latent points that are neighbours on the curve, over the median such distance. */
function largestGap(X: Point[]): number {
  const d = X.map((p, i) => {
    const q = X[(i + X.length - 1) % X.length]
    return Math.hypot(p[0] - q[0], p[1] - q[1])
  })
  const sorted = [...d].sort((a, b) => a - b)
  return Math.max(...d) / sorted[Math.floor(sorted.length / 2)]
}

const Y = loopData()

/**
 * A two-dimensional GP-LVM fitted in the browser to 40 points on a closed curve in four dimensions. The heatmap is the
 * posterior standard deviation of the mapping; the player steps through the optimisation.
 */
export function LatentFit({ variant }: { variant: 'gplvm' | 'back-constrained' }) {
  const isGplvm = variant === 'gplvm'
  const state = useFigureState({
    init: choice<Init>(INIT_OPTIONS, isGplvm ? 'pca' : 'random', { label: 'initialisation', when: () => isGplvm }),
    mapping: choice<Mapping>(MAPPING_OPTIONS, isGplvm ? 'free' : 'back', {
      label: 'latent points',
      when: () => !isGplvm,
    }),
    seed: int(1, { min: 1, max: 6, label: 'random seed', when: (v) => !isGplvm || v.init === 'random' }),
  })
  const { init, mapping } = state
  const seed = { value: state.seed }

  const fit = useMemo(() => {
    const back = mapping === 'back'
    const X0 = init === 'pca' ? pcaScores(Y) : randomLatent(LOOP_N, seed.value)
    const result = fitGplvm(Y, X0, {
      iterations: ITERATIONS,
      snapshotEvery: EVERY,
      backConstraint: back ? BACK_CONSTRAINT_GAMMA : undefined,
      initialAIsX0: back && init === 'random',
    })
    // Fixed axes for the whole run, so the view does not rescale while stepping.
    const all = result.snapshots.flatMap((s) => s.X)
    const lim = Math.max(1, ...all.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1])))) * 1.15
    return { ...result, lim }
  }, [init, mapping, seed.value])

  // The walk-through restarts at iteration 0 whenever the run changes.
  const [position, setPosition] = useState<{ run: typeof fit | null; k: number }>({ run: null, k: 0 })
  const k = position.run === fit ? Math.min(position.k, fit.snapshots.length - 1) : 0
  const snap = fit.snapshots[k]
  const prev = fit.snapshots[Math.max(k - 1, 0)]

  const axis = useMemo(() => toFlat(linspace(-fit.lim, fit.lim, GRID)), [fit.lim])
  const sdGrid = useMemo(() => {
    const p = predictor(Y, snap)
    return axis.map((b) => axis.map((a) => p.sd([a, b]) / snap.sf))
  }, [axis, snap])

  const overlay = useMemo(() => {
    const loop = [...snap.X, snap.X[0]]
    return [
      { name: 'curve order', x: loop.map((p) => p[0]), y: loop.map((p) => p[1]), slot: 2 },
      { name: 'latent points', x: snap.X.map((p) => p[0]), y: snap.X.map((p) => p[1]), slot: 1 },
    ] as const
  }, [snap])
  const vectors = useMemo(
    () =>
      k === 0
        ? []
        : snap.X.flatMap((p, i) => {
            const q = prev.X[i]
            return Math.hypot(p[0] - q[0], p[1] - q[1]) > 0.02 * fit.lim ? [{ from: q, to: p }] : []
          }),
    [k, snap, prev, fit.lim],
  )

  const crossing = Math.hypot(snap.X[0][0] - snap.X[20][0], snap.X[0][1] - snap.X[20][1])

  const xAxis = useAxis({ label: 'x₁' })
  const yAxis = useAxis({ label: 'x₂' })
  return (
    <Figure
      title={isGplvm ? 'Fitting a GP-LVM to a closed curve' : 'Free and back-constrained latent points'}
      caption={
        isGplvm
          ? 'Forty points on a closed curve in four dimensions; the two largest-variance coordinates trace a figure of eight and the third separates its branches. Play or step through the optimisation. Iteration 0 is the initialisation. With PCA, the curve crosses itself at the centre, placing points n = 0 and n = 20 together although they are about 1.3 apart in the data. The GP-LVM pushes them apart, and to do so it tears the loop into arcs: a long line in the curve order joins two points that are neighbours in the data. From a random start the optimiser stops in a worse local optimum. The heatmap is the posterior standard deviation of the mapping as a fraction of σ_f: dark where no data constrain it.'
          : 'The same curve, started from random latent points. Play or step through the optimisation. Free latent points end in a tangle: neighbours on the curve (joined by the lines) sit far apart in the latent space, because nothing in the GP-LVM likelihood keeps them together. The back-constrained model sets X = K_bc A, a smooth kernel map of the data with γ = 4, and optimises A. Points close in the data stay close in the latent space, and the loop comes out whole. Its log posterior is lower: the constraint trades likelihood for local distance preservation.'
      }
      state={state}
      controls={
        <Player
          label="iteration"
          value={k}
          onChange={(v) => setPosition({ run: fit, k: v })}
          count={fit.snapshots.length}
          format={(v) => String(fit.iterations[Math.min(v, fit.iterations.length - 1)])}
        />
      }
      readouts={
        <>
          <Readout label="ln p(Y | X) + ln p(X)" value={formatNumber(snap.logPost)} />
          <Readout label="length-scale ℓ" value={formatNumber(snap.ell)} />
          <Readout label="noise sd σ_n" value={formatNumber(snap.sn)} />
          <Readout label="largest gap between curve neighbours ÷ median gap" value={formatNumber(largestGap(snap.X))} />
          {isGplvm && <Readout label="latent distance, points 0 and 20" value={formatNumber(crossing)} />}
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Plot x={xAxis} y={yAxis} height={420}>
          <Raster x={axis} y={axis} z={sdGrid} scale={'sequential'} range={[0, 1]} valueLabel={'posterior sd ÷ σ_f'} />
          <Curve {...overlay[0]} live />
          <Points {...overlay[1]} live />
          <Vectors vectors={vectors} />
        </Plot>
      </div>
    </Figure>
  )
}
