import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { child, stream, uniform } from 'aifn-compute/foundation/random'
import { decode } from 'aifn-compute/foundation/space'
import { run } from 'aifn-compute/foundation/trace'
import { halton, sobol } from 'aifn-compute/numerics/quadrature'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { bayesianOptimisationSteps } from 'aifn-methods/optimisation/bayesian'
import { bestSoFar, SVM_BEST, SVM_HI, SVM_LO, SVM_SPACE, svmError, svmRaster } from '../_shared/svmSurface'

const MAX_TRIALS = 256
const RUNS = 8

type Design = 'random' | 'latin' | 'shifted' | 'sobol' | 'halton'
type Point = { logC: number; logGamma: number }

/** Unit-cube points, an n × 2 row-major array, decoded into the SVM space. */
function decodeAll(cube: ArrayLike<number>, n: number): Point[] {
  return Array.from({ length: n }, (_, i) => decode(SVM_SPACE, [cube[2 * i], cube[2 * i + 1]]))
}

/**
 * An initial design of the engine's Bayesian-optimisation loop, which places `n` points in the box by a Latin
 * hypercube or by randomly shifted Sobol points: the state before any step holds them.
 */
function loopDesign(design: 'lhs' | 'sobol', n: number, seed: string): Point[] {
  const alg = bayesianOptimisationSteps(
    (x) => svmError(toFlat(x)[0], toFlat(x)[1]),
    { lo: [...SVM_LO], hi: [...SVM_HI] },
    {
      init: n,
      design,
      refit: false,
      grid: 2,
    },
  )
  const x = run(alg, undefined, 0, { stream: stream(seed) }).x.data
  return Array.from({ length: n }, (_, i) => ({ logC: x[2 * i], logGamma: x[2 * i + 1] }))
}

function makeDesign(design: Design, n: number, seed: number): Point[] {
  const key = `quasi-random-designs/${design}/${seed}`
  switch (design) {
    case 'random': {
      const s = stream(key)
      return Array.from({ length: n }, (_, i) => {
        const c = child(s, 'trial', i)
        return decode(SVM_SPACE, [uniform(c), uniform(c)])
      })
    }
    case 'latin':
      return loopDesign('lhs', n, key)
    case 'shifted':
      return loopDesign('sobol', n, key)
    case 'sobol':
      return decodeAll(sobol(n, 2).data, n)
    case 'halton':
      return decodeAll(halton(n, 2, { skip: 1 }).data, n)
  }
}

const errorsOf = (points: Point[]) => points.map((p) => svmError(p.logC, p.logGamma))

/**
 * Space-filling designs on the SVM tuning surface (left) and the best error against the number of trials for random,
 * Sobol and Halton designs (right).
 */
export function QuasiRandomDesigns() {
  const state = useFigureState({
    design: choice<Design>(
      [
        { value: 'random', label: 'independent uniform' },
        { value: 'latin', label: 'Latin hypercube' },
        { value: 'shifted', label: 'Sobol, randomly shifted' },
        { value: 'sobol', label: 'Sobol' },
        { value: 'halton', label: 'Halton (origin skipped)' },
      ],
      'sobol',
      { label: 'design' },
    ),
    trials: int(32, { ge: 2, le: MAX_TRIALS, suggestions: [8, 16, 32, 64, 128], label: 'trials N' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { design, trials, seed } = state
  const raster = useMemo(() => svmRaster(81), [])
  const points = useMemo(() => makeDesign(design, trials, seed), [design, trials, seed])

  const curves = useMemo(() => {
    const n = Array.from({ length: MAX_TRIALS }, (_, t) => t + 1)
    const random = Array.from({ length: RUNS }, (_, r) =>
      bestSoFar(errorsOf(makeDesign('random', MAX_TRIALS, seed + 1000 * r))),
    )
    const average = n.map((_, t) => random.reduce((a, run) => a + run[t], 0) / RUNS)
    return {
      n,
      random,
      average,
      sobol: bestSoFar(errorsOf(makeDesign('sobol', MAX_TRIALS, seed))),
      halton: bestSoFar(errorsOf(makeDesign('halton', MAX_TRIALS, seed))),
    }
  }, [seed])

  const best = Math.min(...errorsOf(points))
  const xAxis = useAxis({ label: 'log₁₀ C', range: [SVM_LO[0], SVM_HI[0]] })
  const yAxis = useAxis({ label: 'log₁₀ γ', range: [SVM_LO[1], SVM_HI[1]] })
  const nAxis = useAxis({ label: 'trials N', log: true, range: [1, MAX_TRIALS] })
  const eAxis = useAxis({ label: 'best error found', range: [0.035, 0.13] })
  return (
    <Figure
      title="Space-filling designs on an SVM's cross-validated error"
      state={state}
      caption="Left: N trials of the chosen design over the tabulated cross-validated error of an RBF support vector machine (colour clipped at 0.15), with each trial's log C repeated along the bottom edge to show the projection onto one axis. Independent draws clump and leave gaps; a Latin hypercube puts one trial in each of N slices of each axis; Sobol and Halton points fill the largest remaining gaps as N grows. Right: the best error within the first N trials. Thin orange lines are eight independent random-search runs and the thick orange line their average; blue is the Sobol sequence and green the Halton sequence, which are fixed. The dashed line is the lowest error in the table."
      readouts={
        <>
          <Readout label={`best error, this design (N = ${trials})`} value={formatNumber(best)} />
          <Readout label="random search, average best" value={formatNumber(curves.average[trials - 1])} />
          <Readout label="Sobol, best" value={formatNumber(curves.sobol[trials - 1])} />
          <Readout label="Halton, best" value={formatNumber(curves.halton[trials - 1])} />
          <Readout label="lowest error in the table" value={formatNumber(SVM_BEST)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Raster
            x={raster.cAxis}
            y={raster.gammaAxis}
            z={raster.z}
            range={[SVM_BEST, 0.15]}
            valueLabel="cross-validated error"
            colorBar={false}
          />
          <Points name="trials" x={points.map((p) => p.logC)} y={points.map((p) => p.logGamma)} emphasis />
          <Points
            name="projection onto log C"
            x={points.map((p) => p.logC)}
            y={points.map(() => SVM_LO[1])}
            muted
            thin
          />
        </Plot>
        <Plot x={nAxis} y={eAxis} height={340}>
          {curves.random.map((r, i) => (
            <Curve key={i} name="random search, one run" x={curves.n} y={r} slot={1} thin silent />
          ))}
          <Curve name="random search, average" x={curves.n} y={curves.average} slot={1} />
          <Curve name="Sobol" x={curves.n} y={curves.sobol} slot={0} />
          <Curve name="Halton" x={curves.n} y={curves.halton} slot={2} />
          <Curve name="lowest error" x={[1, MAX_TRIALS]} y={[SVM_BEST, SVM_BEST]} muted dashed />
        </Plot>
      </div>
    </Figure>
  )
}
