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
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  halton,
  latinHypercube,
  maxGap,
  randomShift,
  scrambledHalton,
  scrambledSobol,
  sobol,
  starDiscrepancy,
} from 'aifn-compute/numerics/quadrature'
import { bestSoFar, SVM_BEST, SVM_HI, SVM_LO, SVM_SPACE, svmError, svmRaster } from '../_shared/svmSurface'

const MAX_TRIALS = 256
const RUNS = 8

type Design = 'random' | 'latin' | 'shifted' | 'scrambled-sobol' | 'scrambled-halton' | 'sobol' | 'halton'
type Point = { logC: number; logGamma: number }

/** The design's N points in the unit square, as [u1, u2] rows. */
function unitDesign(design: Design, n: number, seed: number): number[][] {
  const key = `quasi-random-designs/${design}/${seed}`
  const rows = (flat: ArrayLike<number>) => Array.from({ length: n }, (_, i) => [flat[2 * i], flat[2 * i + 1]])
  switch (design) {
    case 'random': {
      const s = stream(key)
      return Array.from({ length: n }, (_, i) => {
        const c = child(s, 'trial', i)
        return [uniform(c), uniform(c)]
      })
    }
    case 'latin':
      return rows(toFlat(latinHypercube(stream(key), n, 2)))
    case 'shifted':
      return rows(toFlat(randomShift(stream(key), sobol(n, 2))))
    case 'scrambled-sobol':
      return rows(toFlat(scrambledSobol(stream(key), n, 2)))
    case 'scrambled-halton':
      return rows(toFlat(scrambledHalton(stream(key), n, 2)))
    case 'sobol':
      return rows(toFlat(sobol(n, 2)))
    case 'halton':
      return rows(toFlat(halton(n, 2, { skip: 1 })))
  }
}

/** Unit-square points decoded into the SVM space. */
const decodeAll = (cube: number[][]): Point[] => cube.map((u) => decode(SVM_SPACE, u))

const makeDesign = (design: Design, n: number, seed: number) => decodeAll(unitDesign(design, n, seed))

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
        { value: 'scrambled-sobol', label: 'Sobol, scrambled' },
        { value: 'scrambled-halton', label: 'Halton, scrambled' },
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
  const chosen = useMemo(() => {
    const cube = unitDesign(design, trials, seed)
    return {
      points: decodeAll(cube),
      discrepancy: starDiscrepancy(cube),
      gap: maxGap(cube.map((u) => u[0])),
    }
  }, [design, trials, seed])
  const points = chosen.points

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
      caption="Left: N trials of the chosen design over the tabulated cross-validated error of an RBF support vector machine (colour clipped at 0.15), with each trial's log C repeated along the bottom edge to show the projection onto one axis. Independent draws clump and leave gaps; a Latin hypercube puts one trial in each of N slices of each axis; Sobol and Halton points fill the largest remaining gaps as N grows, and a random shift or scramble keeps that evenness while making repeats independent. The readouts give the design's exact star discrepancy D* in the unit square, the area 4D* above which every axis-aligned box must hold a trial, and the largest gap left on the log C axis as a share of its range. Right: the best error within the first N trials. Thin orange lines are eight independent random-search runs and the thick orange line their average; blue is the Sobol sequence and green the Halton sequence, which are fixed. The dashed line is the lowest error in the table."
      readouts={
        <>
          <Readout label={`best error, this design (N = ${trials})`} value={formatNumber(best)} />
          <Readout label="star discrepancy D*" value={formatNumber(chosen.discrepancy)} />
          <Readout label="empty-box bound 4D*" value={formatNumber(Math.min(1, 4 * chosen.discrepancy))} />
          <Readout label="largest gap in log C (share of range)" value={formatNumber(chosen.gap)} />
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
