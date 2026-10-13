import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { bayesianOptimisationSteps, type BayesianObjective } from 'aifn-methods/optimisation/bayesian'
import { svmTuningSurface } from 'aifn-methods/data/objectives'
import { child, stream } from 'aifn-compute/foundation/random'
import { sample } from 'aifn-compute/foundation/space'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { SVM_BEST, SVM_HI, SVM_LO, SVM_SPACE, SVM_SURFACE, svmError, svmRaster } from '../../_shared/svmSurface'
import { FixedHeight } from '../_shared/FixedHeight'

/** Evaluations in the initial design: a Latin hypercube drawn from the run's stream. */
const INIT = 5
/** The colour range of the error maps, clipped so that the valley of good settings shows. */
const ERROR_RANGE: [number, number] = [SVM_BEST, 0.15]

/** The running best of a list of observed values, with the true error of the setting that holds it. */
function incumbentTrace(observed: readonly number[], truth: readonly number[]) {
  const best: number[] = []
  const bestTruth: number[] = []
  observed.forEach((v, i) => {
    const keep = i > 0 && best[i - 1] <= v
    best.push(keep ? best[i - 1] : v)
    bestTruth.push(keep ? bestTruth[i - 1] : truth[i])
  })
  return { best, bestTruth }
}

/** The Player's position, reset to 0 whenever the run changes. */
function useCursor(key: string) {
  const [cursor, setCursor] = useState({ key, step: 0 })
  const step = cursor.key === key ? cursor.step : 0
  return [step, (v: number) => setCursor({ key, step: v })] as const
}

/**
 * One run of Gaussian-process expected improvement on the tabulated SVM error, scrubbed one evaluation at a time,
 * against random search with the same budget.
 */
export function SvmBoWalkthrough() {
  const state = useFigureState({
    budget: int(30, { label: 'evaluations N', min: INIT + 1, max: 50, suggestions: [15, 20, 30, 40] }),
    noise: float(0, { label: 'evaluation noise sd', ge: 0, le: 0.05, suggestions: [0, 0.01, 0.02] }),
    seed: int(1, { label: 'seed', min: 1, max: 99, suggestions: [1, 2, 3, 4] }),
  })
  const { budget, noise, seed } = state

  const runs = useMemo(() => {
    const noisy = svmTuningSurface({ noise })
    const objective: BayesianObjective = noise > 0 ? (x, s) => noisy.value(x, s) : (x) => SVM_SURFACE.mean(x)
    const alg = bayesianOptimisationSteps(
      objective,
      { lo: SVM_LO, hi: SVM_HI },
      { init: INIT, noise: noise > 0 ? 'fit' : 1e-6 },
    )
    const states = trace(alg, undefined, budget - INIT, { stream: stream(`svm-bo/${seed}`), keep: 'all' }).steps
    const grid = toFlat(alg.grid)
    const [nC, nGamma] = alg.shape
    const cAxis = Array.from({ length: nC }, (_, i) => grid[2 * i * nGamma])
    const gammaAxis = Array.from({ length: nGamma }, (_, j) => grid[2 * j + 1])
    const last = states[states.length - 1]
    const xs = toFlat(last.x)
    const boPoints = Array.from({ length: last.evaluations }, (_, i) => [xs[2 * i], xs[2 * i + 1]] as const)
    const bo = incumbentTrace(
      Array.from(toFlat(last.y)),
      boPoints.map(([c, g]) => svmError(c, g)),
    )
    // Random search: N independent uniform draws on the same box, each scored with the same evaluation noise.
    const rs = stream(`svm-random/${seed}`)
    const rsPoints = Array.from({ length: budget }, (_, t) => {
      const p = sample(child(rs, 'trial', t), SVM_SPACE) as Record<string, number>
      return [p.logC, p.logGamma] as const
    })
    const rsTruth = rsPoints.map(([c, g]) => svmError(c, g))
    const rsObserved = rsPoints.map(([c, g], t) =>
      noise > 0 ? noisy.value(tensor([c, g]), child(rs, 'noise', t)) : rsTruth[t],
    )
    return { states, cAxis, gammaAxis, nGamma, boPoints, bo, rsPoints, rs: incumbentTrace(rsObserved, rsTruth) }
  }, [budget, noise, seed])

  const truthMap = useMemo(() => svmRaster(81), [])
  const runKey = `${budget}|${noise}|${seed}`
  const [step, setStep] = useCursor(runKey)
  const at = Math.min(step, runs.states.length - 1)
  const s = runs.states[at]
  const n = s.evaluations

  const maps = useMemo(() => {
    const mean = toFlat(s.mean)
    const acq = toFlat(s.acquisition)
    // The grid's last coordinate (log γ) varies fastest; a Raster's rows run along y.
    const field = (v: ArrayLike<number>) =>
      runs.gammaAxis.map((_, j) => runs.cAxis.map((_, i) => v[i * runs.nGamma + j]))
    return { mean: field(mean), acq: field(acq), next: toFlat(s.next) }
  }, [s, runs])

  const seen = runs.boPoints.slice(0, n)
  const random = runs.rsPoints.slice(0, n)
  const evals = Array.from({ length: budget }, (_, i) => i + 1)
  const c = useAxis({ label: 'log₁₀ C', range: [SVM_LO[0], SVM_HI[0]] })
  const g = useAxis({ label: 'log₁₀ γ', range: [SVM_LO[1], SVM_HI[1]] })
  const nAxis = useAxis({ label: 'evaluations', range: [1, budget], integer: true })
  const eAxis = useAxis({ label: 'error of the incumbent', range: [noise > 0 ? 0 : 0.035, 0.13] })

  return (
    <Figure
      title="Expected improvement against random search on an SVM's error surface"
      state={state}
      caption="Top left: the cross-validated error of an RBF support vector machine over log C and log γ (colour, clipped at 0.15), the evaluations of Bayesian optimisation so far (orange; the latest one larger), the next one it chose (ink) and as many random-search trials (green). Top right: the Gaussian-process posterior mean after those evaluations, on the same colour scale. Bottom left: expected improvement over the 41 by 41 candidate grid; its maximum is the next evaluation. Bottom right: the true error of the configuration each method would return, against the number of evaluations. Step through with the player. Add evaluation noise to see the search chase lucky scores: the observed incumbent falls far below its true error."
      controls={
        <Player
          value={at}
          onChange={setStep}
          count={runs.states.length}
          label="step"
          format={(v) => `${INIT + v} evaluations`}
        />
      }
      readouts={
        <>
          <Readout label="evaluations t" value={String(n)} />
          <Readout label="incumbent f*ₜ (observed)" value={formatNumber(s.best)} />
          <Readout label="its true error" value={formatNumber(runs.bo.bestTruth[n - 1])} />
          <Readout
            label="at (log C, log γ)"
            value={`(${formatNumber(toFlat(s.bestX)[0])}, ${formatNumber(toFlat(s.bestX)[1])})`}
          />
          <Readout label="random search, true error" value={formatNumber(runs.rs.bestTruth[n - 1])} />
          <Readout label="lowest error in the table" value={formatNumber(SVM_BEST)} />
          <Readout label="length-scale (unit box)" value={formatNumber(Number(s.hyper.lengthscale))} />
        </>
      }
    >
      <FixedHeight height={640}>
        <Plots rows={2} cols={2}>
          <Plot x={c} y={g} title="error and evaluations">
            <Raster
              x={truthMap.cAxis}
              y={truthMap.gammaAxis}
              z={truthMap.z}
              range={ERROR_RANGE}
              valueLabel="cross-validated error"
            />
            <Points name="random search" x={random.map((p) => p[0])} y={random.map((p) => p[1])} slot={2} thin />
            <Points name="Bayesian optimisation" x={seen.map((p) => p[0])} y={seen.map((p) => p[1])} slot={1} />
            <Points name="latest evaluation" x={[seen[n - 1][0]]} y={[seen[n - 1][1]]} slot={1} size={12} />
            <Points name="next evaluation" x={[maps.next[0]]} y={[maps.next[1]]} emphasis size={10} />
          </Plot>
          <Plot x={c} y={g} title="posterior mean μₜ">
            <Raster x={runs.cAxis} y={runs.gammaAxis} z={maps.mean} range={ERROR_RANGE} valueLabel="posterior mean" />
            <Points name="Bayesian optimisation" x={seen.map((p) => p[0])} y={seen.map((p) => p[1])} slot={1} />
          </Plot>
          <Plot x={c} y={g} title="expected improvement αₜ">
            <Raster x={runs.cAxis} y={runs.gammaAxis} z={maps.acq} valueLabel="expected improvement" />
            <Points name="next evaluation" x={[maps.next[0]]} y={[maps.next[1]]} emphasis size={10} />
          </Plot>
          <Plot x={nAxis} y={eAxis} title="progress">
            <Curve name="lowest error" x={[1, budget]} y={[SVM_BEST, SVM_BEST]} muted dashed />
            <Curve name="random search" x={evals} y={runs.rs.bestTruth} slot={2} />
            <Curve name="Bayesian optimisation" x={evals.slice(0, n)} y={runs.bo.bestTruth.slice(0, n)} slot={1} />
            {noise > 0 && (
              <Curve
                name="Bayesian optimisation, observed score"
                x={evals.slice(0, n)}
                y={runs.bo.best.slice(0, n)}
                slot={1}
                dashed
              />
            )}
          </Plot>
        </Plots>
      </FixedHeight>
    </Figure>
  )
}
