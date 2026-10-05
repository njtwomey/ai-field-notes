import { useMemo, useState } from 'react'
import {
  fitSparseGp,
  gpPosterior,
  logMarginalLikelihood,
  samplePrior,
  sparseGp,
  type SparseMethod,
} from 'aifn-methods/learning/gaussian-processes'
import { rbf } from 'aifn-compute/learning/kernels'
import { child, normals, stream, uniform } from 'aifn-compute/foundation/random'
import { argmax, linspace, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { Button } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { choice, row, slider, toggle, useFigureState } from 'aifn-render/state'
import { Area, Curve, Handle, Plot, Plots, Points, Readout, useAxis } from 'aifn-render/viz'
import { formatValue } from '@lab/views'

const GRID = linspace(0, 5, 201)
const GRID_X = toFlat(GRID)
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/** The mean and the ± 2 sd edges of a Gaussian prediction on the grid. */
function band(mean: number[], variance: number[]) {
  const sd = variance.map(Math.sqrt)
  return { mean, upper: mean.map((m, i) => m + 2 * sd[i]), lower: mean.map((m, i) => m - 2 * sd[i]) }
}

const START: [number, number][] = [
  [0.6, -0.4],
  [1.5, 0.9],
  [2.4, 0.3],
  [3.6, -1.1],
  [4.3, 0.2],
]

/** GP regression: prior and posterior draws around observations the reader drags. */
export function PosteriorDraws() {
  const state = useFigureState({
    model: row('1 · kernel and noise', {
      lengthscale: slider(0.1, 2, 0.6, { label: 'lengthscale ℓ' }),
      noise: slider(0, 0.5, 0.01, { label: 'noise variance σ²' }),
    }),
    reveal: row('2 · reveal', {
      prior: toggle(false, 'prior draws instead'),
      count: choice([3, 5, 10, 20], 5, { label: 'draws' }),
    }),
  })
  const { lengthscale, noise } = state.model
  const { prior: showPrior, count } = state.reveal
  const [points, setPoints] = useState(START)
  const kernel = useMemo(() => rbf({ lengthscale, variance: 1 }), [lengthscale])
  const x = useMemo(() => tensor(points.map((p) => p[0])), [points])
  const y = useMemo(() => tensor(points.map((p) => p[1])), [points])
  const post = useMemo(() => gpPosterior(kernel, x, y, { noiseVariance: noise }), [kernel, x, y, noise])
  const prediction = useMemo(() => {
    const p = post.predict(GRID)
    return band(toFlat(p.mean), toFlat(p.variance))
  }, [post])
  const draws = useMemo(
    () =>
      showPrior ? samplePrior(stream('gp-draws'), kernel, GRID, count) : post.sample(stream('gp-draws'), GRID, count),
    [showPrior, kernel, post, count],
  )
  const drawRows = useMemo(() => toRows(draws.draws) as number[][], [draws])
  const obs = useMemo(() => ({ x: points.map((p) => p[0]), y: points.map((p) => p[1]) }), [points])
  const xa = useAxis({ label: 'x', range: [0, 5] })
  const ya = useAxis({ label: 'f(x)', range: [-3, 3] })
  return (
    <Figure
      title="Prior and posterior draws"
      purpose="Conditioning a GP on observations pins its draws near the data and leaves them free elsewhere; the band is the posterior mean ± 2 sd."
      state={state}
      readouts={
        <>
          <Readout label="log marginal likelihood" value={formatValue(post.logMarginal.value)} />
          <Readout label="jitter added to K + σ²I" value={formatValue(post.jitter)} />
          <Readout label="jitter for the draws" value={formatValue(draws.jitter)} />
        </>
      }
      caption="Drag the observations. With σ² = 0 the posterior interpolates and its variance collapses at the data; with noise the draws pass near them. A short ℓ lets the mean return to the prior (0) between points. The draws (thin lines) use the same normals throughout, so they move continuously."
    >
      <Plot x={xa} y={ya}>
        {!showPrior && (
          <Area
            name="± 2 sd"
            x={GRID_X}
            y={prediction.upper}
            base={prediction.lower}
            slot={0}
            opacity={0.18}
            line={false}
          />
        )}
        {drawRows.map((d, i) => (
          <Curve key={i} name="draws" x={GRID_X} y={d} thin slot={1} />
        ))}
        {!showPrior && <Curve name="posterior mean" x={GRID_X} y={prediction.mean} slot={0} />}
        <Points name="observations" x={obs.x} y={obs.y} emphasis />
        {points.map((p, i) => (
          <Handle
            key={i}
            kind="point"
            at={p}
            onDrag={([a, b]) =>
              setPoints((ps) => ps.map((q, j): [number, number] => (j === i ? [clamp(a, 0, 5), clamp(b, -3, 3)] : q)))
            }
          />
        ))}
      </Plot>
    </Figure>
  )
}

/** Data for the evidence and sparse figures: a noisy sine on [0, 5]. */
function sineData(n: number, seed: string, sd: number) {
  const s = stream(seed)
  const x = toFlat(uniform(child(s, 'x'), 0, 5, { shape: [n] }) as Tensor).sort((a, b) => a - b)
  const e = toFlat(normals(child(s, 'e'), n, 0, sd))
  return { x, y: x.map((v, i) => Math.sin(1.4 * v) + e[i]) }
}

const LOG_ELL = linspace(-1.7, 1, 110)
const LOG_ELL_X = toFlat(LOG_ELL)

/** The log marginal likelihood against the lengthscale: data fit against complexity. */
export function EvidenceOverLengthscale() {
  const state = useFigureState({
    logEll: slider(-1.7, 1, -0.3, { label: 'log₁₀ ℓ', onChart: true }),
    noise: slider(0.005, 0.5, 0.04, { label: 'noise variance σ²' }),
  })
  const { logEll, noise } = state
  const data = useMemo(() => sineData(14, 'gp-evidence', 0.2), [])
  const x = useMemo(() => tensor(data.x), [data])
  const y = useMemo(() => tensor(data.y), [data])
  const curves = useMemo(() => {
    const parts = LOG_ELL_X.map((l) =>
      logMarginalLikelihood(rbf({ lengthscale: 10 ** l }), x, y, { noiseVariance: noise }),
    )
    const value = parts.map((p) => p.value as number)
    const best = argmax(tensor(value)) as number
    return {
      value,
      fit: parts.map((p) => p.dataFit as number),
      complexity: parts.map((p) => (p.complexity as number) + p.constant),
      best,
      bestAt: { x: [LOG_ELL_X[best]], y: [value[best]] },
    }
  }, [x, y, noise])
  const post = useMemo(() => {
    const p = gpPosterior(rbf({ lengthscale: 10 ** logEll }), x, y, { noiseVariance: noise }).predict(GRID)
    return band(toFlat(p.mean), toFlat(p.variance))
  }, [logEll, x, y, noise])
  const lml = logMarginalLikelihood(rbf({ lengthscale: 10 ** logEll }), x, y, { noiseVariance: noise })
  const ell = useAxis({ label: 'log₁₀ ℓ', range: [-1.7, 1] })
  const nats = useAxis({ label: 'nats', hold: 'union', key: noise })
  const xa = useAxis({ label: 'x', range: [0, 5] })
  const fa = useAxis({ label: 'f(x)', range: [-2.5, 2.5] })
  return (
    <Figure
      title="Evidence against the lengthscale"
      defaultSize="L"
      purpose="The log marginal likelihood trades a data-fit term, which rewards short lengthscales, against a complexity term, which penalises them; its maximum picks ℓ."
      state={state}
      readouts={
        <>
          <Readout label="ℓ" value={formatValue(10 ** logEll)} />
          <Readout label="log p(y | X, ℓ)" value={formatValue(lml.value as number)} />
          <Readout label="best ℓ on the grid" value={formatValue(10 ** LOG_ELL_X[curves.best])} />
        </>
      }
      caption="Drag the vertical line along log₁₀ ℓ and watch the fit below: at short ℓ the mean chases the noise, at long ℓ it flattens to a line. The marked point is the maximum over the grid. The signal variance is fixed at 1."
    >
      <Plots rows={2}>
        <Plot x={ell} y={nats}>
          <Curve name="log p(y | X, ℓ)" x={LOG_ELL_X} y={curves.value} slot={0} />
          <Curve name="data fit −½ yᵀK⁻¹y" x={LOG_ELL_X} y={curves.fit} slot={1} dashed />
          <Curve name="complexity −½ log|K| − (n/2) log 2π" x={LOG_ELL_X} y={curves.complexity} slot={2} dashed />
          <Points name="maximum" x={curves.bestAt.x} y={curves.bestAt.y} emphasis />
          <Handle {...state.handle('logEll', { label: 'ℓ' })} />
        </Plot>
        <Plot x={xa} y={fa}>
          <Area name="± 2 sd" x={GRID_X} y={post.upper} base={post.lower} slot={0} opacity={0.18} line={false} />
          <Curve name="posterior mean" x={GRID_X} y={post.mean} slot={0} />
          <Points name="data" x={data.x} y={data.y} muted />
        </Plot>
      </Plots>
    </Figure>
  )
}

const METHODS: SparseMethod[] = ['vfe', 'fitc', 'dtc', 'sor']

/** Sparse GP regression with inducing inputs the reader drags. */
export function SparseInducingPoints() {
  const state = useFigureState({
    approx: row('1 · approximation', {
      method: choice(METHODS, 'vfe', { label: 'method' }),
      lengthscale: slider(0.2, 1.5, 0.5, { label: 'lengthscale ℓ' }),
    }),
  })
  const method = state.approx.method as SparseMethod
  const { lengthscale } = state.approx
  const [z, setZ] = useState<number[]>([0.3, 0.8, 1.3, 1.8, 2.3])
  const noise = 0.04
  const data = useMemo(() => sineData(60, 'gp-sparse', 0.2), [])
  const x = useMemo(() => tensor(data.x), [data])
  const y = useMemo(() => tensor(data.y), [data])
  const kernel = useMemo(() => rbf({ lengthscale }), [lengthscale])
  const exact = useMemo(() => gpPosterior(kernel, x, y, { noiseVariance: noise }), [kernel, x, y])
  const exactMean = useMemo(() => toFlat(exact.predict(GRID).mean), [exact])
  const sparse = useMemo(
    () => sparseGp(kernel, x, y, tensor(z), { method, noiseVariance: noise }),
    [kernel, x, y, z, method],
  )
  const prediction = useMemo(() => {
    const p = sparse.predict(GRID)
    return band(toFlat(p.mean), toFlat(p.variance))
  }, [sparse])
  const atZ = useMemo(() => toFlat(sparse.predict(tensor(z)).mean), [sparse, z])
  const optimise = () => {
    const fit = fitSparseGp(kernel, x, y, tensor(z), {
      method,
      noiseVariance: noise,
      fitNoise: false,
      fitKernel: false,
      maxSteps: 100,
    })
    setZ(toFlat(fit.model.inducing).map((v) => clamp(v, 0, 5)))
  }
  const xa = useAxis({ label: 'x', range: [0, 5] })
  const fa = useAxis({ label: 'f(x)', range: [-2.5, 2.5] })
  return (
    <Figure
      title="Inducing points"
      purpose="A sparse GP summarises n observations through m inducing inputs; where they are decides where the approximation can follow the data."
      state={state}
      controls={
        <ControlRow label="2 · inducing inputs">
          <Button variant="outline" size="sm" onClick={optimise}>
            Optimise Z (L-BFGS on the bound)
          </Button>
          <Button variant="outline" size="sm" onClick={() => setZ([0.5, 1.5, 2.5, 3.5, 4.5])}>
            Spread evenly
          </Button>
        </ControlRow>
      }
      readouts={
        <>
          <Readout
            label={method === 'vfe' ? 'ELBO' : 'approximate log marginal'}
            value={formatValue(sparse.logMarginal)}
          />
          <Readout label="exact log marginal" value={formatValue(exact.logMarginal.value)} />
          <Readout label="VFE trace penalty" value={formatValue(sparse.terms.trace)} />
          <Readout label="m / n" value={`${z.length} / ${data.x.length}`} />
        </>
      }
      caption="Drag the vertical lines to move the five inducing inputs. Bunched on the left, the sparse mean cannot follow the data on the right, and for VFE the trace penalty shows the lost information; SoR's variance also shrinks to nothing far from Z, the others' does not. Optimise moves Z to maximise the bound."
    >
      <Plot x={xa} y={fa}>
        <Points name="data" x={data.x} y={data.y} muted />
        <Area
          name="± 2 sd"
          x={GRID_X}
          y={prediction.upper}
          base={prediction.lower}
          slot={0}
          opacity={0.18}
          line={false}
        />
        <Curve name="exact GP mean" x={GRID_X} y={exactMean} slot={2} dashed />
        <Curve name={`${method.toUpperCase()} mean`} x={GRID_X} y={prediction.mean} slot={0} />
        <Points name="inducing inputs" x={z} y={atZ} emphasis />
        {z.map((v, i) => (
          <Handle
            key={i}
            kind="x"
            at={v}
            onDrag={(u) => setZ((zs) => zs.map((w, j) => (j === i ? clamp(u, 0, 5) : w)))}
          />
        ))}
      </Plot>
    </Figure>
  )
}
