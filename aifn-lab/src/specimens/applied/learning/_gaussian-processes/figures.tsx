import { useMemo, useState } from 'react'
import {
  fitSparseGp,
  gpPosterior,
  logMarginalLikelihood,
  samplePrior,
  sparseGp,
  type SparseMethod,
} from 'aifn-applied/learning/gaussian-processes'
import { rbf } from 'aifn/learning/kernels'
import { child, normals, stream, uniform } from 'aifn/foundation/random'
import { argmax, linspace, tensor, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { Button, Select, Slider, Switch } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, type Handle, type XYSeries } from '@lab/viz'
import { formatValue } from '@lab/views'

const GRID = linspace(0, 5, 201)
const GRID_X = toFlat(GRID)
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/** Mean and ±2 sd lines of a Gaussian prediction. */
function band(name: string, mean: number[], variance: number[], slot: number): XYSeries[] {
  const sd = variance.map(Math.sqrt)
  return [
    { name, type: 'line', x: GRID_X, y: mean, slot },
    { name: '± 2 sd', type: 'line', x: GRID_X, y: mean.map((m, i) => m + 2 * sd[i]), slot, dashed: true },
    { name: '± 2 sd ', type: 'line', x: GRID_X, y: mean.map((m, i) => m - 2 * sd[i]), slot, dashed: true },
  ]
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
  const [points, setPoints] = useState(START)
  const [lengthscale, setLengthscale] = useState(0.6)
  const [noise, setNoise] = useState(0.01)
  const [showPrior, setShowPrior] = useState(false)
  const kernel = useMemo(() => rbf({ lengthscale, variance: 1 }), [lengthscale])
  const x = useMemo(() => tensor(points.map((p) => p[0])), [points])
  const y = useMemo(() => tensor(points.map((p) => p[1])), [points])
  const post = useMemo(() => gpPosterior(kernel, x, y, { noiseVariance: noise }), [kernel, x, y, noise])
  const prediction = useMemo(() => post.predict(GRID), [post])
  const draws = useMemo(
    () => (showPrior ? samplePrior(stream('gp-draws'), kernel, GRID, 5) : post.sample(stream('gp-draws'), GRID, 5)),
    [showPrior, kernel, post],
  )
  const series: XYSeries[] = [
    ...toRows(draws.draws).map((d, i) => ({
      name: `draw ${i + 1}`,
      type: 'line' as const,
      x: GRID_X,
      y: d,
      thin: true,
      slot: 1,
    })),
    ...(showPrior ? [] : band('posterior mean', toFlat(prediction.mean), toFlat(prediction.variance), 0)),
    { name: 'observations', type: 'scatter', x: points.map((p) => p[0]), y: points.map((p) => p[1]), emphasis: true },
  ]
  const handles: Handle[] = points.map((p, i) => ({
    kind: 'point',
    at: p,
    onDrag: ([a, b]) => setPoints((ps) => ps.map((q, j) => (j === i ? [clamp(a, 0, 5), clamp(b, -3, 3)] : q))),
  }))
  return (
    <Figure
      title="Prior and posterior draws"
      description="Conditioning a GP on observations pins its draws near the data and leaves them free elsewhere; the band is the posterior mean ± 2 sd."
      controls={
        <>
          <ControlRow label="Kernel and noise">
            <Slider label="lengthscale ℓ" value={lengthscale} onChange={setLengthscale} min={0.1} max={2} />
            <Slider label="noise variance σ²" value={noise} onChange={setNoise} min={0} max={0.5} />
          </ControlRow>
          <ControlRow label="Reveal">
            <Switch label="prior draws instead" checked={showPrior} onChange={setShowPrior} />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="log marginal likelihood" value={formatValue(post.logMarginal.value)} />
          <Readout label="jitter added to K + σ²I" value={formatValue(post.jitter)} />
          <Readout label="jitter for the draws" value={formatValue(draws.jitter)} />
        </>
      }
      caption="Drag the observations. With σ² = 0 the posterior interpolates and its variance collapses at the data; with noise the draws pass near them. A short ℓ lets the mean return to the prior (0) between points. The five draws use the same normals throughout, so they move continuously."
    >
      <XYChart series={series} xLabel="x" yLabel="f(x)" xRange={[0, 5]} yRange={[-3, 3]} handles={handles} />
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
  const [logEll, setLogEll] = useState(-0.3)
  const [noise, setNoise] = useState(0.04)
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
    }
  }, [x, y, noise])
  const post = useMemo(
    () => gpPosterior(rbf({ lengthscale: 10 ** logEll }), x, y, { noiseVariance: noise }).predict(GRID),
    [logEll, x, y, noise],
  )
  const lml = logMarginalLikelihood(rbf({ lengthscale: 10 ** logEll }), x, y, { noiseVariance: noise })
  const top: XYSeries[] = [
    { name: 'log p(y | X, ℓ)', type: 'line', x: LOG_ELL_X, y: curves.value, slot: 0 },
    { name: 'data fit −½ yᵀK⁻¹y', type: 'line', x: LOG_ELL_X, y: curves.fit, slot: 1, dashed: true },
    {
      name: 'complexity −½ log|K| − (n/2) log 2π',
      type: 'line',
      x: LOG_ELL_X,
      y: curves.complexity,
      slot: 2,
      dashed: true,
    },
    { name: 'maximum', type: 'scatter', x: [LOG_ELL_X[curves.best]], y: [curves.value[curves.best]], emphasis: true },
  ]
  const bottom: XYSeries[] = [
    ...band('posterior mean', toFlat(post.mean), toFlat(post.variance), 0),
    { name: 'data', type: 'scatter', x: data.x, y: data.y, muted: true },
  ]
  return (
    <Figure
      title="Evidence against the lengthscale"
      defaultSize="L"
      description="The log marginal likelihood trades a data-fit term, which rewards short lengthscales, against a complexity term, which penalises them; its maximum picks ℓ."
      controls={
        <ControlRow label="Model">
          <Slider label="log₁₀ ℓ" value={logEll} onChange={setLogEll} min={-1.7} max={1} />
          <Slider label="noise variance σ²" value={noise} onChange={setNoise} min={0.005} max={0.5} />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="ℓ" value={formatValue(10 ** logEll)} />
          <Readout label="log p(y | X, ℓ)" value={formatValue(lml.value as number)} />
          <Readout label="best ℓ on the grid" value={formatValue(10 ** LOG_ELL_X[curves.best])} />
        </>
      }
      caption="Drag the vertical line (or the slider) along log₁₀ ℓ and watch the fit below: at short ℓ the mean chases the noise, at long ℓ it flattens to a line. The marked point is the maximum over the grid. The variance is fixed at 1."
    >
      <Subplots rows={2} heightRatios={[1, 1]}>
        <Panel>
          <XYChart
            series={top}
            xLabel="log₁₀ ℓ"
            yLabel="nats"
            handles={[{ kind: 'x', at: logEll, label: 'ℓ', onDrag: (v) => setLogEll(clamp(v, -1.7, 1)) }]}
          />
        </Panel>
        <Panel>
          <XYChart series={bottom} xLabel="x" yLabel="f(x)" xRange={[0, 5]} yRange={[-2.5, 2.5]} />
        </Panel>
      </Subplots>
    </Figure>
  )
}

const METHODS: SparseMethod[] = ['vfe', 'fitc', 'dtc', 'sor']

/** Sparse GP regression with inducing inputs the reader drags. */
export function SparseInducingPoints() {
  const [method, setMethod] = useState<SparseMethod>('vfe')
  const [z, setZ] = useState<number[]>([0.3, 0.8, 1.3, 1.8, 2.3])
  const [lengthscale, setLengthscale] = useState(0.5)
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
  const prediction = useMemo(() => sparse.predict(GRID), [sparse])
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
  const series: XYSeries[] = [
    { name: 'data', type: 'scatter', x: data.x, y: data.y, muted: true },
    { name: 'exact GP mean', type: 'line', x: GRID_X, y: exactMean, slot: 2, dashed: true },
    ...band(`${method.toUpperCase()} mean`, toFlat(prediction.mean), toFlat(prediction.variance), 0),
    { name: 'inducing inputs', type: 'scatter', x: z, y: atZ, emphasis: true },
  ]
  const handles: Handle[] = z.map((v, i) => ({
    kind: 'x',
    at: v,
    onDrag: (u) => setZ((zs) => zs.map((w, j) => (j === i ? clamp(u, 0, 5) : w))),
  }))
  return (
    <Figure
      title="Inducing points"
      description="A sparse GP summarises n observations through m inducing inputs; where they are decides where the approximation can follow the data."
      controls={
        <>
          <ControlRow label="Approximation">
            <Select label="method" value={method} onChange={setMethod} options={METHODS} />
            <Slider label="lengthscale ℓ" value={lengthscale} onChange={setLengthscale} min={0.2} max={1.5} />
          </ControlRow>
          <ControlRow label="Inducing inputs">
            <Button variant="outline" size="sm" onClick={optimise}>
              Optimise Z (L-BFGS on the bound)
            </Button>
            <Button variant="outline" size="sm" onClick={() => setZ([0.5, 1.5, 2.5, 3.5, 4.5])}>
              Spread evenly
            </Button>
          </ControlRow>
        </>
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
      <XYChart series={series} xLabel="x" yLabel="f(x)" xRange={[0, 5]} yRange={[-2.5, 2.5]} handles={handles} />
    </Figure>
  )
}
