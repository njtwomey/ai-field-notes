import { moons } from 'aifn-methods/data/synthetic'
import {
  fitGpClassifier,
  gpClassifier,
  gpEpEvidence,
  laplaceLogMarginal,
  type GpClassificationMethod,
} from 'aifn-methods/learning/gaussian-processes'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, linspace, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { gram, rbf } from 'aifn-compute/learning/kernels'
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Player } from 'aifn-render/controls'
import { Dashboard, DashboardCell, DashboardRow, Figure } from 'aifn-render/layout'
import { choice, row, setting, slider, toggle, useComputed, useFigureState } from 'aifn-render/state'
import { Area, Contours, Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from 'aifn-render/viz'
import { formatValue } from '@lab/views'

const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))
const grid = (lo: number, hi: number, n: number) => toFlat(linspace(lo, hi, n))
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
type P2 = [number, number]

const DATA = moons(stream('showcase-gpc'), { n: 50, noise: 0.25 })
const X = DATA.x as Tensor
const Y = fromData(Float64Array.from(toFlat(DATA.y as Tensor)), [X.shape[0]])
const TRAIN = dataset(X, Y)
const XS = toFlat(X)
const LABELS = toFlat(Y)
const BOX = { x: [-1.8, 2.8] as P2, y: [-1.4, 1.9] as P2 }
const GX = grid(BOX.x[0], BOX.x[1], 46)
const GY = grid(BOX.y[0], BOX.y[1], 36)
const GRID = fromData(Float64Array.from(GY.flatMap((y) => GX.flatMap((x) => [x, y]))), [GX.length * GY.length, 2])
const CUT_POINTS = 90
/** The decision boundary is drawn within this distance of a training point. */
const BOUNDARY_REACH = 0.35
/** Per grid cell (row-major, as the field): whether a training point lies within BOUNDARY_REACH. */
const NEAR_DATA = GY.flatMap((y) =>
  GX.map((x) => LABELS.some((_, i) => Math.hypot(XS[2 * i] - x, XS[2 * i + 1] - y) < BOUNDARY_REACH)),
)

const METHODS = [
  { value: 'laplace' as const, label: 'Laplace' },
  { value: 'ep' as const, label: 'EP' },
]
const CLASS_NAMES = ['class 0', 'class 1']
const ELL = { min: 0.05, max: 4, step: 0.01 }
const VAR = { min: 0.3, max: 1000, step: 0.1 }
const METHOD_NAME: Record<GpClassificationMethod, string> = { laplace: 'Laplace', ep: 'EP' }
// Both methods approximate the same model, the probit likelihood Φ(f), so their evidences compare directly.
const LIKELIHOOD = 'probit' as const
const FIT_STEPS_PER_SECOND = 6
const HYPERPRIOR_SD = 2

// The evidence surface over (log₁₀ ℓ, log₁₀ σ²), per method: each cell is one approximation. Laplace's is computed at
// once; EP's (about 10 ms a cell) one row per frame, each row warm-started from the last cell's sites, in a snake.
const LOG_ELL = grid(-1, 0.6, 15)
const LOG_VAR = grid(-0.5, 3, 15)
const kernelAt = (le: number, lv: number) => rbf({ lengthscale: 10 ** le, variance: 10 ** lv })
const LAPLACE_SURFACE = LOG_VAR.map((lv) =>
  LOG_ELL.map((le) => laplaceLogMarginal(kernelAt(le, lv), X, Y, { likelihood: LIKELIHOOD })),
)
let epSurface: number[][] | null = null
const epEvidence = gpEpEvidence(Y, { tolerance: 1e-5 })
const epRow = (r: number) => {
  const order = r % 2 === 0 ? LOG_ELL : [...LOG_ELL].reverse()
  const row = order.map((le) => epEvidence(gram(kernelAt(le, LOG_VAR[r]), X)) as number)
  return r % 2 === 0 ? row : row.reverse()
}

/** The EP surface, filled a row per frame; rows not yet computed are null. */
function useEpSurface(active: boolean): (number[] | null)[] {
  const [rows, setRows] = useState<(number[] | null)[]>(() => epSurface ?? LOG_VAR.map(() => null))
  useEffect(() => {
    if (!active || epSurface) return
    let cancelled = false
    let r = rows.findIndex((row) => row === null)
    const next = () => {
      if (cancelled || r < 0 || r >= LOG_VAR.length) return
      const row = epRow(r)
      const at = r
      setRows((prev) => {
        const out = prev.slice()
        out[at] = row
        if (out.every((x) => x !== null)) epSurface = out as number[][]
        return out
      })
      r++
      requestAnimationFrame(next)
    }
    requestAnimationFrame(next)
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active])
  return rows
}

export function GpClassification() {
  const state = useFigureState({
    approx: row('1 · approximation', {
      method: choice(METHODS, 'laplace', { label: 'posterior approximation' }),
    }),
    kernel: row('2 · kernel', {
      ell: slider(ELL.min, ELL.max, 0.2, { label: 'lengthscale ℓ', step: ELL.step }),
      variance: slider(VAR.min, VAR.max, 4, { label: 'signal variance σ²', step: VAR.step }),
    }),
    reveal: row('3 · reveal', {
      latent: toggle(false, 'latent mean f̄ instead of π*'),
      boundary: toggle(true, 'decision boundary (π* = ½)'),
    }),
    fitting: row('4 · fit', { hyperprior: setting(true, 'weak hyperprior (log θ ~ N(start, 2²))') }),
  })
  const method = state.approx.method as GpClassificationMethod
  const { latent } = state.reveal
  const { hyperprior } = state.fitting
  const [fit, setFit] = useState<ReturnType<typeof fitGpClassifier> | null>(null)
  const [fitStep, setFitStep] = useState(0)
  const [playing, setPlaying] = useState(false)
  const epRows = useEpSurface(method === 'ep')
  const epDone = epRows.filter((r) => r !== null).length
  // Until EP's surface is complete its missing rows show Laplace's (the readout says how far it has got).
  const evidence = useMemo(
    (): number[][] => (method === 'ep' ? epRows.map((r, i) => r ?? LAPLACE_SURFACE[i]) : LAPLACE_SURFACE),
    [method, epRows],
  )

  // While a fit is shown, the hyperparameters are those of its L-BFGS iterate at the played step.
  const fitPath = useMemo(() => {
    if (!fit) return null
    const li = fit.names.findIndex((n) => n.includes('lengthscale'))
    const vi = fit.names.findIndex((n) => n.includes('variance'))
    const logMarginal = toFlat(fit.training.series.logMarginal)
    return fit.training.steps.map((s, i) => {
      const v = toFlat(s.x)
      return { ell: Math.exp(v[li]), variance: Math.exp(v[vi]), logMarginal: logMarginal[i] }
    })
  }, [fit])
  const hp = fitPath ? fitPath[Math.min(fitStep, fitPath.length - 1)] : state.kernel

  // The kernel sliders follow the played fit; moving them (or the evidence handle) by hand leaves the fit. `synced`
  // holds the values last written from the path, so a change that did not come from the path is the reader's.
  const synced = useRef<{ ell: number; variance: number } | null>(null)
  useEffect(() => {
    if (!fitPath) return
    const p = fitPath[Math.min(fitStep, fitPath.length - 1)]
    synced.current = { ell: p.ell, variance: p.variance }
    state.set('kernel.ell', p.ell)
    state.set('kernel.variance', p.variance)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- state.set is stable in effect
  }, [fitPath, fitStep])
  const clearFit = () => {
    setFit(null)
    setPlaying(false)
    synced.current = null
  }
  useEffect(() => {
    const s = synced.current
    if (!s) return
    const moved =
      Math.abs(state.kernel.ell - s.ell) > ELL.step || Math.abs(state.kernel.variance - s.variance) > VAR.step
    if (moved) clearFit()
  }, [state.kernel])
  useEffect(clearFit, [method, hyperprior])

  // "Fit" plays the path from its start; any scrubbing of the player stops the playback.
  const advancing = playing && fitPath !== null && fitStep < fitPath.length - 1
  useEffect(() => {
    if (!advancing) return
    const id = setTimeout(() => setFitStep((s) => s + 1), 1000 / FIT_STEPS_PER_SECOND)
    return () => clearTimeout(id)
  }, [advancing, fitStep])

  // The model and its field are recomputed by the scheduler: at most once a frame while (ℓ, σ²) is dragged.
  const fitted = useComputed(() => {
    const model = gpClassifier({
      kernel: rbf({ lengthscale: hp.ell, variance: hp.variance }),
      method,
      likelihood: LIKELIHOOD,
    }).fit(TRAIN)
    const values = latent ? toFlat(model.latent(GRID).mean) : toFlat(model.expect(GRID))
    const rows = (v: ArrayLike<number>) => GY.map((_, i) => Array.from(v).slice(i * GX.length, (i + 1) * GX.length))
    // The boundary is drawn only near the data (within BOUNDARY_REACH of a training point): far from it π* is ½
    // everywhere, and the ½ contour there only traces noise.
    const informed = Array.from(values, (v, k) => (NEAR_DATA[k] ? v : NaN))
    return { model, field: rows(values), boundary: rows(informed) }
  }, [hp.ell, hp.variance, method, latent])
  const { model, field, boundary } = fitted.value

  const path = useMemo(() => {
    if (!fitPath) return null
    const upto = fitPath.slice(0, fitStep + 1)
    return { x: upto.map((p) => Math.log10(p.ell)), y: upto.map((p) => Math.log10(p.variance)) }
  }, [fitPath, fitStep])

  const runFit = () => {
    setFit(
      fitGpClassifier(rbf({ lengthscale: hp.ell, variance: hp.variance }), X, Y, {
        method,
        likelihood: LIKELIHOOD,
        maxIterations: 60,
        logPriorScale: hyperprior ? HYPERPRIOR_SD : undefined,
      }),
    )
    setFitStep(0)
    setPlaying(true)
  }

  const setHp = useCallback(
    ([a, b]: P2) => {
      state.set('kernel.ell', clamp(10 ** a, ELL.min, ELL.max))
      state.set('kernel.variance', clamp(10 ** b, VAR.min, VAR.max))
    },
    [state.set], // eslint-disable-line react-hooks/exhaustive-deps -- the setter is stable
  )

  return (
    <Figure
      title="Gaussian-process classification by Laplace and EP"
      purpose="The latent posterior's variance pulls the predictive probability π* towards ½ away from the data; the lengthscale sets how far each point's evidence reaches, and the approximate evidence picks it."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <Button variant="outline" size="sm" onClick={runFit} aria-label="Fit hyperparameters">
            Fit hyperparameters (L-BFGS)
          </Button>
          {fitPath && (
            <Player
              value={fitStep}
              onChange={(p) => {
                setPlaying(false)
                setFitStep(p)
              }}
              count={fitPath.length}
              label="L-BFGS step"
              format={(p) => `step ${p}`}
            />
          )}
        </>
      }
      readouts={{
        model: (
          <>
            <Readout label={`${METHOD_NAME[method]} log p(y | X, θ)`} value={f3(model.logMarginal)} />
            <Readout label="ℓ" value={f3(hp.ell)} />
            <Readout label="σ²" value={f3(hp.variance)} />
            <Readout
              label={method === 'ep' ? 'EP sweeps to convergence' : 'Newton steps to the mode'}
              value={method === 'ep' ? Math.ceil(model.training.meta.steps / X.shape[0]) : model.training.meta.steps}
            />
          </>
        ),
        fit: (
          <>
            {fitPath && <Readout label="log evidence at this step" value={f3(fitPath[fitStep].logMarginal)} />}
            {method === 'ep' && epDone < LOG_VAR.length && (
              <Readout label="EP evidence surface" value={`${epDone}/${LOG_VAR.length} rows`} />
            )}
          </>
        ),
      }}
      caption="Left: 50 points from two noisy moons (blue class 0, red class 1) over the predictive probability π*(x) (or, revealed, the latent posterior mean f̄), computed by aifn’s GP classifier with an RBF kernel and the probit likelihood, its posterior approximated by Laplace (Newton to the mode) or EP (moment-matched sites); with the decision boundary on (the default), the ink line is π* = ½, drawn within 0.35 of a training point (far from the data π* is ½ everywhere, so a boundary there means nothing). Drag the two ends of the dashed line cut, and drag the round handle on the evidence surface to set (ℓ, σ²). Top right: along the cut, f̄ with ± 2 posterior sd (the band), and π* scaled by 10 (dashed): between and beyond the moons the band widens and π* falls back towards ½ even where f̄ does not. Bottom right: the chosen approximation’s log marginal likelihood over log₁₀ ℓ and log₁₀ σ² (one fit per cell), the current hyperparameters (the handle) and, after “Fit hyperparameters”, the L-BFGS path, played as it climbs (by default with a weak Gaussian hyperprior on each log hyperparameter, sd 2 about its start, since on these nearly separable moons the evidence keeps rising with σ², EP’s without bound); its gradients come from differentiating the evidence through the Gram matrix (implicitly through the Laplace mode, at the EP fixed point). EP’s evidence is at or above Laplace’s, by up to 20 nats where σ² is large and the posterior skewed. EP’s surface fills in a row at a time. A very short ℓ overfits each point (π* near 0 or 1 at the data, ½ between), a long one underfits to a nearly linear boundary; the evidence peaks between."
    >
      <GpcCharts
        model={model}
        field={field}
        boundary={boundary}
        fieldStale={fitted.stale}
        latent={latent}
        showBoundary={state.reveal.boundary}
        evidence={evidence}
        path={path}
        ell={hp.ell}
        variance={hp.variance}
        setHp={setHp}
      />
    </Figure>
  )
}

type ChartsProps = {
  model: ReturnType<ReturnType<typeof gpClassifier>['fit']>
  field: number[][]
  /** The field where the posterior sd is small (NaN elsewhere), for the decision boundary. */
  boundary: number[][]
  fieldStale: boolean
  latent: boolean
  /** Draw the decision boundary. */
  showBoundary: boolean
  evidence: number[][]
  path: { x: number[]; y: number[] } | null
  ell: number
  variance: number
  setHp: (p: P2) => void
}

/**
 * The charts, apart from the figure: the cut is this component's own state, so dragging its ends re-renders only the
 * charts (not the figure's controls and readouts), and the band along it follows by the scheduler.
 */
const GpcCharts = memo(function GpcCharts({
  model,
  field,
  boundary,
  fieldStale,
  latent,
  showBoundary,
  evidence,
  path,
  ell,
  variance,
  setHp,
}: ChartsProps) {
  const [cut, setCut] = useState<[P2, P2]>([
    [-1.2, 1.2],
    [2.2, -0.8],
  ])
  const moveCut = (k: 0 | 1, [x, y]: P2) =>
    setCut((c) => {
      const next: [P2, P2] = [c[0], c[1]]
      next[k] = [clamp(x, BOX.x[0], BOX.x[1]), clamp(y, BOX.y[0], BOX.y[1])]
      return next
    })
  // The cut's ends and line move on the pointer (live layers); the latent band along it follows by the scheduler.
  const along = useComputed(() => {
    const [a, b] = cut
    const s = grid(0, 1, CUT_POINTS)
    const pts = fromData(Float64Array.from(s.flatMap((u) => [a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])])), [
      CUT_POINTS,
      2,
    ])
    const q = model.latent(pts)
    const mean = toFlat(q.mean)
    const sd = toFlat(q.variance).map(Math.sqrt)
    const length = Math.hypot(b[0] - a[0], b[1] - a[1])
    return {
      s: s.map((u) => u * length),
      mean,
      upper: mean.map((m, i) => m + 2 * sd[i]),
      lower: mean.map((m, i) => m - 2 * sd[i]),
      prob: toFlat(model.expect(pts)).map((p) => 10 * p),
    }
  }, [model, cut[0][0], cut[0][1], cut[1][0], cut[1][1]])

  const cutLine = useMemo(
    () => ({ x: [cut[0][0], cut[1][0]], y: [cut[0][1], cut[1][1]] }),
    [cut[0][0], cut[0][1], cut[1][0], cut[1][1]], // eslint-disable-line react-hooks/exhaustive-deps -- by value
  )
  const points = useMemo(() => ({ x: LABELS.map((_, i) => XS[2 * i]), y: LABELS.map((_, i) => XS[2 * i + 1]) }), [])
  const [lo, hi] = latent ? [-6, 6] : [0, 1]
  const x1 = useAxis({ label: 'x₁', range: BOX.x, nice: false })
  const x2 = useAxis({ label: 'x₂', range: BOX.y, nice: false, equal: x1 })
  const dist = useAxis({ label: 'distance along the cut', hold: 'initial' })
  const fy = useAxis({ label: 'f̄ and 10·π*', range: [-8, 10] })
  const logEll = useAxis({ label: 'log₁₀ ℓ' })
  const logVar = useAxis({ label: 'log₁₀ σ²' })
  return (
    <Dashboard>
      <DashboardRow minHeight={440}>
        {/* As wide as the equal-units field needs at the row's height (its colour bar included), so it fills the cell. */}
        <DashboardCell aspect={1.42} stackAspect={1.3}>
          <Plot x={x1} y={x2}>
            <Raster
              x={GX}
              y={GY}
              z={field}
              scale="diverging"
              range={[lo, hi]}
              valueLabel={latent ? 'f̄' : 'π*'}
              stale={fieldStale}
            />
            {showBoundary && <Contours x={GX} y={GY} z={boundary} levels={latent ? [0] : [0.5]} stale={fieldStale} />}
            <Points name="points" x={points.x} y={points.y} group={LABELS} groupNames={CLASS_NAMES} />
            <Curve name="line cut" x={cutLine.x} y={cutLine.y} emphasis dashed live />
            <Handle kind="point" at={cut[0]} label="cut start" onDrag={(p) => moveCut(0, p)} />
            <Handle kind="point" at={cut[1]} label="cut end" onDrag={(p) => moveCut(1, p)} />
          </Plot>
        </DashboardCell>
        <DashboardCell minWidth={400}>
          <Plots rows={2}>
            <Plot x={dist} y={fy}>
              <Area
                name="f̄ ± 2 sd"
                x={along.value.s}
                y={along.value.upper}
                base={along.value.lower}
                slot={3}
                opacity={0.25}
                line={false}
                stale={along.stale}
                live
              />
              <Curve name="latent mean f̄" x={along.value.s} y={along.value.mean} slot={3} stale={along.stale} live />
              <Curve
                name="predictive π* (×10)"
                x={along.value.s}
                y={along.value.prob}
                slot={4}
                dashed
                stale={along.stale}
                live
              />
            </Plot>
            <Plot x={logEll} y={logVar}>
              <Raster x={LOG_ELL} y={LOG_VAR} z={evidence} valueLabel="log evidence" />
              {path && <Curve name="L-BFGS iterates" x={path.x} y={path.y} slot={1} showPoints />}
              <Handle kind="point" at={[Math.log10(ell), Math.log10(variance)]} label="(ℓ, σ²)" onDrag={setHp} />
            </Plot>
          </Plots>
        </DashboardCell>
      </DashboardRow>
    </Dashboard>
  )
})
