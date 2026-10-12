import { regression1d } from 'aifn-methods/data/synthetic'
import type { RegressionTruth } from 'aifn-methods/data'
import { stream, uniform } from 'aifn-compute/foundation/random'
import { linspace, reshape, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { useMemo, type ReactNode } from 'react'
import { Player } from 'aifn-render/controls'
import { Dashboard, DashboardCell, DashboardRow, Equation, Figure, tex } from 'aifn-render/layout'
import { choice, float, int, row, toggle, useFigureState, type FigureState } from 'aifn-render/state'
import { Area, Curve, Plot, Plots, Points, Readout, Rug, Segments, useAxis, useScaleColor } from 'aifn-render/viz'
import { band, f3, usePlayed } from '../_gaussian-processes/incremental'
import { CLASS_RANGE, CLASSIFICATION, REGRESSION, sigmoid } from './data'
import { predict, splineVmFit, splineVmProblem, type Likelihood, type SplineVmState } from './fit'

const GRID = 301

/** The knot cost λ (nats per knot): 0 is the plain evidence; ½ ln N is BIC's price per parameter. */
const penaltyField = () => float(0, { ge: 0, le: 20, suggestions: [0, 1, 2, 3, 5], label: 'knot cost λ (nats)' })

const modelRow = () =>
  row('2 · spline', {
    degree: choice(
      [
        { value: 1, label: '1 (linear)' },
        { value: 2, label: '2 (quadratic)' },
        { value: 3, label: '3 (cubic)' },
      ],
      3,
      { label: 'degree p' },
    ),
    candidates: int(30, { ge: 3, le: 100, label: 'candidate knots' }),
    penalty: penaltyField(),
    moveKnots: toggle(true, 'move knots'),
  })

// ── Regression ──

type RegressionKey = keyof typeof REGRESSION

const REGRESSION_SCHEMA = {
  data: row('1 · data', {
    dataset: choice(
      (Object.keys(REGRESSION) as RegressionKey[]).map((k) => ({ value: k, label: REGRESSION[k].label })),
      'step',
      { label: 'data' },
    ),
    n: int(150, { ge: 20, le: 400, label: 'n' }),
    noise: float(0.1, { ge: 0.01, le: 0.5, label: 'noise sd' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  }),
  model: modelRow(),
}

export function SplineVmShowcase() {
  const state = useFigureState(REGRESSION_SCHEMA)
  const { n, noise, seed } = state.data
  const key = state.data.dataset as RegressionKey
  const spec = REGRESSION[key]
  const data = useMemo(() => {
    const d = regression1d(stream(`splinevm-${seed}`), { n, fn: spec.fn, range: spec.range, noise })
    const grid = linspace(spec.range[0], spec.range[1], GRID)
    return {
      xs: Array.from(toFlat(d.x as Tensor)),
      ys: Array.from(toFlat(d.y as Tensor)),
      gridX: Array.from(toFlat(grid)),
      truth: Array.from(toFlat((d.meta.truth as RegressionTruth).mean(reshape(grid, [GRID, 1]) as Tensor))),
    }
  }, [n, noise, seed, spec])
  return (
    <SplineVmView
      likelihood="gaussian"
      state={state}
      data={data}
      range={spec.range}
      runKey={JSON.stringify([key, n, noise, seed])}
      title="SplineVM: spline knots chosen and placed by the evidence"
      intro={`${n} noisy observations (aifn’s regression1d), y = f(x) + ε.`}
    />
  )
}

// ── Classification ──

type ClassificationKey = keyof typeof CLASSIFICATION

const CLASSIFICATION_SCHEMA = {
  data: row('1 · data', {
    dataset: choice(
      (Object.keys(CLASSIFICATION) as ClassificationKey[]).map((k) => ({ value: k, label: CLASSIFICATION[k].label })),
      'window',
      { label: 'P(y = 1 | x)' },
    ),
    n: int(300, { ge: 20, le: 800, label: 'n' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  }),
  model: modelRow(),
  run: row('fit', {
    iterations: int(400, { ge: 10, le: 5000, suggestions: [100, 400, 1000, 3000], label: 'iterations (at most)' }),
  }),
}

export function SplineVmClassificationShowcase() {
  const state = useFigureState(CLASSIFICATION_SCHEMA)
  const { n, seed } = state.data
  const key = state.data.dataset as ClassificationKey
  const p = CLASSIFICATION[key].p
  const data = useMemo(() => {
    // x uniform on the range (sorted), a label y ~ Bernoulli(p(x)), and a small vertical jitter for drawing.
    const u = Array.from(toFlat(uniform(stream(`splinevm-class-${seed}`), 0, 1, { shape: [3 * n] }) as Tensor))
    const [a, b] = CLASS_RANGE
    const xs = u.slice(0, n).map((v) => a + (b - a) * v)
    xs.sort((s, t) => s - t)
    const ys = xs.map((x, i) => (u[n + i] < p(x) ? 1 : 0))
    const gridX = Array.from(toFlat(linspace(a, b, GRID)))
    return { xs, ys, shown: ys.map((y, i) => y + 0.08 * (u[2 * n + i] - 0.5)), gridX, truth: gridX.map(p) }
  }, [n, seed, p])
  return (
    <SplineVmView
      likelihood="bernoulli"
      state={state}
      data={data}
      range={CLASS_RANGE}
      runKey={JSON.stringify([key, n, seed])}
      maxIter={state.run.iterations}
      title="SplineVM classification: knots where the class probability changes"
      intro={`${n} labels y ∈ {0, 1} drawn with P(y = 1 | x) = ${key === 'window' ? '0.9 inside |x| < 0.6 and 0.1 outside' : key === 'threshold' ? 'σ(12x)' : '0.5 + 0.4 sin(1.5πx)'}; the model is P(y = 1) = σ(f(x)), σ the logistic sigmoid.`}
    />
  )
}

// ── The shared view ──

type ViewData = { xs: number[]; ys: number[]; shown?: number[]; gridX: number[]; truth: number[] }

function SplineVmView(props: {
  likelihood: Likelihood
  state: FigureState<typeof REGRESSION_SCHEMA> | FigureState<typeof CLASSIFICATION_SCHEMA>
  data: ViewData
  range: readonly [number, number]
  runKey: string
  /** Iterations at most; the run also stops early once 20 iterations together gain under 10⁻³ nats. */
  maxIter?: number
  title: string
  intro: string
}) {
  const { likelihood, state, data, range } = props
  const gaussian = likelihood === 'gaussian'
  const { candidates, moveKnots, penalty } = state.model
  const maxIter = props.maxIter
  const degree = Number(state.model.degree)
  const { xs, ys, gridX, truth } = data

  // ── The fit: every iteration, computed once per setting (milliseconds) ──
  const problem = useMemo(() => splineVmProblem(xs, ys, degree, likelihood), [xs, ys, degree, likelihood])
  const states = useMemo(
    () =>
      splineVmFit(problem, {
        candidates,
        moveKnots,
        penalty,
        ...(maxIter === undefined ? {} : { maxIter, tolerance: 1e-3, window: 20 }),
      }),
    [problem, candidates, moveKnots, penalty, maxIter],
  )
  const runKey = JSON.stringify([props.runKey, degree, candidates, moveKnots, penalty, maxIter])
  const [pos, setPos] = usePlayed(runKey, states.length)
  const s = states[pos]
  const toX = useMemo(() => (t: number) => problem.lo[0] + t * problem.span[0], [problem])

  // ── Colour: a knot's weight wⱼ on the diverging scale (square-root, so small weights still show), full colour at the
  // 95th percentile of |w| over the run ──
  const diverging = useScaleColor('diverging')
  const wFull = useMemo(() => {
    const all = states.flatMap((st) => st.weights.map(Math.abs)).sort((a, b) => a - b)
    return all.length ? Math.max(all[Math.floor(0.95 * (all.length - 1))], 1e-12) : 1
  }, [states])
  const colourOf = useMemo(
    () => (w: number) => diverging(0.5 + 0.5 * Math.sign(w) * Math.sqrt(Math.min(1, Math.abs(w) / wFull))),
    [diverging, wFull],
  )

  const view = useMemo(() => {
    const withNoise = predict(problem, s, gridX, true)
    const latent = predict(problem, s, gridX, false)
    const knotX = knotsOf(s).map(toX)
    const knotColours = s.weights.map(colourOf)
    if (gaussian) return { ...band(latent.mean, withNoise.variance), knotX, knotColours, accuracy: NaN }
    // Classification: the moderated probability, and σ(f) at the latent mean ± 2 sd.
    const b = band(latent.mean, latent.variance)
    const atData = predict(problem, s, xs, false).probability
    const accuracy = ys.filter((y, i) => (atData[i] > 0.5 ? 1 : 0) === y).length / ys.length
    return {
      mean: Array.from(latent.probability),
      upper: b.upper.map(sigmoid),
      lower: b.lower.map(sigmoid),
      knotX,
      knotColours,
      accuracy,
    }
  }, [problem, s, gridX, toX, colourOf, gaussian, xs, ys])

  // ── Knot paths over the iterations: a coloured mark per knot per iteration, and where knots were removed ──
  const paths = useMemo(() => {
    const segments: { from: [number, number]; to: [number, number] }[] = []
    const marks = { x: [] as number[], y: [] as number[], colors: [] as string[] }
    const removed = { x: [] as number[], y: [] as number[] }
    states.forEach((st, i) => {
      knotsOf(st).forEach((t, j) => {
        marks.x.push(toX(t))
        marks.y.push(i)
        marks.colors.push(colourOf(st.weights[j]))
      })
      if (i === 0) return
      const before = knotsOf(states[i - 1])
      const prev = new Map(states[i - 1].ids.map((id, j) => [id, before[j]]))
      const now = knotsOf(st)
      st.ids.forEach((id, j) => {
        const t0 = prev.get(id)
        if (t0 !== undefined) segments.push({ from: [toX(t0), i - 1], to: [toX(now[j]), i] })
      })
      for (const t of st.pruned) {
        removed.x.push(toX(t.at[0]))
        removed.y.push(i)
      }
    })
    return { segments, marks, removed, start: knotsOf(states[0]).map(toX) }
  }, [states, toX, colourOf])
  const series = useMemo(
    () => ({
      x: states.map((_, i) => i),
      objective: states.map((st) => st.objective),
      count: states.map((st) => st.terms.length),
    }),
    [states],
  )

  const yRange: [number, number] = useMemo(() => {
    if (!gaussian) return [-0.12, 1.12]
    const lo = Math.min(...ys)
    const hi = Math.max(...ys)
    return [lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo)]
  }, [ys, gaussian])
  const xa = useAxis({ label: 'x', range: [range[0], range[1]], key: props.runKey })
  const fa = useAxis({ label: gaussian ? 'y' : 'P(y = 1)', range: yRange, key: runKey })
  const iterAxis = useAxis({ label: 'iteration', range: [0, Math.max(1, states.length - 1)], key: runKey })
  const pathAxis = useAxis({ label: 'iteration', range: [0, Math.max(1, states.length - 1)], key: runKey })
  const evAxis = useAxis({
    label: `${gaussian ? 'log p(y | α, β, t)' : 'log p(y | α, t) (Laplace)'}${penalty > 0 ? ' − λK' : ''}`,
    hold: 'initial',
    key: runKey,
  })
  const countAxis = useAxis({ label: 'knots', range: [0, undefined], key: runKey })

  const knotLines = view.knotX.map((x) => ({
    from: [x, yRange[0]] as [number, number],
    to: [x, yRange[1]] as [number, number],
  }))
  const knotFoot = yRange[0] + 0.03 * (yRange[1] - yRange[0])
  const jump = degree === 1 ? 'slope' : degree === 2 ? 'second derivative' : 'third derivative'

  const topLayers: ReactNode = gaussian ? (
    <>
      <Area name="predictive ± 2 sd" x={gridX} y={view.upper} base={view.lower} slot={0} opacity={0.2} line={false} />
      <Curve name="truth" x={gridX} y={truth} emphasis dashed />
      <Curve name="predictive mean" x={gridX} y={view.mean} slot={0} />
      <Points name="data" x={xs} y={ys} muted thin />
    </>
  ) : (
    <>
      <Area name="σ(f) at f ± 2 sd" x={gridX} y={view.upper} base={view.lower} slot={2} opacity={0.2} line={false} />
      <Curve name="true P(y = 1 | x)" x={gridX} y={truth} emphasis dashed />
      <Curve name="predicted P(y = 1)" x={gridX} y={view.mean} slot={2} />
      <Points name="labels" x={xs} y={data.shown ?? ys} group={ys} groupNames={['y = 0', 'y = 1']} thin />
    </>
  )

  return (
    <Figure
      title={props.title}
      purpose="A degree-p spline in truncated-power form gives each knot a weight, the jump in the p-th derivative there. An ARD prior on those weights lets the evidence switch knots off, and its gradient moves the survivors to where the curve bends."
      state={state}
      defaultSize="XL"
      equation={
        <Equation>
          {gaussian
            ? tex`y = f(u) + \varepsilon, \quad f(u) = \sum_{k=0}^{p} c_k u^k + \sum_{j} w_j\,(u - t_j)_+^{p}, \quad w_j \sim \mathcal N(0, \alpha_j^{-1})`
            : tex`P(y = 1) = \sigma\big(f(u)\big), \quad f(u) = \sum_{k=0}^{p} c_k u^k + \sum_{j} w_j\,(u - t_j)_+^{p}, \quad w_j \sim \mathcal N(0, \alpha_j^{-1})`}
        </Equation>
      }
      controls={
        <Player
          label="3 · play"
          value={pos}
          onChange={setPos}
          count={states.length}
          format={(p) => {
            const st = states[p]
            if (!st) return `iteration ${p}`
            const removed = st.pruned.length ? `, ${st.pruned.length} removed` : ''
            return `iteration ${p}: ${st.terms.length} knots${removed}${st.converged ? ' (converged)' : ''}`
          }}
        />
      }
      readouts={{
        'at this iteration': (
          <>
            <Readout label="knots" value={s.terms.length} />
            <Readout label="log evidence" value={f3(s.logEvidence)} />
            {penalty > 0 && <Readout label="objective (− λ per knot)" value={f3(s.objective)} />}
            {gaussian ? (
              <Readout label="σ = 1/√β" value={f3(1 / Math.sqrt(s.beta))} />
            ) : (
              <Readout label="training accuracy" value={f3(view.accuracy)} />
            )}
            <Readout label="largest knot move (x)" value={f3(s.moved * problem.span[0])} />
            <Readout label="|w| at full colour" value={f3(wFull)} />
            <Readout
              label="iterations"
              value={`${states.length - 1}${states.at(-1)?.converged ? ' (converged)' : ''}`}
            />
          </>
        ),
      }}
      caption={`${props.intro} The latent function f is a degree-${degree} spline in u (x rescaled to [0, 1]): a polynomial with a vague prior, plus one truncated power (u − tⱼ)₊ raised to the power ${degree} per knot, whose weight wⱼ is the jump in the ${jump} of f at tⱼ. Each wⱼ has its own precision αⱼ (automatic relevance determination, as in the relevance vector machine). The fit starts from ${candidates} evenly spaced candidate knots and repeats: set every αⱼ to its best value given the others, αⱼ = sⱼ²/(qⱼ² − sⱼ) with sⱼ, qⱼ the sparsity and quality factors of Tipping and Faul's fast RVM (a knot with qⱼ² ≤ sⱼ gets αⱼ = ∞ and is removed)${gaussian ? ', re-estimate the noise precision β' : ''}, and ${moveKnots ? 'take one step tⱼ ← tⱼ + η ∂ln p(y)/∂tⱼ on the knot locations' : 'keep the knots where they started (knot moves are off)'}.${gaussian ? ' Each move raises the evidence.' : ' The posterior is the Laplace approximation at the mode of f (Newton’s method), so the evidence is approximate and can dip by a little between iterations.'} Top: ${gaussian ? 'the predictive mean ± 2 sd with noise (blue), the truth (ink, dashed)' : 'the predicted P(y = 1), moderated by the uncertainty in f (green), with σ(f) at f ± 2 sd shaded, the true P(y = 1 | x) (ink, dashed), and the labels, jittered'}, and the knots at this iteration (dashed lines, with a dot coloured by the knot’s weight wⱼ on the diverging scale: red for wⱼ > 0, where the ${jump} of f increases, blue for wⱼ < 0, pale near 0; square-root scale, full colour at the readout’s |w|). Bottom left: the log evidence${penalty > 0 ? ` minus λ = ${penalty} nats per knot (a prior p(K) ∝ e^(−λK) on the knot count K; λ = 0 leaves the sparsity to the evidence's own Occam factor, and λ = ½ ln N = ${(0.5 * Math.log(xs.length)).toFixed(2)} would be BIC's price)` : ''} and the knot count per iteration. Bottom right: each knot’s path over the iterations, coloured by its weight at each iteration (crosses: removed). Prototype: local numerics in the lab, not yet in the engine.`}
    >
      <Dashboard>
        <DashboardRow ratio={1.3} minHeight={300}>
          <DashboardCell>
            <Plot x={xa} y={fa}>
              {topLayers}
              <Segments name="knots" segments={knotLines} dashed />
              <Points
                name="knot weight"
                x={view.knotX}
                y={view.knotX.map(() => knotFoot)}
                colors={view.knotColours}
                size={11}
                live
              />
              <Rug name="starting candidates" values={paths.start} muted />
            </Plot>
          </DashboardCell>
        </DashboardRow>
        <DashboardRow minHeight={260}>
          <DashboardCell ratio={1.1}>
            <Plots rows={2} tight>
              <Plot x={iterAxis} y={evAxis}>
                <Curve name={penalty > 0 ? 'objective' : 'log evidence'} x={series.x} y={series.objective} slot={0} />
                <Points name="shown" x={[pos]} y={[series.objective[pos]]} emphasis live />
              </Plot>
              <Plot x={iterAxis} y={countAxis}>
                <Curve name="knots" x={series.x} y={series.count} slot={1} />
                <Points name="shown" x={[pos]} y={[series.count[pos]]} emphasis live />
              </Plot>
            </Plots>
          </DashboardCell>
          <DashboardCell>
            <Plot x={xa} y={pathAxis}>
              <Segments name="knot paths" segments={paths.segments} />
              <Points name="weight" x={paths.marks.x} y={paths.marks.y} colors={paths.marks.colors} size={6} />
              {paths.removed.x.length > 0 && (
                <Points name="removed" x={paths.removed.x} y={paths.removed.y} slot={2} shape={4} />
              )}
              <Points
                name="knots now"
                x={view.knotX}
                y={view.knotX.map(() => pos)}
                colors={view.knotColours}
                size={11}
                live
              />
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

/** The knot locations of a 1-D fit's terms (u units). */
const knotsOf = (st: SplineVmState) => st.terms.map((t) => t.at[0])
