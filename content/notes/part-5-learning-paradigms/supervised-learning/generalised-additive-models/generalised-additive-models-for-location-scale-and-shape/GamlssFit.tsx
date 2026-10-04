import { useMemo, useState } from 'react'
import {
  Button,
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { GROWTH_TRUTH, growthChart } from 'aifn-methods/data/synthetic'
import { gamlssModel, gamlssProblem, gamlssTrace, s } from 'aifn-methods/learning/generalised/gam'
import { stream } from 'aifn/foundation/random'
import { fromData, linspace, toFlat } from 'aifn/foundation/tensor'
import { distributionalFamily, type DistributionalFamilyName } from 'aifn/probability/likelihoods'

const AGES = toFlat(linspace(0, 18, 91))
const AGE_GRID = fromData(Float64Array.from(AGES), [AGES.length, 1])
const LEVELS = [0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97]
const NAMES = ['mu', 'sigma', 'nu'] as const
const SYMBOLS = ['μ', 'σ', 'ν']

type Settings = { family: DistributionalFamilyName; smooths: number; n: number; seed: number }

/** Fit by the RS algorithm, keeping every cycle. */
function fit({ family, smooths, n, seed }: Settings) {
  const d = growthChart(stream(seed), { n })
  const data = { x: d.x, y: d.y! }
  const K = distributionalFamily(family).parameters.length
  const parameters = Object.fromEntries(
    NAMES.slice(0, Math.min(smooths, K)).map((name) => [name, { terms: [s(0, { k: 12 })] }]),
  )
  const problem = gamlssProblem({ family, parameters }, data)
  return { settings: { family, smooths, n, seed }, data, problem, trace: gamlssTrace(problem, 60) }
}

/** The true value of each family's parameters, where the family's parameter means what BCCG's does. */
function truth(family: DistributionalFamilyName, k: number, a: number): number {
  if (family === 'box-cox-cole-green') return [GROWTH_TRUTH.mu, GROWTH_TRUTH.sigma, GROWTH_TRUTH.nu][k](a)
  if (k === 0) return GROWTH_TRUTH.centile(a, 0.5)
  return NaN
}

export function GamlssFit() {
  const state = useFigureState({
    family: choice(
      [
        { value: 'normal', label: 'Normal' },
        { value: 'student-t', label: 'Student t' },
        { value: 'box-cox-cole-green', label: 'BCCG (LMS)' },
      ],
      'box-cox-cole-green',
      { label: 'family' },
    ),
    smooths: choice(
      [
        { value: 1, label: 'μ only' },
        { value: 2, label: 'μ and σ' },
        { value: 3, label: 'μ, σ and ν' },
      ],
      3,
      { label: 'smooth parameters' },
    ),
    n: int(1000, { ge: 100, le: 5000, suggestions: [300, 1000, 3000], label: 'n' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })
  const settings: Settings = {
    family: state.family as DistributionalFamilyName,
    smooths: state.smooths as number,
    n: state.n,
    seed: state.seed,
  }
  const [run, setRun] = useState(() => fit(settings))
  const [cycle, setCycle] = useState(0)
  const stale = JSON.stringify(settings) !== JSON.stringify(run.settings)

  const steps = run.trace.steps
  const at = Math.min(cycle, steps.length - 1)
  const model = useMemo(() => gamlssModel(run.problem, steps[at]), [run, steps, at])
  const view = useMemo(() => {
    const centiles = model.centiles(AGE_GRID, LEVELS)
    const parameters = model.parameters(AGE_GRID)
    const fitted = model.centiles(run.data.x, [0.03, 0.97])
    const y = toFlat(run.data.y)
    const a = toFlat(run.data.x)
    // Percentages below the 3rd and above the 97th fitted centile among ages in [lo, hi): 3% each when right.
    const shares = (lo: number, hi: number) => {
      const idx = a.flatMap((v, i) => (v >= lo && v < hi ? [i] : []))
      const pct = (f: (i: number) => boolean) => ((100 * idx.filter(f).length) / idx.length).toFixed(1)
      return `${pct((i) => y[i] < fitted[0][i])}% / ${pct((i) => y[i] > fitted[1][i])}%`
    }
    return { centiles, parameters, young: shares(0, 6), old: shares(12, 19) }
  }, [model, run])
  const deviance = useMemo(() => toFlat(run.trace.series.deviance), [run])
  const cycles = useMemo(() => Array.from(run.trace.index), [run])
  const K = model.family.parameters.length
  const fixed = run.settings.smooths

  const age = useAxis({ label: 'age (years)', range: [0, 18] })
  const yAxis = useAxis({ label: 'measurement', range: [0, 110] })
  const cycleAxis = useAxis({ label: 'RS cycle', hold: 'initial', key: run })
  const devAxis = useAxis({
    label: 'global deviance',
    hold: 'initial',
    key: run,
    range: [deviance[1] - 50, deviance[1] + 300],
  })
  const parameterAxes = [useAxis({ label: 'μ' }), useAxis({ label: 'σ' }), useAxis({ label: 'ν' })]
  const xs = toFlat(run.data.x)
  const ys = toFlat(run.data.y)

  return (
    <Figure
      title="Fitting centile curves by the RS algorithm"
      state={state}
      controls={
        <>
          <Button
            size="sm"
            variant={stale ? 'default' : 'outline'}
            onClick={() => {
              setRun(fit(settings))
              setCycle(0)
            }}
          >
            Fit
          </Button>
          <Player value={at} onChange={setCycle} count={steps.length} label="RS cycle" />
        </>
      }
      caption={
        <>
          Synthetic growth data: a measurement against age drawn from a Box–Cox Cole–Green law whose median, spread and
          skewness all change with age. Choose a family and which parameters get a P-spline smooth (12 basis functions,
          λ by local maximum likelihood), press Fit, then step through the RS cycles. Top: the data with the fitted 3rd,
          10th, 25th, 50th, 75th, 90th and 97th centiles (thin), the median (thick) and the true 3rd, 50th and 97th
          centiles (dashed). Middle: the fitted parameter curves, with the truth dashed where the family&apos;s
          parameter means the same thing. Bottom: the global deviance per cycle. A Normal with constant σ follows the
          centre of the data, but its centiles are parallel and miss the growing, right-skewed spread; BCCG with smooth
          μ, σ and ν puts about 3% of the data below its 3rd centile and 3% above its 97th at every age.
        </>
      }
      readouts={
        <>
          <Readout
            label="cycle"
            value={`${at} of ${steps.length - 1}${stale ? ' (settings changed: press Fit)' : ''}`}
          />
          <Readout label="global deviance" value={formatNumber(model.deviance)} />
          <Readout label="GAIC(2)" value={formatNumber(model.gaic(2))} />
          <Readout label="EDF" value={model.edf.map((e, k) => `${SYMBOLS[k]} ${formatNumber(e)}`).join(', ')} />
          <Readout label="below 3rd / above 97th, ages 0–6" value={view.young} />
          <Readout label="below 3rd / above 97th, ages 12–18" value={view.old} />
        </>
      }
    >
      <Plot x={age} y={yAxis} height={320}>
        <Points name="data" x={xs} y={ys} muted />
        {view.centiles.map((c, j) => (
          <Curve key={j} name="fitted centiles" x={AGES} y={c} slot={0} width={LEVELS[j] === 0.5 ? 2.5 : 1} />
        ))}
        {[0.03, 0.5, 0.97].map((p) => (
          <Curve
            key={p}
            name="true centiles"
            x={AGES}
            y={AGES.map((a) => GROWTH_TRUTH.centile(a, p))}
            emphasis
            dashed
            width={1}
          />
        ))}
      </Plot>
      <div className={K === 3 ? 'grid gap-2 md:grid-cols-3' : 'grid gap-2 md:grid-cols-2'}>
        {Array.from({ length: K }, (_, k) => (
          <Plot
            key={k}
            x={age}
            y={parameterAxes[k]}
            height={170}
            legend={false}
            title={k < fixed ? SYMBOLS[k] : `${SYMBOLS[k]}: constant (not smoothed)`}
          >
            <Curve
              name={`fitted ${SYMBOLS[k]}${k < fixed ? '' : ' (constant: not smoothed)'}`}
              x={AGES}
              y={view.parameters[k]}
              slot={k}
              dashed={k >= fixed}
            />
            <Curve
              name={`true ${SYMBOLS[k]}`}
              x={AGES}
              y={AGES.map((a) => truth(model.family.name, k, a))}
              emphasis
              dashed
            />
          </Plot>
        ))}
      </div>
      <Plot x={cycleAxis} y={devAxis} height={150} legend={false}>
        <Curve name="global deviance" x={cycles} y={deviance} slot={0} showPoints />
        <Points name="this cycle" x={[cycles[at]]} y={[deviance[at]]} emphasis />
      </Plot>
    </Figure>
  )
}
