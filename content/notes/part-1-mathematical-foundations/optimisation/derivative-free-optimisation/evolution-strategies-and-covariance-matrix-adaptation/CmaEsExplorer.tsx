import { useMemo, useState } from 'react'
import { cmaEs, type CmaEsState } from 'aifn/optim/derivative-free'
import { rastrigin, type TestFunction } from 'aifn-applied/data/objectives'
import { stream } from 'aifn/foundation/random'
import { tensor, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import {
  ControlRow,
  Curve,
  Figure,
  formatNumber,
  Player,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Slider,
  useAxis,
} from 'aifn-render'

type Surface = { x: number[]; y: number[]; z: number[][] }

function surfaceOf(fn: TestFunction, n = 90): Surface {
  const [x0, y0] = toFlat(fn.lo)
  const [x1, y1] = toFlat(fn.hi)
  const x = Array.from({ length: n }, (_, i) => x0 + ((x1 - x0) * i) / (n - 1))
  const y = Array.from({ length: n }, (_, i) => y0 + ((y1 - y0) * i) / (n - 1))
  const z = y.map((yi) => x.map((xj) => Math.log10(1 + fn.value(tensor([xj, yi])))))
  return { x, y, z }
}

function pathOf(series: Tensor): { xs: number[]; ys: number[] } {
  const data = toFlat(series)
  const kept = series.shape[0]
  return {
    xs: Array.from({ length: kept }, (_, k) => data[2 * k]),
    ys: Array.from({ length: kept }, (_, k) => data[2 * k + 1]),
  }
}

const RASTRIGIN = rastrigin()

export function CmaEsExplorer() {
  const [sigma, setSigma] = useState(2.0)
  const [seed, setSeed] = useState(1)
  const [step, setStep] = useState(0)

  const surface = useMemo(() => surfaceOf(RASTRIGIN, 100), [])

  const t = useMemo(
    () =>
      trace(
        cmaEs(RASTRIGIN.value, { sigma, populationSize: 12 }),
        { x0: [3, 3] },
        120,
        {
          record: {
            'f(mean)': (s) => s.value,
            'best f': (s) => s.bestValue,
            'step size σ': (s) => s.sigma,
            x: (s) => s.x,
          },
          stream: stream(`cmaes-${seed}`),
        },
      ),
    [sigma, seed],
  )

  const totalSteps = t.steps.length
  const currentStep = Math.min(step, totalSteps - 1)
  const currentState = t.steps[currentStep] as CmaEsState

  const path = useMemo(() => pathOf(t.series.x), [t])
  const clip = (v: number, a: readonly number[]) => Math.min(Math.max(v, a[0]), a[a.length - 1])

  const soFar = useMemo(
    () => ({
      x: path.xs.slice(0, currentStep + 1).map((v) => clip(v, surface.x)),
      y: path.ys.slice(0, currentStep + 1).map((v) => clip(v, surface.y)),
    }),
    [path, currentStep, surface],
  )

  const popXY = useMemo(() => {
    if (!currentState.population) return { x: [], y: [] }
    const p = toFlat(currentState.population)
    const px: number[] = []
    const py: number[] = []
    for (let i = 0; i < p.length; i += 2) {
      px.push(clip(p[i], surface.x))
      py.push(clip(p[i + 1], surface.y))
    }
    return { x: px, y: py }
  }, [currentState, surface])

  const bestSeries = useMemo(() => toFlat(t.series['best f']), [t])
  const iterIndices = useMemo(() => Array.from(t.index), [t])

  const xPlot = useAxis({ label: 'x₀', range: [surface.x[0], surface.x[surface.x.length - 1]] })
  const yPlot = useAxis({ label: 'x₁', equal: xPlot, range: [surface.y[0], surface.y[surface.y.length - 1]] })
  const xGen = useAxis({ label: 'generation' })
  const yMetric = useAxis({ label: 'best f(x) (log scale)', log: true, range: [1e-12, 1e2] })

  return (
    <Figure
      title="CMA-ES on the multimodal Rastrigin function"
      purpose="Simulate covariance matrix adaptation evolution strategies on a multimodal landscape with adaptive search distributions."
      caption="Left: the Rastrigin landscape with its lattice of local minima. CMA-ES samples candidate populations (blue dots) from N(m, σ²C), moves the mean along successful steps, and adapts the covariance C and step size σ. Right: best objective value found over generations. With large initial σ (e.g. 2.0–3.0), the search steps over local basins to reach the global minimum (0,0); with small initial σ (e.g. 0.3), it is trapped in the nearest local minimum."
    >
      <ControlRow>
        <Slider
          label="Initial step size σ"
          value={sigma}
          min={0.2}
          max={3.5}
          step={0.1}
          onChange={setSigma}
        />
        <Slider
          label="Seed"
          value={seed}
          min={1}
          max={20}
          step={1}
          onChange={setSeed}
        />
      </ControlRow>

      <ControlRow>
        <Player
          label="Generation"
          value={currentStep}
          count={totalSteps}
          onChange={setStep}
        />
      </ControlRow>

      <div className="flex flex-wrap gap-4 text-xs font-mono text-muted-foreground my-2">
        <Readout label="generation" value={`${currentStep} / ${totalSteps - 1}`} />
        <Readout label="best f" value={formatNumber(currentState.bestValue)} />
        <Readout label="current σ" value={formatNumber(currentState.sigma)} />
        <Readout
          label="mean position"
          value={`(${formatNumber(path.xs[currentStep])}, ${formatNumber(path.ys[currentStep])})`}
        />
      </div>

      <Plots cols={2}>
        <Plot x={xPlot} y={yPlot} title="Search distribution & population">
          <Raster x={surface.x} y={surface.y} z={surface.z} valueLabel="log₁₀(1 + f)" />
          <Points name="global minimum (0, 0)" x={[0]} y={[0]} emphasis size={8} />
          <Curve name="mean path" x={soFar.x} y={soFar.y} slot={1} showPoints />
          <Points name="population samples" x={popXY.x} y={popXY.y} slot={2} size={6} />
          <Points
            name="current mean"
            x={[clip(path.xs[currentStep], surface.x)]}
            y={[clip(path.ys[currentStep], surface.y)]}
            slot={1}
            size={9}
          />
        </Plot>

        <Plot x={xGen} y={yMetric} title="Best f(x) over generations">
          <Curve name="best f(x)" x={iterIndices} y={bestSeries.map((v) => Math.max(v, 1e-12))} slot={0} />
          <Points
            name="current"
            x={[currentStep]}
            y={[Math.max(currentState.bestValue, 1e-12)]}
            emphasis
            size={7}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
