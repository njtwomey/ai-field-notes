import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { addDiagonal, cholesky, gram, makeKernel, samplesFromFactor, type Kernel } from '../_shared/gp'

type Choice =
  'se' | 'matern12' | 'matern32' | 'matern52' | 'rq' | 'periodic' | 'linear' | 'se-x-periodic' | 'se-plus-linear'

const OPTIONS = [
  { value: 'se' as const, label: 'squared exponential' },
  { value: 'matern12' as const, label: 'Matérn 1/2' },
  { value: 'matern32' as const, label: 'Matérn 3/2' },
  { value: 'matern52' as const, label: 'Matérn 5/2' },
  { value: 'rq' as const, label: 'rational quadratic (α = 0.5)' },
  { value: 'periodic' as const, label: 'periodic (p = 2)' },
  { value: 'linear' as const, label: 'linear' },
  { value: 'se-x-periodic' as const, label: 'SE × periodic' },
  { value: 'se-plus-linear' as const, label: 'SE + linear' },
]

const GRID = linspace(-5, 5, 151)
const X_RANGE: [number, number] = [-5, 5]
const Y_RANGE: [number | undefined, number | undefined] = [-4, 4]
const K_RANGE: [number | undefined, number | undefined] = [undefined, undefined]
const REF = 1
const MAX_SAMPLES = 30
const ZERO = GRID.map(() => 0)
/** Draw k has its own stream of standard normals, shared by every kernel, so raising the count only adds draws. */
const NORMALS = Array.from({ length: MAX_SAMPLES }, (_, k) => {
  const g = rng(5 * 1000 + k)
  return GRID.map(() => g.normal())
})

function kernelFor(choice: Choice, ell: number): Kernel {
  switch (choice) {
    case 'rq':
      return makeKernel('rq', { ell, sf: 1, alpha: 0.5 })
    case 'periodic':
      return makeKernel('periodic', { ell, sf: 1, period: 2 })
    case 'linear':
      return makeKernel('linear', { ell, sf: 0.3 })
    case 'se-x-periodic': {
      // A periodic pattern whose shape drifts: the SE factor decays over a longer distance than one period.
      const se = makeKernel('se', { ell: 3 * ell, sf: 1 })
      const per = makeKernel('periodic', { ell: 1, sf: 1, period: 2 })
      return (a, b) => se(a, b) * per(a, b)
    }
    case 'se-plus-linear': {
      const se = makeKernel('se', { ell, sf: 0.7 })
      const lin = makeKernel('linear', { ell, sf: 0.3 })
      return (a, b) => se(a, b) + lin(a, b)
    }
    default:
      return makeKernel(choice, { ell, sf: 1 })
  }
}

/** Draws from a zero-mean GP prior for each covariance function, beside the kernel as a function of x. */
export function PriorSamples() {
  const [choice, setChoice] = useState<Choice>('se')
  const logEll = useParam(0, { min: -1, max: 0.7, step: 0.02 })
  const count = useParam(3, { min: 1, max: MAX_SAMPLES, step: 1 })
  const ell = 10 ** logEll.value

  // The Cholesky factor depends only on the kernel; changing the number of draws reuses it.
  const r = useMemo(() => {
    const k = kernelFor(choice, ell)
    return { factor: cholesky(addDiagonal(gram(k, GRID, GRID), 1e-6)), shape: GRID.map((x) => k(REF, x)) }
  }, [choice, ell])

  const sampleSeries = useMemo((): XYSeries[] => {
    const many = count.value > 1
    return samplesFromFactor(ZERO, r.factor, NORMALS.slice(0, count.value)).map((d) => ({
      name: many ? 'prior samples' : 'prior sample',
      type: 'line',
      x: GRID,
      y: d,
      slot: 1,
      thin: many,
    }))
  }, [r, count.value])
  const kernelSeries = useMemo(
    (): XYSeries[] => [{ name: `k(${REF}, x)`, type: 'line', x: GRID, y: r.shape, slot: 0 }],
    [r],
  )
  const usesEll = choice !== 'linear'

  return (
    <Interactive
      title="Samples from Gaussian process priors"
      caption="Left: functions drawn from a zero-mean GP prior with the chosen covariance, as light lines, using the same random numbers for every kernel. The draws slider sets how many; a single draw is drawn at full weight. Right: the covariance between f(1) and f(x). The squared exponential gives infinitely smooth samples. Matérn 1/2 samples are continuous but jagged, and Matérn 3/2 and 5/2 are once and twice differentiable. The rational quadratic mixes length-scales, so its samples vary on several scales at once. The periodic kernel repeats exactly every 2 units; multiplying it by a squared exponential lets the repeating shape drift. The linear kernel gives straight lines, and adding it to a squared exponential gives wiggles about a trend."
      controls={
        <>
          <ParamChoice label="covariance function" value={choice} onChange={setChoice} options={OPTIONS} />
          <ParamSlider label="length-scale ℓ" param={logEll} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="draws" param={count} withArrows format={(v) => String(v)} />
        </>
      }
      readout={<Readout label="length-scale ℓ" value={usesEll ? formatNumber(ell) : 'not used'} />}
    >
      <div className="grid gap-4 md:grid-cols-[2fr_1fr]">
        <XYChart series={sampleSeries} xLabel="x" yLabel="f(x)" xRange={X_RANGE} yRange={Y_RANGE} height={340} />
        <XYChart
          series={kernelSeries}
          xLabel="x"
          yLabel={`k(${REF}, x)`}
          xRange={X_RANGE}
          yRange={K_RANGE}
          height={340}
        />
      </div>
    </Interactive>
  )
}
