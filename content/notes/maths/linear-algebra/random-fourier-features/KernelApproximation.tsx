import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace, rng } from '@/lib/math'

const GRID = linspace(-4, 4, 201)

/** k(0, x) for the Gaussian kernel against its estimate z(0)ᵀz(x) from D random Fourier features. */
export function KernelApproximation() {
  const logD = useParam(1.5, { min: 0, max: 3.5, step: 0.1 })
  const width = useParam(1, { min: 0.3, max: 2, step: 0.05 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const D = Math.round(10 ** logD.value)

  const r = useMemo(() => {
    const g = rng(seed.value)
    // Frequencies from the kernel's spectral density, N(0, 1/ℓ²), and uniform phases.
    const w = Array.from({ length: D }, () => g.normal() / width.value)
    const b = Array.from({ length: D }, () => 2 * Math.PI * g.uniform())
    const scale = 2 / D
    const z0 = b.map((bi) => Math.cos(bi))
    const approx = GRID.map((x) => scale * w.reduce((s, wi, i) => s + z0[i] * Math.cos(wi * x + b[i]), 0))
    const exact = GRID.map((x) => Math.exp(-(x * x) / (2 * width.value ** 2)))
    const maxErr = Math.max(...GRID.map((_, i) => Math.abs(approx[i] - exact[i])))
    const series: XYSeries[] = [
      { name: 'exact k(0, x)', type: 'line', x: GRID, y: exact, slot: 0 },
      { name: 'random-feature estimate z(0)ᵀz(x)', type: 'line', x: GRID, y: approx, slot: 1 },
    ]
    return { series, maxErr }
  }, [D, width.value, seed.value])

  return (
    <Interactive
      title="Approximating a Gaussian kernel with random features"
      caption="The exact curve is the Gaussian kernel k(0, x). The estimate is z(0)ᵀz(x), built from D random cosine features. The estimate is unbiased, and its error shrinks like 1/√D: a tenfold increase in D cuts the error by about a factor of three. Change the seed to see a different random draw."
      controls={
        <>
          <ParamSlider
            label="number of features D (log scale)"
            param={logD}
            format={(v) => String(Math.round(10 ** v))}
          />
          <ParamSlider label="kernel width ℓ" param={width} />
          <ParamSlider label="seed" param={seed} />
        </>
      }
      readout={
        <>
          <Readout label="largest error on the plot" value={formatNumber(r.maxErr)} />
        </>
      }
    >
      <XYChart height={320} xLabel="x" yLabel="k(0, x)" series={r.series} xRange={[-4, 4]} yRange={[-0.6, 1.4]} />
    </Interactive>
  )
}
