import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { linspace, rng } from '@/lib/math'

const GRID = linspace(-4, 4, 201)
/** Capped because each draw costs D × 201 cosines, recomputed while D is dragged. */
const MAX_DRAWS = 20

/** k(0, x) for the Gaussian kernel against its estimate z(0)ᵀz(x) from D random Fourier features. */
export function KernelApproximation() {
  const logD = useParam(1.5, { min: 0, max: 3.5, step: 0.1 })
  const width = useParam(1, { min: 0.3, max: 2, step: 0.05 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const count = useParam(5, { min: 1, max: MAX_DRAWS, step: 1 })
  const D = Math.round(10 ** logD.value)

  const r = useMemo(() => {
    const exact = GRID.map((x) => Math.exp(-(x * x) / (2 * width.value ** 2)))
    // Draw k uses its own stream, so raising the count adds curves without changing the earlier ones.
    const approxes = Array.from({ length: count.value }, (_, k) => {
      const g = rng(seed.value * 1000 + k)
      // Frequencies from the kernel's spectral density, N(0, 1/ℓ²), and uniform phases.
      const w = Array.from({ length: D }, () => g.normal() / width.value)
      const b = Array.from({ length: D }, () => 2 * Math.PI * g.uniform())
      const scale = 2 / D
      const z0 = b.map((bi) => Math.cos(bi))
      return GRID.map((x) => scale * w.reduce((s, wi, i) => s + z0[i] * Math.cos(wi * x + b[i]), 0))
    })
    const maxErr =
      approxes.reduce((acc, a) => acc + Math.max(...a.map((v, i) => Math.abs(v - exact[i]))), 0) / approxes.length
    const many = approxes.length > 1
    const series: XYSeries[] = [
      ...approxes.map((y): XYSeries => ({
        name: many ? 'random-feature estimates z(0)ᵀz(x)' : 'random-feature estimate z(0)ᵀz(x)',
        type: 'line',
        x: GRID,
        y,
        slot: 1,
        thin: many,
      })),
      { name: 'exact k(0, x)', type: 'line', x: GRID, y: exact, slot: 0 },
    ]
    return { series, maxErr }
  }, [D, width.value, seed.value, count.value])

  return (
    <Interactive
      title="Approximating a Gaussian kernel with random features"
      caption="The exact curve is the Gaussian kernel k(0, x). Each light curve is one estimate z(0)ᵀz(x), built from its own independent draw of D random cosine features; the draws slider sets how many. The estimates scatter evenly about the exact curve because each is unbiased, and their spread shrinks like 1/√D: a tenfold increase in D cuts the error by about a factor of three. Change the seed to see a different set of draws."
      controls={
        <>
          <ParamSlider
            label="number of features D (log scale)"
            param={logD}
            format={(v) => String(Math.round(10 ** v))}
          />
          <ParamSlider label="kernel width ℓ" param={width} />
          <ParamSlider label="draws" param={count} withArrows format={(v) => String(v)} />
          <ParamSlider label="seed" param={seed} />
        </>
      }
      readout={
        <>
          <Readout label="largest error on the plot (mean over draws)" value={formatNumber(r.maxErr)} />
        </>
      }
    >
      <XYChart height={320} xLabel="x" yLabel="k(0, x)" series={r.series} xRange={[-4, 4]} yRange={[-0.6, 1.4]} />
    </Interactive>
  )
}
