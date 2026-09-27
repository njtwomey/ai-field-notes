import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace, rng } from '@/lib/math'

const MAX_SHOWN = 1500
const SIGMA = 4 * Math.sqrt((Math.PI / 4) * (1 - Math.PI / 4))
const ARC = linspace(0, Math.PI / 2, 90)
const CHECKPOINTS = 160

/**
 * Estimate π as 4 × the fraction of uniform points in the unit square that fall inside the quarter disc. The right
 * panel shows the running estimate against log10(n), inside the band π ± 1.96 σ/√n that the central limit theorem
 * predicts.
 */
export function MonteCarloPi() {
  const logN = useParam(3, { min: 1, max: 5, step: 0.25 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })

  const r = useMemo(() => {
    const n = Math.round(10 ** logN.value)
    const g = rng(seed.value)
    const px: number[] = []
    const py: number[] = []
    const inside: number[] = []
    const tx: number[] = []
    const ty: number[] = []
    const marks = new Set(linspace(0, logN.value, CHECKPOINTS).map((l) => Math.round(10 ** l)))
    let hits = 0
    for (let i = 1; i <= n; i++) {
      const x = g.uniform()
      const y = g.uniform()
      const hit = x * x + y * y <= 1
      if (hit) hits++
      if (i <= MAX_SHOWN) {
        px.push(x)
        py.push(y)
        inside.push(hit ? 0 : 1)
      }
      if (marks.has(i)) {
        tx.push(Math.log10(i))
        ty.push((4 * hits) / i)
      }
    }
    const estimate = (4 * hits) / n
    const p = hits / n
    const se = (4 * Math.sqrt(p * (1 - p))) / Math.sqrt(n)
    return { n, px, py, inside, tx, ty, estimate, se }
  }, [logN.value, seed.value])

  const square: XYSeries[] = useMemo(
    () => [
      {
        name: 'point',
        type: 'scatter',
        x: r.px,
        y: r.py,
        group: r.inside,
        groupNames: ['inside the quarter disc', 'outside'],
      },
      { name: 'x² + y² = 1', type: 'line', x: ARC.map(Math.cos), y: ARC.map(Math.sin), emphasis: true },
    ],
    [r],
  )

  const running: XYSeries[] = useMemo(() => {
    const band = linspace(0, logN.value, 120)
    const half = band.map((l) => (1.96 * SIGMA) / Math.sqrt(10 ** l))
    return [
      { name: 'running estimate', type: 'line', x: r.tx, y: r.ty, slot: 0 },
      { name: 'π', type: 'line', x: [0, logN.value], y: [Math.PI, Math.PI], emphasis: true, dashed: true },
      { name: 'π ± 1.96 σ/√n', type: 'line', x: band, y: band.map((_, i) => Math.PI + half[i]), muted: true },
      { name: 'π − 1.96 σ/√n', type: 'line', x: band, y: band.map((_, i) => Math.PI - half[i]), muted: true },
    ]
  }, [r, logN.value])

  return (
    <Interactive
      title="Estimating π by Monte Carlo"
      caption={`Points are uniform on the unit square; 4 × the fraction inside the quarter disc estimates π. The left panel shows at most ${MAX_SHOWN} of the points. The right panel follows the estimate as samples accumulate, against log₁₀ n. The grey band is π ± 1.96 σ/√n with σ = 1.642: each tenfold increase in n narrows it by √10 ≈ 3.2, whatever the seed.`}
      controls={
        <>
          <ParamSlider label="log₁₀ of sample size n" param={logN} />
          <ParamSlider label="random seed" param={seed} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="n" value={r.n.toLocaleString('en-GB')} />
          <Readout label="estimate" value={formatNumber(r.estimate)} />
          <Readout label="error" value={formatNumber(r.estimate - Math.PI)} />
          <Readout label="standard error" value={formatNumber(r.se)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <XYChart series={square} xLabel="x" yLabel="y" xRange={[0, 1]} yRange={[0, 1]} equalAspect />
        <XYChart series={running} xLabel="log₁₀ n" yLabel="estimate of π" yRange={[2.4, 3.9]} height={300} />
      </div>
    </Interactive>
  )
}
