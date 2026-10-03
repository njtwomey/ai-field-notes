import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'
import { betaHpd, betaPdf, betaQuantile } from '../_shared/beta'

const GRID = linspace(0.0005, 0.9995, 500)

/**
 * Posterior Beta(1 + k, 1 + n − k) from a uniform prior. The equal-tailed interval cuts α/2 from each tail; the HPD
 * interval is the shortest interval with the same mass, and its endpoints have equal density.
 */
export function IntervalCompare() {
  const [n, setN] = useState(12)
  const [k, setK] = useState(1)
  const [level, setLevel] = useState(0.95)

  const r = useMemo(() => {
    const kk = Math.min(k, n)
    const a = 1 + kk
    const b = 1 + n - kk
    const et: [number, number] = [betaQuantile((1 - level) / 2, a, b), betaQuantile((1 + level) / 2, a, b)]
    const hpd = betaHpd(a, b, level)
    const dens = GRID.map((x) => betaPdf(x, a, b))
    const top = Math.max(...dens)
    const cut = Math.min(betaPdf(Math.max(hpd[0], 1e-6), a, b), betaPdf(Math.min(hpd[1], 1 - 1e-6), a, b))
    const series: XYSeries[] = [
      { name: 'posterior density', type: 'line', x: GRID, y: dens, emphasis: true },
      {
        name: `${Math.round(100 * level)}% HPD interval`,
        type: 'line',
        x: GRID,
        y: GRID.map((x, i) => (x >= hpd[0] && x <= hpd[1] ? dens[i] : NaN)),
        slot: 0,
        area: true,
      },
      {
        name: 'equal-tailed interval',
        type: 'line',
        x: [et[0], et[0], NaN, et[1], et[1]],
        y: [0, top * 1.05, NaN, 0, top * 1.05],
        slot: 1,
        dashed: true,
      },
      { name: 'HPD density level', type: 'line', x: [0, 1], y: [cut, cut], muted: true },
    ]
    return { et, hpd, series, top, kk }
  }, [n, k, level])

  return (
    <Interactive
      title="Equal-tailed and highest-density intervals"
      caption="Posterior for a success probability after k successes in n trials with a uniform prior. The shaded region is the HPD interval: every point inside has higher density than every point outside, and its two ends sit on the same density level (grey line). The dashed lines mark the equal-tailed interval, which cuts equal probability from each tail. The two agree for a symmetric posterior (k = n/2) and differ most when k is near 0 or n."
      controls={
        <>
          <ParamSlider label="trials n" value={n} onChange={setN} min={1} max={60} step={1} />
          <ParamSlider label="successes k" value={r.kk} onChange={setK} min={0} max={n} step={1} withArrows />
          <ParamSlider
            label="credible level"
            value={level}
            onChange={setLevel}
            min={0.5}
            max={0.99}
            step={0.01}
            format={(v) => `${Math.round(100 * v)}%`}
          />
        </>
      }
      readout={
        <>
          <Readout
            label="equal-tailed"
            value={`[${formatNumber(r.et[0])}, ${formatNumber(r.et[1])}], width ${formatNumber(r.et[1] - r.et[0])}`}
          />
          <Readout
            label="HPD"
            value={`[${formatNumber(r.hpd[0])}, ${formatNumber(r.hpd[1])}], width ${formatNumber(r.hpd[1] - r.hpd[0])}`}
          />
        </>
      }
    >
      <XYChart
        height={300}
        series={r.series}
        xRange={[0, 1]}
        yRange={[0, r.top * 1.08]}
        xLabel="success probability θ"
        yLabel="posterior density"
      />
    </Interactive>
  )
}
