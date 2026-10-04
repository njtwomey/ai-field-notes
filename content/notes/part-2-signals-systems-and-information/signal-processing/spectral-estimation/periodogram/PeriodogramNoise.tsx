import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { periodogram, whiteNoise } from '../_shared/spectra'

/**
 * Periodograms of unit-variance white noise at increasing N. The true spectrum is flat at 1, but the estimate scatters
 * with the same spread at every N: the periodogram is not consistent.
 */
export function PeriodogramNoise() {
  const state = useFigureState({
    logN: int(8, { min: 6, max: 12, step: 1, label: 'N = 2^k samples, k', format: (v) => `${v}  (N = ${2 ** v})` }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })
  const n = 2 ** state.logN

  const r = useMemo(() => {
    const x = whiteNoise(n, state.seed)
    const p = periodogram(x)
    // Interior bins only: DC and Nyquist are real-valued and follow a χ² with one degree of freedom.
    const interior = p.psd.slice(1, -1)
    const mean = interior.reduce((s, v) => s + v, 0) / interior.length
    const sd = Math.sqrt(interior.reduce((s, v) => s + (v - mean) ** 2, 0) / interior.length)
    return { omega: p.omega.map((w) => w / Math.PI), psd: p.psd, mean, sd }
  }, [n, state.seed])

  const series = [
    { name: 'periodogram', x: r.omega, y: r.psd, slot: 0 },
    { name: 'true spectrum S(ω) = 1', x: [0, 1], y: [1, 1], slot: 1, dashed: true },
  ] as const

  const xAxis = useAxis({ label: 'ω / π', range: [0, 1] })
  const yAxis = useAxis({ label: 'Ŝ(ω)', range: [0, 8] })
  return (
    <Figure
      title="More data, same scatter"
      state={state}
      caption="The periodogram of N samples of unit-variance white noise, whose true spectrum is flat at 1. As N grows the frequency grid gets finer, but each value still scatters as an exponential variable with mean 1 and standard deviation 1. The periodogram is asymptotically unbiased but not consistent."

      readouts={
        <>
          <Readout label="frequency bins" value={r.psd.length} />
          <Readout label="mean of estimate" value={formatNumber(r.mean)} />
          <Readout label="std of estimate" value={formatNumber(r.sd)} />
          <Readout label="theory (both)" value="1" />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
      </Plot>
    </Figure>
  )
}
