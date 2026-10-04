import { useMemo } from 'react'
import { choice, Curve, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { periodogram, powerDb, welch, whiteNoise } from '../_shared/spectra'

const N = 4096
// Two sinusoids 0.02π apart plus a weak one, in unit-variance white noise.
const COMPONENTS = [
  { omega: 0.3 * Math.PI, amp: 1 },
  { omega: 0.32 * Math.PI, amp: 1 },
  { omega: 0.6 * Math.PI, amp: 0.1 },
]

/**
 * Welch's method on two close sinusoids and a weak one in white noise. Long segments resolve the pair but average few
 * periodograms; short segments give a smooth estimate that merges the pair.
 */
export function WelchTradeoff() {
  const state = useFigureState({
    logL: int(8, {
      min: 5,
      max: 12,
      step: 1,
      label: 'segment length L = 2^k, k',
      format: (v) => `${v}  (L = ${2 ** v})`,
    }),
    overlap: choice<'0' | '50'>(
      [
        { value: '0', label: 'none' },
        { value: '50', label: '50%' },
      ],
      '50',
      { label: 'overlap' },
    ),
    seed: int(3, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })
  const seg = 2 ** state.logL

  const r = useMemo(() => {
    const noise = whiteNoise(N, state.seed)
    const x = noise.map((e, n) => e + COMPONENTS.reduce((s, c) => s + c.amp * Math.cos(c.omega * n), 0))
    const w = welch(x, seg, state.overlap === '50' ? seg / 2 : 0, 'hann')
    const p = periodogram(x)
    return {
      welch: { x: w.omega.map((v) => v / Math.PI), y: w.psd.map((v) => powerDb(v)) },
      raw: { x: p.omega.map((v) => v / Math.PI), y: p.psd.map((v) => powerDb(v)) },
      segments: w.segments,
    }
  }, [seg, state.overlap, state.seed])

  const series = [
    { name: 'periodogram (all N samples)', x: r.raw.x, y: r.raw.y, muted: true },
    { name: 'Welch estimate', x: r.welch.x, y: r.welch.y, slot: 0 },
  ] as const
  // Hann: −3 dB main-lobe width is about 1.44 bins of 2π/L, and a pair is resolved when it exceeds the separation.
  const resolution = (1.44 * 2) / seg
  const dof = 2 * r.segments

  const xAxis = useAxis({ label: 'ω / π', range: [0, 1] })
  const yAxis = useAxis({ label: 'power (dB)', range: [-30, 40] })
  return (
    <Figure
      title="Resolution against variance"
      state={state}
      caption="Two equal sinusoids at 0.30π and 0.32π rad/sample, a weak one at 0.60π, and white noise, N = 4,096. Welch's method splits the record into Hann-windowed segments of length L and averages their periodograms. Long segments resolve the close pair but average few periodograms, so the noise floor is jagged; short segments give a smooth floor that merges the pair into one peak. The grey line is the raw periodogram of the whole record."

      readouts={
        <>
          <Readout label="segments averaged K" value={r.segments} />
          <Readout label="resolution ≈ 1.44·2π/L (×π)" value={formatNumber(resolution)} />
          <Readout label="pair separation (×π)" value="0.02" />
          <Readout label="degrees of freedom ≈ 2K" value={dof} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
      </Plot>
    </Figure>
  )
}
