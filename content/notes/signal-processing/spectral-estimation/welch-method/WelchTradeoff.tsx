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
} from '@/components/viz'
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
  const logL = useParam(8, { min: 5, max: 12, step: 1 })
  const [overlap, setOverlap] = useState<'0' | '50'>('50')
  const seed = useParam(3, { min: 1, max: 30, step: 1 })
  const seg = 2 ** logL.value

  const r = useMemo(() => {
    const noise = whiteNoise(N, seed.value)
    const x = noise.map((e, n) => e + COMPONENTS.reduce((s, c) => s + c.amp * Math.cos(c.omega * n), 0))
    const w = welch(x, seg, overlap === '50' ? seg / 2 : 0, 'hann')
    const p = periodogram(x)
    return {
      welch: { x: w.omega.map((v) => v / Math.PI), y: w.psd.map((v) => powerDb(v)) },
      raw: { x: p.omega.map((v) => v / Math.PI), y: p.psd.map((v) => powerDb(v)) },
      segments: w.segments,
    }
  }, [seg, overlap, seed.value])

  const series: XYSeries[] = [
    { name: 'periodogram (all N samples)', type: 'line', x: r.raw.x, y: r.raw.y, muted: true },
    { name: 'Welch estimate', type: 'line', x: r.welch.x, y: r.welch.y, slot: 0 },
  ]
  // Hann: −3 dB main-lobe width is about 1.44 bins of 2π/L, and a pair is resolved when it exceeds the separation.
  const resolution = (1.44 * 2) / seg
  const dof = 2 * r.segments

  return (
    <Interactive
      title="Resolution against variance"
      caption="Two equal sinusoids at 0.30π and 0.32π rad/sample, a weak one at 0.60π, and white noise, N = 4,096. Welch's method splits the record into Hann-windowed segments of length L and averages their periodograms. Long segments resolve the close pair but average few periodograms, so the noise floor is jagged; short segments give a smooth floor that merges the pair into one peak. The grey line is the raw periodogram of the whole record."
      controls={
        <>
          <ParamSlider
            label="segment length L = 2^k, k"
            param={logL}
            format={(v) => `${v}  (L = ${2 ** v})`}
            withArrows
          />
          <ParamChoice
            label="overlap"
            value={overlap}
            onChange={setOverlap}
            options={[
              { value: '0', label: 'none' },
              { value: '50', label: '50%' },
            ]}
          />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="segments averaged K" value={r.segments} />
          <Readout label="resolution ≈ 1.44·2π/L (×π)" value={formatNumber(resolution)} />
          <Readout label="pair separation (×π)" value="0.02" />
          <Readout label="degrees of freedom ≈ 2K" value={dof} />
        </>
      }
    >
      <XYChart series={series} xLabel="ω / π" yLabel="power (dB)" xRange={[0, 1]} yRange={[-30, 40]} height={320} />
    </Interactive>
  )
}
