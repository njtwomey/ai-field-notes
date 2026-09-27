import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, useParam, type XYSeries } from '@/components/viz'
import { magnitudeSpectrum } from '@/lib/dsp'
import { countZeroCrossings, emd, whiteNoise, type StopRule } from '../_shared/emd'

const N = 1024
const IMFS = 6
const BINS = 36
// Log₂ frequency from 2⁻⁹ to 2⁻¹ cycles per sample.
const LOG_LO = -9
const LOG_HI = -1

type RuleName = 'fixed' | 'pyemd'
const RULES: Record<RuleName, StopRule> = { fixed: { kind: 'fixed', sifts: 10 }, pyemd: { kind: 'pyemd' } }

/**
 * EMD of white noise, averaged over realisations: the IMFs' spectra are copies of one band-pass shape, each shifted
 * down by an octave, and their zero-crossing counts halve from one IMF to the next.
 */
export function NoiseFilterBank() {
  const realisations = useParam(20, { min: 5, max: 60, step: 5 })
  const [ruleName, setRuleName] = useState<RuleName>('fixed')

  const r = useMemo(() => {
    const psd = Array.from({ length: IMFS }, () => new Array<number>(BINS).fill(0))
    const counts = Array.from({ length: IMFS }, () => new Array<number>(BINS).fill(0))
    const zeros = new Array<number>(IMFS).fill(0)
    const seen = new Array<number>(IMFS).fill(0)
    for (const w of whiteNoise(realisations.value, N, 11)) {
      const d = emd(w, { rule: RULES[ruleName], maxImfs: IMFS })
      d.imfs.forEach((c, j) => {
        zeros[j] += countZeroCrossings(c)
        seen[j]++
        const mag = magnitudeSpectrum(c)
        for (let k = 1; k < mag.length; k++) {
          const b = Math.floor(((Math.log2(k / N) - LOG_LO) / (LOG_HI - LOG_LO)) * BINS)
          if (b < 0 || b >= BINS) continue
          psd[j][b] += (mag[k] * mag[k]) / N
          counts[j][b]++
        }
      })
    }
    // Only bins that hold at least one DFT frequency; the lowest octaves have fewer DFT bins than plot bins.
    const used = counts[0].flatMap((c, b) => (c > 0 ? [b] : []))
    const x = used.map((b) => LOG_LO + ((b + 0.5) * (LOG_HI - LOG_LO)) / BINS)
    const series: XYSeries[] = psd.map((row, j) => ({
      name: `IMF ${j + 1}`,
      type: 'line',
      x,
      y: used.map((b) => Math.max(row[b] / Math.max(counts[j][b], 1), 1e-4)),
      slot: j,
    }))
    const meanZeros = zeros.map((z, j) => z / Math.max(seen[j], 1))
    return { series, meanZeros }
  }, [realisations.value, ruleName])

  const ratios = r.meanZeros.slice(0, -1).map((z, j) => z / r.meanZeros[j + 1])

  return (
    <Interactive
      title="EMD of white noise is a dyadic filter bank"
      caption="Average power spectra of the first six IMFs of white Gaussian noise (1,024 samples), on a log-frequency axis. Each IMF after the first is a band-pass filter of the same shape as the one before, one octave lower. The readout gives the mean number of zero crossings of each IMF and the ratio between neighbours, which stays close to 2. The first IMF is a high-pass filter, since nothing lies above it."
      controls={
        <>
          <ParamSlider label="noise realisations" param={realisations} format={(v) => `${v}`} />
          <ParamChoice
            label="stopping rule"
            value={ruleName}
            onChange={setRuleName}
            options={[
              { value: 'fixed', label: '10 sifts' },
              { value: 'pyemd', label: 'PyEMD default' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="mean zero crossings" value={r.meanZeros.map((z) => z.toFixed(0)).join(', ')} />
          <Readout label="ratios" value={ratios.map((q) => q.toFixed(2)).join(', ')} />
        </>
      }
    >
      <XYChart
        series={r.series}
        xRange={[LOG_LO, LOG_HI]}
        yLog
        xLabel="log₂ frequency (cycles/sample)"
        yLabel="power"
      />
    </Interactive>
  )
}
