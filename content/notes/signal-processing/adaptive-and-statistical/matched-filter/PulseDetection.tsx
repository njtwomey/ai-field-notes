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
import { rng } from '@/lib/math'

const N = 600
const L = 48

type Template = 'chirp' | 'rectangle' | 'sinc'

function template(kind: Template): number[] {
  const t = Array.from({ length: L }, (_, n) => n / (L - 1))
  const raw =
    kind === 'chirp'
      ? t.map((u) => Math.sin(2 * Math.PI * (2 * u + 6 * u * u)))
      : kind === 'rectangle'
        ? t.map(() => 1)
        : t.map((u) => {
            const z = 8 * (u - 0.5)
            return z === 0 ? 1 : Math.sin(Math.PI * z) / (Math.PI * z)
          })
  const energy = raw.reduce((s, v) => s + v * v, 0)
  return raw.map((v) => v / Math.sqrt(energy)) // unit energy
}

/**
 * A known pulse buried in white noise, and the matched filter's output: correlation with the template, which is
 * convolution with the time-reversed template. The output peak marks the pulse's arrival.
 */
export function PulseDetection() {
  const [kind, setKind] = useState<Template>('chirp')
  const snrDb = useParam(10, { min: -5, max: 25, step: 1 })
  const seed = useParam(3, { min: 1, max: 30, step: 1 })

  const r = useMemo(() => {
    const s = template(kind)
    const g = rng(seed.value)
    const arrival = 150 + Math.floor(g.uniform() * 300)
    // Unit-energy pulse scaled to energy E; noise variance 1, so the peak output SNR is E/σ² = E.
    const energy = 10 ** (snrDb.value / 10)
    const x = Array.from({ length: N }, () => g.normal())
    for (let n = 0; n < L; n++) x[arrival + n] += Math.sqrt(energy) * s[n]
    // y[n] = Σ_k s[k] x[n + k]: the matched filter h[n] = s[L−1−n] with its delay removed.
    const y = Array.from({ length: N - L }, (_, n) => s.reduce((acc, sk, k) => acc + sk * x[n + k], 0))
    const peak = y.reduce((best, v, n) => (v > y[best] ? n : best), 0)
    return { x, y, arrival, peak, energy }
  }, [kind, snrDb.value, seed.value])

  const t = Array.from({ length: N }, (_, n) => n)
  const input: XYSeries[] = [{ name: 'received x[n]', type: 'line', x: t, y: r.x, slot: 0 }]
  const output: XYSeries[] = [
    { name: 'matched-filter output', type: 'line', x: t.slice(0, r.y.length), y: r.y, slot: 1 },
    { name: 'true arrival', type: 'scatter', x: [r.arrival], y: [Math.sqrt(r.energy)], emphasis: true },
  ]

  return (
    <Interactive
      title="Finding a known pulse in noise"
      caption="A unit-energy pulse scaled to energy E arrives at an unknown time in unit-variance white noise, so the matched filter's peak SNR is E/σ². Top: the received signal, where the pulse is often invisible. Bottom: the correlation with the template, peaking at the arrival time with height about √E against noise of standard deviation 1. The chirp and the rectangle have the same energy and the same peak SNR; the chirp's narrower correlation peak locates the arrival more precisely."
      controls={
        <>
          <ParamChoice
            label="pulse"
            value={kind}
            onChange={setKind}
            options={[
              { value: 'chirp', label: 'chirp' },
              { value: 'rectangle', label: 'rectangle' },
              { value: 'sinc', label: 'sinc' },
            ]}
          />
          <ParamSlider label="E/σ² (dB)" param={snrDb} />
          <ParamSlider label="seed" param={seed} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="true arrival" value={r.arrival} />
          <Readout label="output peak at" value={r.peak} />
          <Readout label="expected peak height √E" value={formatNumber(Math.sqrt(r.energy))} />
          <Readout label="detected" value={Math.abs(r.peak - r.arrival) <= 2 ? 'yes' : 'no'} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={input} xLabel="n" yLabel="x[n]" height={200} />
        <XYChart series={output} xLabel="n" yLabel="output" height={220} />
      </div>
    </Interactive>
  )
}
