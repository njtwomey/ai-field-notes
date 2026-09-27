import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'

/** Schedules of Song et al. (2021): β(t) linear from 0.1 to 20 on [0, 1]; σ(t) = 0.01 · 5000ᵗ, from 0.01 to 50. */
const B_MIN = 0.1
const B_MAX = 20
const S_MIN = 0.01
const S_MAX = 50
const TS = linspace(0.002, 1, 250)
const intBeta = (t: number) => B_MIN * t + 0.5 * (B_MAX - B_MIN) * t * t

type Sde = 'vp' | 'subvp' | 've'
const KERNELS: Record<Sde, { label: string; m: (t: number) => number; s: (t: number) => number }> = {
  vp: { label: 'VP', m: (t) => Math.exp(-intBeta(t) / 2), s: (t) => Math.sqrt(1 - Math.exp(-intBeta(t))) },
  subvp: { label: 'sub-VP', m: (t) => Math.exp(-intBeta(t) / 2), s: (t) => 1 - Math.exp(-intBeta(t)) },
  ve: {
    label: 'VE',
    m: () => 1,
    s: (t) => Math.sqrt(Math.max(S_MIN ** 2 * (S_MAX / S_MIN) ** (2 * t) - S_MIN ** 2, 0)),
  },
}
const STEP_CHOICES = ['50', '200', '1000'] as const
type Steps = (typeof STEP_CHOICES)[number]

/**
 * The transition kernel x_t | x₀ ~ N(m(t) x₀, s(t)²) of the three SDE families, with DDPM's discrete √ᾱ_i and
 * √(1 − ᾱ_i) for the matching linear schedule β_i = β(t_i)/N overlaid on the VP curves.
 */
export function NoiseSchedules() {
  const [sde, setSde] = useState<Sde>('vp')
  const [steps, setSteps] = useState<Steps>('50')
  const n = Number(steps)

  const ddpm = useMemo(() => {
    const t: number[] = []
    const m: number[] = []
    const s: number[] = []
    let ab = 1
    let gap = 0
    const every = Math.max(1, Math.round(n / 50))
    for (let i = 1; i <= n; i++) {
      const beta = (B_MIN + ((B_MAX - B_MIN) * (i - 1)) / (n - 1)) / n
      ab *= 1 - beta
      gap = Math.max(gap, Math.abs(Math.sqrt(ab) - KERNELS.vp.m(i / n)))
      if (i % every === 0) {
        t.push(i / n)
        m.push(Math.sqrt(ab))
        s.push(Math.sqrt(1 - ab))
      }
    }
    return { t, m, s, gap }
  }, [n])

  const series = useMemo<XYSeries[]>(() => {
    const k = KERNELS[sde]
    const out: XYSeries[] = [
      { name: 'signal scale m(t)', type: 'line', x: TS, y: TS.map(k.m), slot: 0 },
      { name: 'noise scale s(t)', type: 'line', x: TS, y: TS.map(k.s), slot: 1 },
    ]
    if (sde === 'vp') {
      out.push({ name: `DDPM √ᾱᵢ, N = ${n}`, type: 'scatter', x: ddpm.t, y: ddpm.m, emphasis: true })
      out.push({ name: `DDPM √(1 − ᾱᵢ), N = ${n}`, type: 'scatter', x: ddpm.t, y: ddpm.s, muted: true })
    }
    return out
  }, [sde, n, ddpm])

  const k = KERNELS[sde]
  return (
    <Interactive
      title="How the three forward SDEs noise a data point"
      caption="Each forward SDE turns x₀ into x_t ~ N(m(t) x₀, s(t)²). The variance-preserving (VP) SDE shrinks the signal while the noise grows to 1; the sub-VP SDE uses the same shrinkage with less noise at every time; the variance-exploding (VE) SDE keeps the signal and grows the noise to 50. Schedules are those of Song et al. (2021); the y axis is logarithmic. In the VP view the dots are DDPM's √ᾱᵢ and √(1 − ᾱᵢ) for N steps of the matching discrete schedule: they approach the SDE curves as N grows."
      controls={
        <>
          <ParamChoice
            label="forward SDE"
            value={sde}
            onChange={setSde}
            options={(Object.keys(KERNELS) as Sde[]).map((s) => ({ value: s, label: KERNELS[s].label }))}
          />
          <ParamChoice
            label="DDPM steps N"
            value={steps}
            onChange={setSteps}
            options={STEP_CHOICES.map((s) => ({ value: s, label: s }))}
          />
        </>
      }
      readout={
        <>
          <Readout label="m(1)" value={formatNumber(k.m(1))} />
          <Readout label="s(1)" value={formatNumber(k.s(1))} />
          <Readout label="max |√ᾱᵢ − m(tᵢ)|" value={formatNumber(ddpm.gap)} />
        </>
      }
    >
      <XYChart height={300} xLabel="t" yLabel="scale" yLog series={series} xRange={[0, 1]} yRange={[0.001, 100]} />
    </Interactive>
  )
}
