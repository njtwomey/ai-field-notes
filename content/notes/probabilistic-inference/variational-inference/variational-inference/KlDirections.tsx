import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'

const ZS = linspace(-10, 10, 401)
const DZ = ZS[1] - ZS[0]
// Standard normal quadrature grid for expectations under q = N(m, s²): z = m + s u, weight φ(u) du.
const US = linspace(-6, 6, 121)
const WU = US.map((u) => (Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI)) * (US[1] - US[0]))
const MS = linspace(-6, 6, 121)
const SS = linspace(Math.log(0.3), Math.log(5), 60).map(Math.exp)
const LOG_2PI = Math.log(2 * Math.PI)

const normal = (z: number, m: number, s: number) => Math.exp(-0.5 * ((z - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI))

/**
 * Fit a Gaussian q to the mixture p = (1 − w) N(−d, 1) + w N(d, 1) under each direction of the KL divergence. KL(p ‖ q)
 * is minimised by matching the mean and variance of p. KL(q ‖ p) has no closed form; it is minimised by a grid search
 * over the mean and log standard deviation of q, with E_q[log p] by quadrature.
 */
export function KlDirections() {
  const d = useParam(3, { min: 0, max: 4, step: 0.05 })
  const w = useParam(0.6, { min: 0.1, max: 0.9, step: 0.01 })

  const fit = useMemo(() => {
    const logp = (z: number) => {
      const a = Math.log(1 - w.value) - 0.5 * (z + d.value) ** 2
      const b = Math.log(w.value) - 0.5 * (z - d.value) ** 2
      const top = Math.max(a, b)
      return top + Math.log(Math.exp(a - top) + Math.exp(b - top)) - 0.5 * LOG_2PI
    }
    // KL(q ‖ p) = E_q[log q] − E_q[log p], with E_q[log q] = −½ log(2π e s²).
    const reverseKl = (m: number, s: number) => {
      let cross = 0
      US.forEach((u, i) => (cross += WU[i] * logp(m + s * u)))
      return -0.5 * (LOG_2PI + 1) - Math.log(s) - cross
    }
    let best = { m: 0, s: 1, kl: Infinity }
    for (const m of MS)
      for (const s of SS) {
        const kl = reverseKl(m, s)
        if (kl < best.kl) best = { m, s, kl }
      }
    // Refine the grid optimum by a few rounds of coordinate search on a shrinking step.
    let [step, lstep] = [0.05, 0.05]
    for (let round = 0; round < 30; round++) {
      for (const [dm, dl] of [
        [step, 0],
        [-step, 0],
        [0, lstep],
        [0, -lstep],
      ]) {
        const m = best.m + dm
        const s = best.s * Math.exp(dl)
        const kl = reverseKl(m, s)
        if (kl < best.kl) best = { m, s, kl }
      }
      step *= 0.8
      lstep *= 0.8
    }
    // Moment matching gives the forward fit.
    const mean = (2 * w.value - 1) * d.value
    const sd = Math.sqrt(1 + d.value ** 2 - mean ** 2)
    const pz = ZS.map((z) => Math.exp(logp(z)))
    const negEntropy = pz.reduce((a, v, i) => a + (v > 0 ? v * logp(ZS[i]) : 0), 0) * DZ
    // KL(p ‖ q) for Gaussian q = E_p[log p] + ½ log(2π s²) + E_p[(z − m)²]/(2 s²).
    const forwardKl = (m: number, s: number) =>
      negEntropy + 0.5 * Math.log(2 * Math.PI * s * s) + (1 + d.value ** 2 - 2 * m * mean + m * m) / (2 * s * s)
    return {
      pz,
      rev: best,
      fwd: { m: mean, s: sd },
      revOfFwd: reverseKl(mean, sd),
      fwdOfFwd: forwardKl(mean, sd),
      fwdOfRev: forwardKl(best.m, best.s),
    }
  }, [d.value, w.value])

  const series: XYSeries[] = useMemo(
    () => [
      { name: 'target p', type: 'line', x: ZS, y: fit.pz, slot: 0, area: true },
      {
        name: 'q minimising KL(p ‖ q)',
        type: 'line',
        x: ZS,
        y: ZS.map((z) => normal(z, fit.fwd.m, fit.fwd.s)),
        slot: 1,
      },
      {
        name: 'q minimising KL(q ‖ p)',
        type: 'line',
        x: ZS,
        y: ZS.map((z) => normal(z, fit.rev.m, fit.rev.s)),
        slot: 2,
        dashed: true,
      },
    ],
    [fit],
  )

  return (
    <Interactive
      title="Fitting a Gaussian to two modes, in each direction of the KL divergence"
      caption="The target is a mixture of two unit-variance Gaussians at ±d, with weight w on the right one. Minimising KL(p ‖ q) matches the mean and variance of p: q covers both modes and puts mass in the gap between them. Minimising KL(q ‖ p), the direction variational inference uses, locks onto one mode and ignores the other. Bring the modes together (small d) and the two fits agree; separate them and the reverse fit jumps to a single mode, the heavier one."
      controls={
        <>
          <ParamSlider label="half-separation d" param={d} />
          <ParamSlider label="weight of the right mode w" param={w} />
        </>
      }
      readout={
        <>
          <Readout label="reverse fit mean, sd" value={`${formatNumber(fit.rev.m)}, ${formatNumber(fit.rev.s)}`} />
          <Readout label="KL(q ‖ p) at reverse fit" value={formatNumber(fit.rev.kl)} />
          <Readout label="KL(p ‖ q) at reverse fit" value={formatNumber(fit.fwdOfRev)} />
          <Readout label="forward fit mean, sd" value={`${formatNumber(fit.fwd.m)}, ${formatNumber(fit.fwd.s)}`} />
          <Readout label="KL(p ‖ q) at forward fit" value={formatNumber(fit.fwdOfFwd)} />
          <Readout label="KL(q ‖ p) at forward fit" value={formatNumber(fit.revOfFwd)} />
        </>
      }
    >
      <XYChart series={series} xLabel="z" yLabel="density" xRange={[-8, 8]} height={300} />
    </Interactive>
  )
}
