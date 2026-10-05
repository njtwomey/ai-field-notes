import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  slider,
  type SeriesSpec,
  useAxis,
  useFigureState,
  when,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normalPdf } from 'aifn-compute/numerics/special'

type View = 'fit' | 'pair'

const XS = toFlat(linspace(-14, 14, 1121))
const DX = XS[1] - XS[0]
/** Densities below this count as zero inside logarithms, so the numerical integrals stay finite. */
const FLOOR = 1e-300

const gauss = (m: number, s: number) => XS.map((x) => normalPdf((x - m) / s) / s)

/** ∫ a log(a / b) dx on the grid. */
function kl(a: number[], b: number[]): number {
  let total = 0
  for (let i = 0; i < a.length; i++) {
    if (a[i] > 0) total += a[i] * (Math.log(a[i]) - Math.log(Math.max(b[i], FLOOR)))
  }
  return total * DX
}

/** KL(N(m1, s1²) ‖ N(m2, s2²)) in closed form. */
const klGauss = (m1: number, s1: number, m2: number, s2: number) =>
  Math.log(s2 / s1) + (s1 * s1 + (m1 - m2) ** 2) / (2 * s2 * s2) - 0.5

/**
 * Reverse-KL fit of a Gaussian to p by grid search, then a finer search around the best point. The target is
 * symmetric, so only means m ≥ 0 are searched; the mirror image is an equally good fit. KL(q ‖ p) is computed as
 * −H(q) − E_q[log p], with the Gaussian entropy in closed form.
 */
function reverseFit(p: number[], separation: number): { m: number; s: number } {
  const logP = p.map((v) => Math.log(Math.max(v, FLOOR)))
  const reverseKl = (m: number, s: number) => {
    let cross = 0
    for (let i = 0; i < XS.length; i++) cross += (normalPdf((XS[i] - m) / s) / s) * logP[i]
    return -0.5 * Math.log(2 * Math.PI * Math.E * s * s) - cross * DX
  }
  let best = { m: 0, s: 1, value: Infinity }
  const search = (ms: number[], ss: number[]) => {
    for (const m of ms)
      for (const s of ss) {
        const value = reverseKl(m, s)
        if (value < best.value) best = { m, s, value }
      }
  }
  search(toFlat(linspace(0, separation + 1, 25)), toFlat(linspace(0.3, 2 + separation, 25)))
  const { m, s } = best
  search(toFlat(linspace(Math.max(0, m - 0.2), m + 0.2, 21)), toFlat(linspace(Math.max(0.2, s - 0.2), s + 0.2, 21)))
  return best
}

/**
 * Two views of KL's asymmetry. "fit": the best Gaussian for a two-component mixture under forward and reverse KL.
 * "pair": two Gaussians with the divergence computed in both directions.
 */
export function KlAsymmetry() {
  const state = useFigureState({
    view: choice<View>(
      [
        { value: 'fit', label: 'fit a Gaussian' },
        { value: 'pair', label: 'two Gaussians' },
      ],
      'fit',
      { label: 'view' },
    ),
    separation: slider(0, 4, 2.5, { step: 0.1, label: 'mode separation d (modes at ±d)', when: when('view', 'fit') }),
    mu2: slider(-3, 3, 1, { step: 0.1, label: 'mean of q', when: when('view', 'pair') }),
    sigma2: slider(0.2, 3, 0.6, { step: 0.05, label: 'standard deviation of q', when: when('view', 'pair') }),
  })
  const { separation, mu2, sigma2 } = state

  const fit = useMemo(() => {
    const p = XS.map((x) => 0.5 * normalPdf(x - separation) + 0.5 * normalPdf(x + separation))
    // Forward KL over Gaussians is minimised by matching the mean and variance of p.
    const forward = { m: 0, s: Math.sqrt(1 + separation * separation) }
    const reverse = reverseFit(p, separation)
    const qf = gauss(forward.m, forward.s)
    const qr = gauss(reverse.m, reverse.s)
    const series: SeriesSpec[] = [
      { name: 'target p (mixture)', type: 'line', x: XS, y: p, area: true, slot: 0 },
      { name: 'q minimising KL(p ‖ q)', type: 'line', x: XS, y: qf, slot: 1 },
      { name: 'q minimising KL(q ‖ p)', type: 'line', x: XS, y: qr, dashed: true, slot: 2 },
    ]
    return {
      series,
      forward,
      reverse,
      table: { ff: kl(p, qf), fr: kl(qf, p), rf: kl(p, qr), rr: kl(qr, p) },
    }
  }, [separation])

  const pair = useMemo(() => {
    const series: SeriesSpec[] = [
      { name: 'p = N(0, 1)', type: 'line', x: XS, y: gauss(0, 1), area: true, slot: 0 },
      { name: 'q', type: 'line', x: XS, y: gauss(mu2, sigma2), slot: 1 },
    ]
    return { series, pq: klGauss(0, 1, mu2, sigma2), qp: klGauss(mu2, sigma2, 0, 1) }
  }, [mu2, sigma2])

  const xAxis = useAxis({ label: 'x', range: [-7, 7] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Forward and reverse KL"
      state={state}
      caption={
        state.view === 'fit'
          ? 'The target p is an equal mixture of two unit Gaussians. The best single Gaussian under forward KL(p ‖ q) matches the mean and variance of p and covers both modes. Under reverse KL(q ‖ p) the best Gaussian sits on one mode, because q is penalised for mass where p is small but not for missing a mode. Separate the modes to see the two fits diverge.'
          : 'p is a standard Gaussian and q is another Gaussian. The two directions of KL differ. A q narrower than p makes KL(p ‖ q) grow quickly, because p puts mass where q is tiny.'
      }
      readouts={
        state.view === 'fit' ? (
          <>
            <Readout
              label="forward fit"
              value={`N(0, ${formatNumber(fit.forward.s)}²): KL(p‖q) ${formatNumber(fit.table.ff)}, KL(q‖p) ${formatNumber(fit.table.fr)}`}
            />
            <Readout
              label="reverse fit"
              value={`N(${formatNumber(fit.reverse.m)}, ${formatNumber(fit.reverse.s)}²): KL(p‖q) ${formatNumber(fit.table.rf)}, KL(q‖p) ${formatNumber(fit.table.rr)}`}
            />
          </>
        ) : (
          <>
            <Readout label="KL(p ‖ q)" value={formatNumber(pair.pq)} />
            <Readout label="KL(q ‖ p)" value={formatNumber(pair.qp)} />
          </>
        )
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(state.view === 'fit' ? fit.series : pair.series)}
      </Plot>
    </Figure>
  )
}
