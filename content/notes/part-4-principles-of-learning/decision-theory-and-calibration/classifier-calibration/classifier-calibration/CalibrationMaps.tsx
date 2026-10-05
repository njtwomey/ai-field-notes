import { useMemo } from 'react'
import { choice, Curve, Figure, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { expit, fitBeta, fitIsotonic, fitPlatt, logLoss, logit } from '../../_shared/calibration'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

type Distortion = 'calibrated' | 'overconfident' | 'underconfident' | 'shifted' | 'wavy'
const GRID = toFlat(linspace(0.005, 0.995, 100))
const TEST = 4000
const BINS = 10

/** The raw score a model reports for a case whose true probability is q. Every distortion is increasing in q. */
function distort(kind: Distortion, q: number): number {
  switch (kind) {
    case 'calibrated':
      return q
    case 'overconfident':
      return expit(2.5 * logit(q))
    case 'underconfident':
      return expit(0.45 * logit(q))
    case 'shifted':
      return expit(logit(q) + 1.2)
    case 'wavy':
      return q + 0.04 * Math.sin(6 * Math.PI * q)
  }
}

/** The true calibration map: the q whose distorted score is s, found by bisection (distortions are increasing). */
function trueMap(kind: Distortion, s: number): number {
  let lo = 1e-9
  let hi = 1 - 1e-9
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2
    if (distort(kind, mid) < s) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

function sample(kind: Distortion, n: number, seed: number) {
  const g = stream(seed)
  const scores: number[] = []
  const labels: boolean[] = []
  for (let i = 0; i < n; i++) {
    const q = expit(1.5 * normal(g))
    labels.push(uniform(g) < q)
    scores.push(Math.min(1 - 1e-6, Math.max(1e-6, distort(kind, q))))
  }
  return { scores, labels }
}

/**
 * Logistic (Platt), isotonic and beta calibration maps fitted to the same calibration set, drawn over the raw score,
 * against the true calibration map of a simulated miscalibrated model. The test set is fixed; the calibration set size
 * and the model's distortion are chosen by the reader.
 */
export function CalibrationMaps() {
  const state = useFigureState({
    kind: choice<Distortion>(
      [
        { value: 'calibrated', label: 'none' },
        { value: 'overconfident', label: 'over' },
        { value: 'underconfident', label: 'under' },
        { value: 'shifted', label: 'shifted' },
        { value: 'wavy', label: 'wavy' },
      ],
      'overconfident',
      { label: 'model distortion' },
    ),
    n: int(300, { min: 30, max: 3000, step: 10, label: 'calibration set size', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const cal = sample(state.kind, state.n, 100 + state.seed)
    const test = sample(state.kind, TEST, 7)
    const platt = fitPlatt(cal.scores, cal.labels)
    const iso = fitIsotonic(cal.scores, cal.labels)
    const beta = fitBeta(cal.scores, cal.labels)
    const loss = (f: (s: number) => number) => logLoss(test.scores.map(f), test.labels)
    // Reliability of the raw scores on the calibration set, equal-width bins, plotted at the mean score in each bin.
    const sumS = new Array<number>(BINS).fill(0)
    const sumY = new Array<number>(BINS).fill(0)
    const count = new Array<number>(BINS).fill(0)
    cal.scores.forEach((s, i) => {
      const b = Math.min(BINS - 1, Math.floor(s * BINS))
      sumS[b] += s
      sumY[b] += cal.labels[i] ? 1 : 0
      count[b]++
    })
    const bins = count.flatMap((c, b) => (c ? [{ x: sumS[b] / c, y: sumY[b] / c }] : []))
    return {
      truth: GRID.map((s) => trueMap(state.kind, s)),
      platt: GRID.map(platt),
      iso: GRID.map(iso),
      beta: GRID.map(beta),
      bins,
      losses: {
        raw: loss((s) => s),
        platt: loss(platt),
        iso: loss(iso),
        beta: loss(beta),
        truth: loss((s) => trueMap(state.kind, s)),
      },
    }
  }, [state.kind, state.n, state.seed])

  const xAxis = useAxis({ label: 'raw score s', range: [0, 1] })
  const yAxis = useAxis({ label: 'calibrated probability', range: [0, 1], equal: xAxis })
  return (
    <Figure
      title="Three calibration maps on the same data"
      state={state}
      caption="A simulated model reports a distorted version of each case's true probability. The thick line is the true calibration map from raw score to probability; dots are the observed frequencies of the calibration set in ten bins. Logistic (Platt) calibration fits a sigmoid in the score, isotonic calibration a step function, beta calibration the family σ(a ln s − b ln(1 − s) + c). Log losses are on a separate test set of 4000 cases. Try a calibrated model: beta and isotonic stay near the diagonal, logistic cannot. Shrink the calibration set to see isotonic overfit."

      readouts={
        <>
          <Readout label="test log loss: raw" value={formatNumber(r.losses.raw)} />
          <Readout label="logistic" value={formatNumber(r.losses.platt)} />
          <Readout label="isotonic" value={formatNumber(r.losses.iso)} />
          <Readout label="beta" value={formatNumber(r.losses.beta)} />
          <Readout label="true map" value={formatNumber(r.losses.truth)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve name="identity" x={[0, 1]} y={[0, 1]} dashed muted />
        <Points name="binned frequency" x={r.bins.map((b) => b.x)} y={r.bins.map((b) => b.y)} muted />
        <Curve name="true map" x={GRID} y={r.truth} emphasis />
        <Curve name="logistic (Platt)" x={GRID} y={r.platt} slot={0} />
        <Curve name="isotonic" x={GRID} y={r.iso} slot={1} />
        <Curve name="beta" x={GRID} y={r.beta} slot={2} />
      </Plot>
    </Figure>
  )
}
