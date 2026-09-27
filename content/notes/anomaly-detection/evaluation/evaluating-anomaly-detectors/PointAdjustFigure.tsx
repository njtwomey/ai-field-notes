import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'
import { binomialPmf } from '@/lib/math/tests'

/** Series length and number of labelled anomalous segments. */
const N = 10000
const SEGMENTS = 5
const RATES = linspace(0.0005, 0.2, 400)

const f1 = (precision: number, recall: number) =>
  precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0

/**
 * Expected precision, recall and F1 of a detector that flags each point independently with probability p, scored
 * point-wise, with point adjustment (PA), and with PA%K. Under PA%K a segment counts as fully detected when at least
 * K% of its L points are flagged; otherwise only its flagged points count.
 */
function expectedScores(L: number, p: number, K: number) {
  const positives = SEGMENTS * L
  const fp = p * (N - positives)
  const need = Math.max(1, Math.ceil((K / 100) * L))
  // E[TP per segment] = Σ_d P(D = d)·(L if d ≥ need else d), with D ~ Binom(L, p) flagged points.
  let tpK = 0
  for (let d = 0; d <= L; d++) tpK += binomialPmf(d, L, p) * (d >= need ? L : d)
  const detected = 1 - Math.pow(1 - p, L)
  const tpPa = L * detected
  const point = f1(positives / N, p)
  const pa = f1((SEGMENTS * tpPa) / (SEGMENTS * tpPa + fp), tpPa / L)
  const paK = f1((SEGMENTS * tpK) / (SEGMENTS * tpK + fp), tpK / L)
  return { point, pa, paK }
}

export function PointAdjustFigure() {
  const L = useParam(100, { min: 1, max: 400, step: 1 })
  const p = useParam(0.05, { min: 0.0005, max: 0.2, step: 0.0005 })
  const K = useParam(20, { min: 0, max: 100, step: 5 })

  const curves = useMemo(() => RATES.map((r) => expectedScores(L.value, r, K.value)), [L.value, K.value])
  const series = useMemo(
    (): XYSeries[] => [
      { name: 'point-wise F1', type: 'line', x: RATES, y: curves.map((c) => c.point), slot: 0 },
      { name: 'point-adjusted F1', type: 'line', x: RATES, y: curves.map((c) => c.pa), slot: 1 },
      { name: `PA%K F1 (K = ${K.value}%)`, type: 'line', x: RATES, y: curves.map((c) => c.paK), slot: 2 },
    ],
    [curves, K.value],
  )
  const at = expectedScores(L.value, p.value, K.value)

  return (
    <Interactive
      title="A random detector under point adjustment"
      caption={`A series of ${N.toLocaleString()} points holds ${SEGMENTS} anomalous segments of L points each. The "detector" ignores the data and flags each point independently with probability p. The curves are its F1, computed from expected counts, as p varies. Point-wise, a random detector scores little. Point adjustment credits a whole segment once any point in it is flagged, so long segments hand the random detector a high F1. PA%K credits the segment only when at least K% of it is flagged. Drag the vertical line to set p.`}
      controls={
        <>
          <ParamSlider label="segment length L" param={L} format={(v) => String(v)} />
          <ParamSlider label="flag rate p" param={p} format={(v) => v.toFixed(4)} />
          <ParamSlider label="PA%K threshold K (%)" param={K} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="anomalous share" value={formatNumber((SEGMENTS * L.value) / N)} />
          <Readout label="point-wise F1" value={formatNumber(at.point)} />
          <Readout label="point-adjusted F1" value={formatNumber(at.pa)} />
          <Readout label="PA%K F1" value={formatNumber(at.paK)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="flag rate p"
        yLabel="expected F1"
        xRange={[0, 0.2]}
        yRange={[0, 1]}
        handles={[{ kind: 'x', at: p.value, label: 'p', onDrag: p.set }]}
        height={320}
      />
    </Interactive>
  )
}
