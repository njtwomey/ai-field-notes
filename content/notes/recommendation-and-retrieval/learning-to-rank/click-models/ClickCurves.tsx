import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'

const K = 10
const RANKS = Array.from({ length: K }, (_, k) => k + 1)

/**
 * Examination and click probabilities by rank under the position-based model, π(k) = k^{-η}, and the cascade model, in
 * which the user scans down and stops at the first click, so P(examine k) = Π_{j<k}(1 − r_j).
 */
export function ClickCurves() {
  const eta = useParam(1, { min: 0, max: 2, step: 0.05 })
  const top = useParam(0.4, { min: 0.05, max: 0.9, step: 0.05 })
  const decay = useParam(0.8, { min: 0.3, max: 1, step: 0.05 })

  const { series, cascadeExam, pbmExam, clicksPbm, clicksCascade } = useMemo(() => {
    const r = RANKS.map((k) => top.value * decay.value ** (k - 1))
    const pbmExam = RANKS.map((k) => k ** -eta.value)
    const cascadeExam: number[] = []
    let survive = 1
    for (const rk of r) {
      cascadeExam.push(survive)
      survive *= 1 - rk
    }
    const series: XYSeries[] = [
      { name: 'examination, position-based π(k)', type: 'line', x: RANKS, y: pbmExam, slot: 0 },
      { name: 'examination, cascade', type: 'line', x: RANKS, y: cascadeExam, slot: 1 },
      { name: 'relevance r_k', type: 'line', x: RANKS, y: r, dashed: true, slot: 2 },
    ]
    const clicksPbm = r.reduce((s, rk, k) => s + rk * pbmExam[k], 0)
    const clicksCascade = r.reduce((s, rk, k) => s + rk * cascadeExam[k], 0)
    return { series, cascadeExam, pbmExam, clicksPbm, clicksCascade }
  }, [eta.value, top.value, decay.value])

  return (
    <Interactive
      title="Two models of where users look"
      caption="A ranked list of ten results whose relevance decays with rank (dashed). The position-based model assumes examination depends only on rank, π(k) = k^(−η). The cascade model assumes the user reads from the top and stops at the first click, so reaching rank k requires skipping everything above it. Make the top results more relevant and the cascade's examination curve drops faster, because users stop earlier; the position-based curve does not react to relevance at all."
      controls={
        <>
          <ParamSlider label="position-bias exponent η" param={eta} />
          <ParamSlider label="relevance at rank 1" param={top} />
          <ParamSlider label="relevance decay per rank" param={decay} />
        </>
      }
      readout={
        <>
          <Readout label="P(examine rank 5), position-based" value={formatNumber(pbmExam[4])} />
          <Readout label="P(examine rank 5), cascade" value={formatNumber(cascadeExam[4])} />
          <Readout label="expected clicks, position-based" value={formatNumber(clicksPbm)} />
          <Readout label="expected clicks, cascade" value={formatNumber(clicksCascade)} />
        </>
      }
    >
      <XYChart series={series} xLabel="rank k" yLabel="probability" xRange={[1, K]} yRange={[0, 1]} />
    </Interactive>
  )
}
