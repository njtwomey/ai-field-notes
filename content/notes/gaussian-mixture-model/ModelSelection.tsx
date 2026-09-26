import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, ParamSwitch, Readout, XYChart, type XYSeries } from '@/components/viz'
import type { ModelSelectionTable } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'

/** k means and k diagonal variances in 2-D, plus k − 1 free weights. */
const parameters = (k: number) => 4 * k + (k - 1)

const argmin = (xs: number[]) => xs.indexOf(Math.min(...xs))

/**
 * −2 log L, AIC and BIC against k. The best-fit deviances are precomputed by python/mlc/figures/gmm.py for every n,
 * restart budget and k; the penalties are added here.
 */
export function ModelSelection() {
  const { data } = useFigure<ModelSelectionTable>('gaussian-mixture-model/model-selection')
  const [n, setN] = useState(400)
  const [restarts, setRestarts] = useState(3)
  const [zoom, setZoom] = useState(true)

  const result = useMemo(() => {
    if (!data) return undefined
    const deviance = data.deviance[restarts - 1][data.ns.indexOf(n)]
    const ks = data.ks
    const aic = ks.map((k, i) => deviance[i] + 2 * parameters(k))
    const bic = ks.map((k, i) => deviance[i] + parameters(k) * Math.log(n))
    return { ks, deviance, aic, bic }
  }, [data, n, restarts])

  const series = useMemo((): XYSeries[] => {
    if (!result) return []
    // A marker at each criterion's minimum, sharing the line's name so the legend toggles both.
    const minimum = (name: string, ys: number[], slot: number): XYSeries => ({
      name,
      type: 'scatter',
      x: [result.ks[argmin(ys)]],
      y: [Math.min(...ys)],
      slot,
    })
    return [
      { name: '−2 log L', type: 'line', x: result.ks, y: result.deviance, slot: 0, dashed: true },
      { name: 'AIC', type: 'line', x: result.ks, y: result.aic, slot: 1 },
      { name: 'BIC', type: 'line', x: result.ks, y: result.bic, slot: 2 },
      minimum('AIC', result.aic, 1),
      minimum('BIC', result.bic, 2),
    ]
  }, [result])

  // Zoomed: fit the y-axis to k ≥ 3, where the criteria differ. k = 1 and 2 run off the top.
  const yRange = useMemo((): [number, number] | undefined => {
    if (!result || !zoom) return undefined
    const tail = [...result.deviance, ...result.aic, ...result.bic].filter((_, i) => i % result.ks.length >= 2)
    const lo = Math.min(...tail)
    const hi = Math.max(...tail)
    const pad = (hi - lo) * 0.15
    // Round to multiples of 50 so the axis ends on tidy tick labels.
    return [Math.floor((lo - pad) / 50) * 50, Math.ceil((hi + pad) / 50) * 50]
  }, [result, zoom])

  if (!result) return null
  return (
    <Interactive
      title="Choosing k: likelihood, AIC and BIC"
      caption="Each k is fitted by EM from several k-means++ starts, keeping the best. The dashed line is −2 log L. It only falls, so on its own it always prefers more components; its bend at k = 3 is the elbow. AIC adds 2 per parameter and BIC adds log n per parameter, turning the elbow into a minimum. Lower is better. Change n: BIC's penalty grows with n, AIC's does not."
      controls={
        <>
          <ParamSlider label="points n" value={n} onChange={setN} min={30} max={400} step={10} />
          <ParamSlider label="restarts per k" value={restarts} onChange={setRestarts} min={1} max={5} step={1} />
          <ParamSwitch label="zoom on k ≥ 3" checked={zoom} onChange={setZoom} />
        </>
      }
      readout={
        <>
          <Readout label="AIC picks k" value={result.ks[argmin(result.aic)]} />
          <Readout label="BIC picks k" value={result.ks[argmin(result.bic)]} />
          <Readout label="true k" value={3} />
          <Readout label="BIC penalty per parameter" value={Math.log(n).toFixed(2)} />
        </>
      }
    >
      <XYChart height={340} xLabel="k" yLabel="criterion (lower is better)" series={series} yRange={yRange} />
    </Interactive>
  )
}
