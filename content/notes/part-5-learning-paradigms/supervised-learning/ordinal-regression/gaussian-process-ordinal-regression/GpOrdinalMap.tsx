import { useMemo, useState } from 'react'
import { Interactive, ParamButton, ParamSlider, Readout, formatNumber } from 'aifn-render'
import { MAP_CAPTION } from '../_shared/mapText'
import { OrdinalDataControls } from '../_shared/OrdinalDataControls'
import { OrdinalMap } from '../_shared/OrdinalMap'
import { dataset, ordinalMetrics, TEST_PER_CLASS, type Point } from '../_shared/ordinal'
import { useOrdinalData } from '../_shared/useOrdinalData'
import { fitGp2d, GP_CAP, LENGTHSCALES } from '../_shared/gp2d'

/**
 * Gaussian-process ordinal regression on the shared 2-D data and controls, with its own lengthscale and noise, and a
 * button that picks the lengthscale with the highest Laplace evidence.
 */
export function GpOrdinalMap() {
  const { spec, setSpec, resolution, setResolution, fill, setFill } = useOrdinalData()
  const [query, setQuery] = useState<Point>([0, 1.5])
  const [lengthscale, setLengthscale] = useState(0.8)
  const [sigma, setSigma] = useState(0.3)
  const d = dataset(spec)

  const fit = useMemo(() => fitGp2d(d, spec, lengthscale, sigma), [d, spec, lengthscale, sigma])
  const metrics = useMemo(() => ordinalMetrics(d.test.y, d.test.x.map(fit.fitted.predict), d.k), [d, fit])

  const fitByEvidence = () => {
    const best = LENGTHSCALES.map((l) => ({ l, e: fitGp2d(d, spec, l, sigma).logEvidence })).reduce((a, b) =>
      b.e > a.e ? b : a,
    )
    setLengthscale(best.l)
  }

  return (
    <Interactive
      title="Gaussian-process ordinal regression on shared data"
      caption={`A GP with a squared-exponential kernel, fitted by the Laplace approximation to at most ${GP_CAP} training points (the same number from each class), with thresholds fixed one unit apart. ${MAP_CAPTION} Metrics are computed on ${TEST_PER_CLASS} held-out points per class. The button tries ${LENGTHSCALES.length} lengthscales and keeps the one with the highest approximate evidence.`}
      controls={
        <>
          <OrdinalDataControls
            spec={spec}
            setSpec={setSpec}
            resolution={resolution}
            setResolution={setResolution}
            fill={fill}
            setFill={setFill}
          />
          <ParamSlider
            label="lengthscale ℓ"
            value={lengthscale}
            onChange={setLengthscale}
            min={0.2}
            max={2.5}
            step={0.05}
            debounceMs={150}
          />
          <ParamSlider
            label="noise σ of the likelihood"
            value={sigma}
            onChange={setSigma}
            min={0.05}
            max={1}
            step={0.05}
            debounceMs={150}
          />
          <ParamButton onClick={fitByEvidence}>fit ℓ by evidence</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="accuracy" value={formatNumber(metrics.accuracy)} />
          <Readout label="MAE" value={formatNumber(metrics.mae)} />
          <Readout label="macro-MAE" value={formatNumber(metrics.macroMae)} />
          <Readout label="QWK" value={formatNumber(metrics.qwk)} />
          <Readout label="log evidence (Laplace)" value={formatNumber(fit.logEvidence)} />
          <Readout label="points in the GP" value={fit.n} />
        </>
      }
    >
      <OrdinalMap fitted={fit.fitted} data={d} resolution={resolution} fill={fill} query={query} setQuery={setQuery} />
    </Interactive>
  )
}
