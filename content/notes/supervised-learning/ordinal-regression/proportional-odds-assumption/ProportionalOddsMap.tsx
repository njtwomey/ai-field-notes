import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, Readout, formatNumber } from 'aifn-render'
import { MAP_CAPTION, negativeShare } from '../_shared/mapText'
import { OrdinalDataControls } from '../_shared/OrdinalDataControls'
import { OrdinalMap } from '../_shared/OrdinalMap'
import { fitModel, ordinalMetrics, TEST_PER_CLASS, type Point } from '../_shared/ordinal'
import { useFit } from '../_shared/useFit'
import { useGrid, useOrdinalData } from '../_shared/useOrdinalData'
import { brantTest } from './brantMulti'

type Fit = 'proportional' | 'separate'
const FITS = [
  { value: 'proportional' as const, label: 'proportional odds (one slope vector)' },
  { value: 'separate' as const, label: 'separate slopes per split' },
]
const MODEL = { proportional: 'cumulative-logit', separate: 'binary-decomposition' } as const

/**
 * The proportional-odds fit against separate binary logits per cumulative split on the shared 2-D data, with Brant's
 * test of equal slopes on the current training set.
 */
export function ProportionalOddsMap() {
  const { spec, setSpec, resolution, setResolution, fill, setFill } = useOrdinalData({ shape: 'linear' })
  const [query, setQuery] = useState<Point>([0, 1.5])
  const [which, setWhich] = useState<Fit>('proportional')
  const { fitted, data: d, fitting } = useFit(spec, MODEL[which])
  const grid = useGrid(d.range, resolution)

  const brant = useMemo(() => brantTest(d.train.x, d.train.y, d.k), [d])
  // Both fits' held-out metrics; the one not on screen is fitted (and cached) here too.
  const metrics = useMemo(() => {
    const score = (m: (typeof MODEL)[Fit]) => {
      const f = fitModel({ shape: d.shape, k: d.k, noise: d.noise, seed: d.seed, m: d.m }, m)
      return ordinalMetrics(d.test.y, d.test.x.map(f.predict), d.k)
    }
    return { proportional: score(MODEL.proportional), separate: score(MODEL.separate) }
  }, [d])
  const crossing = useMemo(
    () =>
      negativeShare(fitModel({ shape: d.shape, k: d.k, noise: d.noise, seed: d.seed, m: d.m }, MODEL.separate), grid),
    [d, grid],
  )

  return (
    <Interactive
      title="Proportional odds on shared data"
      caption={`Proportional odds fits one slope vector for every cumulative split, so its class boundaries are parallel lines. The alternative fits a separate logistic regression to each split y > k, the fits Brant's test compares; their boundaries can tilt and cross, and where they cross a class probability is negative. On the linear data the test does not reject; on the arc and the spiral it rejects decisively. ${MAP_CAPTION} Metrics are computed on ${TEST_PER_CLASS} held-out points per class.`}
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
          <div className="sm:col-span-2">
            <ParamChoice label="fit" value={which} onChange={setWhich} options={FITS} />
          </div>
        </>
      }
      readout={
        <>
          {fitting && <Readout label="fitting" value="…" />}
          <Readout label={`Brant χ² (${brant.df} df)`} value={formatNumber(brant.chi2)} />
          <Readout label="p-value" value={brant.p < 1e-4 ? '< 0.0001' : formatNumber(brant.p)} />
          <Readout
            label="MAE, proportional / separate"
            value={`${formatNumber(metrics.proportional.mae)} / ${formatNumber(metrics.separate.mae)}`}
          />
          <Readout
            label="QWK, proportional / separate"
            value={`${formatNumber(metrics.proportional.qwk)} / ${formatNumber(metrics.separate.qwk)}`}
          />
          <Readout label="plane with crossing splits (separate)" value={formatNumber(crossing)} />
        </>
      }
    >
      <OrdinalMap fitted={fitted} data={d} resolution={resolution} fill={fill} query={query} setQuery={setQuery} />
    </Interactive>
  )
}
