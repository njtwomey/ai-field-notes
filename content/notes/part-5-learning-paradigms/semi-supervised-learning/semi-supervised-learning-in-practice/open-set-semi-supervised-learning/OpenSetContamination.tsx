import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  Rug,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { bayesError, mixtureDensity, ruleBoundaries, ruleError } from '../_shared/line-mixtures'
import { contaminationCurve, drawOpenSetSample, fitEm, KNOWN, noveltyKeep, RHOS, UNSEEN_SD } from './open-set-study'

const X_RANGE: [number, number] = [-5, 6]
const XS = Array.from({ length: 221 }, (_, i) => X_RANGE[0] + ((X_RANGE[1] - X_RANGE[0]) * i) / 220)
const gauss = (x: number, m: number, v: number) =>
  Math.exp((-0.5 * ((x - m) * (x - m))) / v) / Math.sqrt(2 * Math.PI * v)

/**
 * An unseen class in the unlabelled data drags a two-class model; a novelty filter fitted to the labels removes the
 * far ones and misses the near ones.
 */
export function OpenSetContamination() {
  const fig = useFigureState({
    rho: float(0.6, {
      min: 0,
      max: 0.9,
      step: 0.05,
      label: 'share ρ of unlabelled points from the unseen class',
      onChart: true,
    }),
    unseen: float(3.5, { min: -4.5, max: 5.5, step: 0.1, label: 'mean of the unseen class', onChart: true }),
    method: choice(
      [
        { value: 'plain', label: 'all unlabelled points' },
        { value: 'filtered', label: 'unlabelled points the novelty filter keeps' },
      ],
      'plain',
      { label: 'EM trains on' },
    ),
    alpha: float(0.01, {
      gt: 0,
      lt: 1,
      scale: 'log10',
      label: 'novelty filter level α',
      suggestions: [0.001, 0.01, 0.05],
    }),
    perClass: int(4, { min: 2, max: 50, label: 'labels per class', suggestions: [2, 4, 10] }),
    seed: int(1, { min: 1, max: 999, label: 'data seed' }),
  })
  const { rho, unseen, method, alpha, perClass, seed } = fig
  const bayes = useMemo(() => bayesError(KNOWN), [])

  const curve = useComputed(() => contaminationCurve(seed, perClass, alpha, unseen), [seed, perClass, alpha, unseen], {
    mode: 'release',
  })

  const shown = useMemo(() => {
    const sample = drawOpenSetSample(seed, 0, perClass, rho, unseen)
    const keep = noveltyKeep(sample, alpha)
    const fit = fitEm(sample, method === 'filtered' ? keep : null)
    const supervised = fitEm(sample, null, 0)
    return { sample, keep, fit, error: ruleError(KNOWN, fit), supervisedError: ruleError(KNOWN, supervised) }
  }, [seed, perClass, rho, unseen, alpha, method])
  const { sample, keep, fit } = shown

  const xa = useAxis({ label: 'x', range: X_RANGE })
  const ya = useAxis({ label: 'density', range: [0, 0.5] })
  const xb = useAxis({ label: 'share ρ from the unseen class', range: [0, 0.9] })
  const yb = useAxis({ label: 'test error on the known classes', range: [0, 0.5] })

  const unlabelledKept = sample.x.filter((_, i) => sample.labels[i] < 0 && keep[i])
  const unlabelledDropped = sample.x.filter((_, i) => sample.labels[i] < 0 && !keep[i])
  const bounds = ruleBoundaries(fit).filter((b) => b > X_RANGE[0] && b < X_RANGE[1])
  const unseenCount = sample.isUnseen.filter(Boolean).length
  const removed = sample.x.filter((_, i) => sample.isUnseen[i] && !keep[i]).length
  const c = curve.value

  return (
    <Figure
      title="An unseen class in the unlabelled data"
      state={fig}
      caption="Two known classes on the line (means ∓1.2, standard deviation 0.6) with a few labels each, and 300 unlabelled points of which a share ρ comes from an unseen class (standard deviation 0.5). Semi-supervised EM fits one Gaussian per known class. Left: the true densities (dashed: the known classes and, in grey, the unseen one), the fitted class Gaussians (solid) and the fitted boundary (ink), with the unlabelled points as ticks along the bottom: grey ticks are used, red ticks are removed by the novelty filter, which drops a point when a chi-squared test against the nearest labelled class mean rejects it at level α. Drag the vertical line to move the unseen class. Right: exact test error on the known classes, averaged over 8 samples, for EM on all unlabelled points, EM on the filtered points, and the labels alone (dashed); drag the vertical line to set ρ. An unseen class far from both known classes is absorbed by the nearer class's Gaussian, which stretches towards it and moves the boundary; the filter removes most of its points. An unseen class close to a known class, or between the two, passes the filter and still does harm."
      readouts={
        <>
          <Readout label="error of this fit" value={formatNumber(shown.error)} />
          <Readout label="labels alone" value={formatNumber(shown.supervisedError)} />
          <Readout label="unseen points removed by the filter" value={`${removed} of ${unseenCount}`} />
          <Readout label="Bayes error" value={formatNumber(bayes)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya}>
          {[0, 1].map((k) => (
            <Curve
              key={`t${k}`}
              name={`class ${k}, true`}
              x={XS}
              y={XS.map((x) => ((1 - rho) / 2) * mixtureDensity(KNOWN.classes[k], x))}
              slot={k}
              dashed
              thin
            />
          ))}
          <Curve
            name="unseen class, true"
            x={XS}
            y={XS.map((x) => rho * gauss(x, unseen, UNSEEN_SD ** 2))}
            muted
            dashed
          />
          {[0, 1].map((k) => (
            <Curve
              key={`f${k}`}
              name={`class ${k}, fitted`}
              x={XS}
              y={XS.map((x) => fit.weights[k] * gauss(x, fit.means[k], fit.covariances[k]))}
              slot={k}
            />
          ))}
          {bounds.map((b, i) => (
            <Curve key={`b${i}`} name="fitted boundary" x={[b, b]} y={[0, 0.5]} emphasis />
          ))}
          {unlabelledKept.length > 0 && <Rug name="unlabelled, used" values={unlabelledKept} muted length={6} />}
          {unlabelledDropped.length > 0 && (
            <Rug name="unlabelled, removed by the filter" values={unlabelledDropped} tone="destructive" length={9} />
          )}
          <Handle
            kind="x"
            at={unseen}
            label="unseen class"
            onDrag={(v) => fig.set('unseen', Math.round(Math.min(5.5, Math.max(-4.5, v)) * 10) / 10)}
          />
        </Plot>
        <Plot x={xb} y={yb}>
          <Curve
            name="EM on all unlabelled points"
            x={RHOS}
            y={c.map((r) => r.plain)}
            slot={2}
            showPoints
            stale={curve.stale}
          />
          <Curve
            name="EM on the filtered points"
            x={RHOS}
            y={c.map((r) => r.filtered)}
            slot={3}
            showPoints
            stale={curve.stale}
          />
          <Curve name="labels alone" x={RHOS} y={c.map((r) => r.supervised)} muted dashed stale={curve.stale} />
          <Curve name="Bayes error" x={[0, 0.9]} y={[bayes, bayes]} muted thin />
          <Handle
            kind="x"
            at={rho}
            label="ρ"
            onDrag={(v) => fig.set('rho', Math.round(Math.min(0.9, Math.max(0, v)) * 20) / 20)}
          />
        </Plot>
      </div>
    </Figure>
  )
}
