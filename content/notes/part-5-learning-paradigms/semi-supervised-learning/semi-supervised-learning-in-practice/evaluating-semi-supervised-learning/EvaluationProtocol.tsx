import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Plots,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { CONFIGS, LABEL_COUNTS, REPEATS, runStudy, summarise, TEST_POINTS } from './protocol-study'

/**
 * The gain of a semi-supervised method over its supervised baseline depends on how well the baseline was tuned and on
 * how large the validation set was: Laplacian regularised least squares against kernel ridge regression on two moons.
 */
export function EvaluationProtocol() {
  const fig = useFigureState({
    budget: int(1, {
      min: 1,
      max: CONFIGS.length,
      label: 'settings tried for the baseline',
      suggestions: [1, 2, 3, CONFIGS.length],
    }),
    validation: choice(
      [
        { value: 'large', label: 'large validation set (500 labelled points)' },
        { value: 'small', label: 'realistic validation set (as many points as labels)' },
      ],
      'large',
      { label: 'hyperparameters picked on' },
    ),
    seed: int(1, { min: 1, max: 999, label: 'data seed' }),
  })
  const { budget, validation, seed } = fig

  const study = useComputed(() => runStudy(seed), [seed], { mode: 'release' })
  const rows = useMemo(
    () => summarise(study.value, budget, validation as 'large' | 'small'),
    [study.value, budget, validation],
  )

  const xa = useAxis({ label: 'labels per class', log: true, range: [0.8, 20] })
  const ya = useAxis({ label: 'test accuracy', range: [0.6, 1] })
  const yg = useAxis({ label: 'gain in test accuracy', range: [-0.1, 0.3] })
  const m = LABEL_COUNTS
  const apparent = rows.map((r) => r.semiSupervised - r.baseline)
  const fair = rows.map((r) => r.semiSupervised - r.tunedBaseline)
  const last = rows.length - 1

  return (
    <Figure
      title="A weak baseline inflates the gain"
      state={fig}
      caption={`Laplacian regularised least squares (the semi-supervised method) and kernel ridge regression on the labelled points alone (its supervised baseline) on 100 points of two moons, with 1 to 16 labels per class, averaged over ${REPEATS} draws and scored on ${TEST_POINTS} test points. Both methods have the same ${CONFIGS.length} hyperparameter settings (kernel lengthscale and ridge weight). The semi-supervised method is always tuned over all of them; the baseline is tuned over a random subset of the chosen size, averaged over every such subset. Left: test accuracy. Right: the gain of the semi-supervised method over this baseline (solid) and over the baseline tuned on all ${CONFIGS.length} settings (dashed). With one setting tried, the gain looks positive at every label count; tuned equally, the baseline catches up at 8 to 16 labels per class, and with data seed 1 it is slightly ahead there. Picking hyperparameters on a validation set as small as the labelled set picks worse settings for the semi-supervised method, which has more to tune, and shrinks its gain at one label per class.`}
      readouts={
        <>
          <Readout label="gain at 1 label per class" value={formatNumber(apparent[0])} />
          <Readout label="gain at 1, baseline tuned" value={formatNumber(fair[0])} />
          <Readout label={`gain at ${m[last]} labels per class`} value={formatNumber(apparent[last])} />
          <Readout label={`gain at ${m[last]}, baseline tuned`} value={formatNumber(fair[last])} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={xa} y={ya} title="test accuracy">
          <Curve
            name="semi-supervised (tuned on all settings)"
            x={m}
            y={rows.map((r) => r.semiSupervised)}
            slot={0}
            showPoints
            stale={study.stale}
          />
          <Curve
            name={`baseline (${budget} setting${budget > 1 ? 's' : ''} tried)`}
            x={m}
            y={rows.map((r) => r.baseline)}
            slot={1}
            showPoints
            stale={study.stale}
          />
          <Curve
            name="baseline tuned on all settings"
            x={m}
            y={rows.map((r) => r.tunedBaseline)}
            slot={1}
            dashed
            stale={study.stale}
          />
        </Plot>
        <Plot x={xa} y={yg} title="gain of the semi-supervised method">
          <Curve name="over this baseline" x={m} y={apparent} slot={2} showPoints stale={study.stale} />
          <Curve name="over the tuned baseline" x={m} y={fair} slot={2} dashed stale={study.stale} />
          <Curve name="no gain" x={[0.8, 20]} y={[0, 0]} muted />
        </Plot>
      </Plots>
    </Figure>
  )
}
