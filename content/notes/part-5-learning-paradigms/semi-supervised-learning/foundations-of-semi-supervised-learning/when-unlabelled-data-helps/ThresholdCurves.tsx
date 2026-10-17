import { useMemo } from 'react'
import {
  Area,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
  variants,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { unlabelledDensity } from 'aifn-methods/learning/semi-supervised'
import {
  clustersProblem,
  drawLine,
  fitBoth,
  thresholdLearningCurve,
  uniformProblem,
  type LineProblem,
} from 'aifn-methods/data/synthetic'

const SIZES = [1, 2, 4, 8, 16, 32, 64]
const CURVE_POINTS = 200

/**
 * Learning a threshold on the line with and without unlabelled data, on a problem whose classes are separated by a
 * density trough and on one whose input density is flat.
 */
export function ThresholdCurves() {
  const state = useFigureState({
    problem: variants(
      {
        clusters: {
          label: 'two Gaussian classes',
          params: {
            separation: float(4, { ge: 0, le: 8, label: 'separation s (sd)', suggestions: [1, 2, 3, 4, 6] }),
          },
        },
        uniform: {
          label: 'uniform inputs',
          params: { boundary: float(0.3, { ge: 0.05, le: 0.95, label: 'true boundary t*', onChart: true }) },
        },
      },
      { choiceLabel: 'problem' },
    ),
    unlabelled: int(200, { ge: 0, le: 2000, label: 'unlabelled points u', suggestions: [0, 20, 200, 1000] }),
    perClass: int(1, { ge: 1, le: 64, label: 'labels per class m (left)', suggestions: [1, 2, 4, 16, 64] }),
    repeats: int(100, { ge: 10, le: 400, label: 'repeats', suggestions: [50, 100, 200] }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const kind = state.problem.key
  const separation = kind === 'clusters' ? (state.problem.values as { separation: number }).separation : 4
  const boundary = kind === 'uniform' ? (state.problem.values as { boundary: number }).boundary : 0.3
  const { unlabelled, perClass, repeats, seed } = state

  const problem: LineProblem = useMemo(
    () => (kind === 'clusters' ? clustersProblem(separation) : uniformProblem(boundary)),
    [kind, separation, boundary],
  )

  // The first repeat of the experiment, shown on the left.
  const shown = useMemo(() => {
    const draw = drawLine(child(stream(seed), 'repeat', 0), problem, unlabelled, Math.max(...SIZES))
    const density = unlabelledDensity(draw.unlabelled, problem.domain, { points: CURVE_POINTS })
    const fit = fitBoth(problem, draw, perClass, density)
    return { draw, density, fit }
  }, [problem, unlabelled, perClass, seed])

  const curves = useComputed(
    () => thresholdLearningCurve(stream(seed), problem, { perClass: SIZES, unlabelled, repeats }),
    [problem, unlabelled, repeats, seed],
    { mode: 'release' },
  )

  const [a, b] = problem.domain
  const xs = useMemo(
    () => Array.from({ length: CURVE_POINTS }, (_, j) => a + (j * (b - a)) / (CURVE_POINTS - 1)),
    [a, b],
  )
  const truth = xs.map((x) => problem.density(x))
  const top = Math.max(...truth, ...(shown.density?.f ?? [0])) * 1.15
  const rug = -0.04 * top

  const xa = useAxis({ label: 'x', range: [a, b] })
  const ya = useAxis({ label: 'density', range: [2 * rug, top] })
  const xb = useAxis({ label: 'labels per class m', range: [1, 64], log: true })
  const yb = useAxis({ label: 'risk of the threshold', range: [0, undefined] })

  const c = curves.value
  const band = (pts: { mean: number; se: number }[]) => ({
    y: pts.map((p) => p.mean + 2 * p.se),
    base: pts.map((p) => Math.max(0, p.mean - 2 * p.se)),
  })
  const sup = band(c.supervised)
  const semi = band(c.semiSupervised)
  const m = shown.draw.class0.slice(0, perClass)
  const n = shown.draw.class1.slice(0, perClass)
  const vline = (t: number) => ({ x: [t, t], y: [0, top] })

  return (
    <Figure
      title="Learning a threshold with and without unlabelled data"
      state={state}
      caption="Left: one draw. Grey ticks are the unlabelled inputs, the curve through them their kernel density estimate, and the dashed curve the true input density; coloured markers are the m labelled points per class. The supervised threshold is the midpoint of the interval of thresholds with the fewest training errors; the semi-supervised threshold is the density trough with the fewest training errors. Right: the risk of each threshold, averaged over repeated draws (bands are ± 2 standard errors), against the number of labels per class; the dashed line is the Bayes risk. Drag the vertical line on the right to set m, and on the uniform problem drag the true boundary on the left. With two separated classes one label per class and a few hundred unlabelled points reach the Bayes risk. With uniform inputs the troughs are noise, the right threshold is not among them, and the semi-supervised learner stays wrong however many labels it gets."
      readouts={
        <>
          <Readout label="Bayes risk" value={formatNumber(problem.bayesRisk)} />
          <Readout label="supervised risk (this draw)" value={formatNumber(problem.risk(shown.fit.supervised))} />
          <Readout
            label="semi-supervised risk (this draw)"
            value={formatNumber(problem.risk(shown.fit.semiSupervised))}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya}>
          <Curve name="true input density" x={xs} y={truth} dashed muted />
          {shown.density && <Curve name="estimated density" x={shown.density.x} y={shown.density.f} thin />}
          {shown.draw.unlabelled.length > 0 && (
            <Points name="unlabelled" x={shown.draw.unlabelled} y={shown.draw.unlabelled.map(() => rug)} muted thin />
          )}
          <Points name="class 0" x={m} y={m.map(() => 0)} slot={0} size={10} />
          <Points name="class 1" x={n} y={n.map(() => 0)} slot={1} size={10} />
          <Curve name="supervised threshold" {...vline(shown.fit.supervised)} slot={2} />
          <Curve name="semi-supervised threshold" {...vline(shown.fit.semiSupervised)} slot={3} />
          {kind === 'clusters' ? (
            <Curve name="Bayes threshold" {...vline(problem.bayes)} dashed muted />
          ) : (
            <Handle
              kind="x"
              at={boundary}
              label="t*"
              onDrag={(v) => state.set('problem.boundary', Math.round(Math.min(0.95, Math.max(0.05, v)) * 100) / 100)}
            />
          )}
        </Plot>
        <Plot x={xb} y={yb}>
          <Area
            name="supervised ± 2 se"
            x={SIZES}
            y={sup.y}
            base={sup.base}
            slot={2}
            line={false}
            stale={curves.stale}
          />
          <Area
            name="semi-supervised ± 2 se"
            x={SIZES}
            y={semi.y}
            base={semi.base}
            slot={3}
            line={false}
            stale={curves.stale}
          />
          <Curve
            name="supervised"
            x={SIZES}
            y={c.supervised.map((p) => p.mean)}
            slot={2}
            showPoints
            stale={curves.stale}
          />
          <Curve
            name="semi-supervised"
            x={SIZES}
            y={c.semiSupervised.map((p) => p.mean)}
            slot={3}
            showPoints
            stale={curves.stale}
          />
          <Curve name="Bayes risk" x={[1, 64]} y={[problem.bayesRisk, problem.bayesRisk]} dashed muted />
          <Handle
            kind="x"
            at={perClass}
            label="m"
            onDrag={(v) => state.set('perClass', Math.round(Math.min(64, Math.max(1, v))))}
          />
        </Plot>
      </div>
    </Figure>
  )
}
