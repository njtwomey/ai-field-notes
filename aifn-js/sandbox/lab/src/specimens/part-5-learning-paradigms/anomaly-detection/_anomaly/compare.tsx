/**
 * Showcase: anomaly detection compared. Two-dimensional data with planted anomalies (`plantedAnomalies`); one detector
 * of aifn-methods `anomalyScores` fitted in the worker scores the points and a grid, and `compareDetectors` ranks every
 * detector by AUROC and average precision. The threshold is the share of points flagged, set on a slider or by dragging
 * it on the score histogram; precision and recall at it come from `aifn/learning/metrics`.
 */
import { useMemo } from 'react'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { f1, precision, precisionRecallCurve, recall } from 'aifn/learning/metrics'
import { ANOMALY_SHAPES, plantedAnomalies } from 'aifn-methods/data/synthetic'
import {
  anomalyThreshold,
  DETECTORS,
  type AnomalyScores,
  type DetectorComparison,
  type DetectorKind,
} from 'aifn-methods/unsupervised/anomaly'
import { Figure } from '@lab/layout'
import { call, choice, float, int, row, useComputed, useFigureState } from '@lab/state'
import { formatValue } from '@lab/views'
import { Bars, Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'

const f3 = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—'
const BOX: [number, number] = [-2.6, 2.6]
const G = 48
const AXIS = Array.from({ length: G }, (_, i) => BOX[0] + ((BOX[1] - BOX[0]) * i) / (G - 1))
const GRID = AXIS.flatMap((y) => AXIS.map((x) => [x, y]))
const MODULE = 'applied/unsupervised/anomaly'
const SHORT = ['IF', 'LOF', 'k-NN', 'OCSVM', 'SVDD', 'Mahal.', 'MCD', 'PCA', 'ensemble']

export function AnomalyComparison() {
  const state = useFigureState({
    data: row('1 · data', {
      shape: choice([...ANOMALY_SHAPES], 'clusters', { label: 'inliers' }),
      n: int(300, { ge: 20, le: 2000, suggestions: [150, 300, 600], label: 'points' }),
      contamination: float(0.05, { ge: 0, le: 0.4, suggestions: [0.02, 0.05, 0.1], label: 'share of anomalies' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    detector: row('2 · detector', {
      kind: choice(
        DETECTORS.map((d) => ({ value: d.kind, label: d.name })),
        'isolation-forest',
        { label: 'detector' },
      ),
      k: int(10, { ge: 1, le: 100, suggestions: [5, 10, 20], label: 'neighbours (k-NN, LOF)' }),
      nu: float(0.1, { gt: 0, le: 1, suggestions: [0.05, 0.1, 0.2], label: 'ν (one-class SVM, SVDD)' }),
      gamma: float(1, { gt: 0, scale: 'log10', suggestions: [0.3, 1, 3], label: 'kernel γ' }),
      components: int(1, { ge: 1, le: 2, label: 'components (PCA)' }),
    }),
    threshold: row('3 · threshold', {
      risk: float(0.05, { gt: 0, lt: 1, suggestions: [0.02, 0.05, 0.1], label: 'share of points flagged' }),
    }),
  })
  const { shape, n, contamination, seed } = state.data
  const data = useMemo(
    () => plantedAnomalies(stream(seed), { shape, n, contamination }),
    [shape, n, contamination, seed],
  )
  const X = useMemo(
    () => Array.from({ length: n }, (_, i) => [toFlat(data.x)[2 * i], toFlat(data.x)[2 * i + 1]]),
    [data, n],
  )
  const y = useMemo(() => Array.from(toFlat(data.y!)), [data])
  const kind = state.detector.kind as DetectorKind
  const options = {
    k: state.detector.k,
    nu: state.detector.nu,
    gamma: state.detector.gamma,
    components: state.detector.components,
    seed: 1,
  }
  const dataKey = JSON.stringify([shape, n, contamination, seed])
  const fitKey = JSON.stringify([dataKey, kind, options])
  const fit = useComputed(() => call<AnomalyScores>(`${MODULE}/anomalyScores`, kind, X, GRID, options), [fitKey], {
    mode: 'worker',
    initial: null as { key: string; scores: AnomalyScores } | null,
    then: (r) => ({ key: fitKey, scores: r }),
    cancelAfter: 2000,
  })
  const all = useComputed(
    () => call<DetectorComparison[]>(`${MODULE}/compareDetectors`, X, y, options),
    [JSON.stringify([dataKey, options])],
    {
      mode: 'worker',
      initial: null as DetectorComparison[] | null,
      cancelAfter: 4000,
    },
  )
  const scores = fit.value?.key === fitKey ? fit.value.scores : null
  const risk = state.threshold.risk
  const threshold = useMemo(() => (scores ? anomalyThreshold(scores.train, { risk }) : NaN), [scores, risk])
  const field = useMemo(
    () => (scores?.queries ? AXIS.map((_, r) => Array.from(scores.queries!.subarray(r * G, (r + 1) * G))) : null),
    [scores],
  )
  const flagged = useMemo(
    () => (scores ? Array.from(scores.train, (s): number => (s > threshold ? 1 : 0)) : null),
    [scores, threshold],
  )
  const pr = useMemo(() => (scores ? precisionRecallCurve(y, scores.train) : null), [scores, y])
  const at = flagged
    ? { p: precision(y, flagged), r: recall(y, flagged), f: f1(y, flagged), count: flagged.reduce((a, b) => a + b, 0) }
    : null

  // The training scores of inliers and anomalies as two histograms on shared bins.
  const hist = useMemo(() => {
    if (!scores) return null
    const s = scores.train
    let lo = Infinity
    let hi = -Infinity
    for (const v of s) {
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
    const B = 30
    const w = (hi - lo) / B || 1
    const edges = Array.from({ length: B + 1 }, (_, b) => lo + b * w)
    const counts = [new Array<number>(B).fill(0), new Array<number>(B).fill(0)]
    s.forEach((v, i) => counts[y[i]][Math.min(B - 1, Math.floor((v - lo) / w))]++)
    return { edges, inliers: counts[0], anomalies: counts[1], centres: edges.slice(0, B).map((e) => e + w / 2) }
  }, [scores, y])
  // Dragging the threshold on the histogram sets the share of training scores above it.
  const setThreshold = (t: number) => {
    if (!scores) return
    const above = Array.from(scores.train).filter((v) => v > t).length / scores.train.length
    state.set('threshold.risk', Math.min(0.99, Math.max(0.001, above)))
  }

  const px = useAxis({ label: 'x₁', range: BOX })
  const py = useAxis({ label: 'x₂', range: BOX, equal: px })
  const sx = useAxis({ label: 'anomaly score', key: fitKey })
  const cy = useAxis({ label: 'points', hold: 'union', key: fitKey })
  const rx = useAxis({ label: 'recall', range: [0, 1.02] })
  const ry = useAxis({ label: 'precision', range: [0, 1.02] })
  const bx = useAxis({ label: 'detector', categories: SHORT })
  const by = useAxis({ label: 'AUROC / average precision', range: [0, 1.02] })
  const name = DETECTORS.find((d) => d.kind === kind)?.name ?? kind
  const pending = !scores ? 'fitting…' : undefined
  const mine = all.value?.find((c) => c.kind === kind)
  return (
    <Figure
      title="Anomaly detection compared"
      purpose="Each detector turns 'unlike the rest' into a score: distance to neighbours, local density, isolation depth, a kernel boundary, a robust Gaussian or a subspace. Which one finds the planted anomalies depends on the shape of the inliers."
      state={state}
      defaultSize="XL"
      readouts={{
        threshold: (
          <>
            <Readout label="threshold" value={f3(threshold)} />
            <Readout label="flagged" value={at ? `${at.count} of ${n}` : '—'} />
            <Readout label="precision" value={f3(at?.p)} />
            <Readout label="recall" value={f3(at?.r)} />
            <Readout label="F1" value={f3(at?.f)} />
          </>
        ),
        ranking: (
          <>
            <Readout label="AUROC" value={f3(mine?.auroc)} />
            <Readout label="average precision" value={f3(mine?.averagePrecision ?? pr?.area)} />
            <Readout label="planted anomalies" value={y.reduce((a, b) => a + b, 0)} />
            {(!scores || !all.value) && (
              <Readout label="computing" value={<span aria-busy="true">in a worker…</span>} />
            )}
          </>
        ),
      }}
      caption={
        <>
          aifn-methods <code>plantedAnomalies</code> ({shape}) scored by <code>anomalyScores</code> with {name}. Top
          left: the score field over the plane (its scale beside it; higher is more anomalous) with the boundary at the
          threshold, the inliers and the planted anomalies in their class colours, shaped by whether the threshold flags
          them. Top right: the training scores of inliers and anomalies; drag the threshold line, or set the share
          flagged on its slider. Bottom left: the precision–recall curve of the scores with the operating point at the
          threshold. Bottom right: every detector&apos;s AUROC and average precision on the same data (
          <code>compareDetectors</code>), the chosen one marked.
        </>
      }
    >
      <Plots rows={2} cols={2} heights={[3, 2]}>
        <Plot x={px} y={py} title={pending ?? `${name}: the score field`}>
          {field && (
            <Raster
              x={AXIS}
              y={AXIS}
              z={field}
              scale="sequential"
              valueLabel="anomaly score"
              fillOpacity={0.7}
              boundary={threshold}
            />
          )}
          <Points
            name="points"
            x={X.map((p) => p[0])}
            y={X.map((p) => p[1])}
            group={y}
            groupNames={['inlier', 'planted anomaly']}
            shape={flagged ?? 0}
            shapeNames={['below the threshold', 'flagged']}
            size={7}
          />
        </Plot>
        <Plot x={sx} y={cy} title={pending ?? 'training scores'}>
          {hist && <Bars name="inliers" x={hist.centres} y={hist.inliers} edges={hist.edges} slot={0} />}
          {hist && <Bars name="planted anomalies" x={hist.centres} y={hist.anomalies} edges={hist.edges} slot={1} />}
          {Number.isFinite(threshold) && (
            <Handle kind="x" at={threshold} onDrag={setThreshold} label={`threshold ${f3(threshold)}`} />
          )}
        </Plot>
        <Plot x={rx} y={ry} title={pending ?? 'precision against recall'}>
          {pr && <Curve name="precision–recall" x={Array.from(toFlat(pr.x))} y={Array.from(toFlat(pr.y))} slot={2} />}
          {at && <Points name="at the threshold" x={[at.r]} y={[at.p]} emphasis size={12} />}
        </Plot>
        <Plot x={bx} y={by} title={all.value ? 'every detector on this data' : 'comparing…'}>
          {all.value && (
            <Bars
              name="AUROC"
              x={all.value.map((_, i) => i - 0.2)}
              y={all.value.map((c) => c.auroc)}
              width={0.38}
              slot={3}
            />
          )}
          {all.value && (
            <Bars
              name="average precision"
              x={all.value.map((_, i) => i + 0.2)}
              y={all.value.map((c) => c.averagePrecision)}
              width={0.38}
              slot={4}
            />
          )}
          {all.value && (
            <Points name="chosen" x={[DETECTORS.findIndex((d) => d.kind === kind)]} y={[1.0]} emphasis size={10} />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
