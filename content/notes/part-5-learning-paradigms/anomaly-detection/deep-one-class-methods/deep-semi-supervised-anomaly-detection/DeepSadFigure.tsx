import { useDeferredValue, useMemo, useState } from 'react'
import { Figure, float, Handle, Plot, Points, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal as drawNormal, stream, uniform as drawUniform } from 'aifn-compute/foundation/random'
import {
  auc,
  embed,
  initNet,
  initialCentre,
  normalData,
  scoreGrid,
  sqDist,
  train,
  uniformAnomalies,
  type Point,
} from '../_shared/deepOneClass'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

/** Uniform and normal draws from one aifn stream, in the shape the shared helpers take. */
const rand = (seed: number) => {
  const s = stream(seed)
  return { uniform: () => drawUniform(s), normal: () => drawNormal(s) }
}

const LIM = 4
const AXIS = toFlat(linspace(-LIM, LIM, 41))
const LOG_RANGE: [number, number] = [-3, 1]
const log10 = (v: number) => Math.log10(Math.max(v, 1e-12))

/** Normal data, an unseen anomaly class between the two blobs, and uniform anomalies for testing. */
const DATA = (() => {
  const r = rand(1)
  const x = normalData('blobs', 150, r)
  const testNormal = normalData('blobs', 100, r)
  const uniform = uniformAnomalies(100, LIM, x, 0.5, r)
  const cluster: Point[] = Array.from({ length: 50 }, () => [0.25 * r.normal(), 1.3 + 0.25 * r.normal()])
  return { x, testNormal, uniform, cluster }
})()
const START: Point[] = DATA.cluster.slice(0, 3)
const clamp = (v: number) => Math.max(-LIM, Math.min(LIM, v))

/** Deep SAD: Deep SVDD plus an inverse-distance term that pushes a few labelled anomalies away from c. */
export function DeepSadFigure() {
  const [labelled, setLabelled] = useState<Point[]>(START)
  const state = useFigureState({
    eta: float(1, {
      min: 0,
      max: 5,
      step: 0.1,
      label: 'η (weight of labelled anomalies)',
      format: (v) => v.toFixed(1),
    }),
  })
  const deferred = useDeferredValue({ labelled, eta: state.eta })

  const r = useMemo(() => {
    const net = initNet([2, 16, 16, 2], false, rand(7))
    const c = initialCentre(net, DATA.x)
    const snaps = train(net, DATA.x, c, {
      objective: { kind: 'semi-supervised', eta: deferred.eta, labelled: deferred.labelled },
      learnCentre: false,
      epochs: 200,
      lr: 0.01,
      weightDecay: 0.03,
      every: 200,
    })
    const s = snaps[snaps.length - 1]
    const score = (p: Point) => sqDist(embed(s.net, p), s.c)
    const trainScores = DATA.x.map(score).sort((a, b) => a - b)
    // Threshold: the 95th percentile of the training scores, a 5 % false-alarm rate on normal data.
    const tau = trainScores[Math.floor(0.95 * trainScores.length)]
    const sn = DATA.testNormal.map(score)
    const sc = DATA.cluster.slice(3).map(score)
    return {
      z: scoreGrid(s.net, s.c, AXIS).map((row) => row.map(log10)),
      flagged: sc.filter((v) => v > tau).length / sc.length,
      aucCluster: auc(sn, sc),
      aucUniform: auc(sn, DATA.uniform.map(score)),
    }
  }, [deferred])

  const overlay = useMemo(
    () =>
      [
        { name: 'unlabelled normal data', x: DATA.x.map((p) => p[0]), y: DATA.x.map((p) => p[1]) },
        {
          name: 'unseen anomalies (test)',
          x: DATA.cluster.slice(3).map((p) => p[0]),
          y: DATA.cluster.slice(3).map((p) => p[1]),
        },
      ] as const,
    [],
  )

  const handles: Handle[] = labelled.map((p, i) => ({
    kind: 'point',
    at: p,
    label: `labelled anomaly ${i + 1}`,
    onDrag: ([a, b]) => setLabelled((prev) => prev.map((q, j) => (j === i ? [clamp(a), clamp(b)] : q))),
  }))

  const xAxis = useAxis({ label: 'x₁' })
  const yAxis = useAxis({ label: 'x₂' })
  return (
    <Figure
      title="Deep SAD: a few labelled anomalies reshape the score"
      state={state}
      caption="The network and centre of the Deep SVDD figure, trained for 200 epochs on 150 unlabelled normal points plus the three labelled anomalies, drawn as large round markers. The colour is log₁₀ ‖φ(x) − c‖². A cluster of unseen anomalies sits between the two normal blobs, where a network trained on the blobs alone interpolates between them and gives many of the anomalies low scores. At η = 0 the labels are ignored and the model is Deep SVDD. Raise η, or drag the labelled anomalies, and watch the high-score region follow them. The readout counts how many of the unseen cluster exceed the 95th percentile of the training scores."

      readouts={
        <>
          <Readout label="unseen cluster flagged" value={`${Math.round(100 * r.flagged)} %`} />
          <Readout label="AUC, cluster" value={r.aucCluster.toFixed(3)} />
          <Readout label="AUC, uniform anomalies" value={r.aucUniform.toFixed(3)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis} height={420}>
          <Raster x={AXIS} y={AXIS} z={r.z} range={LOG_RANGE} valueLabel={'log₁₀ s(x)'} />
          <Points {...overlay[0]} live />
          <Points {...overlay[1]} live />
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
      </div>
    </Figure>
  )
}
