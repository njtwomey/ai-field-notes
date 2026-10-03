import { useMemo, useState } from 'react'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat } from 'aifn/foundation/tensor'
import type { InverseTruth } from 'aifn-applied/data'
import { bishopInverse } from 'aifn-applied/data/synthetic'
import {
  mdnModel,
  mdnPredict,
  mixtureDensityRun,
  type MdnSnapshot,
} from 'aifn-applied/learning/mixture-density'
import {
  ControlGroup,
  Curve,
  Figure,
  Handle,
  NumberSelector,
  Player,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Select,
  formatNumber,
  useAxis,
} from 'aifn-render'

const TASKS = [
  { value: 'bishop', label: 'Folded sine wave (multimodal inverse branches)' },
]

const X_RANGE: [number, number] = [-0.15, 1.15]
const T_RANGE: [number, number] = [-0.1, 1.1]
const G = 91

const grid = (lo: number, hi: number, n: number) =>
  Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))

const XS = grid(...X_RANGE, G)
const TS = grid(...T_RANGE, G)
const XS_TENSOR = fromData(Float64Array.from(XS), [G, 1])

export function MdnExplorer() {
  const [task, setTask] = useState('bishop')
  const [components, setComponents] = useState(3)
  const [width, setWidth] = useState(20)
  const [steps, setSteps] = useState(1000)
  const [noise, setNoise] = useState(0.05)
  const [checkpointIdx, setCheckpointIdx] = useState(0)
  const [probeX, setProbeX] = useState(0.5)

  // Synthetic dataset
  const data = useMemo(() => {
    return bishopInverse(stream(`bishop-${noise}`), { n: 350, noise })
  }, [noise])

  const truth = data.meta.truth as InverseTruth

  const dataPoints = useMemo(() => {
    return {
      x: Array.from(toFlat(data.x)),
      t: Array.from(toFlat(data.y!)),
    }
  }, [data])

  // Ground truth true branches
  const branches = useMemo(() => {
    const t = grid(0, 1, 201)
    return { x: t.map((v) => truth.forward([v])[0]), t }
  }, [truth])

  // Run training generator
  const snapshots = useMemo(() => {
    const gen = mixtureDensityRun({
      data: { x: data.x, y: data.y! },
      components,
      hidden: [width],
      steps,
      every: Math.max(1, Math.round(steps / 40)),
      seed: 42,
    })
    const list: MdnSnapshot[] = []
    for (const snap of gen) {
      list.push(snap)
    }
    return list
  }, [data, components, width, steps])

  const currentSnap = snapshots[Math.min(checkpointIdx, snapshots.length - 1)] ?? snapshots[snapshots.length - 1]
  const shot = currentSnap?.checkpoints[currentSnap.checkpoints.length - 1]

  // Evaluate both models at checkpoint
  const evaluation = useMemo(() => {
    if (!currentSnap || !shot) return null
    const mdn = mdnModel(currentSnap.spec)
    const meanModel = mdnModel(currentSnap.meanSpec)

    const mdnPred = mdnPredict(mdn, shot.mixture, XS_TENSOR)
    const meanPred = mdnPredict(meanModel, shot.mean, XS_TENSOR)

    const mix = mdnPred.mixture!
    const z = TS.map(() => new Array<number>(G).fill(0))
    for (let j = 0; j < G; j++) {
      const col = TS.map((t) => Math.exp(mix.logDensity(j, [t])))
      const top = Math.max(...col) || 1
      col.forEach((v, i) => {
        z[i][j] = v / top
      })
    }

    const K = mix.components
    const means: { x: number[]; t: number[] } = { x: [], t: [] }
    for (let j = 0; j < G; j += 2) {
      for (let k = 0; k < K; k++) {
        const w = mix.weights[j * K + k]
        if (w > 0.05) {
          means.x.push(XS[j])
          means.t.push(mix.means[j * K + k])
        }
      }
    }

    const meanCurve = {
      x: XS,
      t: Array.from(meanPred.mean) as number[],
    }

    // Cross-sectional conditional distribution p(t | probeX)
    const probeTensor = fromData(Float64Array.from([probeX]), [1, 1])
    const probePred = mdnPredict(mdn, shot.mixture, probeTensor)
    const probeMix = probePred.mixture!
    const densitySlice = TS.map((t) => Math.exp(probeMix.logDensity(0, [t])))
    const maxDensity = Math.max(...densitySlice) || 1
    const normSlice = densitySlice.map((v) => v / maxDensity)

    return {
      heatmap: { x: XS, y: TS, z },
      mdnMeans: means,
      meanCurve,
      slice: { t: TS, p: normSlice },
    }
  }, [currentSnap, shot, probeX])

  const xAxis = useAxis({ label: 'observed feature x', range: X_RANGE })
  const tAxis = useAxis({ label: 'target variable t', range: T_RANGE, equal: xAxis })
  const sliceT = useAxis({ label: 'target t', range: T_RANGE })
  const sliceP = useAxis({ label: 'conditional density p(t | x)', range: [0, 1.1] })

  const latestHistory = currentSnap?.history
  const mdnNll = latestHistory?.nll[latestHistory.nll.length - 1] ?? 0
  const mseLoss = latestHistory?.meanMse[latestHistory.meanMse.length - 1] ?? 0

  return (
    <Figure
      title="Mixture Density Networks vs Standard Squared-Error Regression"
      purpose="When an inverse problem is multi-valued, standard least-squares neural networks predict the conditional expectation E[t | x], which cuts disastrously between branches. An MDN predicts a full mixture of Gaussians, capturing every solution branch."
      defaultSize="L"
      controls={
        <>
          <ControlGroup title="Problem Setup">
            <Select label="Inverse Task" value={task} onChange={setTask} options={TASKS} />
            <NumberSelector
              label="Mixture components K"
              value={components}
              onChange={setComponents}
              min={1}
              max={6}
              step={1}
              suggestions={[1, 2, 3, 5]}
            />
            <NumberSelector
              label="Hidden width"
              value={width}
              onChange={setWidth}
              min={8}
              max={64}
              step={4}
              suggestions={[12, 20, 32]}
            />
            <NumberSelector
              label="Noise level σ"
              value={noise}
              onChange={setNoise}
              min={0.02}
              max={0.15}
              step={0.01}
              spacing="lin"
              points={0.01}
            />
            <NumberSelector
              label="Training steps"
              value={steps}
              onChange={setSteps}
              min={300}
              max={2000}
              step={100}
              suggestions={[500, 1000, 1500]}
            />
          </ControlGroup>
          <ControlGroup title="Playback Checkpoint">
            <Player
              value={Math.min(checkpointIdx, snapshots.length - 1)}
              onChange={setCheckpointIdx}
              count={snapshots.length}
              label="Training steps"
            />
          </ControlGroup>
        </>
      }
      readouts={{
        'Model Comparison': (
          <>
            <Readout label="step" value={`${shot?.step ?? 0} / ${steps}`} />
            <Readout label="MDN negative log-likelihood" value={formatNumber(mdnNll)} />
            <Readout label="MSE network loss" value={formatNumber(mseLoss)} />
          </>
        ),
        'Interactive Probe': (
          <>
            <Readout label="probe x₀" value={formatNumber(probeX)} />
            <Readout label="components K" value={components} />
          </>
        ),
      }}
      caption="Left: Training pairs (points), the multi-valued ground-truth curve (dashed line), the standard MLP mean prediction (orange line, which cuts between the branches), and the MDN conditional density heatmap with component modes (green). Drag the probe handle x₀ to view the live multi-peaked conditional density p(t | x₀) on the right."
    >
      <Plots cols={2} widths={[1.4, 1]}>
        <Plot x={xAxis} y={tAxis}>
          {evaluation && (
            <Raster
              x={evaluation.heatmap.x}
              y={evaluation.heatmap.y}
              z={evaluation.heatmap.z}
              scale="sequential"
              range={[0, 1]}
              valueLabel="p(t | x)"
            />
          )}
          <Curve name="true branches x = t + 0.3 sin(2πt)" x={branches.x} y={branches.t} dashed muted />
          <Points name="training samples" x={dataPoints.x} y={dataPoints.t} />
          {evaluation && (
            <>
              <Curve name="MSE MLP mean E[t | x]" x={evaluation.meanCurve.x} y={evaluation.meanCurve.t} slot={1} />
              <Points name="MDN mixture modes" x={evaluation.mdnMeans.x} y={evaluation.mdnMeans.t} slot={2} emphasis />
            </>
          )}
          <Handle kind="x" at={probeX} onDrag={setProbeX} label="x₀" />
        </Plot>
        <Plot x={sliceT} y={sliceP}>
          {evaluation && (
            <Curve name="conditional density p(t | x₀)" x={evaluation.slice.t} y={evaluation.slice.p} slot={2} emphasis />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
