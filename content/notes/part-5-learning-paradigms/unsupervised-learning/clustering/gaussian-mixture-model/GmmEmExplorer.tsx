import { useMemo, useState } from 'react'
import {
  Figure,
  ControlGroup,
  Select,
  NumberSelector,
  Player,
  Plots,
  Plot,
  Points,
  Curve,
  Readout,
  Handle,
  Button,
  useAxis,
  type Vec2,
} from 'aifn-render'
import { blobs } from 'aifn-methods/data/synthetic'
import { gaussianMixtureSteps, type CovarianceType } from 'aifn-methods/unsupervised/clustering'
import { covarianceEllipse } from 'aifn-compute/numerics/geometry'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'

const fmt = (v: number | null | undefined, digits = 3): string => {
  if (v == null || !Number.isFinite(v)) return '—'
  return v.toFixed(digits)
}

const columns = (x: Tensor) => {
  const rows = toRows(x)
  return { x0: rows.map((r) => r[0]), x1: rows.map((r) => r[1]) }
}

const blobData = () =>
  blobs(stream('gmm/em/blobs'), {
    n: [60, 60, 50, 40],
    centers: [
      [-3, -2],
      [2.5, -2.5],
      [0, 2.5],
      [4, 2],
    ],
    sd: [0.8, 1.1, 0.7, 0.5],
  })

const defaultStartsForK = (k: number): Vec2[] => {
  if (k === 2)
    return [
      [-2, 3.5],
      [2, 3.5],
    ]
  if (k === 3)
    return [
      [-2.5, 3.5],
      [0, 3.5],
      [2.5, 3.5],
    ]
  if (k === 4)
    return [
      [-3, 3.5],
      [-1, 3.5],
      [1, 3.5],
      [3, 3.5],
    ]
  if (k === 5)
    return [
      [-3.5, 3.5],
      [-1.8, 3.5],
      [0, 3.5],
      [1.8, 3.5],
      [3.5, 3.5],
    ]
  return Array.from({ length: k }, (_, i) => [-3 + (6 * i) / (k - 1), 3.5] as Vec2)
}

export function GmmEmExplorer() {
  const data = useMemo(() => blobData(), [])
  const cols = useMemo(() => columns(data.x), [data])

  const [k, setK] = useState(4)
  const [covariance, setCovariance] = useState<CovarianceType>('full')
  const [start, setStart] = useState<Vec2[]>(() => defaultStartsForK(4))
  const [step, setStep] = useState(0)

  const handleKChange = (newK: number) => {
    setK(newK)
    setStart(defaultStartsForK(newK))
    setStep(0)
  }

  const handleCovarianceChange = (cov: string) => {
    setCovariance(cov as CovarianceType)
    setStep(0)
  }

  const run = useMemo(() => {
    const kVal = start.length
    const weightsInit = fromData(new Float64Array(kVal).fill(1 / kVal), [kVal])
    const meansInit = fromData(Float64Array.from(start.flat()), [kVal, 2])
    const covsFlat = new Float64Array(kVal * 4)
    for (let j = 0; j < kVal; j++) {
      covsFlat[j * 4] = 4
      covsFlat[j * 4 + 1] = 0
      covsFlat[j * 4 + 2] = 0
      covsFlat[j * 4 + 3] = 4
    }
    const covsInit = fromData(covsFlat, [kVal, 2, 2])

    return trace(
      gaussianMixtureSteps(data.x, { k: kVal, covariance, tolerance: 1e-6 }),
      { weights: weightsInit, means: meansInit, covariances: covsInit },
      150,
      { record: { ll: (s) => s.logLikelihood } },
    )
  }, [data, start, covariance])

  const currentStep = Math.min(step, run.steps.length - 1)
  const state = run.steps[currentStep]

  const owner = useMemo(() => toRows(state.responsibilities).map((r) => r.indexOf(Math.max(...r))), [state])

  const means = useMemo(() => toRows(state.means) as Vec2[], [state])

  const ellipses = useMemo(() => {
    const covs = toFlat(state.covariances)
    return means.flatMap((m, j) =>
      [1, 2].map((sd) => {
        const e = toRows(
          covarianceEllipse(
            m,
            [
              [covs[j * 4], covs[j * 4 + 1]],
              [covs[j * 4 + 2], covs[j * 4 + 3]],
            ],
            { k: sd },
          ).points,
        )
        return { j, sd, x: e.map((p) => p[0]), y: e.map((p) => p[1]) }
      }),
    )
  }, [means, state])

  const ll = useMemo(() => ({ x: Array.from(run.index), y: toFlat(run.series.ll) }), [run])
  const now = useMemo(() => ({ x: [ll.x[currentStep]], y: [ll.y[currentStep]] }), [ll, currentStep])

  const x0 = useAxis({ label: 'x₀', range: [-6, 7] })
  const x1 = useAxis({ label: 'x₁', range: [-6, 6], equal: x0 })
  const stepAxis = useAxis({ label: 'step', hold: 'initial', key: `${run.steps.length}-${covariance}-${k}` })
  const llAxis = useAxis({ label: 'log-likelihood', hold: 'initial', key: `${run.steps.length}-${covariance}-${k}` })

  const groupNames = useMemo(() => Array.from({ length: k }, (_, j) => `component ${j + 1}`), [k])

  return (
    <Figure
      title="Expectation-maximisation for Gaussian mixtures"
      purpose="The E-step computes the responsibilities of each component for every data point; the M-step refits mixing weights, means, and covariances to those soft assignments, monotonically increasing the log-likelihood."
      defaultSize="L"
      controls={
        <>
          <ControlGroup title="1 · Model structure">
            <Select
              label="covariance"
              value={covariance}
              onChange={handleCovarianceChange}
              options={[
                { value: 'full', label: 'full covariance (rotated ellipses)' },
                { value: 'diagonal', label: 'diagonal covariance (axis-aligned)' },
                { value: 'spherical', label: 'spherical covariance (isotropic circles)' },
              ]}
            />
            <NumberSelector
              label="k components"
              value={k}
              onChange={handleKChange}
              min={2}
              max={6}
              step={1}
              suggestions={[2, 3, 4, 5]}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setStart(defaultStartsForK(k))
                setStep(0)
              }}
            >
              Reset starts
            </Button>
          </ControlGroup>
          <ControlGroup title="2 · EM steps">
            <Player value={currentStep} onChange={setStep} count={run.steps.length} label="step" />
          </ControlGroup>
        </>
      }
      readouts={
        <>
          <Readout label="step" value={`${currentStep} of ${run.steps.length - 1}`} />
          <Readout label="mean log-likelihood" value={fmt(state.logLikelihood, 4)} />
          <Readout
            label="weights π"
            value={toFlat(state.weights)
              .map((w) => w.toFixed(2))
              .join(', ')}
          />
          <Readout label="converged" value={state.converged ? 'yes' : 'no'} />
        </>
      }
      caption="Ellipses display the 1σ (solid) and 2σ (dashed) contours of each component; each point takes the colour of its most responsible component. Drag the starting centroid handles (at top) to explore how initialisation affects the reached local optimum. Full covariance rotates arbitrarily to capture correlation; diagonal stays axis-aligned; spherical remains circular."
    >
      <Plots rows={2} heights={[3, 1]}>
        <Plot x={x0} y={x1} legend={false}>
          <Points name="points" x={cols.x0} y={cols.x1} group={owner} groupNames={groupNames} />
          {ellipses.map((e) => (
            <Curve
              key={`${e.j}-${e.sd}`}
              name={`component ${e.j + 1}`}
              x={e.x}
              y={e.y}
              slot={e.j}
              dashed={e.sd === 2}
            />
          ))}
          <Points name="means" x={means.map((m) => m[0])} y={means.map((m) => m[1])} emphasis />
          {start.map((p, j) => (
            <Handle
              key={j}
              kind="point"
              at={p}
              label={`start ${j + 1}`}
              onDrag={(q) => {
                setStart((s) => s.map((c, i) => (i === j ? q : c)))
                setStep(0)
              }}
            />
          ))}
        </Plot>
        <Plot x={stepAxis} y={llAxis} legend={false}>
          <Curve name="mean log-likelihood" x={ll.x} y={ll.y} />
          <Points name="current" x={now.x} y={now.y} emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}

export { GmmEmExplorer as EmStepper }
