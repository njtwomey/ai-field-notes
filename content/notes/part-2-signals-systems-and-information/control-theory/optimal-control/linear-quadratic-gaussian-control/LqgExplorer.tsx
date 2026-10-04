import { useMemo, useState } from 'react'
import {
  ControlRow,
  Curve,
  Figure,
  Player,
  Plot,
  Plots,
  Points,
  Readout,
  Select,
  formatNumber,
  useAxis,
} from 'aifn-render'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { lqg, lqgSimulation, type LqgSimulationState } from 'aifn/dynamics/control'

const DT = 0.1
const A = [
  [1, DT],
  [0, 1],
]
const B = [[0.5 * DT * DT], [DT]]
const C = [[1, 0]]
const Q = [
  [1, 0],
  [0, 0.1],
]
const X0 = [3, 0]

const NOISE_SCALES = [
  { value: 'low', label: 'Low noise (W = 10⁻³, V = 10⁻²)' },
  { value: 'medium', label: 'Moderate noise (W = 10⁻², V = 10⁻¹)' },
  { value: 'high', label: 'High noise (W = 10⁻¹, V = 1)' },
]

const R_SCALES = [
  { value: '0.01', label: 'Aggressive (R = 0.01, high actuation force)' },
  { value: '0.1', label: 'Balanced (R = 0.1)' },
  { value: '1.0', label: 'Cautious (R = 1.0, penalises large force)' },
]

export function LqgExplorer() {
  const [noiseSetting, setNoiseSetting] = useState<'low' | 'medium' | 'high'>('medium')
  const [rChoice, setRChoice] = useState<'0.01' | '0.1' | '1.0'>('0.1')
  const [mismatch, setMismatch] = useState<number>(0)
  const [playhead, setPlayhead] = useState(100)

  const steps = 100

  const simulation = useMemo(() => {
    let logW = -2
    let logV = -1
    if (noiseSetting === 'low') {
      logW = -3
      logV = -2
    } else if (noiseSetting === 'high') {
      logW = -1
      logV = 0
    }

    const W = Math.pow(10, logW)
    const V = Math.pow(10, logV)
    const filterV = Math.pow(10, logV + mismatch)
    const R = [[Number(rChoice)]]

    const plantNoise = [
      [W * 0.1, 0],
      [0, W],
    ]
    const filterNoise = [
      [W * 0.1, 0],
      [0, W],
    ]

    const plant = { A, B, C }
    const weights = { Q, R, W: plantNoise, V: [[V]] }
    const design = lqg(plant, { Q, R, W: filterNoise, V: [[filterV]] }, { discrete: true })

    const s1 = stream('lqg-sim-noise')
    const loopEstimate = lqgSimulation(plant, weights, design, {
      feedback: 'estimate',
    })
    const traceEstimate = trace(loopEstimate, { x0: X0 }, steps, { keep: 'all', stream: s1 })

    const s2 = stream('lqg-sim-noise')
    const loopTrue = lqgSimulation(plant, weights, design, {
      feedback: 'state',
    })
    const traceTrue = trace(loopTrue, { x0: X0 }, steps, { keep: 'all', stream: s2 })

    const extract = (tr: readonly LqgSimulationState[]) => {
      const t: number[] = []
      const p: number[] = []
      const v: number[] = []
      const pHat: number[] = []
      const vHat: number[] = []
      const u: number[] = []
      const y: number[] = []
      const cost: number[] = []

      let runningCost = 0
      for (let i = 0; i < tr.length; i++) {
        const item = tr[i]
        t.push(i * DT)
        const xArr = toFlat(item.x)
        p.push(xArr[0])
        v.push(xArr[1])

        const xHatArr = toFlat(item.xHat)
        pHat.push(xHatArr[0])
        vHat.push(xHatArr[1])

        const uVal = toFlat(item.u)[0]
        u.push(uVal)

        const yVal = toFlat(item.y)[0]
        y.push(yVal)

        const stateCost = xArr[0] * xArr[0] + 0.1 * xArr[1] * xArr[1]
        const ctrlCost = Number(rChoice) * uVal * uVal
        runningCost += stateCost + ctrlCost
        cost.push(runningCost / (i + 1))
      }
      return { t, p, v, pHat, vHat, u, y, cost }
    }

    return {
      design,
      estimateLoop: extract(traceEstimate.steps),
      trueLoop: extract(traceTrue.steps),
    }
  }, [noiseSetting, rChoice, mismatch, steps])

  const { design, estimateLoop, trueLoop } = simulation
  const numSteps = estimateLoop.t.length
  const currentStep = Math.min(playhead, Math.max(0, numSteps - 1))

  const curT = useMemo(() => estimateLoop.t.slice(0, currentStep + 1), [estimateLoop.t, currentStep])
  const curP = useMemo(() => estimateLoop.p.slice(0, currentStep + 1), [estimateLoop.p, currentStep])
  const curPHat = useMemo(() => estimateLoop.pHat.slice(0, currentStep + 1), [estimateLoop.pHat, currentStep])
  const curStateP = useMemo(() => trueLoop.p.slice(0, currentStep + 1), [trueLoop.p, currentStep])
  const curY = useMemo(() => estimateLoop.y.slice(0, currentStep + 1), [estimateLoop.y, currentStep])
  const curU = useMemo(() => estimateLoop.u.slice(0, currentStep + 1), [estimateLoop.u, currentStep])
  const curCost = useMemo(() => estimateLoop.cost.slice(0, currentStep + 1), [estimateLoop.cost, currentStep])

  const timeAxisX = useAxis({ label: 'Time t (s)', range: [0, steps * DT] })
  const stateAxisY = useAxis({ label: 'Position p(t)', range: [-1, 4] })

  const costAxisX = useAxis({ label: 'Time t (s)', range: [0, steps * DT] })
  const costAxisY = useAxis({ label: 'Control u(t) / Cost J', range: [-4, 6] })

  const kGain = toFlat(design.K)
  const lGain = toFlat(design.L)

  return (
    <Figure
      title="Linear-Quadratic-Gaussian (LQG) Optimal Control"
      purpose="Simulate separation principle dynamics combining LQR state feedback and Kalman filter state estimation under process and measurement noise."
      caption={
        'Linear-Quadratic-Gaussian (LQG) closed-loop regulation of a double integrator (Anderson & Moore, 2007). Position p(0) = 3 must be driven to the origin under velocity disturbances W and noisy position measurements y = p + v. By the separation theorem, the optimal stochastic controller pairs the deterministic LQR feedback gain K with the Kalman filter estimate x̂, achieving certainty equivalence.'
      }
    >
      <ControlRow label="Disturbances & cost">
        <Select
          label="Noise level"
          value={noiseSetting}
          options={NOISE_SCALES}
          onChange={(v) => setNoiseSetting(v as 'low' | 'medium' | 'high')}
        />
        <Select
          label="Control penalty R"
          value={rChoice}
          options={R_SCALES}
          onChange={(v) => setRChoice(v as '0.01' | '0.1' | '1.0')}
        />
        <Select
          label="Filter noise mismatch log₁₀(V̂/V)"
          value={String(mismatch)}
          options={[
            { value: '-1', label: '-1 (underestimate noise)' },
            { value: '0', label: '0 (matched filter)' },
            { value: '1', label: '1 (overestimate noise)' },
          ]}
          onChange={(v) => setMismatch(Number(v))}
        />
      </ControlRow>

      <ControlRow label="Closed-loop execution">
        <Player
          count={numSteps}
          value={currentStep}
          onChange={setPlayhead}
        />
      </ControlRow>

      <Plots>
        <Plot x={timeAxisX} y={stateAxisY} title="State trajectories: true p(t), estimated p̂(t) & full-state LQR">
          <Curve
            x={curT}
            y={curP}
            slot={0}
          />
          <Curve
            x={curT}
            y={curPHat}
            slot={1}
          />
          <Curve
            x={curT}
            y={curStateP}
            slot={2}
            thin={true}
          />
          <Points
            x={curT}
            y={curY}
            slot={3}
            size={3}
            thin={true}
          />
        </Plot>

        <Plot x={costAxisX} y={costAxisY} title="Control input u(t) and average quadratic cost J(t)">
          <Curve
            x={curT}
            y={curU}
            slot={0}
          />
          <Curve
            x={curT}
            y={curCost}
            slot={1}
          />
        </Plot>
      </Plots>

      <ControlRow label="Diagnostics">
        <Readout
          label="Position error |p - p̂|"
          value={formatNumber(Number(Math.abs(estimateLoop.p[currentStep] - estimateLoop.pHat[currentStep]).toFixed(3)))}
        />
        <Readout
          label="Mean cost J"
          value={formatNumber(Number(estimateLoop.cost[currentStep].toFixed(2)))}
        />
        <Readout
          label="LQR gain K"
          value={`[${formatNumber(Number(kGain[0].toFixed(2)))}, ${formatNumber(Number(kGain[1].toFixed(2)))}]`}
        />
        <Readout
          label="Kalman gain L"
          value={`[${formatNumber(Number(lGain[0].toFixed(2)))}, ${formatNumber(Number(lGain[1].toFixed(2)))}]ᵀ`}
        />
      </ControlRow>
    </Figure>
  )
}
