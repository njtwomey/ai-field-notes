import { useMemo, useState } from 'react'
import {
  ControlRow,
  Curve,
  Figure,
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
import { normals, stream } from 'aifn/foundation/random'
import { mul, toFlat } from 'aifn/foundation/tensor'
import {
  ddimSampler,
  gaussianMixtureData,
  linearSchedule,
  mixtureLogDensity,
  mixtureNoisePredictor,
  probabilityFlowSampler,
  reverseSdeSampler,
  vpSde,
  type SamplerState,
} from 'aifn-methods/generative/diffusion'

const MIXTURE = gaussianMixtureData(
  [0.35, 0.35, 0.3],
  [
    [-1.2, -1.0],
    [1.2, -0.8],
    [0.0, 1.4],
  ],
  [0.25, 0.25, 0.3],
)

const G = 32
const GX = Array.from({ length: G }, (_, i) => -3 + (6 * (i + 0.5)) / G)

const SDE = vpSde()
const T = 100
const SCHEDULE = linearSchedule(T, { betaStart: 0.1 / T, betaEnd: 20 / T })

export function ScoreBasedExplorer() {
  const [samplerChoice, setSamplerChoice] = useState<'ode' | 'sde' | 'ddim'>('ode')
  const [numParticles, setNumParticles] = useState(40)
  const [stepIndex, setStepIndex] = useState(0)

  // Pre-generate reverse trajectories from noise t=1 down to t=0
  const trajectories = useMemo(() => {
    const s = stream('score-particles')
    const points = mul(normals(s, [numParticles, 2]), 1.0)
    const predictor = mixtureNoisePredictor(MIXTURE)

    let sampler
    if (samplerChoice === 'ode') {
      sampler = probabilityFlowSampler(predictor, SDE, { steps: 30, method: 'rk4' })
    } else if (samplerChoice === 'sde') {
      sampler = reverseSdeSampler(predictor, SDE, { steps: 30 })
    } else {
      sampler = ddimSampler(predictor, SCHEDULE, { steps: 30, eta: 0 })
    }

    const runnerStream = stream('sampler-stepping')
    let state = sampler.init({ x: points }, runnerStream)
    const states: SamplerState[] = [state]
    const isDone = sampler.done ?? ((st: SamplerState) => st.t >= 30)
    while (!isDone(state)) {
      state = sampler.step(state, { stream: runnerStream, t: state.t })
      states.push(state)
    }
    return states
  }, [samplerChoice, numParticles])

  const numSteps = trajectories.length
  const currentStep = Math.min(stepIndex, Math.max(0, numSteps - 1))
  const currentT = numSteps > 1 ? 1 - currentStep / (numSteps - 1) : 0

  // Points at the current step
  const currentPoints = useMemo(() => {
    if (!trajectories[currentStep]) return { x: [], y: [] }
    const flat = toFlat(trajectories[currentStep].x)
    const n = flat.length / 2
    const xs: number[] = []
    const ys: number[] = []
    for (let i = 0; i < n; i++) {
      xs.push(flat[2 * i])
      ys.push(flat[2 * i + 1])
    }
    return { x: xs, y: ys }
  }, [trajectories, currentStep])

  // Continuous smoothed density raster p_t(x)
  const densityRaster = useMemo(() => {
    const meanS = SDE.meanScale(currentT)
    const stdS = SDE.std(currentT)
    const gridPoints: number[] = []
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        gridPoints.push(GX[j], GX[i])
      }
    }
    const gridTensor = mul(normals(stream(1), [G * G, 2]), 0) // allocate shape
    const dataFlat = toFlat(gridTensor)
    for (let k = 0; k < gridPoints.length; k++) dataFlat[k] = gridPoints[k]

    const logp = toFlat(mixtureLogDensity(MIXTURE, gridTensor, meanS, stdS))

    const m: number[][] = []
    for (let i = 0; i < G; i++) {
      const row: number[] = []
      for (let j = 0; j < G; j++) {
        row.push(Math.exp(logp[i * G + j]))
      }
      m.push(row)
    }
    return m
  }, [currentT])

  // Particle path history for trace lines
  const pathLines = useMemo(() => {
    const lines: { x: number[]; y: number[] }[] = []
    const P = Math.min(15, numParticles)
    for (let p = 0; p < P; p++) {
      const px: number[] = []
      const py: number[] = []
      for (let s = 0; s <= currentStep; s++) {
        const flat = toFlat(trajectories[s].x)
        px.push(flat[2 * p])
        py.push(flat[2 * p + 1])
      }
      lines.push({ x: px, y: py })
    }
    return lines
  }, [trajectories, currentStep, numParticles])

  const plotX = useAxis({ label: 'x₁', range: [-3, 3] })
  const plotY = useAxis({ label: 'x₂', range: [-3, 3] })

  return (
    <Figure
      title="Score-Based Generative Diffusion & Probability Flow ODE"
      purpose="Visualise continuous-time reverse SDE sampling, Probability Flow ODE determinism, and exact time-dependent density evolution."
      caption={
        'Interactive score-based reverse diffusion (Song et al., 2021). As time reverses from t = 1 (pure isotropic Gaussian noise) to t = 0 (data manifold), particles flow towards high-density mixture modes under the exact score field ∇log p_t(x). The Probability Flow ODE integrates deterministically along smooth trajectories, while reverse SDE injects Brownian noise at each Euler–Maruyama step.'
      }
    >
      <ControlRow label="Diffusion setup">
        <Select
          label="Reverse trajectory integrator"
          value={samplerChoice}
          options={[
            { value: 'ode', label: 'Probability Flow ODE (deterministic RK4)' },
            { value: 'sde', label: 'Continuous Reverse VP SDE (Euler–Maruyama)' },
            { value: 'ddim', label: 'DDIM (deterministic η = 0)' },
          ]}
          onChange={(v) => setSamplerChoice(v as 'ode' | 'sde' | 'ddim')}
        />
        <Select
          label="Particle count"
          value={String(numParticles)}
          options={[
            { value: '20', label: '20 particles' },
            { value: '40', label: '40 particles' },
            { value: '80', label: '80 particles' },
          ]}
          onChange={(v) => setNumParticles(Number(v))}
        />
      </ControlRow>

      <ControlRow label="Reverse time progression">
        <Player count={Math.max(1, numSteps)} value={currentStep} onChange={setStepIndex} />
      </ControlRow>

      <Plots>
        <Plot x={plotX} y={plotY} title="State space: evolving smoothed density p_t(x) & particle flows">
          <Raster x={GX} y={GX} z={densityRaster} scale="sequential" />
          {pathLines.map((line, idx) => (
            <Curve key={idx} x={line.x} y={line.y} slot={0} thin={true} />
          ))}
          <Points x={currentPoints.x} y={currentPoints.y} slot={1} size={6} />
        </Plot>
      </Plots>

      <ControlRow label="Diffusion time state">
        <Readout label="Continuous time t" value={formatNumber(Number(currentT.toFixed(2)))} />
        <Readout label="Noise std σ(t)" value={formatNumber(Number(SDE.std(currentT).toFixed(3)))} />
        <Readout label="Signal scale s(t)" value={formatNumber(Number(SDE.meanScale(currentT).toFixed(3)))} />
        <Readout label="Step" value={`${currentStep} / ${Math.max(1, numSteps - 1)}`} />
      </ControlRow>
    </Figure>
  )
}
