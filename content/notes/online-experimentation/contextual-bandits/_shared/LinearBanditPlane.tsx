import { useEffect, useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import {
  HORIZON,
  dot,
  ellipse,
  estimate,
  inverse,
  play,
  width,
  type ArmMode,
  type LinearPolicy,
  type Vec,
} from './linear'

const RUNS = 12
const RANGE: [number, number] = [-1.6, 1.6]
const POLICIES: { id: LinearPolicy; label: string; slot: number }[] = [
  { id: 'linucb', label: 'LinUCB', slot: 0 },
  { id: 'lints', label: 'linear Thompson sampling', slot: 1 },
  { id: 'greedy', label: 'greedy ridge', slot: 3 },
]

/** Mean cumulative regret of every policy over RUNS seeds. Plain values in, so the memo is exact. */
function averageRegret(t0: number, t1: number, mode: ArmMode, lambda: number, alpha: number, v: number, seed: number) {
  const settings = { mode, lambda, alpha, v }
  return new Map(
    POLICIES.map((p) => {
      const sum = new Array<number>(HORIZON).fill(0)
      for (let run = 0; run < RUNS; run++) {
        play(p.id, [t0, t1], settings, seed * 100 + run, rng, false).regret.forEach((r, i) => (sum[i] += r / RUNS))
      }
      return [p.id, sum] as const
    }),
  )
}

/**
 * A two-dimensional linear bandit seen in parameter space: the true θ* (draggable), the ridge estimate, the confidence
 * ellipse (LinUCB) or posterior ellipse (linear Thompson sampling), the arms of the current round as arrows, and the
 * regret of LinUCB, linear Thompson sampling and greedy ridge regression averaged over seeded runs.
 */
export function LinearBanditPlane({ policy }: { policy: 'linucb' | 'lints' }) {
  const [theta, setTheta] = useState<Vec>([-0.25, 0.7])
  const [mode, setMode] = useState<ArmMode>('fixed')
  const [round, setRound] = useState(10)
  const [playing, setPlaying] = useState(false)
  const alpha = useParam(1, { min: 0, max: 3, step: 0.05 })
  const v = useParam(0.3, { min: 0, max: 1.5, step: 0.05 })
  const lambda = useParam(1, { min: 0.05, max: 5, step: 0.05 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })

  const [t0, t1] = theta
  const run = useMemo(
    () =>
      play(
        policy,
        [t0, t1],
        { mode, lambda: lambda.value, alpha: alpha.value, v: v.value },
        seed.value * 100,
        rng,
        true,
      ),
    [policy, t0, t1, mode, lambda.value, alpha.value, v.value, seed.value],
  )
  const regret = useMemo(
    () => averageRegret(t0, t1, mode, lambda.value, alpha.value, v.value, seed.value),
    [t0, t1, mode, lambda.value, alpha.value, v.value, seed.value],
  )

  const isPlaying = playing && round < HORIZON - 1
  useEffect(() => {
    if (!isPlaying) return
    const id = setInterval(() => setRound((r) => Math.min(HORIZON - 1, r + 1)), 150)
    return () => clearInterval(id)
  }, [isPlaying])

  const current = run.rounds[round]
  const s = current.stats
  const hat = estimate(s)
  const x = current.arms[current.chosen]
  const radius = policy === 'linucb' ? alpha.value : v.value
  const ring = ellipse(s, hat, Math.max(radius, 1e-6))
  const slot = policy === 'linucb' ? 0 : 1
  const values = current.arms.map((a) => dot(a, theta))
  const bestArm = values.indexOf(Math.max(...values))

  // LinUCB's optimistic parameter for the chosen arm: the point of the ellipse furthest along x, θ̂ + α V⁻¹x / ‖x‖.
  const [a, c, d] = inverse(s)
  const w = width(s, x)
  const optimistic: Vec = [
    hat[0] + (alpha.value * (a * x[0] + c * x[1])) / w,
    hat[1] + (alpha.value * (c * x[0] + d * x[1])) / w,
  ]

  const plane: XYSeries[] = [
    {
      name: policy === 'linucb' ? 'confidence ellipse ‖θ − θ̂‖_V = α' : 'posterior ellipse, one standard deviation',
      type: 'line',
      x: ring.map((p) => p[0]),
      y: ring.map((p) => p[1]),
      slot,
    },
    ...(policy === 'lints'
      ? [
          {
            name: 'two standard deviations',
            type: 'line' as const,
            x: ellipse(s, hat, 2 * Math.max(radius, 1e-6)).map((p) => p[0]),
            y: ellipse(s, hat, 2 * Math.max(radius, 1e-6)).map((p) => p[1]),
            slot,
            dashed: true,
          },
        ]
      : []),
    { name: 'ridge estimate θ̂', type: 'scatter', x: [hat[0]], y: [hat[1]], slot },
    policy === 'linucb'
      ? { name: 'optimistic θ for the pulled arm', type: 'scatter', x: [optimistic[0]], y: [optimistic[1]], slot: 2 }
      : { name: 'posterior draw θ̃', type: 'scatter', x: [current.draw![0]], y: [current.draw![1]], slot: 2 },
    { name: 'pulled arm', type: 'line', x: [0, x[0]], y: [0, x[1]], slot: 4 },
    { name: 'true θ* (drag)', type: 'scatter', x: [theta[0]], y: [theta[1]], emphasis: true },
  ]
  const handles: Handle[] = [
    {
      kind: 'point',
      at: theta,
      label: 'θ*',
      onDrag: ([px, py]) => {
        const n = Math.hypot(px, py)
        const scale = n > 1.2 ? 1.2 / n : 1
        setTheta([Math.round(px * scale * 100) / 100, Math.round(py * scale * 100) / 100])
      },
    },
  ]

  const ts = useMemo(() => Array.from({ length: HORIZON }, (_, i) => i + 1), [])
  const regretSeries: XYSeries[] = POLICIES.map((p) => ({
    name: p.label,
    type: 'line',
    x: ts,
    y: regret.get(p.id)!,
    slot: p.slot,
  }))

  const scores =
    policy === 'linucb'
      ? current.arms.map((arm) => `${formatNumber(dot(arm, hat))} + ${formatNumber(alpha.value * width(s, arm))}`)
      : current.arms.map((arm) => formatNumber(dot(arm, current.draw!)))

  return (
    <Interactive
      title={policy === 'linucb' ? 'LinUCB in parameter space' : 'Linear Thompson sampling in parameter space'}
      caption={
        policy === 'linucb'
          ? 'Five arms (arrows) with rewards xᵀθ* plus Gaussian noise; drag the true θ* (diamond). The ellipse is the confidence set around the ridge estimate θ̂. Each arm’s index is its score at the point of the ellipse furthest in its direction, xᵀθ̂ + α‖x‖_{V⁻¹}; the marked point is that point for the pulled arm. The ellipse shrinks fastest along the directions of pulled arms. With fixed arms, greedy ridge regression (α = 0) often settles on a wrong arm; LinUCB keeps the other directions open until they are ruled out. With arms that change every round, the arms’ own variety explores, and greedy does nearly as well. Right: regret averaged over seeded runs; drag the guide to move through the rounds.'
          : 'Five arms (arrows) with rewards xᵀθ* plus Gaussian noise; drag the true θ* (diamond). The ellipses are one and two standard deviations of the Gaussian posterior N(θ̂, v²V⁻¹). Each round draws one θ̃ from it (the marked point) and pulls the arm with the largest xᵀθ̃. Early on the draws scatter widely and different arms win; as the posterior shrinks the draws cluster near θ* and the best arm wins nearly always. Right: regret averaged over seeded runs; drag the guide to move through the rounds.'
      }
      controls={
        <>
          <ParamSlider
            label="round t"
            value={round}
            onChange={(r) => {
              setPlaying(false)
              setRound(r)
            }}
            min={0}
            max={HORIZON - 1}
            step={1}
            format={(r) => String(r)}
            withArrows
          />
          {policy === 'linucb' ? (
            <ParamSlider label="confidence radius α" param={alpha} />
          ) : (
            <ParamSlider label="posterior scale v" param={v} />
          )}
          <ParamSlider label="ridge penalty λ" param={lambda} />
          <ParamChoice
            label="arms"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'fixed', label: 'fixed' },
              { value: 'context', label: 'new each round' },
            ]}
          />
          <ParamSlider label="seed" param={seed} format={(r) => String(r)} withArrows />
          <ParamButton
            onClick={() => {
              if (isPlaying) return setPlaying(false)
              if (round >= HORIZON - 1) setRound(0)
              setPlaying(true)
            }}
          >
            {isPlaying ? 'Pause' : 'Play'}
          </ParamButton>
        </>
      }
      readout={
        <>
          <Readout
            label={policy === 'linucb' ? 'arm scores xᵀθ̂ + bonus' : 'arm scores xᵀθ̃'}
            value={scores.join(' | ')}
          />
          <Readout
            label="pulled arm"
            value={`${current.chosen + 1}${current.chosen === bestArm ? ' (best)' : ` (best is ${bestArm + 1})`}`}
          />
          <Readout label="θ̂" value={`(${formatNumber(hat[0])}, ${formatNumber(hat[1])})`} />
          {POLICIES.map((p) => (
            <Readout key={p.id} label={`${p.label}: regret at T`} value={formatNumber(regret.get(p.id)!.at(-1)!)} />
          ))}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={plane}
          vectors={current.arms.map((arm) => ({ from: [0, 0] as [number, number], to: arm }))}
          xLabel="θ₁"
          yLabel="θ₂"
          xRange={RANGE}
          yRange={RANGE}
          equalAspect
          handles={handles}
        />
        <XYChart
          series={regretSeries}
          xLabel="round t"
          yLabel="cumulative regret"
          xRange={[0, HORIZON]}
          yRange={[0, undefined]}
          handles={[
            {
              kind: 'x',
              at: round,
              label: 'round',
              onDrag: (r) => {
                setPlaying(false)
                setRound(Math.max(0, Math.min(HORIZON - 1, Math.round(r))))
              },
            },
          ]}
        />
      </div>
    </Interactive>
  )
}
