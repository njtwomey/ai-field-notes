import { useContext, useMemo, useState, type ReactNode } from 'react'
import {
  Annotation,
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  FrameContext,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  useScaleColor,
} from 'aifn-render'
import {
  coverageProblem,
  grpoTrace,
  passAtKExact,
  solveProbabilities,
  type GrpoOptions,
} from 'aifn-methods/neural/post-training'

const PROMPTS = 40
const STRATEGIES = 8
const K_MAX = 256
const KS = Array.from({ length: K_MAX }, (_, i) => i + 1)
/** A prompt counts as abandoned when its single-sample success rate falls below this. */
const ABANDONED = 0.01
const TOP = 230
const BOTTOM = 170

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

type Variant = 'grpo' | 'dr-grpo' | 'dapo'

function variantOptions(variant: Variant): Partial<GrpoOptions> {
  if (variant === 'dr-grpo') return { advantage: { scale: 'none' } }
  if (variant === 'dapo') return { clip: { low: 0.2, high: 0.28 }, dynamicSampling: true }
  return {}
}

const mean = (xs: ArrayLike<number>) => {
  let s = 0
  for (let i = 0; i < xs.length; i++) s += xs[i]
  return s / xs.length
}

/** pass@k averaged over prompts, from each prompt's single-sample success rate. */
const passAt = (p: Float64Array, k: number) => mean(Array.from(p, (px) => passAtKExact(px, k)))

export function SharpeningCoverage() {
  const state = useFigureState({
    variant: choice(
      [
        { value: 'grpo', label: 'GRPO' },
        { value: 'dr-grpo', label: 'Dr. GRPO (no std scaling)' },
        { value: 'dapo', label: 'DAPO (clip 0.2 / 0.28, dynamic sampling)' },
      ],
      'grpo',
      { label: 'variant' },
    ),
    G: int(8, { ge: 2, le: 64, suggestions: [2, 4, 8, 16, 32, 64], label: 'group size G' }),
    lr: float(0.1, { gt: 0, le: 1, scale: 'log10', suggestions: [0.03, 0.1, 0.3], label: 'learning rate' }),
    steps: int(80, { ge: 10, le: 300, suggestions: [40, 80, 160, 300], label: 'steps' }),
    beta: float(0, { ge: 0, le: 1, suggestions: [0, 0.01, 0.1, 0.5], label: 'KL weight β' }),
    density: float(0.25, { ge: 0.05, le: 0.9, suggestions: [0.1, 0.25, 0.5], label: 'coverage density' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
    k: slider(1, K_MAX, 64, { step: 1, onChart: true }),
  })
  const { variant, G, lr, steps, beta, density, seed } = state
  const k = Math.round(state.k)

  const run = useComputed(
    () => {
      const problem = coverageProblem({ prompts: PROMPTS, strategies: STRATEGIES, density, seed })
      const options: GrpoOptions = { groupSize: G, learningRate: lr, seed, beta, ...variantOptions(variant) }
      const snaps = [...grpoTrace(problem, new Float64Array(STRATEGIES), options, steps)]
      const solves = (x: number, s: number) => problem.solves[x * STRATEGIES + s]
      const solvedBy = Array.from({ length: STRATEGIES }, (_, s) => {
        let c = 0
        for (let x = 0; x < PROMPTS; x++) c += solves(x, s)
        return c
      })
      const unsolvable = Array.from({ length: PROMPTS }, (_, x) => x).filter((x) =>
        solvedBy.every((_, s) => !solves(x, s)),
      ).length
      const p = snaps.map((s) => solveProbabilities(problem, s.probs))
      // Probability that a group of G is all failures or all successes: 1 − pass@G + p^G, averaged over prompts.
      const expectedZero = p.map((px) => mean(Array.from(px, (v) => 1 - passAtKExact(v, G) + v ** G)))
      // Prompts in a fixed order, by their success rate under the base policy.
      const order = Array.from({ length: PROMPTS }, (_, x) => x).sort((a, b) => p[0][b] - p[0][a])
      return {
        solvedBy,
        unsolvable,
        p,
        order,
        pi: snaps.map((s) => Array.from(s.probs[0])),
        reward: snaps.map((s) => s.meanReward),
        zero: snaps.map((s) => s.fracZeroStd),
        expectedZero,
        entropy: snaps.map((s) => s.entropy),
        kl: snaps.map((s) => s.kl),
        base: KS.map((kk) => passAt(p[0], kk)),
      }
    },
    [variant, G, lr, steps, beta, density, seed],
    { mode: 'release' },
  )
  const r = run.value

  const key = `${variant}|${G}|${lr}|${steps}|${beta}|${density}|${seed}`
  const [pos, setPos] = useState({ key, step: 0 })
  const count = r ? r.p.length : 1
  const step = pos.key === key ? Math.min(pos.step, count - 1) : 0

  const current = useMemo(() => (r ? KS.map((kk) => passAt(r.p[step], kk)) : []), [r, step])
  const stepsX = useMemo(() => (r ? r.p.map((_, i) => i) : []), [r])
  const sampledX = useMemo(() => stepsX.slice(1), [stepsX])
  const sampledZero = useMemo(() => (r ? r.zero.slice(1) : []), [r])

  const scale = useScaleColor('sequential')
  const strategies = useMemo(() => {
    if (!r) return null
    const most = Math.max(...r.solvedBy, 1)
    return {
      names: r.solvedBy.map((c, s) => `${String.fromCharCode(65 + s)}\n${c}`),
      colors: r.solvedBy.map((c) => scale(0.25 + (0.75 * c) / most)),
      x: r.solvedBy.map((_, s) => s),
    }
  }, [r, scale])

  const prompts = useMemo(() => {
    if (!r) return null
    const x = r.order.map((_, i) => i + 1)
    return { x, base: r.order.map((i) => r.p[0][i]), now: r.order.map((i) => r.p[step][i]) }
  }, [r, step])

  const abandoned = r ? r.p[0].filter((v, x) => v > 0 && r.p[step][x] < ABANDONED).length : 0

  const kx = useAxis({ label: 'samples k', log: true, range: [1, K_MAX] })
  const ky = useAxis({ label: 'pass@k (mean over prompts)', range: [0, 1] })
  const sx = useAxis({ label: 'step', integer: true, range: [0, steps], key: steps })
  const ry = useAxis({ label: 'fraction', range: [0, 1] })
  const ny = useAxis({ label: 'nats', range: [0, undefined] })
  const px = useAxis({
    label: 'strategy, and the prompts it solves',
    categories: strategies?.names ?? [],
    key,
  })
  const py = useAxis({ label: 'π(strategy)', range: [0, 1] })
  const qx = useAxis({ label: 'prompt, by base success rate', integer: true, range: [0.5, PROMPTS + 0.5] })
  const qy = useAxis({ label: 'success rate pₓ', range: [0, 1] })

  return (
    <Figure
      title="Sharpening and coverage"
      state={state}
      defaultSize="XL"
      caption={`A toy RLVR problem: ${PROMPTS} prompts and ${STRATEGIES} strategies A–H; each strategy solves each prompt with probability equal to the coverage density, drawn once from the seed. The policy is one softmax over the strategies, shared by every prompt, trained by GRPO with groups of G samples per prompt and every prompt in each step. Top left: pass@k = 1 − (1 − pₓ)ᵏ averaged over prompts, for the base policy (grey) and the policy at the step played (blue); drag the dashed line to choose k. Top right: the expected reward, which is pass@1, the fraction of groups whose rewards were all equal (dots: sampled at each step; line: expected, 1 − (1 − pₓ)ᴳ + pₓᴳ), and below it the policy's entropy and its KL divergence from the base policy. Bottom: the policy over strategies, each bar darker the more prompts its strategy solves, and each prompt's single-sample success rate pₓ under the base policy (grey) and at the step played (blue). Play the steps: the policy collapses onto the strategy that solves the most prompts, pass@1 rises towards the fraction of prompts that strategy solves, and pass@k at large k falls, because prompts that only other strategies solve lose their probability. The curves cross at small k. Prompts that no strategy solves always give equal rewards and never move the policy; as the policy collapses, most of the rest do too, solved by every sample or by none. A KL weight β of 0.05 or more keeps probability on every strategy and holds pass@256 near its base value. At one update per batch every probability ratio is 1, so DAPO's wider clip never acts; the three variants give similar curves here.`}
      controls={
        <Player
          value={step}
          onChange={(s) => setPos({ key, step: s })}
          count={count}
          label="step"
          format={(s) => (s === 0 ? 'base policy' : `step ${s} of ${count - 1}`)}
        />
      }
      readouts={
        r && (
          <>
            <Readout label={`pass@${k}, base`} value={formatNumber(r.base[k - 1])} />
            <Readout label={`pass@${k}, step ${step}`} value={formatNumber(current[k - 1])} />
            <Readout label="pass@1, base → now" value={`${formatNumber(r.base[0])} → ${formatNumber(current[0])}`} />
            <Readout label="entropy (nats)" value={formatNumber(r.entropy[step])} />
            <Readout label="prompts no strategy solves" value={String(r.unsolvable)} />
            <Readout label="prompts abandoned (pₓ < 0.01 from > 0)" value={String(abandoned)} />
          </>
        )
      }
    >
      {!r || !strategies || !prompts ? (
        <p className="py-12 text-center text-sm text-muted-foreground">Training…</p>
      ) : (
        <div className="flex flex-col gap-3">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <FixedHeight height={TOP + BOTTOM + 12}>
              <Plot x={kx} y={ky}>
                <Curve name="base policy" x={KS} y={r.base} muted width={2} stale={run.stale} />
                <Curve name={`step ${step}`} x={KS} y={current} slot={0} width={2} stale={run.stale} />
                <Points name={`pass@${k}`} x={[k, k]} y={[r.base[k - 1], current[k - 1]]} emphasis size={8} live />
                <Handle {...state.handle('k', { label: 'k' })} />
              </Plot>
            </FixedHeight>
            <div className="flex flex-col gap-3">
              <FixedHeight height={TOP}>
                <Plot x={sx} y={ry}>
                  <Curve name="expected reward (pass@1)" x={stepsX} y={r.reward} slot={0} width={2} />
                  <Points name="equal-reward groups (sampled)" x={sampledX} y={sampledZero} slot={1} size={4} />
                  <Curve name="equal-reward groups (expected)" x={stepsX} y={r.expectedZero} slot={1} width={1.5} />
                  <Annotation x={step} dashed live />
                </Plot>
              </FixedHeight>
              <FixedHeight height={BOTTOM}>
                <Plot x={sx} y={ny}>
                  <Curve name="entropy" x={stepsX} y={r.entropy} slot={2} width={2} />
                  <Curve name="KL from base" x={stepsX} y={r.kl} slot={3} width={2} />
                  <Annotation x={step} dashed live />
                </Plot>
              </FixedHeight>
            </div>
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-[2fr_3fr]">
            <FixedHeight height={BOTTOM}>
              <Plot x={px} y={py} legend={false}>
                <Bars name="π(strategy)" x={strategies.x} y={r.pi[step]} colors={strategies.colors} />
                <Annotation y={1 / STRATEGIES} dashed muted text="base" />
              </Plot>
            </FixedHeight>
            <FixedHeight height={BOTTOM}>
              <Plot x={qx} y={qy}>
                <Bars name="base policy" x={prompts.x} y={prompts.base} width={0.8} muted />
                <Bars name={`step ${step}`} x={prompts.x} y={prompts.now} width={0.45} slot={0} />
              </Plot>
            </FixedHeight>
          </div>
        </div>
      )}
    </Figure>
  )
}
