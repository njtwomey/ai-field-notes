import { normals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { studentTCdf } from 'aifn-compute/numerics/special'
import { Normal } from 'aifn-compute/probability/distributions'
import { confidenceSequence, msprt, oneSampleTTest, sprt, waldBoundaries } from 'aifn-compute/probability/tests'
import { useMemo } from 'react'
import { Player, usePlayhead } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { row, slider, useFigureState } from 'aifn-render/state'
import { Annotation, Area, Curve, formatNumber, Plot, Plots, Points, Readout, useAxis } from 'aifn-render/viz'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/** n standard-normal observations shifted by δ, one stream per simulated experiment. */
const observations = (key: string, n: number, delta: number) =>
  Float64Array.from(toFlat(normals(stream(key), n, delta, 1)))

// ── 1 · Type-I error under peeking: a simulation ───────────────────────────────────────────────────────────────────

export function PeekingSimulationSpecimen() {
  const state = useFigureState({
    design: row('1 · experiment', {
      horizon: slider(20, 1000, 300, { step: 10, label: 'horizon N' }),
      every: slider(1, 50, 10, { step: 1, label: 'peek every k observations' }),
      delta: slider(0, 0.6, 0, { step: 0.01, label: 'true effect δ (0: an A/A test)' }),
    }),
    test: row('2 · tests', {
      alpha: slider(0.01, 0.2, 0.05, { step: 0.01, label: 'level α' }),
      tau: slider(0.05, 2, 0.3, { step: 0.05, label: 'mixture sd τ (mSPRT)' }),
      runs: slider(100, 1000, 400, { step: 50, label: 'simulated experiments' }),
      seed: slider(1, 20, 1, { step: 1, label: 'seed' }),
    }),
  })
  const { horizon: N, every, delta } = state.design
  const { alpha, tau, runs, seed } = state.test

  const sim = useMemo(() => {
    // The first n at which each procedure rejects (Infinity when it never does).
    const naiveAt: number[] = []
    const mixtureAt: number[] = []
    let fixed = 0
    for (let r = 0; r < runs; r++) {
      const x = observations(`peeking/${seed}/${r}`, N, delta)
      // Naive: a two-sided one-sample t-test at every peek, stopping at the first p ≤ α.
      let sum = 0
      let sq = 0
      let first = Infinity
      for (let n = 1; n <= N; n++) {
        sum += x[n - 1]
        sq += x[n - 1] ** 2
        if (n < 2 || (n % every !== 0 && n !== N)) continue
        const mean = sum / n
        const sd = Math.sqrt(Math.max(0, (sq - n * mean * mean) / (n - 1)))
        const t = mean / (sd / Math.sqrt(n))
        const p = 2 * (studentTCdf(-Math.abs(t), n - 1) as number)
        if (p <= alpha && first === Infinity) first = n
        if (n === N && p <= alpha) fixed++
      }
      naiveAt.push(first)
      const m = run(msprt(x, { sigma: 1, tau, alpha }), undefined, N)
      mixtureAt.push(m.rejected ? m.n : Infinity)
    }
    const grid = Array.from({ length: Math.ceil(N / every) }, (_, i) => Math.min(N, (i + 1) * every))
    const share = (at: number[]) => grid.map((n) => at.filter((a) => a <= n).length / runs)
    return { grid, naive: share(naiveAt), mixture: share(mixtureAt), fixed: fixed / runs }
  }, [N, every, delta, alpha, tau, runs, seed])

  const xa = useAxis({ label: 'observations so far n', range: [0, N] })
  const ya = useAxis({
    label: delta === 0 ? 'share rejected (type-I error)' : 'share rejected (power)',
    range: [0, delta === 0 ? undefined : 1],
    hold: 'union',
    key: `${delta === 0}`,
  })
  const last = sim.grid.length - 1
  return (
    <Figure
      title="Peeking inflates the type-I error; an always-valid test does not"
      purpose="A fixed-horizon test holds α only when it is read once. Reading it at every peek and stopping at the first p ≤ α gives the null hypothesis many chances to look rejected, so the error grows with the number of peeks. The mixture SPRT is valid at any stopping time: its error stays below α however often it is read."
      state={state}
      readouts={
        <>
          <Readout label="naive peeking, rejected by N" value={fmt(sim.naive[last])} />
          <Readout label="mSPRT, rejected by N" value={fmt(sim.mixture[last])} />
          <Readout label="one test at N" value={fmt(sim.fixed)} />
          <Readout label="peeks" value={sim.grid.length} />
        </>
      }
      caption="Each experiment draws N observations from N(δ, 1). With δ = 0 every rejection is a false positive: the naive curve climbs well past α as peeks accumulate (more peeks, higher curve), while the mSPRT curve stays under the α line. With δ > 0 the curves become power: the always-valid test pays for its guarantee with some power, more for a mixture sd τ far from δ."
    >
      <Plot x={xa} y={ya}>
        <Curve name="naive t-test at every peek" x={sim.grid} y={sim.naive} slot={0} />
        <Curve name="mSPRT (always valid)" x={sim.grid} y={sim.mixture} slot={1} />
        <Points name="one t-test at N" x={[N]} y={[sim.fixed]} emphasis />
        <Annotation y={alpha} text={`α = ${alpha}`} dashed />
      </Plot>
    </Figure>
  )
}

// ── 2 · One experiment, observation by observation ─────────────────────────────────────────────────────────────────

export function PeekingPlayerSpecimen() {
  const state = useFigureState({
    data: row('1 · data', {
      delta: slider(0, 1, 0.25, { step: 0.01, label: 'true effect δ' }),
      horizon: slider(20, 500, 200, { step: 10, label: 'observations N' }),
      seed: slider(1, 50, 3, { step: 1, label: 'seed' }),
    }),
    test: row('2 · tests', {
      alpha: slider(0.01, 0.2, 0.05, { step: 0.01, label: 'level α' }),
      tau: slider(0.05, 2, 0.3, { step: 0.05, label: 'mixture sd τ' }),
      effect: slider(0.05, 1, 0.3, { step: 0.05, label: 'SPRT alternative δ₁' }),
      beta: slider(0.05, 0.5, 0.2, { step: 0.05, label: 'SPRT β' }),
    }),
  })
  const { delta, horizon: N, seed } = state.data
  const { alpha, tau, effect, beta } = state.test
  const x = useMemo(() => observations(`peek-player/${seed}`, N, delta), [N, delta, seed])

  const paths = useMemo(() => {
    const cs = trace(confidenceSequence(x, { sigma: 1, tau, alpha }), undefined, N, { keep: 'all' }).steps.slice(1)
    // The always-valid p-value of the mixture: the running minimum of 1/Λₙ.
    const mixtureP: number[] = []
    for (const s of cs) mixtureP.push(Math.min(mixtureP[mixtureP.length - 1] ?? 1, Math.exp(-s.logEValue)))
    const naive = cs.map((_, i) => (i < 1 ? null : oneSampleTTest(x.subarray(0, i + 1), { level: 1 - alpha })))
    const sp = trace(sprt(x, { h0: Normal(0, 1), h1: Normal(effect, 1), alpha, beta }), undefined, N, { keep: 'all' })
    return {
      n: cs.map((s) => s.n),
      mean: cs.map((s) => s.mean),
      csLower: cs.map((s) => s.lower),
      csUpper: cs.map((s) => s.upper),
      tLower: naive.map((r) => r?.ci?.lower ?? NaN),
      tUpper: naive.map((r) => r?.ci?.upper ?? NaN),
      naiveP: naive.map((r) => r?.pValue ?? NaN),
      mixtureP,
      llr: sp.steps.slice(1).map((s) => s.llr),
      sprtStop: sp.final.decision === 'continue' ? null : { n: sp.final.t, decision: sp.final.decision },
    }
  }, [x, N, tau, alpha, effect, beta])

  const [at, setAt] = usePlayhead(N)
  const upto = at + 1
  const cut = <T,>(a: T[]) => a.slice(0, upto)
  const firstNaive = paths.naiveP.findIndex((p) => p <= alpha)
  const firstMixture = paths.mixtureP.findIndex((p) => p <= alpha)
  const wald = waldBoundaries(alpha, beta)

  const nAxis = useAxis({ label: 'observations n', range: [1, N] })
  const meanAxis = useAxis({ label: 'mean', range: [-1.5, 1.5] })
  const pAxis = useAxis({ label: 'p-value', log: true, range: [1e-4, 1] })
  const llrAxis = useAxis({ label: 'SPRT log-LR', range: [wald.lower - 1, wald.upper + 1] })
  const clipP = (p: number[]) => p.map((v) => (Number.isFinite(v) ? Math.max(1e-4, v) : NaN))
  return (
    <Figure
      title="Peeking at one experiment"
      purpose="As observations arrive, the fixed-n t interval and p-value are recomputed at every n; the confidence sequence and the mixture p-value are built to be read at every n."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="3 · observations">
          <Player
            className="col-span-full"
            value={at}
            onChange={setAt}
            count={N}
            label="n"
            format={(i) => `${i + 1}`}
          />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="n" value={upto} />
          <Readout label="mean" value={fmt(paths.mean[at])} />
          <Readout label="t-test p (this n)" value={fmt(paths.naiveP[at])} />
          <Readout label="mSPRT p (always valid)" value={fmt(paths.mixtureP[at])} />
          <Readout label="first naive p ≤ α" value={firstNaive >= 0 ? `n = ${firstNaive + 1}` : 'never'} />
          <Readout label="first mSPRT p ≤ α" value={firstMixture >= 0 ? `n = ${firstMixture + 1}` : 'never'} />
          <Readout
            label="SPRT"
            value={paths.sprtStop ? `${paths.sprtStop.decision} at n = ${paths.sprtStop.n}` : 'no decision'}
          />
        </>
      }
      caption="Play, or step with the arrows. Top: the running mean with the 1 − α t interval (recomputed at each n, so it may exclude 0 at some n by chance) and the confidence sequence (valid at every n at once, so wider). Middle: the t-test's p-value wanders and dips below α by chance under the null; the mixture p-value only falls. Bottom: Wald's SPRT log-likelihood ratio of N(δ₁, 1) against N(0, 1), which stops at the first boundary it crosses. Set δ = 0 and change the seed to watch false alarms appear in the naive p-value."
    >
      <Plots rows={3} heights={[40, 32, 28]} hoverGroup>
        <Plot x={nAxis} y={meanAxis}>
          <Area
            name="t interval (fixed n)"
            x={cut(paths.n)}
            y={cut(paths.tUpper)}
            base={cut(paths.tLower)}
            slot={0}
            opacity={0.2}
            line={false}
          />
          <Area
            name="confidence sequence"
            x={cut(paths.n)}
            y={cut(paths.csUpper).map((v) => Math.min(v, 5))}
            base={cut(paths.csLower).map((v) => Math.max(v, -5))}
            slot={1}
            opacity={0.25}
            line={false}
          />
          <Curve name="running mean" x={cut(paths.n)} y={cut(paths.mean)} emphasis />
          <Annotation y={0} text="null mean 0" dashed />
          <Annotation y={delta} text="true δ" />
        </Plot>
        <Plot x={nAxis} y={pAxis}>
          <Curve name="t-test p at this n" x={cut(paths.n)} y={clipP(cut(paths.naiveP))} slot={0} />
          <Curve name="mSPRT always-valid p" x={cut(paths.n)} y={clipP(cut(paths.mixtureP))} slot={1} />
          <Annotation y={alpha} text={`α = ${alpha}`} dashed />
        </Plot>
        <Plot x={nAxis} y={llrAxis}>
          <Curve
            name="SPRT log-likelihood ratio"
            x={cut(paths.n).slice(0, paths.llr.length)}
            y={cut(paths.llr)}
            slot={2}
          />
          <Annotation y={wald.upper} text="reject H₀" dashed />
          <Annotation y={wald.lower} text="accept H₀" dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
