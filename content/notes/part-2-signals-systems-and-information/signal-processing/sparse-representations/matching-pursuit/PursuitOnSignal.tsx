import { useMemo, useState } from 'react'
import { matchingPursuitSteps, orthogonalMatchingPursuitSteps, type PursuitState } from 'aifn-compute/signal/sparse'
import { dctMatrix } from 'aifn-compute/foundation/fourier'
import { child, normals, permutation, stream, uniform } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'

/** Signal length; the dictionary is the n cosines of the DCT followed by the n spikes, 2n atoms. */
const N = 64
const MAX_STEPS = 40
const FLOOR = 1e-12
const SAMPLES = Array.from({ length: N }, (_, i) => i)
const COSINE_IDS = SAMPLES
const SPIKE_IDS = SAMPLES.map((i) => N + i)

/** The dictionary [Cᵀ | I], m × 2m, atoms as columns: cosine k is row k of the orthonormal DCT-II matrix C. */
function cosinesAndSpikes(): number[][] {
  const C = toFlat(dctMatrix(N))
  return SAMPLES.map((i) => [...SAMPLES.map((k) => C[k * N + i]), ...SAMPLES.map((j) => (i === j ? 1 : 0))])
}

const atomLabel = (j: number) => (j < N ? `cosine ${j}` : `spike at ${j - N}`)

/** Matching pursuit and orthogonal matching pursuit on a signal built from a few cosines and a few spikes. */
export function PursuitOnSignal() {
  const state = useFigureState({
    cosines: int(2, { min: 0, max: 12, label: 'cosines in the signal' }),
    spikes: int(2, { min: 0, max: 12, label: 'spikes in the signal' }),
    noise: float(0, { min: 0, max: 0.5, step: 0.01, suggestions: [0, 0.05, 0.2], label: 'noise standard deviation' }),
    seed: int(1, { min: 1, max: 99, label: 'seed' }),
    method: choice(
      [
        { value: 'mp', label: 'matching pursuit' },
        { value: 'omp', label: 'orthogonal matching pursuit' },
      ],
      'mp',
      { label: 'pursuit drawn' },
    ),
  })
  const [step, setStep] = useState(0)
  const D = useMemo(() => cosinesAndSpikes(), [])

  // The true code: a few low-to-mid frequency cosines and a few spikes, with amplitudes of either sign.
  const problem = useMemo(() => {
    const s = stream(`matching-pursuit/signal/${state.seed}`)
    const freq = toFlat(permutation(child(s, 'cosines'), 24)).slice(0, state.cosines)
    const where = toFlat(permutation(child(s, 'spikes'), N)).slice(0, state.spikes)
    const size = toFlat(uniform(child(s, 'size'), 1, 3, { shape: [24] }))
    const x0 = new Array<number>(2 * N).fill(0)
    freq.forEach((k, i) => (x0[k + 1] = 4 * size[i] * (i % 2 ? -1 : 1)))
    where.forEach((p, i) => (x0[N + p] = size[12 + i] * (i % 2 ? 1 : -1)))
    const e = toFlat(normals(child(s, 'noise'), N, 0, 1))
    const y = SAMPLES.map((i) => D[i].reduce((acc, v, j) => acc + v * x0[j], 0) + state.noise * e[i])
    const truth = x0.flatMap((v, j) => (v !== 0 ? [j] : []))
    return { y, truth }
  }, [D, state.cosines, state.spikes, state.noise, state.seed])

  const runs = useMemo(() => {
    const steps = (alg: Parameters<typeof trace>[0]) => trace(alg, undefined, MAX_STEPS).steps as PursuitState[]
    return {
      mp: steps(matchingPursuitSteps(D, problem.y)),
      omp: steps(orthogonalMatchingPursuitSteps(D, problem.y, { sparsity: MAX_STEPS })),
    }
  }, [D, problem])

  const count = Math.max(runs.mp.length, runs.omp.length)
  const t = Math.min(step, count - 1)
  const at = (list: PursuitState[]) => list[Math.min(t, list.length - 1)]
  const states = state.method === 'mp' ? runs.mp : runs.omp
  const current = at(states)
  const next = t + 1 < states.length ? states[t + 1].support.at(-1) : undefined
  const slot = state.method === 'mp' ? 0 : 1

  const view = useMemo(() => {
    const r = toFlat(current.residual)
    const approx = problem.y.map((v, i) => v - r[i])
    const c = toFlat(current.correlations).map(Math.abs)
    return { r, approx, cosines: c.slice(0, N), spikes: c.slice(N), c }
  }, [current, problem])

  const decay = useMemo(() => {
    const line = (list: PursuitState[]) => ({
      x: list.map((s) => s.t),
      y: list.map((s) => Math.max(s.residualNorm, FLOOR)),
    })
    return { mp: line(runs.mp), omp: line(runs.omp) }
  }, [runs])
  const mpNow = at(runs.mp)
  const ompNow = at(runs.omp)

  const peak = Math.max(...problem.y.map(Math.abs), 1)
  // The correlation axis fits the current step, so the next pick stands out even when every correlation is small.
  const top = Math.max(...view.c, 1e-12)
  const tAxis = useAxis({ label: 'sample i', range: [-0.5, N - 0.5] })
  const vAxis = useAxis({ label: 'value', range: [-1.15 * peak, 1.15 * peak], key: problem.y })
  const jAxis = useAxis({ label: 'atom j (cosines 0–63, spikes 64–127)', range: [-1, 2 * N] })
  const cAxis = useAxis({ label: '|correlation with r|', range: [0, 1.05 * top], key: `${state.method}/${t}` })
  const sAxis = useAxis({ label: 'step t', range: [0, count - 1], integer: true })
  const nAxis = useAxis({ label: 'residual norm ‖r‖', log: true, range: [FLOOR, 2 * runs.mp[0].residualNorm] })

  const distinct = new Set(current.support).size
  const found = runs.omp.at(-1)?.support ?? []
  const exact = found.length === problem.truth.length && problem.truth.every((j) => found.includes(j))
  return (
    <Figure
      title="Pursuit on a signal of cosines and spikes"
      state={state}
      defaultSize="L"
      caption="The dictionary holds the 64 cosines of the discrete cosine transform and the 64 spikes, 128 unit atoms for a signal of 64 samples. Top: the signal, the current approximation and the residual. Bottom left: the absolute correlation of the residual with every atom, cosines then spikes, on an axis that fits each step; the large dot marks the atom picked next, and the dots on the axis mark the atoms the signal was built from. Bottom right: the residual norm of both pursuits. For a signal of a few atoms, orthogonal matching pursuit picks exactly those atoms and stops, while matching pursuit keeps revisiting them. With ten or more cosines and spikes, orthogonal matching pursuit sometimes picks a wrong atom and needs more steps (try several seeds). With noise, both keep picking atoms to fit it."
      controls={<Player value={t} onChange={setStep} count={count} label="step" />}
      readouts={
        <>
          <Readout label="next atom" value={next === undefined ? '·' : atomLabel(next)} />
          <Readout label="atoms used / steps" value={`${distinct} / ${current.t}`} />
          <Readout label="‖r‖, matching pursuit" value={formatNumber(mpNow.residualNorm)} />
          <Readout label="‖r‖, orthogonal matching pursuit" value={formatNumber(ompNow.residualNorm)} />
          <Readout label="orthogonal matching pursuit found the signal's atoms" value={exact ? 'yes' : 'no'} />
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <Plot x={tAxis} y={vAxis} height={240}>
          <Curve name="residual r" x={SAMPLES} y={view.r} slot={2} dashed />
          <Curve name="signal y" x={SAMPLES} y={problem.y} muted width={3} />
          <Curve name="approximation Dx" x={SAMPLES} y={view.approx} slot={slot} />
        </Plot>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Plot x={jAxis} y={cAxis} height={240}>
            <Bars name="cosine atoms" x={COSINE_IDS} y={view.cosines} slot={3} width={0.9} />
            <Bars name="spike atoms" x={SPIKE_IDS} y={view.spikes} slot={4} width={0.9} />
            {next !== undefined && <Points name="picked next" x={[next]} y={[view.c[next]]} emphasis size={10} live />}
            <Points name="atoms in the signal" x={problem.truth} y={problem.truth.map(() => 0)} muted size={6} />
          </Plot>
          <Plot x={sAxis} y={nAxis} height={240}>
            <Curve name="matching pursuit" x={decay.mp.x} y={decay.mp.y} slot={0} />
            <Curve name="orthogonal matching pursuit" x={decay.omp.x} y={decay.omp.y} slot={1} />
            <Points
              name="step t"
              x={[mpNow.t, ompNow.t]}
              y={[Math.max(mpNow.residualNorm, FLOOR), Math.max(ompNow.residualNorm, FLOOR)]}
              emphasis
              size={8}
              live
            />
          </Plot>
        </div>
      </div>
    </Figure>
  )
}
