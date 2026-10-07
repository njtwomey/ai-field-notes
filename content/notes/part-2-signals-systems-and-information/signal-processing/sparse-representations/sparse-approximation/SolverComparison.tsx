import { useMemo } from 'react'
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  basisPursuit,
  basisPursuitDenoising,
  iterativeHardThresholding,
  matchingPursuit,
  mutualCoherence,
  orthogonalMatchingPursuit,
  type SparseApproximation,
} from 'aifn-compute/signal/sparse'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  when,
} from 'aifn-render'
import { cosineSignal, dictionary, I, M, makeTruth, split, type DictId } from './mixture'

type Method = 'omp' | 'mp' | 'bp' | 'lasso' | 'iht'

function solve(
  D: number[][],
  y: number[],
  method: Method,
  s: number,
  steps: number,
  lambda: number,
): SparseApproximation {
  switch (method) {
    case 'omp':
      return orthogonalMatchingPursuit(D, y, { sparsity: Math.min(s, M) })
    case 'mp':
      return matchingPursuit(D, y, { maxSteps: steps })
    case 'bp':
      // The interior-point method: the simplex method is far slower on a 64 × 128 basis pursuit.
      return basisPursuit(D, y, { method: 'interior-point' })
    case 'lasso':
      return basisPursuitDenoising(D, y, { lambda })
    case 'iht':
      return iterativeHardThresholding(D, y, { sparsity: s })
  }
}

export function SolverComparison() {
  const state = useFigureState({
    dict: choice(
      [
        { value: 'spikes', label: 'spikes (64)' },
        { value: 'cosines', label: 'cosines (64)' },
        { value: 'both', label: 'spikes and cosines (128)' },
      ],
      'both',
      { label: 'dictionary' },
    ),
    method: choice(
      [
        { value: 'omp', label: 'orthogonal matching pursuit' },
        { value: 'mp', label: 'matching pursuit' },
        { value: 'bp', label: 'basis pursuit' },
        { value: 'lasso', label: 'basis pursuit denoising (lasso)' },
        { value: 'iht', label: 'iterative hard thresholding' },
      ],
      'omp',
      { label: 'method' },
    ),
    s: slider(1, 32, 7, { step: 1, label: 'non-zeros s', when: (v) => v.method === 'omp' || v.method === 'iht' }),
    steps: int(20, {
      ge: 1,
      le: 500,
      scale: 'log10',
      suggestions: [5, 20, 100, 500],
      label: 'steps',
      when: when('method', 'mp'),
    }),
    lambda: float(0.1, {
      ge: 0.001,
      le: 10,
      scale: 'log10',
      suggestions: [0.01, 0.1, 1],
      label: 'penalty λ',
      when: when('method', 'lasso'),
    }),
    cosines: slider(0, 16, 3, { step: 1, label: 'cosines in the signal' }),
    spikes: slider(0, 16, 4, { step: 1, label: 'spikes in the signal' }),
    noise: float(0, { ge: 0, le: 0.5, suggestions: [0, 0.05, 0.2], label: 'noise σ' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const dict = state.dict as DictId
  const method = state.method as Method
  const { D, offset } = useMemo(() => dictionary(dict), [dict])
  const mu = useMemo(() => mutualCoherence(D), [D])
  const truth = useMemo(
    () => makeTruth(state.cosines, state.spikes, state.seed),
    [state.cosines, state.spikes, state.seed],
  )
  const y = useMemo(() => truth.clean.map((v, i) => v + state.noise * truth.noise[i]), [truth, state.noise])

  const result = useComputed(
    () => {
      const r = solve(D, y, method, state.s, state.steps, state.lambda)
      return { ...r, x: toFlat(r.x) }
    },
    [D, y, method, state.s, state.steps, state.lambda],
    { mode: 'release' },
  ).value

  const view = useMemo(() => {
    const x = result.x
    const { cos, spk } = split(x, offset)
    const cosSignal = cosineSignal(cos)
    const approx = I.map((i) => cosSignal[i] + spk[i])
    const nonzeros = x.filter((v) => Math.abs(v) > 1e-6 * Math.max(...x.map(Math.abs))).length
    const err = Math.hypot(...I.map((i) => approx[i] - truth.clean[i])) / Math.hypot(...truth.clean)
    const trueCos = I.filter((k) => truth.cos[k] !== 0)
    const trueSpk = I.filter((i) => truth.spk[i] !== 0)
    return { cos, spk, cosSignal, approx, nonzeros, err, trueCos, trueSpk }
  }, [result, offset, truth])

  const tx = useAxis({ label: 'sample i', range: [-0.5, M - 0.5] })
  const ty = useAxis({ label: 'value' })
  const kx = useAxis({ label: 'atom: cosine frequency k, or spike position i', range: [-0.5, M - 0.5] })
  const ky = useAxis({ label: 'coefficient' })
  return (
    <Figure
      title="One signal, five solvers, three dictionaries"
      state={state}
      caption="A signal of 64 samples made of a few cosines and a few spikes (grey dots), approximated over the spikes, the cosines of the orthonormal DCT, or both together (128 atoms). Top: the approximation (dashed) and its cosine part (orange line) and spike part (blue stems). Bottom: the coefficients each solver finds, cosines in orange and spikes in blue, with the true coefficients as black dots. Over one basis alone the code is dense; over the union, every method separates the two parts when the signal has few enough atoms, and fails in its own way when it has too many or when the penalty or step budget is wrong."
      readouts={
        <>
          <Readout label="non-zeros found" value={`${view.nonzeros} (true ${state.cosines + state.spikes})`} />
          <Readout label="relative error against the clean signal" value={formatNumber(view.err)} />
          <Readout label="residual norm" value={formatNumber(result.residualNorm)} />
          <Readout label="coherence μ" value={formatNumber(mu)} />
          <Readout label="guaranteed recovery below" value={`${formatNumber(0.5 * (1 + 1 / mu))} non-zeros`} />
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <Plot x={tx} y={ty} height={240}>
          <Points name="signal y" x={I} y={y} muted size={4} />
          {offset.cos >= 0 && <Curve name="cosine part" x={I} y={view.cosSignal} slot={1} width={2.5} />}
          {offset.spk >= 0 && <Bars name="spike part" x={I} y={view.spk} slot={0} width={0.35} />}
          <Curve name="approximation Dx" x={I} y={view.approx} emphasis width={1} dashed />
        </Plot>
        <Plot x={kx} y={ky} height={220}>
          {offset.cos >= 0 && (
            <Bars name="cosine coefficients" x={I.map((k) => k - 0.2)} y={view.cos} slot={1} width={0.4} />
          )}
          {offset.spk >= 0 && (
            <Bars name="spike coefficients" x={I.map((i) => i + 0.2)} y={view.spk} slot={0} width={0.4} />
          )}
          <Points
            name="true cosine coefficients"
            x={view.trueCos.map((k) => k - 0.2)}
            y={view.trueCos.map((k) => truth.cos[k])}
            emphasis
            size={6}
          />
          <Points
            name="true spikes"
            x={view.trueSpk.map((i) => i + 0.2)}
            y={view.trueSpk.map((i) => truth.spk[i])}
            emphasis
            size={6}
          />
        </Plot>
      </div>
    </Figure>
  )
}
