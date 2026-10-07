import { useMemo } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { orthogonalMatchingPursuitSteps, type PursuitState } from 'aifn-compute/signal/sparse'
import {
  Annotation,
  Bars,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { cosineSignal, COS, dictionary, I, M, makeTruth, split, SPK, type DictId } from './mixture'

const MAX_STEPS = 40
const MAX_ROWS = 12
const DICTS: { id: DictId; label: string }[] = [
  { id: 'spikes', label: 'spikes only' },
  { id: 'cosines', label: 'cosines only' },
  { id: 'both', label: 'spikes and cosines' },
]

/** Orthogonal matching pursuit's state after every step, held flat after it stops. */
function pursuitPath(D: number[][], y: number[]): PursuitState[] {
  const tr = trace(orthogonalMatchingPursuitSteps(D, y, { sparsity: MAX_STEPS }), undefined, MAX_STEPS, {
    keep: 'all',
    stream: stream('sparse-approximation/mixture/omp'),
  })
  const states = [...tr.steps]
  while (states.length <= MAX_STEPS) states.push(states[states.length - 1])
  return states
}

export function MixtureSeparation() {
  const state = useFigureState({
    cosines: slider(0, 12, 4, { step: 1, label: 'cosines in the signal' }),
    spikes: slider(0, 12, 5, { step: 1, label: 'spikes in the signal' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
    s: slider(0, MAX_STEPS, 9, { step: 1, label: 'atoms used s' }),
  })
  const truth = useMemo(
    () => makeTruth(state.cosines, state.spikes, state.seed),
    [state.cosines, state.spikes, state.seed],
  )
  const norm = Math.hypot(...truth.clean) || 1

  const paths = useMemo(
    () =>
      DICTS.map((d) => {
        const { D, offset } = dictionary(d.id)
        const states = pursuitPath(D, truth.clean)
        return { ...d, offset, states, error: states.map((st) => Math.max(st.residualNorm / norm, 1e-6)) }
      }),
    [truth, norm],
  )

  const union = paths[2]
  const view = useMemo(() => {
    const st = union.states[state.s]
    const { cos, spk } = split(toFlat(st.x), union.offset)
    const smooth = cosineSignal(cos)
    // The atoms in the order the pursuit picked them, each with its coefficient times its shape.
    const rows = st.support.slice(0, MAX_ROWS).map((j, r) => {
      const isCos = j < M
      const c = isCos ? cos[j] : spk[j - M]
      const shape = I.map((i) => c * (isCos ? COS[i][j] : SPK[i][j - M]))
      const peak = Math.max(...shape.map(Math.abs), 1e-12)
      return {
        j,
        isCos,
        offset: -r,
        y: shape.map((v) => -r + (0.4 * v) / peak),
        label: `${r + 1}. ${isCos ? `cosine k = ${j}` : `spike at i = ${j - M}`}, coefficient ${formatNumber(c)}`,
      }
    })
    const found = new Set(st.support)
    const wanted = [...I.filter((k) => truth.cos[k] !== 0), ...I.filter((i) => truth.spk[i] !== 0).map((i) => i + M)]
    const exact = wanted.length === found.size && wanted.every((j) => found.has(j))
    return { smooth, spk, approx: I.map((i) => smooth[i] + spk[i]), rows, exact }
  }, [union, state.s, truth])

  const steps = useMemo(() => Array.from({ length: MAX_STEPS + 1 }, (_, s) => s), [])
  const total = state.cosines + state.spikes

  const tx = useAxis({ label: 'sample i', range: [-0.5, M - 0.5] })
  const ty = useAxis({ label: 'value', key: `${state.cosines}/${state.spikes}/${state.seed}` })
  const ex = useAxis({ label: 'atoms used s', range: [0, MAX_STEPS] })
  const ey = useAxis({ label: 'relative error', log: true, range: [1e-6, 1] })
  const ax = useAxis({ label: 'sample i', range: [0, M - 1] })
  const ay = useAxis({ label: 'atoms picked, in order', key: view.rows.length })
  return (
    <Figure
      title="An overcomplete dictionary separates a mixture"
      state={state}
      caption="A signal of 64 samples (grey dots) made of a few cosines and a few spikes, coded by orthogonal matching pursuit with s atoms. Top right: the relative error after s atoms over the 64 spikes (blue), the 64 cosines (orange), and their union of 128 atoms (black). Drag the vertical line to change s. Over one basis the error falls slowly, because each spike needs many cosines and the smooth part needs many spikes. Over the union it drops to zero once s reaches the number of atoms in the signal. Top left: the union's approximation (dashed) split into its cosine part (orange line) and its spike part (blue stems). Bottom: the atoms the union's pursuit has picked, in order, each multiplied by its coefficient and scaled to fit its row."
      readouts={
        <>
          <Readout label="atoms in the signal" value={`${total} (${state.cosines} cosines, ${state.spikes} spikes)`} />
          {paths.map((p) => (
            <Readout
              key={p.id}
              label={`error with ${state.s} atoms, ${p.label}`}
              value={formatNumber(p.states[state.s].residualNorm / norm)}
            />
          ))}
          <Readout label="union found exactly the true atoms" value={view.exact ? 'yes' : 'no'} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={tx} y={ty} height={260}>
          <Points name="signal" x={I} y={truth.clean} muted size={4} />
          <Curve name="cosine part" x={I} y={view.smooth} slot={1} width={2.5} />
          <Bars name="spike part" x={I} y={view.spk} slot={0} width={0.4} />
          <Curve name="approximation" x={I} y={view.approx} emphasis width={1} dashed />
        </Plot>
        <Plot x={ex} y={ey} height={260}>
          <Curve name="spikes only" x={steps} y={paths[0].error} slot={0} />
          <Curve name="cosines only" x={steps} y={paths[1].error} slot={1} />
          <Curve name="spikes and cosines" x={steps} y={paths[2].error} emphasis width={2.5} />
          <Handle {...state.handle('s', { label: 'atoms used' })} />
        </Plot>
      </div>
      {view.rows.length > 0 && (
        <Plot x={ax} y={ay} height={60 + 40 * view.rows.length} legend={false}>
          {view.rows.map((r) => (
            <Annotation key={`base-${r.j}`} y={r.offset} text={r.label} muted />
          ))}
          {view.rows.map((r) => (
            <Curve key={r.j} name={r.label} x={I} y={r.y} slot={r.isCos ? 1 : 0} width={1.5} />
          ))}
        </Plot>
      )}
    </Figure>
  )
}
