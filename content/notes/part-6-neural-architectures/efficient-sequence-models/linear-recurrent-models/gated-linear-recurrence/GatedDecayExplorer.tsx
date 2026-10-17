import { useMemo } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Raster,
  Readout,
  slider,
  toggle,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { eye, mul, ones, sub, type Tensor, tensor, toFlat, zeros } from 'aifn-compute/foundation/tensor'
import { linearRecurrence, rgLru, rgLruLambda } from 'aifn-compute/nn/sequence'

const C = 8
/** a^c per channel, log-spaced in 1 − a^c across Griffin's initial range [0.9, 0.999]. */
const DECAYS = [0.9, 0.9631, 0.9864, 0.995, 0.9982, 0.999]
const D = DECAYS.length
const LAMBDA = rgLruLambda(tensor(DECAYS), C)

/** Small noise everywhere, a burst of three tokens of height 2 at `burst`, and one token of height 1 later on. */
function input(T: number, burst: number): number[] {
  const noise = toFlat(normals(stream(7), [T], 0, 0.15))
  const late = Math.min(T - 1, Math.round(0.75 * T))
  return noise.map((z, t) => (t >= burst && t < burst + 3 ? 2 : t === late ? 1 : z))
}

const halfLife = (a: number) => Math.min(1e6, Math.log(2) / -Math.log(Math.min(a, 1 - 1e-15)))

/** RG-LRU decays a_t = a^{c r_t} over a sequence, with the states they produce. */
export function GatedDecayExplorer() {
  const state = useFigureState({
    w: slider(0, 8, 4, { step: 0.5, label: 'gate weight w (r = σ(w x + b))' }),
    b: slider(-6, 2, -3, { step: 0.5, label: 'gate bias b' }),
    channel: choice<number>(
      DECAYS.map((a, i) => ({ value: i, label: `aᶜ = ${a}` })),
      0,
      { label: 'channel shown' },
    ),
    T: int(64, { min: 16, max: 256, suggestions: [32, 64, 128, 256], label: 'length T' }),
    mamba: toggle(false, 'Mamba-style input weight 1 − aₜ'),
    burst: slider(0, 252, 12, { step: 1, onChart: true }),
  })
  const { w, b, channel, T, mamba } = state
  const burst = Math.min(state.burst, T - 3)

  const r = useMemo(() => {
    const xs = input(T, burst)
    const x = tensor(
      xs.flatMap((v) => DECAYS.map(() => v)),
      [T, D],
    )
    const gated = rgLru(x, {
      Wa: mul(w, eye(D)),
      ba: mul(b, ones([D])),
      Wx: zeros([D, D]),
      bx: mul(20, ones([D])),
      Lambda: LAMBDA,
    })
    // The same channels with the recurrence gate held open (r = 1): a fixed decay a^c, as in the LRU.
    const fixed = rgLru(x, {
      Wa: zeros([D, D]),
      ba: mul(40, ones([D])),
      Wx: zeros([D, D]),
      bx: mul(20, ones([D])),
      Lambda: LAMBDA,
    })
    const decay = gated.decay as Tensor
    // Mamba's zero-order-hold weighting with the same decays: h_t = a_t h_{t−1} + (1 − a_t) x_t.
    const zoh = linearRecurrence(decay, mul(sub(1, decay), x))
    const col = (v: unknown) => {
      const f = toFlat(v as Tensor)
      return Array.from({ length: T }, (_, t) => f[t * D + channel])
    }
    const a = toFlat(decay)
    const z = DECAYS.map((_, d) => Array.from({ length: T }, (_, t) => Math.log10(halfLife(a[t * D + d]))))
    const rest = col(decay)
    return {
      xs,
      gated: col(gated.outputs),
      fixed: col(fixed.outputs),
      zoh: col(zoh),
      gate: col(gated.recurrenceGate),
      z,
      restHalfLife: halfLife(rest[Math.max(0, burst - 1)]),
      burstHalfLife: halfLife(rest[burst]),
    }
  }, [T, burst, w, b, channel])

  const t = useMemo(() => Array.from({ length: T }, (_, i) => i), [T])
  const rows = useMemo(() => DECAYS.map((_, i) => i), [])
  const xAxis = useAxis({ label: 'token t', range: [-0.5, T - 0.5] })
  const yTop = useAxis({ label: 'value' })
  const yBottom = useAxis({ label: 'channel (aᶜ from 0.9 up to 0.999)', integer: true })
  const color = useAxis({ hold: 'initial' })

  return (
    <Figure
      title="Input-dependent decay in a gated linear recurrence"
      state={state}
      caption="Six RG-LRU channels read the same input: small noise, a burst of three tokens, and one later token. The recurrence gate r = σ(w x + b) is high on large tokens and low on noise, so each channel's decay aₜ = a^(c rₜ) drops at the burst (short memory, the state is overwritten) and stays near 1 elsewhere (long memory, the state is held). The bottom panel colours each channel's memory half-life, log₁₀ of ln 2 / (−ln aₜ) tokens. The top panel shows one channel: the gated state holds the burst through the noise, while the same channel with the gate held open (fixed decay aᶜ) forgets it at its fixed rate. Drag the burst along the sequence; set w to 0 to remove the input dependence. The Mamba-style weighting uses 1 − aₜ in place of √(1 − aₜ²) and keeps a smaller state when aₜ is near 1."
      readouts={
        <>
          <Readout label="gate rₜ at the burst" value={formatNumber(r.gate[burst])} />
          <Readout label="half-life before the burst" value={`${formatNumber(r.restHalfLife)} tokens`} />
          <Readout label="half-life at the burst" value={`${formatNumber(r.burstHalfLife)} tokens`} />
          <Readout label="state at the end, gated" value={formatNumber(r.gated[T - 1])} />
          <Readout label="state at the end, fixed decay" value={formatNumber(r.fixed[T - 1])} />
        </>
      }
    >
      <Plots rows={2} heights={[3, 2]}>
        <Plot x={xAxis} y={yTop}>
          <Curve name="input xₜ" x={t} y={r.xs} muted />
          <Curve name="state hₜ, gated" x={t} y={r.gated} slot={0} />
          <Curve name="state hₜ, fixed decay aᶜ" x={t} y={r.fixed} slot={1} dashed />
          {mamba && <Curve name="state hₜ, weight 1 − aₜ" x={t} y={r.zoh} slot={2} />}
          <Handle {...state.handle('burst', { label: 'burst' })} />
        </Plot>
        <Plot x={xAxis} y={yBottom}>
          <Raster x={t} y={rows} z={r.z} colorAxis={color} valueLabel="log₁₀ half-life (tokens)" />
          <Annotation y={channel} text="shown" dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
