import { useMemo } from 'react'
import { basisPursuit, iterativeHardThresholding, orthogonalMatchingPursuit } from 'aifn-compute/signal/sparse'
import { dctMatrix } from 'aifn-compute/foundation/fourier'
import { child, permutation, stream, uniform } from 'aifn-compute/foundation/random'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import {
  Bars,
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

const N = 128
/** The cosines are drawn from DCT indices 1 to this; most lie above the Nyquist rate of a few dozen even samples. */
const BAND = 100
const C = toArray(dctMatrix(N)) as number[][]
const TIME = Array.from({ length: N }, (_, i) => i)

type Method = 'bp' | 'omp' | 'iht' | 'l2'

/** An s-sparse DCT spectrum and m sample times, random or evenly spaced, from one seed. */
function problem(s: number, m: number, seed: number, sampling: 'random' | 'even') {
  const r = stream(seed)
  const freqs = toFlat(permutation(child(r, 'freqs'), BAND))
    .slice(0, s)
    .map((f) => f + 1)
  const amps = child(r, 'amps')
  const c = new Array<number>(N).fill(0)
  for (const f of freqs) c[f] = (uniform(amps) < 0.5 ? -1 : 1) * (0.6 + uniform(amps))
  // x = Cᵀc: the signal is the inverse DCT of its spectrum.
  const x = TIME.map((i) => C.reduce((acc, row, k) => acc + row[i] * c[k], 0))
  const times =
    sampling === 'even'
      ? Array.from({ length: m }, (_, j) => Math.floor((j * N) / m))
      : toFlat(permutation(child(r, 'times'), N))
          .slice(0, m)
          .sort((a, b) => a - b)
  // Row r of A is the inverse DCT evaluated at time t_r: y = A c holds the samples.
  const A = times.map((t) => C.map((row) => row[t]))
  return { c, x, times, A, y: times.map((t) => x[t]) }
}

function recover(method: Method, A: number[][], y: number[], s: number): number[] {
  if (method === 'l2') {
    // The rows of A are orthonormal (rows of an orthogonal matrix), so the minimum-norm solution is Aᵀy.
    const c = new Array<number>(N).fill(0)
    A.forEach((row, r) => row.forEach((v, k) => (c[k] += v * y[r])))
    return c
  }
  const res =
    method === 'bp'
      ? basisPursuit(A, y, { method: 'interior-point' })
      : method === 'omp'
        ? orthogonalMatchingPursuit(A, y, { tolerance: 1e-6 })
        : iterativeHardThresholding(A, y, { sparsity: s, maxSteps: 2000 })
  return toFlat(res.x).map((v) => (Number.isFinite(v) ? v : 0))
}

export function FewSamples() {
  const state = useFigureState({
    m: int(32, { min: 4, max: N, label: 'samples m', suggestions: [16, 24, 32, 48, 64] }),
    s: int(5, { min: 1, max: 30, label: 'cosines s', suggestions: [2, 5, 10, 15] }),
    sampling: choice(
      [
        { value: 'random', label: 'random times' },
        { value: 'even', label: 'evenly spaced' },
      ],
      'random',
      { label: 'sampling' },
    ),
    method: choice(
      [
        { value: 'bp', label: 'basis pursuit (least ℓ1)' },
        { value: 'omp', label: 'orthogonal matching pursuit' },
        { value: 'iht', label: 'iterative hard thresholding (knows s)' },
        { value: 'l2', label: 'least energy (least ℓ2)' },
      ],
      'bp',
      { label: 'recovery' },
    ),
    seed: int(1, { min: 1, max: 9999, label: 'seed' }),
  })
  const { m, s, sampling, method, seed } = state
  const p = useMemo(() => problem(s, m, seed, sampling), [s, m, seed, sampling])
  const ch = useComputed(() => recover(method, p.A, p.y, s), [method, p, s]).value
  const xh = useMemo(() => TIME.map((i) => C.reduce((acc, row, k) => acc + row[i] * ch[k], 0)), [ch])

  const err = Math.sqrt(p.c.reduce((a, v, k) => a + (v - ch[k]) ** 2, 0) / p.c.reduce((a, v) => a + v * v, 0))
  const nonzero = ch.filter((v) => Math.abs(v) > 1e-6).length
  const truthK = useMemo(() => TIME.filter((k) => p.c[k] !== 0), [p])
  const top = Math.max(...p.c.map(Math.abs), ...ch.map(Math.abs)) * 1.1
  const lim = Math.max(...p.x.map(Math.abs), ...xh.map(Math.abs)) * 1.1

  const tx = useAxis({ label: 'time sample i', range: [-1, N] })
  const ty = useAxis({ label: 'signal', range: [-lim, lim], key: `${seed}|${s}|${method}` })
  const kx = useAxis({ label: 'DCT index k', range: [-1, N] })
  const ky = useAxis({ label: 'coefficient', range: [-top, top], key: `${seed}|${s}|${method}` })

  return (
    <Figure
      title="A whole signal from a few samples"
      state={state}
      defaultSize="L"
      caption="A signal of 128 samples made of s cosines (its DCT has s non-zero coefficients) is observed only at the m dots. Top: the true signal (grey) and the reconstruction (blue). Bottom: the true DCT coefficients (dots) and the recovered ones (bars). Basis pursuit and orthogonal matching pursuit recover the signal exactly once m is a few times s, even though most cosines oscillate faster than the dots could follow if they were evenly spaced. The least-energy solution never does: it is zero at every unobserved time. Switch to evenly spaced samples and raise the seed to see the regular pattern fail more often."
      readouts={
        <>
          <Readout label="m / n" value={formatNumber(m / N)} />
          <Readout label="relative error" value={err < 1e-6 ? '< 10⁻⁶' : formatNumber(err)} />
          <Readout label="non-zero coefficients" value={`${nonzero} (true ${s})`} />
        </>
      }
    >
      <div className="flex flex-col gap-2">
        <Plot x={tx} y={ty} height={230}>
          <Curve name="true signal" x={TIME} y={p.x} muted />
          <Curve name="reconstruction" x={TIME} y={xh} slot={0} />
          <Points name="samples" x={p.times} y={p.y} emphasis size={8} />
        </Plot>
        <Plot x={kx} y={ky} height={200}>
          <Bars name="recovered" x={TIME} y={ch} slot={0} width={0.8} />
          <Points name="true" x={truthK} y={truthK.map((k) => p.c[k])} emphasis size={7} />
        </Plot>
      </div>
    </Figure>
  )
}
