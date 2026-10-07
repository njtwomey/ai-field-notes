import { basisPursuit, basisPursuitDenoising, iterativeHardThresholding } from 'aifn-compute/signal/sparse'
import { dctMatrix } from 'aifn-compute/foundation/fourier'
import { child, normals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  Annotation,
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
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

/** Signal length m; the dictionary holds the m DCT cosines and the m spikes, k = 2m atoms. */
const M = 64
const K = 2 * M
/** The λ values and sparsities at which the error is swept. */
const LAMBDAS = Array.from({ length: 15 }, (_, i) => 0.02 * 250 ** (i / 14))
const SPARSITIES = [...Array.from({ length: 20 }, (_, i) => i + 1), 25, 30, 35, 40]
/** Coefficients below this size count as zero (the solvers return exact zeros except basis pursuit's interior point). */
const ZERO = 1e-6

const COS = toFlat(dctMatrix(M))
/** D[i][j]: entry i of atom j, cosines first (COS holds one cosine per row), then spikes. */
const D: number[][] = Array.from({ length: M }, (_, i) => [
  ...Array.from({ length: M }, (_, j) => COS[j * M + i]),
  ...Array.from({ length: M }, (_, j) => (i === j ? 1 : 0)),
])
/** Three cosines and two spikes. */
const TRUTH = new Map([
  [3, 4],
  [9, -3],
  [17, 2],
  [M + 20, 1.5],
  [M + 45, -2],
])
const TRUE_X = Array.from({ length: K }, (_, j) => TRUTH.get(j) ?? 0)
const TRUE_AT = [...TRUTH.keys()]
const apply = (x: ArrayLike<number>) => D.map((row) => row.reduce((s, v, j) => s + v * x[j], 0))
const CLEAN = apply(TRUE_X)
const NOISE = toFlat(normals(child(stream('basis-pursuit/denoising'), 'noise'), M))
const SAMPLES = Array.from({ length: M }, (_, i) => i)
const ATOMS = Array.from({ length: K }, (_, j) => j)
const norm = (v: readonly number[]) => Math.hypot(...v)

type Method = 'lasso' | 'iht' | 'bp'

const noisy = (sd: number) => CLEAN.map((c, i) => c + sd * NOISE[i])
const error = (x: ArrayLike<number>) => norm(apply(x).map((r, i) => r - CLEAN[i])) / norm(CLEAN)

function code(method: Method, y: number[], lambda: number, s: number) {
  if (method === 'lasso') return toFlat(basisPursuitDenoising(D, y, { lambda }).x)
  if (method === 'iht') return toFlat(iterativeHardThresholding(D, y, { sparsity: s }).x)
  return toFlat(basisPursuit(D, y, { method: 'interior-point' }).x)
}

/** One solve, with the atoms' correlations with the noisy signal and the error measures. */
function solve(method: Method, lambda: number, s: number, sd: number) {
  const y = noisy(sd)
  const corr = ATOMS.map((j) => Math.abs(D.reduce((acc, row, i) => acc + row[j] * y[i], 0)))
  const x = code(method, y, lambda, s)
  return {
    y,
    corr,
    x,
    recon: apply(x),
    nonZeros: x.filter((v) => Math.abs(v) > ZERO).length,
    found: TRUE_AT.filter((j) => Math.abs(x[j]) > ZERO).length,
    lambdaMax: Math.max(...corr),
    kept: [...corr].sort((a, b) => b - a)[Math.min(s, K) - 1],
    error: error(x),
    noise: norm(y.map((v, i) => v - CLEAN[i])) / norm(CLEAN),
  }
}

/** The reconstruction error across λ (the lasso; basis pursuit is its limit as λ → 0) or across s (IHT). */
function sweep(method: Method, sd: number) {
  const y = noisy(sd)
  if (method === 'iht') return { x: SPARSITIES, y: SPARSITIES.map((s) => error(code('iht', y, 0, s))) }
  return { x: LAMBDAS, y: LAMBDAS.map((lambda) => error(code('lasso', y, lambda, 0))) }
}

/** A noisy signal that is sparse over cosines and spikes, denoised by the lasso, hard thresholding or an exact fit. */
export function SparseDenoising() {
  const state = useFigureState({
    method: choice(
      [
        { value: 'lasso', label: 'basis pursuit denoising (lasso)' },
        { value: 'iht', label: 'iterative hard thresholding' },
        { value: 'bp', label: 'basis pursuit (exact fit)' },
      ],
      'lasso',
      { label: 'method' },
    ),
    lambda: float(0.3, {
      ge: 0.02,
      le: 5,
      scale: 'log10',
      suggestions: [0.03, 0.1, 0.3, 1, 3],
      label: 'λ',
      when: when('method', 'lasso'),
    }),
    s: int(5, { ge: 1, le: 40, suggestions: [1, 3, 5, 10, 20], label: 'sparsity s', when: when('method', 'iht') }),
    sd: slider(0, 0.5, 0.15, { step: 0.01, label: 'noise standard deviation' }),
  })
  const method = state.method as Method
  const { lambda, s, sd } = state
  const { value: r, stale } = useComputed(() => solve(method, lambda, s, sd), [method, lambda, s, sd])
  // The sweep depends only on the method family and the noise, so dragging λ or s does not redo it.
  const family = method === 'iht' ? 'iht' : 'lasso'
  const swept = useComputed(() => sweep(family, sd), [family, sd], { mode: 'release' })

  const sx = useAxis({ label: 'sample i', range: [-1, M] })
  const sy = useAxis({ label: 'signal', range: [-2.2, 2.2] })
  const cx = useAxis({ label: 'atom j (cosines, then spikes)', range: [-1, K] })
  const cy = useAxis({ label: 'coefficient', range: [-5, 5] })
  const ky = useAxis({ label: '|dⱼᵀy|', range: [0, 6] })
  const lx = useAxis({ label: 'λ', log: true, range: [0.01, 10] })
  const nx = useAxis({ label: 'sparsity s', range: [0, 41] })
  const ey = useAxis({ label: '‖Dx̂ − clean‖ / ‖clean‖', range: [0, 1.1] })
  return (
    <Figure
      title="Denoising a signal that is sparse over cosines and spikes"
      state={state}
      defaultSize="L"
      caption="The clean signal (dashed) is three cosines and two spikes: five atoms of a dictionary of 64 cosines and 64 spikes. Top left: the noisy samples and the reconstruction. Top right: the coefficients found (bars) and the true ones (dots). Bottom left: how strongly each atom correlates with the noisy signal; for the lasso, drag the horizontal line to set λ, and with the line above the tallest bar every coefficient is zero. Bottom right: the reconstruction error across λ (or across s for hard thresholding), with the current setting marked; the dashed line is the error of the noisy signal itself. The exact fit of basis pursuit uses 64 atoms and reproduces the noise. As λ grows the lasso drops atoms until about the five true ones remain, but it also shrinks every kept coefficient towards zero, so its error climbs again. Hard thresholding at s = 5 keeps the five true atoms at full size and has the smallest error."
      readouts={
        <>
          <Readout label="non-zero coefficients" value={`${r.nonZeros} of ${K} (true: ${TRUTH.size})`} />
          <Readout label="true atoms found" value={`${r.found} of ${TRUTH.size}`} />
          <Readout label="‖Dx̂ − clean‖ / ‖clean‖" value={formatNumber(r.error)} />
          <Readout label="‖y − clean‖ / ‖clean‖ (the noise)" value={formatNumber(r.noise)} />
          {method === 'lasso' && <Readout label="λ at which x̂ = 0: ‖Dᵀy‖∞" value={formatNumber(r.lambdaMax)} />}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={sx} y={sy} height={240}>
          <Points name="noisy signal y" x={SAMPLES} y={r.y} muted thin />
          <Curve name="clean signal" x={SAMPLES} y={CLEAN} emphasis dashed />
          <Curve name="reconstruction Dx̂" x={SAMPLES} y={r.recon} slot={0} stale={stale} />
        </Plot>
        <Plot x={cx} y={cy} height={240}>
          <Bars name="coefficients x̂" x={ATOMS} y={r.x} slot={0} stale={stale} />
          <Points name="true coefficients" x={TRUE_AT} y={TRUE_AT.map((j) => TRUE_X[j])} emphasis size={7} />
          <Annotation x={M - 0.5} text="cosines | spikes" dashed muted />
        </Plot>
        <Plot x={cx} y={ky} height={240}>
          <Bars name="|dⱼᵀy|" x={ATOMS} y={r.corr} muted />
          {method === 'lasso' && <Handle {...state.handle('lambda', { axis: 'y', label: 'λ' })} />}
          {method === 'iht' && (
            <Annotation y={r.kept} text={`the first step keeps the ${s} atoms on or above`} dashed />
          )}
        </Plot>
        <Plot x={method === 'iht' ? nx : lx} y={ey} height={240}>
          <Curve
            name="reconstruction error"
            x={swept.value.x}
            y={swept.value.y}
            slot={0}
            showPoints
            stale={swept.stale}
          />
          <Annotation y={r.noise} text="the noise: error of y itself" dashed muted />
          {method === 'bp' ? (
            <Annotation at={[LAMBDAS[0], r.error]} text="basis pursuit: the limit λ → 0" />
          ) : (
            <Points name="current setting" x={[method === 'iht' ? s : lambda]} y={[r.error]} emphasis size={10} live />
          )}
        </Plot>
      </div>
    </Figure>
  )
}
