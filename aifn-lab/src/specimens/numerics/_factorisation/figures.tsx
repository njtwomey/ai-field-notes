/**
 * Matrix factorisations: FastICA unmixing a cocktail party (`aifn-applied/unsupervised/embedding/linear`), NMF finding
 * the strokes glyphs are made of (`aifn/numerics/factorisation`), and factor analysis against probabilistic PCA on
 * data whose noise differs by feature. The data come from `aifn-applied/data/synthetic`; the page only aligns
 * recovered components with the true ones for display.
 */
import { cocktailParty, latentFactorModel, latentFactors, strokeGlyphs } from 'aifn-applied/data/synthetic'
import {
  factorAnalysis,
  fastIca,
  latentGaussianSteps,
  pca,
  probabilisticPca,
} from 'aifn-applied/unsupervised/embedding/linear'
import { stream } from 'aifn/foundation/random'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { dataset } from 'aifn/learning/estimators'
import { nmfSteps, type NmfLoss, type NmfSolver } from 'aifn/numerics/factorisation'
import { Player, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, int, row, slider, useComputed, useFigureState } from '@lab/state'
import { Annotation, Bars, Curve, formatNumber, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

/** Column c of a row-major matrix [n, d]. */
const column = (v: ArrayLike<number>, n: number, d: number, c: number) =>
  Array.from({ length: n }, (_, i) => v[i * d + c])

const standardise = (a: number[]) => {
  const m = a.reduce((s, u) => s + u, 0) / a.length
  const sd = Math.sqrt(a.reduce((s, u) => s + (u - m) ** 2, 0) / a.length) || 1
  return a.map((u) => (u - m) / sd)
}

const correlation = (a: number[], b: number[]) => {
  const x = standardise(a)
  const y = standardise(b)
  return x.reduce((s, u, i) => s + u * y[i], 0) / x.length
}

/** Match each estimated component to the true source it correlates with most, signed and standardised for display. */
function align(estimates: number[][], truth: number[][]) {
  const used = new Set<number>()
  const out: { source: number; y: number[]; r: number }[] = []
  const order = estimates.map((e, k) => ({ k, best: Math.max(...truth.map((t) => Math.abs(correlation(e, t)))) }))
  order.sort((a, b) => b.best - a.best)
  for (const { k } of order) {
    let best = -1
    let r = 0
    truth.forEach((t, j) => {
      if (used.has(j)) return
      const c = correlation(estimates[k], t)
      if (best < 0 || Math.abs(c) > Math.abs(r)) {
        best = j
        r = c
      }
    })
    used.add(best)
    out.push({ source: best, y: standardise(estimates[k]).map((u) => (r < 0 ? -u : u)), r: Math.abs(r) })
  }
  return out.sort((a, b) => a.source - b.source)
}

const SOURCE_NAMES = ['sine', 'square wave', 'sawtooth']

// ── 1 · Cocktail party ───────────────────────────────────────────────────────────────────────────────────────────────

export function CocktailPartySpecimen() {
  const state = useFigureState({
    data: row('1 · mixing', {
      n: int(400, { ge: 50, le: 4000, suggestions: [200, 400, 1000], label: 'samples n' }),
      noise: slider(0, 0.5, 0, { step: 0.01, label: 'sensor noise sd' }),
      seed: slider(0, 20, 0, { step: 1, label: 'mixing seed' }),
    }),
    ica: row('2 · FastICA', {
      iterations: int(30, { ge: 1, le: 500, suggestions: [10, 30, 100], label: 'iterations' }),
    }),
  })
  const { n, noise, seed } = state.data
  const { iterations } = state.ica
  const run = useComputed(() => {
    const party = cocktailParty(stream(`cocktail-${seed}`), { n, noise })
    const x = dataset(party.mixed)
    const truth = [0, 1, 2].map((c) => column(toFlat(party.sources), n, 3, c))
    // One fit per iteration count from the same initial unmixing matrix (the same init stream): the player's steps.
    const steps = Array.from({ length: iterations + 1 }, (_, t) => {
      const m = fastIca({ maxSteps: t, tolerance: 0 }).fit(x, { stream: stream(`ica-${seed}`) })
      const s = toFlat(m.transform(party.mixed))
      return {
        aligned: align(
          [0, 1, 2].map((c) => column(s, n, 3, c)),
          truth,
        ),
        change: m.training.final.change,
      }
    })
    const p = toFlat(pca({ components: 3 }).fit(x).transform(party.mixed))
    const principal = align(
      [0, 1, 2].map((c) => column(p, n, 3, c)),
      truth,
    )
    return {
      t: Array.from(toFlat(party.t)),
      mixed: [0, 1, 2].map((c) => column(toFlat(party.mixed), n, 3, c)),
      truth: truth.map(standardise),
      steps,
      principal,
    }
  }, [n, noise, seed, iterations])
  const r = run.value
  const [at, setAt] = usePlayhead(r.steps.length)
  const now = r.steps[Math.min(at, r.steps.length - 1)]
  const tAxis = useAxis({ label: 'time t', range: [0, 8] })
  const micAxis = useAxis({ label: 'microphone', hold: 'union', key: `${n}-${noise}-${seed}` })
  const srcAxis = useAxis({ label: 'source (standardised)', range: [-2.6, 2.6] })
  const pcAxis = useAxis({ label: 'principal component', range: [-2.6, 2.6] })
  return (
    <Figure
      title="FastICA separates three voices that PCA cannot"
      purpose="Mixtures of independent non-Gaussian sources are more Gaussian than the sources, so turning the whitened data until each output is as non-Gaussian as possible recovers the sources; PCA only decorrelates, which leaves any rotation of them."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="3 · iterations">
          <Player className="col-span-full" value={at} onChange={setAt} count={r.steps.length} label="iteration" />
        </ControlRow>
      }
      readouts={{
        [`FastICA after ${at} iteration${at === 1 ? '' : 's'}`]: (
          <>
            {now.aligned.map((a) => (
              <Readout key={a.source} label={`|corr| with the ${SOURCE_NAMES[a.source]}`} value={fmt(a.r)} />
            ))}
            <Readout label="max | |⟨wᵢ, wᵢ′⟩| − 1 |" value={fmt(now.change, 2)} />
          </>
        ),
        PCA: (
          <>
            {r.principal.map((a) => (
              <Readout key={a.source} label={`|corr| with the ${SOURCE_NAMES[a.source]}`} value={fmt(a.r)} />
            ))}
          </>
        ),
      }}
      caption={`Data: aifn cocktailParty (seeded): a sine, a square wave and a sawtooth, each mixed into three microphones by a random 3 × 3 matrix, plus sensor noise. Top: what the microphones hear. Middle: FastICA's sources at the chosen iteration (solid) over the true ones (dashed, same colour); play to watch the unmixing matrix turn from its random start. Bottom: the principal components, which are uncorrelated but still mixtures. Each estimate is matched to the source it correlates with most and shown standardised and signed, since ICA recovers sources only up to order, sign and scale.`}
    >
      <Plots rows={3} hoverGroup>
        <Plot x={tAxis} y={micAxis} title="microphones">
          {r.mixed.map((m, k) => (
            <Curve key={k} name={`microphone ${k + 1}`} x={r.t} y={m} slot={k + 3} width={1.5} stale={run.stale} />
          ))}
        </Plot>
        <Plot x={tAxis} y={srcAxis} title="FastICA">
          {r.truth.map((s, k) => (
            // Same name as the estimate: one legend entry per source (the caption says dashed is the truth).
            <Curve key={`t${k}`} name={SOURCE_NAMES[k]} x={r.t} y={s} slot={k} dashed width={1} silent />
          ))}
          {now.aligned.map((a) => (
            <Curve key={a.source} name={SOURCE_NAMES[a.source]} x={r.t} y={a.y} slot={a.source} width={1.5} />
          ))}
        </Plot>
        <Plot x={tAxis} y={pcAxis} title="PCA">
          {r.principal.map((a) => (
            <Curve
              key={a.source}
              name={`PC near the ${SOURCE_NAMES[a.source]}`}
              x={r.t}
              y={a.y}
              slot={a.source}
              width={1.5}
            />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2 · NMF on glyphs ────────────────────────────────────────────────────────────────────────────────────────────────

const SOLVERS: { value: string; label: string; solver: NmfSolver; loss: NmfLoss }[] = [
  { value: 'mu-f', label: 'multiplicative, squared error', solver: 'multiplicative', loss: 'frobenius' },
  { value: 'mu-kl', label: 'multiplicative, KL divergence', solver: 'multiplicative', loss: 'kullback-leibler' },
  { value: 'hals', label: 'HALS, squared error', solver: 'hals', loss: 'frobenius' },
]

/** A size × size image from a flat row, top row first, as Raster rows (row 0 at the bottom of the plot). */
const image = (v: ArrayLike<number>, offset: number, size: number) =>
  Array.from({ length: size }, (_, r) => Array.from({ length: size }, (_, c) => v[offset + (size - 1 - r) * size + c]))

export function NmfGlyphsSpecimen() {
  const state = useFigureState({
    data: row('1 · glyphs', {
      n: int(200, { ge: 20, le: 3000, suggestions: [100, 200, 1000], label: 'glyphs n' }),
      p: slider(0.1, 0.8, 0.3, { step: 0.05, label: 'stroke probability' }),
      noise: slider(0, 0.3, 0.05, { step: 0.01, label: 'noise sd' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
    model: row('2 · factorisation', {
      solver: choice(
        SOLVERS.map(({ value, label }) => ({ value, label })),
        'mu-f',
        { label: 'solver' },
      ),
      rank: int(8, { ge: 1, le: 16, suggestions: [4, 6, 8, 12], label: 'rank k' }),
      sweeps: int(200, { ge: 1, le: 3000, suggestions: [50, 200, 1000], label: 'sweeps' }),
    }),
  })
  const { n, p, noise, seed } = state.data
  const { solver: solverKey, rank, sweeps } = state.model
  const size = 7
  const run = useComputed(
    () => {
      const g = strokeGlyphs(stream(`glyphs-${seed}`), { n, p, noise, size })
      const s = SOLVERS.find((x) => x.value === solverKey) ?? SOLVERS[0]
      const tr = trace(nmfSteps(g.x, { rank, solver: s.solver, loss: s.loss, tolerance: 0 }), undefined, sweeps, {
        stream: stream(`nmf-${seed}`),
      })
      const steps = tr.steps.map((st) => ({ t: st.t, objective: st.objective, H: Float64Array.from(toFlat(st.H)) }))
      const P = pca({ components: rank }).fit(dataset(g.x)).components
      return { steps, examples: Array.from(toFlat(g.x)).slice(0, 6 * size * size), pca: Float64Array.from(toFlat(P)) }
    },
    [n, p, noise, seed, solverKey, rank, sweeps],
    { mode: 'release' },
  )
  const r = run.value
  const [at, setAt] = usePlayhead(r.steps.length)
  const now = r.steps[Math.min(at, r.steps.length - 1)]
  const cells = Array.from({ length: size }, (_, i) => i)
  const px = useAxis({ label: '', range: [-0.5, size - 0.5], nice: false })
  const py = useAxis({ label: '', range: [-0.5, size - 0.5], nice: false, equal: px })
  const sweepAxis = useAxis({ label: 'sweep', range: [0, Math.max(1, sweeps)] })
  const objAxis = useAxis({
    label: 'objective',
    log: true,
    hold: 'union',
    key: `${n}-${p}-${noise}-${seed}-${solverKey}-${rank}`,
  })
  const partMax = Math.max(1e-9, ...now.H)
  const pcaMax = Math.max(1e-9, ...Array.from(r.pca, Math.abs))
  const shown = Math.min(rank, 8)
  return (
    <Figure
      title="NMF learns the strokes; PCA learns signed blends"
      purpose="With W and H held non-negative, each glyph is a sum of parts with no cancellation, so the parts NMF finds are the strokes the glyphs were drawn from; PCA's components mix strokes with positive and negative weights."
      state={state}
      defaultSize="XL"
      controls={
        <ControlRow label="3 · sweeps">
          <Player className="col-span-full" value={at} onChange={setAt} count={r.steps.length} label="sweep" />
        </ControlRow>
      }
      readouts={{
        [`sweep ${now.t}`]: <Readout label="objective" value={fmt(now.objective, 5)} />,
      }}
      caption={`Data: aifn strokeGlyphs (seeded): ${n} glyphs on a ${size} × ${size} grid, each the union of strokes (the middle bars, the four edges and the two diagonals) switched on with probability ${p}, plus noise clipped at 0. First row: six glyphs. Second row: the first ${shown} rows of H, the parts NMF has found at the chosen sweep (white is 0, each part scaled to its largest pixel); play from the random start to watch them sharpen into strokes. Third row: the first ${shown} principal axes (red positive, blue negative), which overlay strokes with opposite signs. Bottom: the objective, which never increases. HALS solves each part exactly in turn and usually gets lower in the same number of sweeps than the multiplicative updates.`}
    >
      <Plots rows={3} cols={8} heights={[1, 1, 1]}>
        {Array.from({ length: 8 }, (_, k) => (
          <Plot key={`g${k}`} x={px} y={py} bare>
            {k < 6 && (
              <Raster
                x={cells}
                y={cells}
                z={image(r.examples, k * size * size, size)}
                range={[0, 1]}
                colorBar={false}
              />
            )}
          </Plot>
        ))}
        {Array.from({ length: 8 }, (_, k) => (
          <Plot key={`h${k}`} x={px} y={py} bare title={k < shown ? `part ${k + 1}` : undefined}>
            {k < shown && (
              <Raster
                x={cells}
                y={cells}
                z={image(
                  now.H.map((v) => v / partMax),
                  k * size * size,
                  size,
                )}
                range={[0, 1]}
                colorBar={false}
                stale={run.stale}
              />
            )}
          </Plot>
        ))}
        {Array.from({ length: 8 }, (_, k) => (
          <Plot key={`p${k}`} x={px} y={py} bare title={k < shown ? `PC ${k + 1}` : undefined}>
            {k < shown && (
              <Raster
                x={cells}
                y={cells}
                z={image(r.pca, k * size * size, size)}
                scale="diverging"
                range={[-pcaMax, pcaMax]}
                colorBar={false}
                stale={run.stale}
              />
            )}
          </Plot>
        ))}
      </Plots>
      <Plots rows={1} scale={0.45}>
        <Plot x={sweepAxis} y={objAxis}>
          <Curve
            name="objective"
            x={r.steps.map((s) => s.t)}
            y={r.steps.map((s) => s.objective)}
            slot={0}
            stale={run.stale}
          />
          <Points name="now" x={[now.t]} y={[now.objective]} emphasis live />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 3 · Factor analysis against probabilistic PCA ────────────────────────────────────────────────────────────────────

export function FactorAnalysisSpecimen() {
  const state = useFigureState({
    data: row('1 · data', {
      n: int(400, { ge: 30, le: 5000, suggestions: [100, 400, 2000], label: 'samples n' }),
      spread: slider(0.05, 4, 2, { step: 0.05, label: 'largest noise variance' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
    model: row('2 · EM', {
      iterations: int(100, { ge: 1, le: 2000, suggestions: [30, 100, 500], label: 'EM iterations' }),
    }),
  })
  const { n, spread, seed } = state.data
  const { iterations } = state.model
  const d = 8
  const run = useComputed(() => {
    const data = latentFactors(stream(`fa-${seed}`), { n, d, latent: 2, spread })
    const truth = latentFactorModel({ d, latent: 2, spread })
    const tr = trace(
      latentGaussianSteps(data.x, { latent: 2, noise: 'diagonal', tolerance: 0 }),
      undefined,
      iterations,
      {
        stream: stream(`fa-em-${seed}`),
      },
    )
    const communality = (W: Tensor) => {
      const w = toFlat(W)
      return Array.from({ length: d }, (_, i) => w[i * 2] ** 2 + w[i * 2 + 1] ** 2)
    }
    const fa = factorAnalysis({ latent: 2, maxSteps: iterations, tolerance: 0 }).fit(dataset(data.x), {
      stream: stream(`fa-em-${seed}`),
    })
    const pp = probabilisticPca({ latent: 2 }).fit(dataset(data.x))
    return {
      steps: tr.steps.map((s) => ({
        t: s.t,
        ll: s.logLikelihood,
        noise: Array.from(toFlat(s.noise)),
        communality: communality(s.loadings),
      })),
      trueNoise: Array.from(toFlat(truth.noise)),
      trueCommunality: communality(truth.loadings),
      ppcaNoise: pp.noiseVariance,
      ppcaCommunality: communality(pp.loadings),
      ppcaLl: pp.logLikelihood,
      faLl: fa.logLikelihood,
    }
  }, [n, spread, seed, iterations])
  const r = run.value
  const [at, setAt] = usePlayhead(r.steps.length)
  const now = r.steps[Math.min(at, r.steps.length - 1)]
  const features = Array.from({ length: d }, (_, i) => i + 1)
  const fAxis = useAxis({ label: 'feature', range: [0.4, d + 0.6], integer: true })
  // Held over every iteration, so the initial guess (each feature's total variance), where the player opens, fits too.
  const top = Math.max(...r.trueNoise, ...r.steps.flatMap((s) => s.noise), r.ppcaNoise) * 1.1
  const ctop = Math.max(...r.trueCommunality, ...r.ppcaCommunality, ...r.steps.flatMap((s) => s.communality)) * 1.1
  const noiseAxis = useAxis({ label: 'noise variance ψᵢ', range: [0, top], key: `${spread}-${seed}-${n}` })
  const commAxis = useAxis({ label: 'communality Σⱼ Wᵢⱼ²', range: [0, ctop], key: `${spread}-${seed}-${n}` })
  const itAxis = useAxis({ label: 'EM iteration', range: [0, Math.max(1, iterations)] })
  const llAxis = useAxis({ label: 'log-likelihood per row', hold: 'union', key: `${spread}-${seed}-${n}` })
  const shift = (k: number) => features.map((f) => f + (k - 1) * 0.27)
  return (
    <Figure
      title="Factor analysis models each feature's own noise; PPCA shares one"
      purpose="Both fit x = Wz + ε with two factors; factor analysis gives every feature its own noise variance, while probabilistic PCA forces one shared σ², so on features with large noise its loadings absorb noise as if it were signal."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="3 · EM">
          <Player className="col-span-full" value={at} onChange={setAt} count={r.steps.length} label="iteration" />
        </ControlRow>
      }
      readouts={{
        [`EM iteration ${now.t}`]: (
          <>
            <Readout label="FA log-likelihood per row" value={fmt(now.ll, 5)} />
            <Readout label="PPCA (closed form)" value={fmt(r.ppcaLl, 5)} />
            <Readout label="PPCA shared σ²" value={fmt(r.ppcaNoise)} />
          </>
        ),
      }}
      caption={`Data: aifn latentFactors (seeded): ${n} draws of x = Wz + ε in ${d} features with two factors loading on blocks of features, and noise variances rising geometrically from 0.05 (feature 1) to ${spread} (feature ${d}). Top: the noise variance of each feature: the truth (ink), factor analysis at the chosen EM iteration (blue) and PPCA's single σ² (orange). Middle: each feature's communality, the variance the factors explain, which is the part of the loadings that does not depend on their rotation. Bottom: factor analysis's log-likelihood per EM iteration, which never decreases, against PPCA's closed-form maximum (dashed). Play from the random start; raise the largest noise variance to widen the gap.`}
    >
      <Plots rows={3} heights={[35, 35, 30]} hoverGroup>
        <Plot x={fAxis} y={noiseAxis}>
          <Bars name="true ψᵢ" x={shift(0)} y={r.trueNoise} width={0.25} emphasis />
          <Bars name="factor analysis" x={shift(1)} y={now.noise} width={0.25} slot={0} stale={run.stale} />
          <Bars
            name="PPCA σ²"
            x={shift(2)}
            y={features.map(() => r.ppcaNoise)}
            width={0.25}
            slot={1}
            stale={run.stale}
          />
        </Plot>
        <Plot x={fAxis} y={commAxis}>
          <Bars name="true" x={shift(0)} y={r.trueCommunality} width={0.25} emphasis />
          <Bars name="factor analysis" x={shift(1)} y={now.communality} width={0.25} slot={0} stale={run.stale} />
          <Bars name="PPCA" x={shift(2)} y={r.ppcaCommunality} width={0.25} slot={1} stale={run.stale} />
        </Plot>
        <Plot x={itAxis} y={llAxis}>
          <Curve name="factor analysis (EM)" x={r.steps.map((s) => s.t)} y={r.steps.map((s) => s.ll)} slot={0} />
          <Annotation y={r.ppcaLl} dashed text="PPCA maximum" />
          <Points name="now" x={[now.t]} y={[now.ll]} emphasis live />
        </Plot>
      </Plots>
    </Figure>
  )
}
