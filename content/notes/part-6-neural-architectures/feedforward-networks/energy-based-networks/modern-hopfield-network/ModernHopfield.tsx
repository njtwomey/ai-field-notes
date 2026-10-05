import { useMemo } from 'react'
import {
  Bars,
  Figure,
  formatNumber,
  int,
  Pixels,
  Plot,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform } from 'aifn-compute/foundation/random'
import { GLYPH_SIZE, LETTERS, glyph } from '../_shared/glyphs'
import { corrupt, overlap } from '../_shared/spins'
import { attentionWeights, denseRecall, softmaxUpdate, type Separation } from './dense'

const N = GLYPH_SIZE * GLYPH_SIZE
const SUPERSCRIPT = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸']

/** What a recalled state is: a stored letter exactly, or the nearest one and its overlap. */
function verdict(state: ArrayLike<number>, letters: string[], patterns: Int8Array[]): { ok: boolean; text: string } {
  const signs = Int8Array.from(state, (v) => (v >= 0 ? 1 : -1))
  const m = patterns.map((p) => overlap(p, signs))
  const best = m.reduce((b, v, k) => (v > m[b] ? k : b), 0)
  return m[best] === 1
    ? { ok: true, text: letters[best] }
    : { ok: false, text: `not stored (nearest ${letters[best]}, m = ${formatNumber(m[best])})` }
}

/** Seeded uniform draws. */
function draws(seed: number) {
  const g = stream(seed)
  return () => uniform(g)
}

const PIXEL_RANGE: [number, number] = [-1, 1]

function Panel({ title, values, note }: { title: string; values: ArrayLike<number>; note: string }) {
  // Row 0 at the top and square pixels.
  const x = useAxis({ range: [-0.5, GLYPH_SIZE - 0.5], nice: false })
  const y = useAxis({ range: [-0.5, GLYPH_SIZE - 0.5], nice: false, inverse: true, equal: x })
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="text-xs text-muted-foreground">{title}</span>
      <Plot x={x} y={y} bare height={160} ariaLabel={title}>
        <Pixels width={GLYPH_SIZE} height={GLYPH_SIZE} values={values} scale="diverging" range={PIXEL_RANGE} />
      </Plot>
      <span className="text-xs">{note}</span>
    </div>
  )
}

/**
 * The same corrupted letter recalled by four memories that store the first K letters: the classical Hopfield network,
 * a dense associative memory with F(x) = xⁿ, the exponential memory, and one softmax update of the continuous modern
 * Hopfield network, whose weights over the stored letters are shown as attention weights.
 */
export function ModernHopfield() {
  const state = useFigureState({
    K: int(10, {
      min: 1,
      max: LETTERS.length,
      step: 1,
      label: 'stored letters K',
      format: (v) => (v === 1 ? 'A' : `${v} (A–${LETTERS[v - 1]})`),
    }),
    cue: slider(0, LETTERS.length - 1, 3, {
      step: 1,
      label: 'cue letter (within the stored ones)',
      format: (v) => LETTERS[v],
    }),
    noise: slider(0, 0.4, 0.15, { step: 0.01, label: 'fraction of pixels flipped' }),
    n: int(4, { min: 2, max: 8, step: 1, label: 'dense exponent n' }),
    beta: slider(0, 0.3, 0.1, { step: 0.005, label: 'inverse temperature β' }),
    hideHalf: setting(false, 'blank the lower half'),
    seed: int(1, { ge: 0, label: 'noise seed' }),
  })
  const { K, cue: cueIndex, noise, n, beta, hideHalf, seed } = state

  const letters = useMemo(() => LETTERS.slice(0, K), [K])
  const patterns = useMemo(() => LETTERS.slice(0, K).map(glyph), [K])
  const cueAt = Math.min(cueIndex, K - 1)

  const cue = useMemo(() => {
    const s = corrupt(glyph(LETTERS[cueAt]), noise, draws(seed))
    if (hideHalf) for (let i = N / 2; i < N; i++) s[i] = -1
    return s
  }, [cueAt, noise, hideHalf, seed])

  const classical = useMemo(
    () => denseRecall(patterns, cue, { kind: 'power', n: 2 }, draws(seed + 1)),
    [patterns, cue, seed],
  )
  const dense = useMemo(() => {
    const f: Separation = { kind: 'power', n }
    return denseRecall(patterns, cue, f, draws(seed + 1))
  }, [patterns, cue, n, seed])
  const exponential = useMemo(() => denseRecall(patterns, cue, { kind: 'exp' }, draws(seed + 1)), [patterns, cue, seed])
  const attended = useMemo(() => softmaxUpdate(patterns, cue, beta), [patterns, cue, beta])
  const weights = useMemo(() => attentionWeights(patterns, cue, beta), [patterns, cue, beta])

  const positions = useMemo(() => letters.map((_, k) => k), [letters])
  const ranked = weights
    .map((p, k) => [p, k] as const)
    .sort((a, b) => b[0] - a[0])
    .slice(0, 3)

  const describe = (s: ArrayLike<number>) => {
    const v = verdict(s, letters, patterns)
    return v.ok ? `recalls ${v.text}` : v.text
  }

  const xAxis = useAxis({ label: 'stored letter', categories: letters })
  const yAxis = useAxis({ label: 'softmax weight', range: [0, 1] })
  return (
    <Figure
      title="Classical and modern Hopfield networks recall a letter"
      state={state}
      caption={
        <>
          The first K capital letters are stored as 10 × 10 patterns of ±1 (red +1, blue −1). The cue is one of them
          with a fraction of its pixels flipped, or with its lower half blanked. Each memory runs from the cue: the
          classical network is F(x) = x², the dense memory F(x) = xⁿ, the exponential memory F(x) = eˣ, each updated one
          neuron at a time until nothing changes; the softmax update makes one step ξ ← X softmax(β Xᵀ ξ). Letters
          overlap heavily, so the classical network fails with only a few stored, while the exponential memory recalls
          all 26. The bars are the softmax weights on the stored letters, in alphabetical order: at small β they spread
          over many letters and the update returns their average; at large β one letter takes all the weight.
        </>
      }
      readouts={
        <>
          <Readout
            label="largest weights"
            value={ranked.map(([p, k]) => `${letters[k]} ${formatNumber(p)}`).join(', ')}
          />
          <Readout label="β × N" value={formatNumber(beta * N)} />
        </>
      }
    >
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <Panel title="cue" values={cue} note={`${LETTERS[cueAt]}, corrupted`} />
        <Panel title="classical, F(x) = x²" values={classical} note={describe(classical)} />
        <Panel title={`dense, F(x) = x${SUPERSCRIPT[n]}`} values={dense} note={describe(dense)} />
        <Panel title="exponential, F(x) = eˣ" values={exponential} note={describe(exponential)} />
        <Panel title="softmax, one step" values={attended} note={describe(attended)} />
      </div>
      <Plot x={xAxis} y={yAxis} height={200} ariaLabel="Softmax weights of the cue on each stored letter">
        <Bars name="attention weight" x={positions} y={weights} slot={0} />
      </Plot>
    </Figure>
  )
}
