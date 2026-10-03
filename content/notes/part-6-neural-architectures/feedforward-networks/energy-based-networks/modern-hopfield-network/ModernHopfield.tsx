import { RotateCcw } from 'lucide-react'
import { useMemo, useState } from 'react'
import {
  ImagePlot,
  Interactive,
  ParamButton,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
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

function Panel({ title, values, note }: { title: string; values: ArrayLike<number>; note: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="text-xs text-muted-foreground">{title}</span>
      <ImagePlot
        width={GLYPH_SIZE}
        height={GLYPH_SIZE}
        values={values}
        scale="diverging"
        range={[-1, 1]}
        ariaLabel={title}
      />
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
  const [K, setK] = useState(10)
  const [cueIndex, setCueIndex] = useState(3)
  const [noise, setNoise] = useState(0.15)
  const [hideHalf, setHideHalf] = useState(false)
  const [n, setN] = useState(4)
  const [beta, setBeta] = useState(0.1)
  const [seed, setSeed] = useState(1)

  const letters = LETTERS.slice(0, K)
  const patterns = useMemo(() => LETTERS.slice(0, K).map(glyph), [K])
  const cueAt = Math.min(cueIndex, K - 1)

  const cue = useMemo(() => {
    const s = corrupt(glyph(LETTERS[cueAt]), noise, rng(seed).uniform)
    if (hideHalf) for (let i = N / 2; i < N; i++) s[i] = -1
    return s
  }, [cueAt, noise, hideHalf, seed])

  const classical = useMemo(
    () => denseRecall(patterns, cue, { kind: 'power', n: 2 }, rng(seed + 1).uniform),
    [patterns, cue, seed],
  )
  const dense = useMemo(() => {
    const f: Separation = { kind: 'power', n }
    return denseRecall(patterns, cue, f, rng(seed + 1).uniform)
  }, [patterns, cue, n, seed])
  const exponential = useMemo(
    () => denseRecall(patterns, cue, { kind: 'exp' }, rng(seed + 1).uniform),
    [patterns, cue, seed],
  )
  const attended = useMemo(() => softmaxUpdate(patterns, cue, beta), [patterns, cue, beta])
  const weights = useMemo(() => attentionWeights(patterns, cue, beta), [patterns, cue, beta])

  const bars = useMemo(
    (): XYSeries[] => [{ name: 'attention weight', type: 'bar', x: weights.map((_, k) => k + 1), y: weights, slot: 0 }],
    [weights],
  )
  const ranked = weights
    .map((p, k) => [p, k] as const)
    .sort((a, b) => b[0] - a[0])
    .slice(0, 3)

  const describe = (s: ArrayLike<number>) => {
    const v = verdict(s, letters, patterns)
    return v.ok ? `recalls ${v.text}` : v.text
  }

  return (
    <Interactive
      title="Classical and modern Hopfield networks recall a letter"
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
      controls={
        <>
          <ParamSlider
            label="stored letters K"
            value={K}
            onChange={setK}
            min={1}
            max={LETTERS.length}
            step={1}
            format={(v) => (v === 1 ? 'A' : `${v} (A–${LETTERS[v - 1]})`)}
          />
          <ParamSlider
            label="cue letter"
            value={cueAt}
            onChange={setCueIndex}
            min={0}
            max={K - 1}
            step={1}
            withArrows
            format={(v) => LETTERS[v]}
          />
          <ParamSlider
            label="fraction of pixels flipped"
            value={noise}
            onChange={setNoise}
            min={0}
            max={0.4}
            step={0.01}
          />
          <ParamSlider label="dense exponent n" value={n} onChange={setN} min={2} max={8} step={1} />
          <ParamSlider label="inverse temperature β" value={beta} onChange={setBeta} min={0} max={0.3} step={0.005} />
          <ParamSwitch label="blank the lower half" checked={hideHalf} onChange={setHideHalf} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>
            <RotateCcw /> New noise
          </ParamButton>
        </>
      }
      readout={
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
      <XYChart
        height={200}
        xLabel="stored letter (A = 1)"
        yLabel="softmax weight"
        xRange={[0.5, K + 0.5]}
        yRange={[0, 1]}
        integerX
        series={bars}
        ariaLabel="Softmax weights of the cue on each stored letter"
      />
    </Interactive>
  )
}
