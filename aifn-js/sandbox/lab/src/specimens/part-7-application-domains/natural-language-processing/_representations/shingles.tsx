import { useMemo, useState } from 'react'
import {
  characterShingles,
  jaccardSimilarity,
  minHashSignature,
  minHashSignatures,
  minHashSimilarity,
  minHashStandardError,
  wordShingles,
} from 'aifn/text/features'
import { lshCandidates, lshProbability, lshThreshold } from 'aifn/numerics/neighbours'
import { Figure } from '@lab/layout'
import { choice, slider, useFigureState, when } from '@lab/state'
import { Button } from '@lab/ui/button'
import { Annotation, Area, Curve, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'
import type { Corpus } from './showcase'

/** The candidate list for B: the documents most similar to A. */
const CANDIDATES = 8
/** Similarity bins for the empirical candidate rate. */
const BINS = 10

export function ShingleFigure({ corpus }: { corpus: Corpus }) {
  const state = useFigureState({
    unit: choice(['characters', 'words'] as const, 'characters', { label: 'shingles of' }),
    k: slider(2, 8, 4, { label: 'k (characters)', step: 1, when: when('unit', 'characters') }),
    w: slider(1, 4, 2, { label: 'w (words)', step: 1, when: when('unit', 'words') }),
    hashes: slider(16, 512, 256, { label: 'hashes k', step: 16 }),
    bands: slider(1, 40, 16, { label: 'bands b', step: 1 }),
    rows: slider(1, 10, 4, { label: 'rows per band r', step: 1 }),
  })
  const { unit, k, w, hashes, bands, rows } = state
  const D = corpus.docs.length
  const [aRaw, setA] = useState(0)
  const [bPick, setB] = useState<number | null>(null)
  const a = Math.min(aRaw, Math.max(0, D - 1))

  const sets = useMemo(
    () =>
      corpus.docs.map((d, i) => (unit === 'characters' ? characterShingles(d, k) : wordShingles(corpus.tokens[i], w))),
    [corpus, unit, k, w],
  )
  // Documents ranked by true Jaccard similarity to A.
  const ranked = useMemo(
    () =>
      sets
        .map((s, i) => ({ i, j: i === a ? -1 : jaccardSimilarity(sets[a] ?? [], s) }))
        .filter((e) => e.i !== a)
        .sort((p, q) => q.j - p.j || p.i - q.i),
    [sets, a],
  )
  const b = bPick !== null && bPick !== a && bPick < D ? bPick : (ranked[0]?.i ?? a)
  const truth = D > 1 ? jaccardSimilarity(sets[a], sets[b]) : 0

  const banding = { bands, rows }
  const length = Math.max(hashes, bands * rows)
  const pair = useMemo(() => {
    if (D < 2) return null
    const sa = minHashSignature(sets[a], { hashes: length })
    const sb = minHashSignature(sets[b], { hashes: length })
    const js = Array.from({ length: hashes }, (_, j) => j + 1)
    const estimate = js.map((j) => minHashSimilarity(sa, sb, { hashes: j }))
    return { js, estimate }
  }, [sets, a, b, hashes, length, D])

  // The whole corpus under banding: which pairs become candidates, against their true similarity.
  const corpusLsh = useMemo(() => {
    if (D < 2 || D > 1000) return null
    const sig = minHashSignatures(sets, { hashes: bands * rows })
    const candidates = new Set(lshCandidates(sig, { bands, rows }).map((c) => c.i * D + c.j))
    const seen = new Array<number>(BINS).fill(0)
    const hit = new Array<number>(BINS).fill(0)
    const t = lshThreshold({ bands, rows })
    let near = 0
    let found = 0
    for (let i = 0; i < D; i++)
      for (let j = i + 1; j < D; j++) {
        const s = jaccardSimilarity(sets[i], sets[j])
        const bin = Math.min(BINS - 1, Math.floor(s * BINS))
        const isCandidate = candidates.has(i * D + j)
        seen[bin]++
        if (isCandidate) hit[bin]++
        if (s >= t) {
          near++
          if (isCandidate) found++
        }
      }
    const x: number[] = []
    const y: number[] = []
    seen.forEach((n, bin) => {
      if (n > 0) {
        x.push((bin + 0.5) / BINS)
        y.push(hit[bin] / n)
      }
    })
    return { candidates: candidates.size, pairs: (D * (D - 1)) / 2, near, found, x, y }
  }, [sets, bands, rows, D])

  const sGrid = useMemo(() => Array.from({ length: 101 }, (_, i) => i / 100), [])
  const sCurve = useMemo(() => sGrid.map((s) => lshProbability(s, { bands, rows })), [sGrid, bands, rows])
  const threshold = lshThreshold(banding)
  const band = pair ? pair.js.map((j) => 2 * minHashStandardError(truth, j)) : []

  const hashAxis = useAxis({ label: 'hashes used', integer: true, key: hashes })
  const estAxis = useAxis({ label: 'Jaccard similarity', range: [0, 1] })
  const sAxis = useAxis({ label: 'true Jaccard similarity s', range: [0, 1] })
  const pAxis = useAxis({ label: 'P(candidate)', range: [0, 1] })

  const shared = new Set((sets[a] ?? []).filter((x) => (sets[b] ?? []).includes(x)))
  const step = (d: number) => {
    setA((x) => (Math.min(x, D - 1) + d + D) % D)
    setB(null)
  }

  return (
    <Figure
      title="Shingles, MinHash and LSH"
      purpose="Near-duplicate detection compares shingle sets by Jaccard similarity; MinHash estimates it from k hashes, and banding the signature turns that estimate into an S-shaped chance of being compared at all."
      state={state}
      defaultSize="L"
      hoverReadout={false}
      readouts={
        D > 1 && (
          <>
            <Readout label="true Jaccard J(A, B)" value={truth.toFixed(3)} />
            <Readout
              label={`MinHash estimate, k = ${hashes}`}
              value={pair ? pair.estimate[hashes - 1].toFixed(3) : '–'}
            />
            <Readout label="threshold (1/b)^{1/r}" value={threshold.toFixed(3)} />
            <Readout label="P(A, B candidate)" value={lshProbability(truth, banding).toFixed(3)} />
            {corpusLsh && (
              <Readout
                label="corpus: candidates / pairs"
                value={`${corpusLsh.candidates} / ${corpusLsh.pairs}, ${corpusLsh.found} of ${corpusLsh.near} with s ≥ threshold`}
              />
            )}
          </>
        )
      }
      caption="Step document A with the arrows; B is the document most similar to A, or click another in the list (ranked by true Jaccard similarity). Shingles both documents share are filled. Left: the MinHash estimate (share of agreeing minima among the first j hashes) converges to the true Jaccard similarity J, inside ±2√(J(1 − J)/j). Right: with b bands of r rows, a pair becomes a candidate when any band agrees, with probability 1 − (1 − sʳ)ᵇ; the dots are the share of the corpus's pairs in each similarity bin that did become candidates, and the black diamond is the pair A, B. More rows per band sharpen the curve and move it right; more bands move it left."
    >
      {D < 2 ? (
        <div className="text-sm text-muted-foreground">At least two documents are needed.</div>
      ) : (
        <div className="flex flex-col gap-3">
          <div className="grid gap-3 text-xs md:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <Button size="xs" variant="outline" onClick={() => step(-1)} aria-label="previous document">
                  ‹
                </Button>
                <Button size="xs" variant="outline" onClick={() => step(1)} aria-label="next document">
                  ›
                </Button>
                <span className="text-muted-foreground">A = d{a + 1}</span>
                <span className="font-mono">{corpus.docs[a]}</span>
              </div>
              <ShingleChips label={`A: ${sets[a].length} shingles`} items={sets[a]} shared={shared} />
              <div className="flex items-center gap-2">
                <span className="text-muted-foreground">B = d{b + 1}</span>
                <span className="font-mono">{corpus.docs[b]}</span>
              </div>
              <ShingleChips label={`B: ${sets[b].length} shingles`} items={sets[b]} shared={shared} />
            </div>
            <div className="flex max-h-40 flex-col gap-0.5 overflow-auto">
              <span className="text-muted-foreground">most similar to A</span>
              {ranked.slice(0, CANDIDATES).map(({ i, j }) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setB(i)}
                  className={
                    'flex gap-2 rounded px-1 text-left hover:bg-muted ' + (i === b ? 'bg-muted font-semibold' : '')
                  }
                >
                  <span className="w-10 shrink-0 font-mono tabular-nums">{j.toFixed(2)}</span>
                  <span className="truncate font-mono">{corpus.docs[i]}</span>
                </button>
              ))}
            </div>
          </div>
          {pair && (
            <Plots cols={2} scale={0.62}>
              <Plot x={hashAxis} y={estAxis} title="MinHash estimate as hashes grow">
                <Area
                  name="±2 standard errors"
                  x={pair.js}
                  y={band.map((e) => Math.min(1, truth + e))}
                  base={band.map((e) => Math.max(0, truth - e))}
                  muted
                  line={false}
                  opacity={0.25}
                />
                <Curve name="true J" x={[1, hashes]} y={[truth, truth]} emphasis dashed />
                <Curve name="estimate" x={pair.js} y={pair.estimate} slot={0} />
              </Plot>
              <Plot x={sAxis} y={pAxis} title={`banding: b = ${bands}, r = ${rows}`}>
                <Curve name="1 − (1 − sʳ)ᵇ" x={sGrid} y={sCurve} slot={0} />
                <Annotation x={threshold} text="threshold" dashed />
                {corpusLsh && (
                  <Points name="corpus pairs, share that are candidates" x={corpusLsh.x} y={corpusLsh.y} slot={1} />
                )}
                <Points name="A, B" x={[truth]} y={[lshProbability(truth, banding)]} emphasis />
              </Plot>
            </Plots>
          )}
        </div>
      )}
    </Figure>
  )
}

function ShingleChips({
  label,
  items,
  shared,
}: {
  label: string
  items: readonly string[]
  shared: ReadonlySet<string>
}) {
  return (
    <div className="flex max-h-20 flex-wrap items-center gap-1 overflow-auto">
      <span className="mr-1 text-muted-foreground">{label}</span>
      {items.map((x) => (
        <span
          key={x}
          className={
            'rounded border px-1 font-mono whitespace-pre ' +
            (shared.has(x) ? 'border-primary bg-primary/20' : 'border-border text-muted-foreground')
          }
        >
          {x.replaceAll(' ', '␣')}
        </span>
      ))}
    </div>
  )
}
