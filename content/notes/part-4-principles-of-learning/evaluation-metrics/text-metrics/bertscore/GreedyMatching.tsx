import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamChoice, Readout, formatNumber, type HeatmapOverlay } from 'aifn-render'

/**
 * Toy two-dimensional "contextual embeddings": each word is a unit vector at an angle, and related words sit at nearby
 * angles. Real BERTScore uses high-dimensional contextual vectors from a pretrained model; the matching is the same.
 */
const ANGLE: Record<string, number> = {
  the: 0,
  a: 10,
  cat: 60,
  kitten: 70,
  dog: 85,
  sat: 120,
  sits: 115,
  down: 170,
}
const vec = (w: string): [number, number] => {
  const t = (ANGLE[w] * Math.PI) / 180
  return [Math.cos(t), Math.sin(t)]
}
const cos = (a: string, b: string) => {
  const [x1, y1] = vec(a)
  const [x2, y2] = vec(b)
  return x1 * x2 + y1 * y2
}

const REFERENCE = ['the', 'cat', 'sat']
const CANDIDATES = {
  paraphrase: ['a', 'kitten', 'sits', 'down'],
  exact: ['the', 'cat', 'sat'],
  wrong: ['a', 'dog', 'sat'],
} as const
type Candidate = keyof typeof CANDIDATES

/** Pairwise cosine similarities, with each token's best match marked: the greedy matching behind BERTScore. */
export function GreedyMatching() {
  const [choice, setChoice] = useState<Candidate>('paraphrase')
  const cand = CANDIDATES[choice]

  const r = useMemo(() => {
    // z[i][j]: candidate token i (rows) against reference token j (columns).
    const z = cand.map((c) => REFERENCE.map((w) => cos(c, w)))
    const rowBest = z.map((row) => row.indexOf(Math.max(...row)))
    const colBest = REFERENCE.map((_, j) => {
      const col = z.map((row) => row[j])
      return col.indexOf(Math.max(...col))
    })
    const precision = z.reduce((s, row) => s + Math.max(...row), 0) / cand.length
    const recall = REFERENCE.reduce((s, _, j) => s + Math.max(...z.map((row) => row[j])), 0) / REFERENCE.length
    const f = (2 * precision * recall) / (precision + recall)
    return { z, rowBest, colBest, precision, recall, f }
  }, [cand])

  const overlay: HeatmapOverlay[] = [
    {
      name: 'best reference match for each candidate token (precision)',
      type: 'scatter',
      x: r.rowBest,
      y: cand.map((_, i) => i),
      slot: 1,
    },
    {
      name: 'best candidate match for each reference token (recall)',
      type: 'scatter',
      x: REFERENCE.map((_, j) => j),
      y: r.colBest,
      emphasis: true,
    },
  ]

  return (
    <Interactive
      title="Greedy matching of token embeddings"
      caption={`Rows are candidate tokens (${cand.join(', ')}), columns reference tokens (${REFERENCE.join(', ')}); each cell is the cosine similarity of their embeddings. Precision averages each row's maximum, recall each column's maximum. Toy two-dimensional embeddings stand in for a pretrained model's contextual vectors, so the paraphrase scores high although it shares no word with the reference.`}
      controls={
        <ParamChoice
          label="candidate"
          value={choice}
          onChange={setChoice}
          options={[
            { value: 'paraphrase', label: 'a kitten sits down' },
            { value: 'exact', label: 'the cat sat' },
            { value: 'wrong', label: 'a dog sat' },
          ]}
        />
      }
      readout={
        <>
          <Readout label="precision" value={formatNumber(r.precision)} />
          <Readout label="recall" value={formatNumber(r.recall)} />
          <Readout label="F1" value={formatNumber(r.f)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Heatmap
          x={REFERENCE.map((_, j) => j)}
          y={cand.map((_, i) => i)}
          z={r.z}
          range={[-1, 1]}
          scale="diverging"
          xLabel={`reference token (${REFERENCE.join(' · ')})`}
          yLabel="candidate token index"
          valueLabel="cosine"
          overlay={overlay}
          height={300}
        />
      </div>
    </Interactive>
  )
}
