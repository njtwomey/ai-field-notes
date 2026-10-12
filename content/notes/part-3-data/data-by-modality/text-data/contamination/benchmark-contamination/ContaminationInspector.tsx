import { useMemo } from 'react'
import { choice, Curve, Figure, Handle, int, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'

const BENCHMARK_ITEM =
  'A train leaves Chicago travelling east at 60 miles per hour. Two hours later, a second train leaves Chicago on a parallel track travelling at 80 miles per hour. How many hours after the second train departs does it overtake the first train?'

const DOCUMENTS = {
  contaminated: {
    label: 'verbatim benchmark leak',
    text: 'Exam problem 4: A train leaves Chicago travelling east at 60 miles per hour. Two hours later, a second train leaves Chicago on a parallel track travelling at 80 miles per hour. How many hours after the second train departs does it overtake the first train? Solution: In two hours the first train travels 120 miles. The relative speed is 20 miles per hour, so it takes 6 hours.',
  },
  paraphrased: {
    label: 'paraphrased variant',
    text: 'A locomotive departs Chicago headed eastward moving at 60 mph. After 2 hours, another train starts from Chicago on an adjacent track at 80 mph. Determine the duration in hours until the faster train catches the initial train.',
  },
  clean: {
    label: 'clean topical text',
    text: 'High-speed passenger rail networks connecting Chicago and Detroit operate multiple train services daily. Operating speeds exceed 80 miles per hour along dedicated freight corridors, with automated scheduling ensuring safe track separation.',
  },
} as const

type DocKey = keyof typeof DOCUMENTS

function getNGrams(tokens: string[], n: number): Set<string> {
  const set = new Set<string>()
  for (let i = 0; i + n <= tokens.length; i++) {
    set.add(tokens.slice(i, i + n).join(' '))
  }
  return set
}

function tokenize(text: string): string[] {
  const clean = text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .trim()
  return clean ? clean.split(/\s+/) : []
}

export function ContaminationInspector() {
  const state = useFigureState({
    docKey: choice<DocKey>(
      [
        { value: 'contaminated', label: 'verbatim test leak' },
        { value: 'paraphrased', label: 'paraphrased variant' },
        { value: 'clean', label: 'clean topical text' },
      ],
      'contaminated',
      { label: 'training document' },
    ),
    n: int(8, { min: 2, max: 15, step: 1, label: 'n-gram length n' }),
    threshold: slider(0.05, 0.5, 0.15, { step: 0.05, label: 'contamination threshold' }),
  })

  const { docKey, n, threshold } = state

  const benchTokens = useMemo(() => tokenize(BENCHMARK_ITEM), [])

  // Overlap curves across n = 2..15 for all three documents
  const curves = useMemo(() => {
    const ns = Array.from({ length: 14 }, (_, i) => i + 2)
    const computeOverlap = (targetTokens: string[]) => {
      return ns.map((currN) => {
        const bSet = getNGrams(benchTokens, currN)
        if (bSet.size === 0) return 0
        const dSet = getNGrams(targetTokens, currN)
        let matches = 0
        for (const gram of bSet) {
          if (dSet.has(gram)) matches++
        }
        return matches / bSet.size
      })
    }

    const contTokens = tokenize(DOCUMENTS.contaminated.text)
    const paraTokens = tokenize(DOCUMENTS.paraphrased.text)
    const clnTokens = tokenize(DOCUMENTS.clean.text)

    return {
      ns,
      contaminated: computeOverlap(contTokens),
      paraphrased: computeOverlap(paraTokens),
      clean: computeOverlap(clnTokens),
    }
  }, [benchTokens])

  const activeCurve = curves[docKey]
  const activeRate = activeCurve[n - 2]
  const isContaminated = activeRate >= threshold

  const xAxis = useAxis({ label: 'n-gram length n', range: [2, 15] })
  const yAxis = useAxis({ label: 'benchmark n-gram overlap rate', range: [0, 1] })

  return (
    <Figure
      title="Benchmark contamination detection by n-gram overlap"
      state={state}
      hoverReadout={false}
      caption="Overlap fraction between a test benchmark prompt and three candidate training documents as n-gram length n varies from 2 to 15. Common vocabulary causes clean text to share 2-grams and 3-grams by coincidence, but overlap collapses to zero for n ≥ 5. Verbatim contamination retains high overlap even at n = 8 or n = 13. Drag the n-gram handle to observe the sensitivity of contamination auditing."
      readouts={
        <>
          <Readout label="active n-gram size" value={String(n)} />
          <Readout label="overlap rate" value={`${(activeRate * 100).toFixed(1)}%`} />
          <Readout label="audit status" value={isContaminated ? 'flagged contaminated' : 'clean'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="overlap decay as n-gram length increases">
        <Curve name="verbatim leak" x={curves.ns} y={curves.contaminated} slot={0} />
        <Curve name="paraphrased" x={curves.ns} y={curves.paraphrased} slot={1} />
        <Curve name="clean topical" x={curves.ns} y={curves.clean} muted dashed />
        <Curve name="threshold" x={[2, 15]} y={[threshold, threshold]} slot={2} dashed />
        <Handle
          kind="x"
          at={n}
          label={`n=${n}`}
          onDrag={(val) => state.set('n', Math.max(2, Math.min(15, Math.round(val))))}
        />
      </Plot>
    </Figure>
  )
}
