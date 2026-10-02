import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'

type Word = 'rib' | 'rob'
const P_AXIS = linspace(0.01, 0.99, 99)

/**
 * The label-bias example of Lafferty, McCallum and Pereira. Two paths leave the start on "r": 1→2→3 spells "rib",
 * 4→5→3 spells "rob". States 1 and 4 each have one successor, so an MEMM's local softmax gives that transition
 * probability 1 whatever the second letter is; only the first transition, fixed by the training share p of "rib",
 * decides. A CRF normalises over whole paths, so the evidence feature on the second letter (weight w) is not cancelled.
 */
function probabilities(p: number, w: number, word: Word) {
  // MEMM: P(path | x) is the product of local transition probabilities; the middle letter never enters.
  const memm = { rib: p, rob: 1 - p }
  // CRF: path scores add the start bias and the evidence feature, then normalise over both paths.
  const matchRib = word === 'rib' ? w : 0
  const matchRob = word === 'rob' ? w : 0
  const sRib = Math.log(p) + matchRib
  const sRob = Math.log(1 - p) + matchRob
  const top = Math.max(sRib, sRob)
  const eRib = Math.exp(sRib - top)
  const eRob = Math.exp(sRob - top)
  const crf = { rib: eRib / (eRib + eRob), rob: eRob / (eRib + eRob) }
  return { memm, crf }
}

export function LabelBias() {
  const p = useParam(0.6, { min: 0.01, max: 0.99, step: 0.01 })
  const w = useParam(4, { min: 0, max: 8, step: 0.1 })
  const [word, setWord] = useState<Word>('rob')

  const series = useMemo((): XYSeries[] => {
    const curves = P_AXIS.map((q) => probabilities(q, w.value, word))
    return [
      { name: 'MEMM: P(correct path | x)', type: 'line', x: P_AXIS, y: curves.map((c) => c.memm[word]), slot: 1 },
      { name: 'CRF: P(correct path | x)', type: 'line', x: P_AXIS, y: curves.map((c) => c.crf[word]), slot: 0 },
    ]
  }, [w.value, word])

  const here = probabilities(p.value, w.value, word)
  const decode = (d: { rib: number; rob: number }) => (d.rib >= d.rob ? 'rib' : 'rob')
  const handles: Handle[] = [{ kind: 'x', at: p.value, label: 'p', onDrag: (x) => p.set(x) }]

  return (
    <Interactive
      title="Label bias"
      caption="Two words, rib and rob, share their first and last letters. In training, a fraction p of the words starting with r were rib. An MEMM decides between the two paths at the first letter and then passes probability 1 along each path, because each middle state has only one successor, so the second letter cannot change its mind. A CRF weighs the evidence of the second letter (weight w) against the same prior, over whole paths. Choose the observed word and drag p."
      controls={
        <>
          <ParamChoice
            label="observed word x"
            value={word}
            onChange={setWord}
            options={[
              { value: 'rob', label: 'r o b' },
              { value: 'rib', label: 'r i b' },
            ]}
          />
          <ParamSlider label="training share of rib, p" param={p} />
          <ParamSlider label="CRF evidence weight w" param={w} />
        </>
      }
      readout={
        <>
          <Readout label="MEMM decodes" value={decode(here.memm)} />
          <Readout label="MEMM P(correct)" value={formatNumber(here.memm[word])} />
          <Readout label="CRF decodes" value={decode(here.crf)} />
          <Readout label="CRF P(correct)" value={formatNumber(here.crf[word])} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="training share of rib, p"
        yLabel="P(correct path | x)"
        xRange={[0, 1]}
        yRange={[0, 1]}
        handles={handles}
        height={300}
      />
    </Interactive>
  )
}
