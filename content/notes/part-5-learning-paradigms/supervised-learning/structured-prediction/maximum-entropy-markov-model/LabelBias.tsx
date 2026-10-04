import { useMemo } from 'react'
import { choice, Curve, Figure, float, formatNumber, Handle, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

type Word = 'rib' | 'rob'
const P_AXIS = toFlat(linspace(0.01, 0.99, 99))

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
  const state = useFigureState({
    word: choice<Word>(
      [
        { value: 'rob', label: 'r o b' },
        { value: 'rib', label: 'r i b' },
      ],
      'rob',
      { label: 'observed word x' },
    ),
    p: float(0.6, { min: 0.01, max: 0.99, step: 0.01, label: 'training share of rib, p' }),
    w: float(4, { min: 0, max: 8, step: 0.1, label: 'CRF evidence weight w' }),
  })

  const series = useMemo(() => {
    const curves = P_AXIS.map((q) => probabilities(q, state.w, state.word))
    return [
      { name: 'MEMM: P(correct path | x)', x: P_AXIS, y: curves.map((c) => c.memm[state.word]), slot: 1 },
      { name: 'CRF: P(correct path | x)', x: P_AXIS, y: curves.map((c) => c.crf[state.word]), slot: 0 },
    ] as const
  }, [state.w, state.word])

  const here = probabilities(state.p, state.w, state.word)
  const decode = (d: { rib: number; rob: number }) => (d.rib >= d.rob ? 'rib' : 'rob')

  const xAxis = useAxis({ label: 'training share of rib, p', range: [0, 1] })
  const yAxis = useAxis({ label: 'P(correct path | x)', range: [0, 1] })
  return (
    <Figure
      title="Label bias"
      state={state}
      caption="Two words, rib and rob, share their first and last letters. In training, a fraction p of the words starting with r were rib. An MEMM decides between the two paths at the first letter and then passes probability 1 along each path, because each middle state has only one successor, so the second letter cannot change its mind. A CRF weighs the evidence of the second letter (weight w) against the same prior, over whole paths. Choose the observed word and drag p."

      readouts={
        <>
          <Readout label="MEMM decodes" value={decode(here.memm)} />
          <Readout label="MEMM P(correct)" value={formatNumber(here.memm[state.word])} />
          <Readout label="CRF decodes" value={decode(here.crf)} />
          <Readout label="CRF P(correct)" value={formatNumber(here.crf[state.word])} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Handle {...state.handle('p', { label: 'p' })} />
      </Plot>
    </Figure>
  )
}
