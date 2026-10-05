import { useMemo, useState } from 'react'
import type { Tensor } from 'aifn-compute/foundation/tensor'
import { analogy } from 'aifn-compute/text/representations'
import { tokenId } from 'aifn-compute/text/vocabulary'
import { Figure } from 'aifn-render/layout'
import { choice, useFigureState } from 'aifn-render/state'
import { Input } from 'aifn-render/ui/input'
import { Bars, Plot, Readout, useAxis } from 'aifn-render/viz'
import type { Model } from './showcase'

/** Analogies the topic corpus can support, with the answer its frames imply. */
const PRESETS = {
  'cats − cat + dog': ['cats', 'cat', 'dog', 'dogs'],
  'buses − bus + car': ['buses', 'bus', 'car', 'cars'],
  'chase − chases + bakes': ['chase', 'chases', 'bakes', 'bake'],
  'paints − painter + baker': ['paints', 'painter', 'baker', 'bakes'],
  'fish − cat + chef': ['fish', 'cat', 'chef', 'bread'],
  custom: ['', '', '', ''],
} as const
type Preset = keyof typeof PRESETS
const COUNT = 8

export function AnalogyFigure({ model, vectors, rank }: { model: Model; vectors: Tensor; rank: number }) {
  const state = useFigureState({
    preset: choice(Object.keys(PRESETS) as Preset[], 'cats − cat + dog', { label: 'analogy' }),
  })
  const { preset } = state
  const [typed, setTyped] = useState<[string, string, string]>(['cats', 'cat', 'dog'])
  const [a, b, c] = preset === 'custom' ? typed : (PRESETS[preset].slice(0, 3) as unknown as [string, string, string])
  const expected = preset === 'custom' ? null : PRESETS[preset][3]

  const result = useMemo(() => {
    if (model.itemKind !== 'word' || !model.vocabulary)
      return { error: 'Analogies need word vectors: pick a word representation above.' }
    const missing = [a, b, c].filter((x) => tokenId(model.vocabulary!, x) < 0)
    if (missing.length > 0) return { error: `Not in the vocabulary: ${missing.join(', ')}.` }
    const all = analogy(vectors, model.vocabulary, a, b, c, { count: model.items.length - 3 })
    const at = expected ? all.findIndex((x) => x.word === expected) : -1
    return { answers: all.slice(0, COUNT), rankOfExpected: at >= 0 ? at + 1 : null }
  }, [model, vectors, a, b, c, expected])

  const answers = 'answers' in result ? result.answers! : []
  const names = answers.map((x) => x.word)
  const wordAxis = useAxis({ categories: [...names].reverse() })
  const cosAxis = useAxis({ label: 'cosine to â − b̂ + ĉ', range: [undefined, 1] })

  return (
    <Figure
      title="Analogies by vector offset"
      purpose="If a relation is a consistent direction in the space, a − b + c lands near the word that completes the analogy; in a small corpus only relations its frames repeat (number agreement, agent and action) come out."
      state={state}
      defaultSize="M"
      controls={
        preset === 'custom' ? (
          <div className="col-span-full flex flex-wrap items-center gap-2 text-sm">
            {(['a', 'b', 'c'] as const).map((name, i) => (
              <label key={name} className="flex items-center gap-1">
                <span className="text-muted-foreground">{i === 0 ? 'a' : i === 1 ? '− b' : '+ c'}</span>
                <Input
                  aria-label={name}
                  value={typed[i]}
                  onChange={(e) =>
                    setTyped(
                      (t) =>
                        t.map((x, j) => (j === i ? e.target.value.trim().toLowerCase() : x)) as [
                          string,
                          string,
                          string,
                        ],
                    )
                  }
                  className="h-8 w-28"
                />
              </label>
            ))}
          </div>
        ) : undefined
      }
      readouts={
        <>
          <Readout label="query" value={`${a} − ${b} + ${c}`} />
          <Readout label="vectors" value={`${model.rep}, rank ${rank}`} />
          {expected && 'rankOfExpected' in result && (
            <Readout label={`rank of “${expected}”`} value={result.rankOfExpected ?? 'absent'} />
          )}
        </>
      }
      caption="The answer is the word whose unit vector has the highest cosine with â − b̂ + ĉ, the three inputs excluded (3CosAdd), using the rank-k vectors of the representation chosen above. The presets are relations the corpus's frames repeat: plural nouns take base-form verbs, each agent has its own actions and objects. Expect partial success: with a few hundred sentences and a hundred-odd words the expected answer is often near the top rather than first, and it changes with the weighting, window and rank. Co-occurrence with PPMI and k near 10 does best here; one-hot cannot do it at all, since every offset is equally far from every word."
    >
      {'error' in result ? (
        <div className="text-sm text-muted-foreground">{result.error}</div>
      ) : (
        <Plot x={cosAxis} y={wordAxis}>
          <Bars
            name="cosine"
            x={names.map((_, i) => names.length - 1 - i)}
            y={answers.map((x) => x.cosine)}
            orient="y"
            slot={0}
            width={0.7}
          />
          {expected && names.includes(expected) && (
            <Bars
              name="expected answer"
              x={[names.length - 1 - names.indexOf(expected)]}
              y={[answers[names.indexOf(expected)].cosine]}
              orient="y"
              emphasis
              width={0.7}
            />
          )}
        </Plot>
      )}
    </Figure>
  )
}
