import { ArrowDown, ArrowUp, Minus, Plus } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Interactive, ParamButton, ParamSlider, Readout, XYChart, formatNumber, useParam } from '@/components/viz'
import { averagePrecision, isRelevant, ndcg, precisionAt, recallAt, reciprocalRank } from './ranking'

const START = [2, 0, 3, 1, 0, 0, 2, 0, 1, 0]
const MAX_GRADE = 3

/**
 * One query's ranked list, editable: move documents up and down, change their relevance grades, and set how many
 * relevant documents the ranking missed. Every ranking metric in this subtopic is computed from the same list.
 */
export function RankingExplorer() {
  const [grades, setGrades] = useState<number[]>(START)
  const k = useParam(5, { min: 1, max: START.length, step: 1 })
  const missed = useParam(1, { min: 0, max: 5, step: 1 })

  const retrieved = grades.filter(isRelevant).length
  const totalRelevant = retrieved + missed.value
  const move = (i: number, d: -1 | 1) =>
    setGrades((g) => {
      const j = i + d
      if (j < 0 || j >= g.length) return g
      const next = [...g]
      ;[next[i], next[j]] = [next[j], next[i]]
      return next
    })
  const regrade = (i: number, d: -1 | 1) =>
    setGrades((g) => g.map((v, j) => (j === i ? Math.min(MAX_GRADE, Math.max(0, v + d)) : v)))

  const ranks = grades.map((_, i) => i + 1)
  const inTop = (i: number) => i < k.value

  return (
    <Interactive
      title="One ranked list, every ranking metric"
      caption="Each row is a retrieved document, in rank order, with a relevance grade from 0 (not relevant) to 3. Move documents up or down, change their grades, and set how many relevant documents the ranking missed entirely. Precision, recall, average precision and reciprocal rank treat any grade above 0 as relevant; NDCG uses the grades themselves. The bars show the grades, with the top k highlighted."
      controls={
        <>
          <ParamSlider label="cut-off k" param={k} format={(v) => String(v)} withArrows />
          <ParamSlider label="relevant documents not retrieved" param={missed} format={(v) => String(v)} withArrows />
          <ParamButton onClick={() => setGrades([...grades].sort((a, b) => b - a))}>
            Sort into the ideal order
          </ParamButton>
          <ParamButton onClick={() => setGrades([...grades].reverse())}>Reverse</ParamButton>
          <ParamButton onClick={() => setGrades(START)}>Reset</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label={`P@${k.value}`} value={formatNumber(precisionAt(grades, k.value))} />
          <Readout label={`R@${k.value}`} value={formatNumber(recallAt(grades, k.value, totalRelevant))} />
          <Readout label="AP" value={formatNumber(averagePrecision(grades, totalRelevant))} />
          <Readout label="reciprocal rank" value={formatNumber(reciprocalRank(grades))} />
          <Readout label={`NDCG@${k.value}`} value={formatNumber(ndcg(grades, k.value))} />
          <Readout
            label={`R-precision (R = ${totalRelevant})`}
            value={formatNumber(precisionAt(grades, totalRelevant))}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <ol className="space-y-1 text-sm">
          {grades.map((g, i) => (
            <li
              key={i}
              className={`flex items-center gap-2 rounded-md px-2 py-1 ${inTop(i) ? 'bg-muted' : 'text-muted-foreground'}`}
            >
              <span className="w-6 text-right font-mono text-xs tabular-nums">{i + 1}</span>
              <span className="flex-1">
                grade <span className="font-mono tabular-nums">{g}</span>
                {isRelevant(g) ? '' : ' (not relevant)'}
              </span>
              <Button variant="ghost" size="icon-xs" aria-label="Lower grade" onClick={() => regrade(i, -1)}>
                <Minus />
              </Button>
              <Button variant="ghost" size="icon-xs" aria-label="Raise grade" onClick={() => regrade(i, 1)}>
                <Plus />
              </Button>
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label="Move up"
                onClick={() => move(i, -1)}
                disabled={i === 0}
              >
                <ArrowUp />
              </Button>
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label="Move down"
                onClick={() => move(i, 1)}
                disabled={i === grades.length - 1}
              >
                <ArrowDown />
              </Button>
            </li>
          ))}
        </ol>
        <XYChart
          height={320}
          xLabel="rank"
          yLabel="relevance grade"
          yRange={[0, MAX_GRADE]}
          series={[
            {
              name: 'in the top k',
              type: 'bar',
              x: ranks.filter((_, i) => inTop(i)),
              y: grades.filter((_, i) => inTop(i)),
              slot: 0,
            },
            {
              name: 'below the cut-off',
              type: 'bar',
              x: ranks.filter((_, i) => !inTop(i)),
              y: grades.filter((_, i) => !inTop(i)),
              muted: true,
            },
          ]}
        />
      </div>
    </Interactive>
  )
}
