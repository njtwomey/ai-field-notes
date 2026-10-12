import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SearchRelevanceVisualizer() {
  const state = useFigureState({
    modelType: choice<'cascade' | 'dbn'>(
      [
        { value: 'cascade', label: 'Cascade Model (Craswell et al. 2008: stops at first satisfied click)' },
        { value: 'dbn', label: 'Dynamic Bayesian Network (Chapelle & Zhang 2009: multi-click continuation)' },
      ],
      'cascade',
      { label: 'click model' },
    ),
    persistenceGamma: slider(0.4, 0.95, 0.75, {
      step: 0.05,
      label: 'user browsing persistence (continuation probability gamma)',
    }),
  })

  const results = useMemo(() => {
    const ranks = [1, 2, 3, 4, 5]
    // 5 documents with graded relevance [0..4]
    // Rank 1: Fair (1)
    // Rank 2: Excellent (3)
    // Rank 3: Irrelevant (0)
    // Rank 4: Perfect (4)
    // Rank 5: Good (2)
    const grades = [1, 3, 0, 4, 2]
    // Normalized attractiveness / click probability given examination: alpha_k = grades[k] / 4.0
    const attractiveness = grades.map((g) => Math.max(0.1, g / 4.0))
    // Satisfaction probability given click: s_k
    const satisfaction = grades.map((g) => g / 4.0)

    const gamma = state.persistenceGamma
    const examinationProb: number[] = []
    const clickProb: number[] = []

    let currentExam = 1.0 // P(E_1 = 1) = 1.0

    ranks.forEach((_, idx) => {
      examinationProb.push(currentExam)
      const a = attractiveness[idx]
      const s = satisfaction[idx]
      const c = currentExam * a
      clickProb.push(c)

      if (state.modelType === 'cascade') {
        // Cascade: user abandons if satisfied upon click
        // Next doc examined if current doc not clicked, or clicked but not satisfied
        currentExam = currentExam * (1.0 - a * s)
      } else {
        // DBN: if not clicked, continues with prob gamma. If clicked and not satisfied, continues with prob gamma.
        currentExam = currentExam * (1.0 - a * s) * gamma
      }
    })

    // Compute DCG and IDCG for current ranking
    let dcg = 0
    ranks.forEach((k, idx) => {
      const gain = Math.pow(2, grades[idx]) - 1
      dcg += gain / Math.log2(k + 1)
    })

    const sortedGrades = [...grades].sort((a, b) => b - a)
    let idcg = 0
    ranks.forEach((k, idx) => {
      const gain = Math.pow(2, sortedGrades[idx]) - 1
      idcg += gain / Math.log2(k + 1)
    })
    const ndcg = idcg > 0 ? dcg / idcg : 1.0

    const totalClicks = clickProb.reduce((acc, v) => acc + v, 0)
    const rank4ClickProb = clickProb[3] // The Perfect item at rank 4

    return {
      ranks,
      examinationProb,
      clickProb,
      ndcg,
      totalClicks,
      rank4ClickProb,
    }
  }, [state.modelType, state.persistenceGamma])

  const axisX = useAxis({
    label: 'SERP Display Rank (1 to 5)',
    range: [0.8, 5.2],
  })

  const axisY = useAxis({
    label: 'Probability',
    range: [0, 1.05],
  })

  return (
    <Figure
      title="User Browsing and Cascade Click Models on Search Result Pages"
      state={state}
      hoverReadout={false}
      caption="Under cascade and DBN click models, user examination decays sequentially down the SERP. Even though Rank 4 holds the highest graded relevance (Perfect grade 4), its observed click rate is suppressed because satisfied clicks at earlier ranks terminate the search session."
      readouts={
        <>
          <Readout label="SERP NDCG@5" value={results.ndcg.toFixed(3)} />
          <Readout label="Rank 1 Click P(C1)" value={`${(results.clickProb[0] * 100).toFixed(1)}%`} />
          <Readout label="Rank 4 Click P(C4) [Grade 4]" value={`${(results.rank4ClickProb * 100).toFixed(1)}%`} />
          <Readout label="Expected Total Clicks" value={results.totalClicks.toFixed(2)} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Examination Decay vs Document Click Likelihood">
        <Curve name="Examination P(Examined | k)" x={results.ranks} y={results.examinationProb} slot={0} width={2.5} />
        <Points name="Observed Click P(Click | k)" x={results.ranks} y={results.clickProb} slot={1} size={8} />
      </Plot>
    </Figure>
  )
}
