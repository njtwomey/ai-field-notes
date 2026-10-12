import { useMemo } from 'react'
import { Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function PatternMiningVisualizer() {
  const state = useFigureState({
    minSupport: slider(2, 20, 6, {
      step: 1,
      label: 'minimum support threshold (%)',
    }),
    minConfidence: slider(20, 85, 45, {
      step: 5,
      label: 'minimum confidence threshold (%)',
    }),
    minLift: slider(0.8, 3.5, 1.2, {
      step: 0.1,
      label: 'minimum lift threshold',
    }),
  })

  const { minSupport, minConfidence, minLift } = state

  const results = useMemo(() => {
    // 6 association rules:
    // R1: Diapers => Beer (Supp 8%, Conf 62%, Lift 4.8)
    // R2: Bread => Butter (Supp 16%, Conf 72%, Lift 3.6)
    // R3: Pasta => Tomato Sauce (Supp 11%, Conf 65%, Lift 3.8)
    // R4: Coffee => Bag (Supp 19%, Conf 88%, Lift 0.98 - FALSE POSITIVE LIFT <= 1)
    // R5: Coke => Pepsi (Supp 3%, Conf 15%, Lift 0.45 - SUBSTITUTE)
    // R6: Cereal => Milk (Supp 14%, Conf 78%, Lift 3.1)
    const rules = [
      { id: 1, name: 'Diapers ⇒ Beer', supp: 8.0, conf: 62.0, lift: 4.8 },
      { id: 2, name: 'Bread ⇒ Butter', supp: 16.0, conf: 72.0, lift: 3.6 },
      { id: 3, name: 'Pasta ⇒ Sauce', supp: 11.0, conf: 65.0, lift: 3.8 },
      { id: 4, name: 'Coffee ⇒ Carrier Bag', supp: 19.0, conf: 88.0, lift: 0.98 },
      { id: 5, name: 'Coke ⇒ Pepsi', supp: 3.0, conf: 15.0, lift: 0.45 },
      { id: 6, name: 'Cereal ⇒ Milk', supp: 14.0, conf: 78.0, lift: 3.1 },
    ]

    const suppVals = rules.map((r) => r.supp)
    const confVals = rules.map((r) => r.conf)
    const validRules = rules.filter((r) => r.supp >= minSupport && r.conf >= minConfidence && r.lift >= minLift)

    // Threshold lines
    // Vertical line at minSupport for Y in [0, 100]
    // Horizontal line at minConfidence for X in [0, 25]
    const suppLineX = [minSupport, minSupport]
    const suppLineY = [0, 100]
    const confLineX = [0, 25]
    const confLineY = [minConfidence, minConfidence]

    const r4Valid = rules[3].supp >= minSupport && rules[3].conf >= minConfidence && rules[3].lift >= minLift

    return {
      x: suppVals,
      y: confVals,
      suppLineX,
      suppLineY,
      confLineX,
      confLineY,
      validCount: validRules.length,
      r4Valid,
    }
  }, [minSupport, minConfidence, minLift])

  const axisX = useAxis({
    label: 'Rule Support (%) — Observed Co-occurrence Frequency',
    range: [0, 25],
  })

  const axisY = useAxis({
    label: 'Rule Confidence (%) — P(Consequent | Antecedent)',
    range: [0, 100],
  })

  return (
    <Figure
      title="Association Rule Quality: Support, Confidence, and Lift Explorer"
      state={state}
      hoverReadout={false}
      caption="Evaluating candidate association rules across support, confidence, and lift. Note how Coffee ⇒ Carrier Bag exhibits 88% confidence yet fails Lift filtering because bags are ubiquitously purchased independently."
      readouts={
        <>
          <Readout label="Active Valid Rules" value={`${results.validCount} / 6 rules`} />
          <Readout
            label="Carrier Bag Rule (R4)"
            value={results.r4Valid ? 'Retained' : 'PRUNED BY LIFT (Lift = 0.98 ≤ 1.0)'}
            color={results.r4Valid ? 'var(--amber-600)' : 'var(--blue-600)'}
          />
          <Readout
            label="Lift Importance"
            value="Lift filters out high-confidence rules driven solely by consequent popularity"
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Rule Evaluation Space">
        <Points name="Candidate Association Rules" x={results.x} y={results.y} slot={0} size={8} />
        <Curve name="Min Support Threshold" x={results.suppLineX} y={results.suppLineY} slot={2} dashed width={1.8} />
        <Curve
          name="Min Confidence Threshold"
          x={results.confLineX}
          y={results.confLineY}
          slot={1}
          dashed
          width={1.8}
        />
      </Plot>
    </Figure>
  )
}
