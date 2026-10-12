import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DatasetDocumentationVisualizer() {
  const state = useFigureState({
    datasetMaturity: choice<'raw_scrape' | 'academic_dump' | 'enterprise_datasheet'>(
      [
        { value: 'raw_scrape', label: 'Uncurated Web Scrape (Zero documentation, high liability)' },
        { value: 'academic_dump', label: 'Academic Benchmark (Paper published, ambiguous license)' },
        { value: 'enterprise_datasheet', label: 'Audited Enterprise Asset (Full Gebru et al. Datasheet)' },
      ],
      'academic_dump',
      { label: 'documentation regime' },
    ),
    complianceHurdle: slider(40, 90, 70, {
      step: 5,
      label: 'production compliance threshold score (%)',
    }),
  })

  const results = useMemo(() => {
    // 5 pillars of Gebru et al. 2021 Datasheet standard:
    // 1: Motivation & Intent, 2: Composition & Labels, 3: Collection & Consent, 4: Cleaning & Pipeline, 5: Licensing & Distribution
    const pillars = [1, 2, 3, 4, 5]

    let scores: number[] = []
    let license = ''
    let piiRisk = ''

    if (state.datasetMaturity === 'raw_scrape') {
      scores = [15, 20, 10, 25, 5]
      license = 'High Risk (Implied copyright / ToS breach)'
      piiRisk = 'Severe (Unchecked raw user telemetry)'
    } else if (state.datasetMaturity === 'academic_dump') {
      scores = [80, 65, 50, 60, 45]
      license = 'Ambiguous (CC-BY-NC / Non-commercial conflict)'
      piiRisk = 'Moderate (Anonymized without formal verification)'
    } else {
      scores = [98, 95, 92, 94, 96]
      license = 'Verified (Apache 2.0 / Commercial safe)'
      piiRisk = 'Low (Automated redaction + signed consent)'
    }

    const hurdle = state.complianceHurdle
    const hurdleLine = pillars.map(() => hurdle)

    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length
    const passesAudit = avgScore >= hurdle && scores.every((s) => s >= hurdle - 15)

    return {
      pillars,
      scores,
      hurdleLine,
      avgScore,
      passesAudit,
      license,
      piiRisk,
    }
  }, [state.complianceHurdle, state.datasetMaturity])

  const axisX = useAxis({
    label: 'Datasheet Audit Pillar (1: Intent, 2: Composition, 3: Collection, 4: Cleaning, 5: Licensing)',
    range: [0.8, 5.2],
  })

  const axisY = useAxis({
    label: 'Governance Completeness Score (%)',
    range: [0, 105],
  })

  return (
    <Figure
      title="Dataset Provenance, Datasheet Completeness and Governance Auditing"
      state={state}
      hoverReadout={false}
      caption="Documenting dataset provenance, collection mechanisms, labeling methodologies, and license terms (Gebru et al. 2021) is critical for enterprise safety. Unaudited web scrapes introduce severe copyright, ToS, and regulatory liabilities when ingested into production pipelines."
      readouts={
        <>
          <Readout label="Overall Audit Score" value={`${results.avgScore.toFixed(0)}%`} />
          <Readout
            label="Compliance Decision"
            value={results.passesAudit ? 'Approved for Production' : 'Blocked by Governance'}
          />
          <Readout label="Licensing Risk" value={results.license} />
          <Readout label="Personal Data / PII Risk" value={results.piiRisk} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Datasheet Pillar Compliance vs Hurdle">
        <Points name="Pillar Audit Score" x={results.pillars} y={results.scores} slot={0} size={8} />
        <Curve name="Pillar Trajectory" x={results.pillars} y={results.scores} slot={0} width={2} />
        <Curve
          name="Enterprise Compliance Hurdle"
          x={results.pillars}
          y={results.hurdleLine}
          slot={2}
          dashed
          width={2}
        />
      </Plot>
    </Figure>
  )
}
