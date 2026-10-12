import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function UnitOfObservationVisualizer() {
  const state = useFigureState({
    plotView: choice<'effective_n' | 'design_effect'>(
      [
        { value: 'effective_n', label: 'Effective Sample Size (N_eff) vs Correlation' },
        { value: 'design_effect', label: 'Kish Design Effect (Deff = 1 + (m - 1)*rho)' },
      ],
      'effective_n',
      { label: 'statistical view' },
    ),
    clusterSize: slider(1, 50, 10, {
      step: 1,
      label: 'cluster size m (average rows recorded per independent entity)',
    }),
    icc: slider(0.0, 0.5, 0.15, {
      step: 0.01,
      label: 'intra-class correlation rho (within-cluster dependence)',
    }),
  })

  const results = useMemo(() => {
    const nominalN = 10000
    const m = state.clusterSize
    const rho = state.icc
    const numEntities = Math.round(nominalN / m)

    // Current point
    const currentDeff = 1.0 + (m - 1) * rho
    const currentNeff = nominalN / currentDeff
    const sampleLossPct = ((nominalN - currentNeff) / nominalN) * 100.0

    // Sweep across rho [0 .. 0.5]
    const rhoGrid: number[] = []
    const neffCurve: number[] = []

    for (let r = 0; r <= 0.5; r += 0.02) {
      rhoGrid.push(r)
      const deff = 1.0 + (m - 1) * r
      neffCurve.push(nominalN / deff)
    }

    // Sweep across m [1 .. 50] for design effect
    const mGrid: number[] = []
    const deffCurve: number[] = []
    for (let cm = 1; cm <= 50; cm += 1) {
      mGrid.push(cm)
      deffCurve.push(1.0 + (cm - 1) * rho)
    }

    return {
      nominalN,
      numEntities,
      currentDeff,
      currentNeff: Math.round(currentNeff),
      sampleLossPct,
      rhoGrid,
      neffCurve,
      mGrid,
      deffCurve,
    }
  }, [state.clusterSize, state.icc])

  const axisX = useAxis({
    label: state.plotView === 'effective_n' ? 'Intra-Class Correlation (rho)' : 'Cluster Size m (Rows per Entity)',
    range: state.plotView === 'effective_n' ? [0, 0.52] : [1, 51],
  })

  const axisY = useAxis({
    label: state.plotView === 'effective_n' ? 'Effective Sample Size (N_eff)' : 'Design Effect (Deff)',
    range: state.plotView === 'effective_n' ? [0, 10500] : [1, 15],
  })

  return (
    <Figure
      title="Row Grain and Effective Sample Size Under Clustered Dependence"
      state={state}
      hoverReadout={false}
      caption="When rows represent repeated measurements of the same underlying entity (user, patient, vehicle), within-cluster correlation deflates statistical degrees of freedom. A dataset with 10,000 nominal rows provides far less statistical precision than 10,000 independent samples."
      readouts={
        <>
          <Readout label="Nominal Row Count (N)" value="10,000 rows" />
          <Readout label="Unique Entities (K)" value={`${results.numEntities} entities`} />
          <Readout
            label="Effective Sample Size"
            value={`${results.currentNeff.toLocaleString()} rows (-${results.sampleLossPct.toFixed(1)}%)`}
          />
          <Readout label="Design Effect (Deff)" value={`${results.currentDeff.toFixed(2)}x`} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Clustered Dependence vs Statistical Information Content">
        {state.plotView === 'effective_n' ? (
          <>
            <Curve name="Effective Sample Size N_eff" x={results.rhoGrid} y={results.neffCurve} slot={0} width={2.5} />
            <Points name="Current Operating Point" x={[state.icc]} y={[results.currentNeff]} slot={1} size={8} />
          </>
        ) : (
          <>
            <Curve name="Kish Design Effect Deff" x={results.mGrid} y={results.deffCurve} slot={2} width={2.5} />
            <Points
              name="Current Operating Point"
              x={[state.clusterSize]}
              y={[results.currentDeff]}
              slot={1}
              size={8}
            />
          </>
        )}
      </Plot>
    </Figure>
  )
}
