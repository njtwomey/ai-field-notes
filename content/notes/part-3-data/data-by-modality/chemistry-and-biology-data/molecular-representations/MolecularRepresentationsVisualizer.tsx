import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function MolecularRepresentationsVisualizer() {
  const state = useFigureState({
    mode: choice<'fingerprint_collision' | 'conformational_energy' | 'information_tradeoff'>(
      [
        {
          value: 'fingerprint_collision',
          label: 'Morgan / ECFP: Bit Collisions vs Bit-Vector Size B & Substructure Count M (Rogers 2010)',
        },
        {
          value: 'conformational_energy',
          label: '3D Conformers: Lennard-Jones Potential & SE(3) Invariance vs Distance r (Gilmer 2017)',
        },
        {
          value: 'information_tradeoff',
          label: 'Modality Tradeoffs: Graph vs 3D Point Cloud vs Fingerprint vs SMILES',
        },
      ],
      'fingerprint_collision',
      { label: 'representation analysis' },
    ),
    param1: slider(512, 4096, 1024, {
      step: 512,
      label: 'fingerprint bit length B / equilibrium radius r_0 / feature dimension',
    }),
    param2: slider(1, 4, 2, {
      step: 1,
      label: 'ECFP radius r (1=ECFP2, 2=ECFP4, 3=ECFP6) / well depth epsilon / model complexity',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'fingerprint_collision') {
      const bitLength = param1 // e.g. 1024
      const ecfpRadius = param2 // 1, 2, 3, 4
      const baseSubstructures = Math.round(15 * (1 + 1.8 * ecfpRadius))

      const xVals: number[] = []
      const collisionVals: number[] = []
      const densityVals: number[] = []

      for (let m = 5; m <= 200; m += 5) {
        const pCollision = 1 - Math.pow((bitLength - 1) / bitLength, m - 1)
        const bitDensity = 1 - Math.pow(1 - 1 / bitLength, m)

        xVals.push(m)
        collisionVals.push(Math.min(100, pCollision * 100))
        densityVals.push(bitDensity * 100)
      }

      const activeCollision = (1 - Math.pow((bitLength - 1) / bitLength, baseSubstructures - 1)) * 100
      const activeDensity = (1 - Math.pow(1 - 1 / bitLength, baseSubstructures)) * 100

      return {
        x: xVals,
        y1: collisionVals,
        y2: densityVals,
        stat1: `ECFP${ecfpRadius * 2} (${bitLength}-bit vector)`,
        stat2: `${baseSubstructures} sub-environments extracted`,
        stat3: `${activeCollision.toFixed(1)}% collision rate`,
        insight: `Bit vector sparsity: ${(100 - activeDensity).toFixed(1)}% zeros`,
        axisXLabel: 'Extracted Substructures M',
        axisYLabel: 'Rate / Vector Density (%)',
        xRange: [0, 205] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    } else if (mode === 'conformational_energy') {
      const sigma = 3.4 // Angstroms
      const epsilon = param2 * 0.5 // kcal/mol
      const rMin = Math.pow(2, 1 / 6) * sigma // ~ 3.816 A

      const xVals: number[] = []
      const energyVals: number[] = []
      const forceVals: number[] = []

      for (let r = 2.8; r <= 8.0; r += 0.1) {
        const sr6 = Math.pow(sigma / r, 6)
        const sr12 = sr6 * sr6
        const energy = 4 * epsilon * (sr12 - sr6)
        const force = (4 * epsilon * (12 * sr12 - 6 * sr6)) / r

        xVals.push(r)
        energyVals.push(Math.max(-2.5, Math.min(8.0, energy)))
        forceVals.push(Math.max(-2.5, Math.min(8.0, force)))
      }

      return {
        x: xVals,
        y1: energyVals,
        y2: forceVals,
        stat1: `r_eq = ${rMin.toFixed(2)} Å`,
        stat2: `Well depth = ${epsilon.toFixed(2)} kcal/mol`,
        stat3: 'V(RQ + t) = V(R)',
        insight: 'SE(3) invariant scalar energy, SE(3) equivariant forces',
        axisXLabel: 'Interatomic Distance r (Å)',
        axisYLabel: 'Potential V(r) (kcal/mol) & Force (arb)',
        xRange: [2.5, 8.2] as [number, number],
        yRange: [-3, 8] as [number, number],
      }
    } else {
      const xVals = [1, 2, 3, 4]
      // 1: SMILES, 2: Circular Fingerprint, 3: 2D Graph (MPNN), 4: 3D Point Cloud
      const confAwareness = [10, 5, 20, 100]
      const sampleEfficiency = [50, 85, 75, 40]

      return {
        x: xVals,
        y1: confAwareness,
        y2: sampleEfficiency,
        stat1: '1=SMILES, 2=ECFP4, 3=MPNN, 4=SE(3)',
        stat2: 'Conformation vs Tabular Simplicity',
        stat3: 'MPNN preserves full 2D topology',
        insight: '3D requires SE(3) equivariance & conformer ensemble',
        axisXLabel: 'Representation Modality (1 to 4)',
        axisYLabel: 'Conformational Awareness & Sample Efficiency (%)',
        xRange: [0.5, 4.5] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Molecular Representations: Fingerprint Bit Collisions, 3D Potentials, and Modality Trade-offs"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Representation Config" value={results.stat1} />
          <Readout label="Dimensionality / Equilibrium" value={results.stat2} />
          <Readout label="Primary Metric / Symmetry" value={results.stat3} />
          <Readout label="Inductive Bias Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Molecular Feature Behaviour and Physical Invariance Response">
        <Curve
          name={
            mode === 'fingerprint_collision'
              ? 'Hash Bit Collision Probability (%)'
              : mode === 'conformational_energy'
                ? 'Lennard-Jones Potential V(r) (kcal/mol)'
                : 'Conformational Spatial Awareness (%)'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            mode === 'fingerprint_collision'
              ? 'Bit Vector Density (Fraction of 1s, %)'
              : mode === 'conformational_energy'
                ? 'Interatomic Restoring Force -dV/dr'
                : 'Small-Dataset Sample Efficiency (%)'
          }
          x={results.x}
          y={results.y2}
          slot={1}
          dashed
          width={2.0}
        />
        <Points name="Operating Points" x={results.x} y={results.y1} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
