import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function CausalDiscoveryVisualizer() {
  const state = useFigureState({
    discoveryLens: choice<'v_structure' | 'lingam' | 'notears'>(
      [
        { value: 'v_structure', label: 'PC Algorithm: Collider (v-Structure) Conditioning' },
        { value: 'lingam', label: 'LiNGAM: Non-Gaussian Residual Asymmetry (X → Y vs Y → X)' },
        { value: 'notears', label: 'NOTEARS: Continuous Acyclicity & Sparsity Pruning' },
      ],
      'v_structure',
      { label: 'causal discovery mechanism' },
    ),
    colliderConditioned: choice<'unconditioned' | 'conditioned'>(
      [
        { value: 'unconditioned', label: 'Unconditioned: P(X1, X2) (Marginally Independent)' },
        { value: 'conditioned', label: 'Conditioned on Z: P(X1, X2 | Z = z) (Induced Collider Bias)' },
      ],
      'unconditioned',
      { label: 'conditioning state on collider Z' },
    ),
    sparsityWeight: slider(0.05, 0.8, 0.35, {
      step: 0.05,
      label: 'L1 regularisation penalty lambda',
    }),
  })

  const { discoveryLens, colliderConditioned, sparsityWeight } = state

  const results = useMemo(() => {
    if (discoveryLens === 'v_structure') {
      // 30 points of X1 vs X2
      // True DAG: X1 -> Z <- X2 (Collider Z = X1 + X2 + noise)
      // If unconditioned: X1 and X2 are independent spherical Gaussian noise
      // If conditioned on Z in [-0.3, 0.3]: X1 + X2 ~ 0 => X2 ~ -X1 (negative correlation!)
      const pointsX: number[] = []
      const pointsY: number[] = []

      for (let i = 0; i < 35; i++) {
        const x1 = -2.0 + (i / 34) * 4.0
        if (colliderConditioned === 'unconditioned') {
          const x2 = Math.sin(i * 11) * 1.5
          pointsX.push(x1)
          pointsY.push(x2)
        } else {
          // Conditioned on Z ~ 0: x2 = -x1 + minor noise
          const x2 = -x1 + Math.sin(i * 9) * 0.35
          pointsX.push(x1)
          pointsY.push(x2)
        }
      }

      const corr = colliderConditioned === 'unconditioned' ? '0.02' : '-0.88'
      const status =
        colliderConditioned === 'unconditioned'
          ? 'Marginally Independent: X1 ⫫ X2 (SepSet does NOT contain Z)'
          : 'DEPENDENT GIVEN Z: X1 ⫫̸ X2 | Z => Unambiguous Collider X1 → Z ← X2'

      return {
        x: pointsX,
        y: pointsY,
        lineX: [-2.2, 2.2],
        lineY: colliderConditioned === 'conditioned' ? [2.2, -2.2] : [0, 0],
        corr,
        status,
      }
    } else if (discoveryLens === 'lingam') {
      // LiNGAM: True model is X -> Y (X ~ Uniform[-2, 2], Y = 1.2 * X + Uniform[-0.5, 0.5])
      // In forward regression (Y on X): residual e_Y is independent of X
      // In reverse regression (X on Y): residual e_X shows diamond/heteroscedastic dependence with Y
      const pointsX: number[] = []
      const pointsY: number[] = []

      for (let i = 0; i < 35; i++) {
        const yVal = -2.5 + (i / 34) * 5.0
        // reverse residual e_X against Y: non-constant variance (diamond bound)
        const spread = 2.5 - Math.abs(yVal) * 0.4
        const eX = Math.sin(i * 13) * spread * 0.7
        pointsX.push(yVal)
        pointsY.push(eX)
      }

      return {
        x: pointsX,
        y: pointsY,
        lineX: [-2.5, 2.5],
        lineY: [0, 0],
        corr: 'Residual depends on Y',
        status: 'Reverse regression (Y → X) produces dependent residuals; breaks ICA => Confirms X → Y',
      }
    } else {
      // NOTEARS: 5 candidate edges as lambda increases
      // Edge 1: X1 -> X2 (True, weight 1.5)
      // Edge 2: X2 -> X3 (True, weight 1.2)
      // Edge 3: X1 -> X3 (True direct, weight 0.8)
      // Edge 4: Spurious cycle X3 -> X1 (pruned by h(W) constraint)
      // Edge 5: Spurious noise correlation X4 -> X2 (pruned by lambda)
      const lambdas = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
      // Show weight of spurious edge 5 dropping to 0 at lambda = 0.3
      const spurWeight = lambdas.map((l) => Math.max(0, 0.6 - l * 1.8))
      const trueWeight = lambdas.map((l) => Math.max(0, 1.4 - l * 0.4))

      return {
        x: lambdas,
        y: spurWeight,
        lineX: lambdas,
        lineY: trueWeight,
        corr: `λ = ${sparsityWeight.toFixed(2)}`,
        status:
          sparsityWeight > 0.3
            ? 'Acyclic DAG recovered: Spurious reverse edges pruned to exactly 0.0'
            : 'Under-penalised: Dense graph with potential cyclic leaks',
      }
    }
  }, [discoveryLens, colliderConditioned, sparsityWeight])

  const axisX = useAxis({
    label:
      discoveryLens === 'v_structure'
        ? 'Variable X1'
        : discoveryLens === 'lingam'
          ? 'Target Variable Y (Reverse Predictor)'
          : 'L1 Regularisation Penalty (λ)',
    range: discoveryLens === 'notears' ? [0, 0.85] : [-3.0, 3.0],
  })

  const axisY = useAxis({
    label:
      discoveryLens === 'v_structure'
        ? 'Variable X2'
        : discoveryLens === 'lingam'
          ? 'Reverse Residual ε_X (Conditioned on Y)'
          : 'Learned Adjacency Matrix Weight (W_ij)',
    range: discoveryLens === 'notears' ? [0, 1.6] : [-3.0, 3.0],
  })

  return (
    <Figure
      title="Algorithmic Causal Discovery Explorer"
      state={state}
      hoverReadout={false}
      caption="Evaluating causal structure estimation: PC algorithm collider orientation via conditioning, LiNGAM non-Gaussian residual independence testing, and NOTEARS continuous acyclic optimization."
      readouts={
        <>
          <Readout label="Discovery Principle" value={results.status} />
          {discoveryLens === 'v_structure' && <Readout label="Sample Correlation" value={`r = ${results.corr}`} />}
          {discoveryLens === 'lingam' && (
            <Readout label="Noise Asymmetry" value="Non-Gaussian residuals identify unique causal direction" />
          )}
          {discoveryLens === 'notears' && (
            <Readout label="Acyclicity Constraint" value="h(W) = Tr(exp(W ∘ W)) - D = 0 strictly enforced" />
          )}
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Causal Discovery Geometry">
        <Points name="Observed Instances / Adjacency Points" x={results.x} y={results.y} slot={0} size={7} />
        <Curve
          name="Theoretical Reference / True Weight Path"
          x={results.lineX}
          y={results.lineY}
          slot={2}
          dashed
          width={1.8}
        />
      </Plot>
    </Figure>
  )
}
