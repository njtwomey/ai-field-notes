import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ImageDatasetBiasVisualizer() {
  const state = useFigureState({
    biasMode: choice<'spurious_background' | 'site_scanner' | 'framing_pose' | 'subgroup_disparity'>(
      [
        {
          value: 'spurious_background',
          label: 'Spurious Background: In-Distribution vs Worst-Group Accuracy (Beery 2018)',
        },
        {
          value: 'site_scanner',
          label: 'Site & Scanner Bias: Multi-Hospital Radiography Generalization Gap (Zech 2018)',
        },
        {
          value: 'framing_pose',
          label: 'Capture Signatures: Canonical Framing Bias vs Viewpoint Variation',
        },
        {
          value: 'subgroup_disparity',
          label: 'Subgroup Representation: Fitzpatrick Phototype Disparity (Buolamwini 2018)',
        },
      ],
      'spurious_background',
      { label: 'dataset bias mechanism' },
    ),
    spuriousStrength: slider(50, 99, 90, {
      step: 5,
      label: 'spurious correlation / confounder strength rho (%)',
    }),
    regularizationLevel: slider(0, 10, 2, {
      step: 1,
      label: 'robustness / group-DRO weight parameter',
    }),
  })

  const { biasMode, spuriousStrength, regularizationLevel } = state

  const results = useMemo(() => {
    if (biasMode === 'spurious_background') {
      // Background correlation rho from 50% (uncorrelated) to 99% (severe shortcut)
      const corrLevels = [50, 60, 70, 80, 85, 90, 95, 99]
      // In-distribution accuracy remains very high
      const inDistAcc = corrLevels.map((c) => 92 + (c - 50) * 0.08)
      // Worst-group (flipped background) accuracy plunges as correlation increases
      const worstGroupAcc = corrLevels.map((c) => {
        const base = Math.max(12, 88 - Math.pow((c - 50) / 10, 1.6) * 6.5)
        return Math.min(90, base + regularizationLevel * 2.8)
      })

      const currInDist = 92 + (spuriousStrength - 50) * 0.08
      const currWorstGroup = Math.min(
        90,
        Math.max(12, 88 - Math.pow((spuriousStrength - 50) / 10, 1.6) * 6.5) + regularizationLevel * 2.8,
      )
      const robustGap = currInDist - currWorstGroup

      return {
        x: corrLevels,
        y: worstGroupAcc,
        baseline: inDistAcc,
        stat1: `Average / ID Acc: ${currInDist.toFixed(1)}%`,
        stat2: `Worst-Group Acc: ${currWorstGroup.toFixed(1)}%`,
        stat3: `Generalization Gap: ${robustGap.toFixed(1)}%`,
        warning:
          robustGap > 30
            ? 'Severe shortcut learning; the model relies predominantly on spurious background scene context rather than foreground object semantics'
            : 'Balanced representations; robust regularization maintains acceptable worst-group accuracy across counterfactual backgrounds',
        axisXLabel: 'Spurious Background Correlation in Training Set (%)',
        axisYLabel: 'Test Classification Accuracy (%)',
        xRange: [48, 101] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    } else if (biasMode === 'site_scanner') {
      // Number of training hospital sites (1 to 10)
      const siteCounts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      // Internal validation performance (same hospital)
      const internalAuc = siteCounts.map(() => 94)
      // External hospital site generalization AUC as a function of multi-site pooling
      const externalAuc = siteCounts.map((s) => {
        const base = 62 + 25 * (1 - Math.exp(-0.35 * s))
        return Math.min(94, base + regularizationLevel * 0.8)
      })

      const currSites = Math.min(10, Math.max(1, Math.round(spuriousStrength / 10)))
      const currExternalAuc = Math.min(94, 62 + 25 * (1 - Math.exp(-0.35 * currSites)) + regularizationLevel * 0.8)
      const siteGap = 94 - currExternalAuc

      return {
        x: siteCounts,
        y: externalAuc,
        baseline: internalAuc,
        stat1: `Internal Hospital AUC: 94.0%`,
        stat2: `External Site AUC: ${currExternalAuc.toFixed(1)}%`,
        stat3: `Cross-Site Gap: ${siteGap.toFixed(1)}%`,
        warning:
          siteGap > 15
            ? 'Classifier exploits site-specific scanner artifacts, radiopaque marker fonts, and patient orientation tokens rather than pathology'
            : 'Multi-center training pools diverse acquisition signatures, attenuating scanner-specific confounding',
        axisXLabel: 'Number of Participating Clinical Sites in Training Cohort',
        axisYLabel: 'Diagnostic Pathology AUC ROC (%)',
        xRange: [0.8, 10.2] as [number, number],
        yRange: [50, 100] as [number, number],
      }
    } else if (biasMode === 'framing_pose') {
      // Viewpoint / crop deviation angle theta from canonical center (0 deg to 90 deg)
      const angles = [0, 10, 20, 30, 45, 60, 75, 90]
      // Canonical ImageNet framing accuracy vs off-center / unusual angle
      const canonicalBaseline = angles.map(() => 86)
      const offCanonicalAcc = angles.map((a) => {
        const drop = Math.pow(a / 90, 1.8) * 58
        return Math.min(86, Math.max(20, 86 - drop + regularizationLevel * 2.2))
      })

      const currAngle = ((spuriousStrength - 50) / 49) * 90
      const currDrop = Math.pow(currAngle / 90, 1.8) * 58
      const currAcc = Math.min(86, Math.max(20, 86 - dropSafe(currDrop, regularizationLevel * 2.2)))

      return {
        x: angles,
        y: offCanonicalAcc,
        baseline: canonicalBaseline,
        stat1: `Canonical Centered Acc: 86.0%`,
        stat2: `Non-Canonical Pose Acc: ${currAcc.toFixed(1)}%`,
        stat3: `Viewpoint Vulnerability: ${(86 - currAcc).toFixed(1)}%`,
        warning:
          86 - currAcc > 25
            ? 'Photographer framing bias induces spatial fragility; targets missing canonical scale and center centering suffer sharp confidence collapse'
            : 'Scale and viewpoint augmentations mitigate canonical framing assumptions',
        axisXLabel: 'Object Framing / Viewpoint Deviation from Canonical Center (Degrees)',
        axisYLabel: 'Top-1 Classification Accuracy (%)',
        xRange: [0, 92] as [number, number],
        yRange: [15, 95] as [number, number],
      }
    } else {
      // Subgroup representation: Fitzpatrick skin types I through VI
      const skinTypes = [1, 2, 3, 4, 5, 6]
      const lightSkinError = skinTypes.map(() => 4.2)
      // Error disparity increases for underrepresented darker skin types (Types V-VI)
      const observedError = skinTypes.map((t) => {
        const base = t <= 2 ? 3.8 : 3.8 + Math.pow(t - 2, 1.9) * 4.8
        return Math.max(2.5, base - regularizationLevel * 1.2)
      })

      const currType = Math.min(6, Math.max(1, Math.round(((spuriousStrength - 50) / 49) * 5 + 1)))
      const currError = Math.max(
        2.5,
        (currType <= 2 ? 3.8 : 3.8 + Math.pow(currType - 2, 1.9) * 4.8) - regularizationLevel * 1.2,
      )

      return {
        x: skinTypes,
        y: observedError,
        baseline: lightSkinError,
        stat1: `Fitzpatrick Type ${currType} Error: ${currError.toFixed(1)}%`,
        stat2: `Type I-II Baseline Error: 4.2%`,
        stat3: `Disparity Ratio: ${(currError / 4.2).toFixed(2)}x`,
        warning:
          currError / 4.2 > 2.0
            ? 'Severe demographic disparity; underrepresentation and dynamic range calibration failures in dark skintones amplify error rates'
            : 'Subpopulation reweighting and balanced calibration minimize demographic performance disparities',
        axisXLabel: 'Fitzpatrick Phototype Scale (I=Pale White, VI=Deeply Pigmented)',
        axisYLabel: 'Facial Analysis Classification Error Rate (%)',
        xRange: [0.8, 6.2] as [number, number],
        yRange: [0, 35] as [number, number],
      }
    }
  }, [biasMode, spuriousStrength, regularizationLevel])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Image Dataset Bias: Confounders, Shortcuts, and Generalization Breakdown"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Target Subgroup / Metric" value={results.stat1} />
          <Readout label="Counterfactual / External Metric" value={results.stat2} />
          <Readout label="Disparity / Generalization Gap" value={results.stat3} />
          <Readout label="Bias Diagnostic" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Performance Degradation under Confounded Visual Distributions">
        <Curve
          name={
            biasMode === 'subgroup_disparity'
              ? 'Majority Subgroup Reference Error (%)'
              : 'In-Distribution / Canonical Baseline (%)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            biasMode === 'subgroup_disparity'
              ? 'Empirical Subgroup Error Rate (%)'
              : 'Out-of-Distribution / Worst-Group Performance (%)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Cohort Data Points" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}

function dropSafe(drop: number, reg: number): number {
  return drop - reg
}
