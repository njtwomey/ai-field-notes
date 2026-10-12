import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function RemoteSensingImageryVisualizer() {
  const state = useFigureState({
    mode: choice<'spectral_signatures' | 'atmospheric_scattering' | 'temporal_compositing'>(
      [
        {
          value: 'spectral_signatures',
          label: 'Spectral Signatures: Vegetation Red Edge, Soil & Water Reflectance (Rouse 1974)',
        },
        {
          value: 'atmospheric_scattering',
          label: 'Atmospheric Correction: Rayleigh Scattering Path Radiance ~ lambda^(-4) (Drusch 2012)',
        },
        {
          value: 'temporal_compositing',
          label: 'Seasonal Phenology: NDVI Time-Series Profile vs Cloud Masking Composites',
        },
      ],
      'spectral_signatures',
      { label: 'remote sensing analysis' },
    ),
    param1: slider(1, 4, 1, {
      step: 1,
      label: 'land cover target (1=Vegetation, 2=Soil, 3=Water, 4=Snow) / aerosol optical depth / DOY scale',
    }),
    param2: slider(10, 100, 50, {
      step: 10,
      label: 'vegetation canopy density (%) / solar zenith angle theta_s / cloud cover threshold (%)',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'spectral_signatures') {
      const targetType = param1 // 1: Veg, 2: Soil, 3: Water, 4: Snow
      const canopy = param2 / 100 // 0.1 to 1.0

      // Wavelengths from 450 nm (Blue) to 2200 nm (SWIR2)
      // Key satellite bands: Blue (490), Green (560), Red (665), RedEdge (705), NIR (842), SWIR1 (1610), SWIR2 (2190)
      const wavelengths = [450, 490, 560, 665, 705, 740, 783, 842, 945, 1375, 1610, 2190]
      const targetReflectance: number[] = []
      const baselineSoil: number[] = []

      for (const w of wavelengths) {
        // Soil reflectance rises monotonically
        const soil = Math.min(45, 10 + 35 * ((w - 450) / 1750))
        baselineSoil.push(soil)

        if (targetType === 1) {
          // Healthy vegetation: low Blue (5%), Green peak (12%), deep Red absorption (4%),
          // steep Red Edge up to high NIR plateau (55% * canopy), SWIR dips
          let veg = 5
          if (w === 560) veg = 14
          else if (w === 665) veg = 5
          else if (w >= 705 && w <= 783) veg = 25 + 30 * ((w - 705) / 80)
          else if (w === 842) veg = 58 * canopy
          else if (w === 945) veg = 45 * canopy
          else if (w === 1375) veg = 2
          else if (w === 1610) veg = 28 * canopy
          else if (w === 2190) veg = 14 * canopy
          targetReflectance.push(veg)
        } else if (targetType === 2) {
          targetReflectance.push(soil)
        } else if (targetType === 3) {
          // Clear water: high Blue/Green, absorbs all NIR and SWIR
          let water = 0.5
          if (w <= 490) water = 8
          else if (w === 560) water = 5
          else if (w === 665) water = 2
          else if (w >= 842) water = 0.5
          targetReflectance.push(water)
        } else {
          // Snow: very high visible (90%), drops severely in SWIR
          let snow = 85
          if (w >= 1375) snow = 8
          else if (w >= 1610) snow = 4
          targetReflectance.push(snow)
        }
      }

      // Compute indices for target:
      // Red = index 3 (665 nm), NIR = index 7 (842 nm), Green = index 2 (560 nm)
      const redVal = targetReflectance[3] ?? 5
      const nirVal = targetReflectance[7] ?? 50
      const greenVal = targetReflectance[2] ?? 12
      const ndvi = (nirVal - redVal) / Math.max(0.1, nirVal + redVal)
      const ndwi = (greenVal - nirVal) / Math.max(0.1, greenVal + nirVal)

      const targetLabel =
        targetType === 1 ? 'Vegetation' : targetType === 2 ? 'Bare Soil' : targetType === 3 ? 'Clear Water' : 'Snow'

      return {
        x: wavelengths,
        y1: targetReflectance,
        y2: baselineSoil,
        stat1: `${targetLabel} Target (canopy ${param2}%)`,
        stat2: `NDVI = ${ndvi >= 0 ? '+' : ''}${ndvi.toFixed(2)}`,
        stat3: `NDWI = ${ndwi >= 0 ? '+' : ''}${ndwi.toFixed(2)}`,
        insight: 'The steep 700-800 nm red-edge separates photosynthetically active chlorophyll from bare ground',
        axisXLabel: 'Spectral Wavelength λ (nm)',
        axisYLabel: 'Surface Reflectance ρ (%)',
        xRange: [420, 2250] as [number, number],
        yRange: [0, 95] as [number, number],
      }
    } else if (mode === 'atmospheric_scattering') {
      // Rayleigh molecular scattering sigma_R ~ 1 / lambda^4
      const wavelengths = [400, 450, 490, 560, 665, 705, 842, 1000, 1200, 1610]
      const rayleighScattering: number[] = []
      const pathRadiancePct: number[] = []

      for (const w of wavelengths) {
        // Rayleigh relative scattering normalized to 550 nm = 1.0
        const rayleigh = Math.pow(550 / w, 4)
        // Path radiance fraction of total TOA signal (e.g. up to 70% in blue over dark targets)
        const pathFraction = Math.min(85, 25 * rayleigh)

        rayleighScattering.push(Math.min(10, rayleigh))
        pathRadiancePct.push(pathFraction)
      }

      return {
        x: wavelengths,
        y1: pathRadiancePct,
        y2: rayleighScattering.map((v) => v * 8),
        stat1: 'Rayleigh Scattering: σ_R ∝ λ^(-4)',
        stat2: 'Blue (490 nm): ~60% atmospheric haze',
        stat3: 'NIR (842 nm): < 10% atmospheric haze',
        insight: 'Level-2A BOA surface reflectance correction removes wavelength-dependent Rayleigh path radiance',
        axisXLabel: 'Spectral Wavelength λ (nm)',
        axisYLabel: 'Path Radiance Fraction (%) & Relative Scattering',
        xRange: [380, 1650] as [number, number],
        yRange: [0, 90] as [number, number],
      }
    } else {
      // Seasonal phenology NDVI profile across day of year (DOY 1 to 365)
      const doys = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]
      const rawCloudyNDVI: number[] = []
      const compositeNDVI: number[] = []

      for (const doy of doys) {
        // True agricultural phenology: bell curve peaking at DOY 200 (summer)
        const trueNdvi = 0.2 + 0.65 * Math.exp(-Math.pow((doy - 205) / 50, 2))
        // Cloudy scenes randomly drop NDVI to negative / low values
        const isCloudy = doy === 105 || doy === 195
        rawCloudyNDVI.push((isCloudy ? 0.05 : trueNdvi) * 100)
        compositeNDVI.push(trueNdvi * 100)
      }

      return {
        x: doys,
        y1: rawCloudyNDVI,
        y2: compositeNDVI,
        stat1: 'Annual Phenological Cycle (DOY)',
        stat2: 'Summer Peak NDVI: ~0.85',
        stat3: 'Winter Baseline: ~0.20',
        insight: 'Temporal cloud-mask compositing cleans sudden dips caused by undetected cloud contamination',
        axisXLabel: 'Day of Year (DOY)',
        axisYLabel: 'Vegetation Index NDVI x100',
        xRange: [0, 365] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Remote Sensing Imagery: Spectral Profiles, Atmospheric Scattering, and Seasonal Phenology"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Target Condition" value={results.stat1} />
          <Readout label="Primary Index / Band" value={results.stat2} />
          <Readout label="Secondary Diagnostic" value={results.stat3} />
          <Readout label="Remote Sensing Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Multispectral Reflectance Signatures and Radiometric Corrections">
        <Curve
          name={
            mode === 'spectral_signatures'
              ? 'Target Surface Reflectance Signature (%)'
              : mode === 'atmospheric_scattering'
                ? 'Atmospheric Path Radiance Fraction (%)'
                : 'Raw Instantaneous Observations (Cloud Distorted NDVI x100)'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            mode === 'spectral_signatures'
              ? 'Reference Bare Soil Reflectance Signature (%)'
              : mode === 'atmospheric_scattering'
                ? 'Rayleigh Relative Scattering Magnitude (Scaled)'
                : 'Cloud-Masked Best-Pixel Temporal Composite (Clean NDVI x100)'
          }
          x={results.x}
          y={results.y2}
          slot={1}
          dashed
          width={2.0}
        />
        <Points name="Evaluated Bands / Dates" x={results.x} y={results.y1} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
