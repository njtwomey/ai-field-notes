import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SpecialisedImagingDataVisualizer() {
  const state = useFigureState({
    imagingMode: choice<'ct_windowing' | 'multispectral' | 'wsi_pyramid' | 'depth_sensing'>(
      [
        {
          value: 'ct_windowing',
          label: 'CT Volumes: Hounsfield Unit (HU) Windowing & Leveling (Hounsfield 1973)',
        },
        {
          value: 'multispectral',
          label: 'Multispectral Imagery: Sentinel-2 Surface Reflectance Bands (Drusch 2012)',
        },
        {
          value: 'wsi_pyramid',
          label: 'Digital Pathology: Gigapixel Pyramidal Multi-Resolution Tiling (Macenko 2009)',
        },
        {
          value: 'depth_sensing',
          label: 'Range & Depth: Active IR / Time-of-Flight Quadratic Error Growth (Newcombe 2011)',
        },
      ],
      'ct_windowing',
      { label: 'specialised modality' },
    ),
    paramLevel: slider(-600, 800, 40, {
      step: 20,
      label: 'window level L / target band / distance parameter',
    }),
    paramWidth: slider(100, 2000, 400, {
      step: 50,
      label: 'window width W / patch size / baseline parameter',
    }),
  })

  const { imagingMode, paramLevel, paramWidth } = state

  const results = useMemo(() => {
    if (imagingMode === 'ct_windowing') {
      // HU scale from -1000 (Air) to +1200 (Cortical Bone)
      const huValues = [-1000, -800, -600, -400, -200, -100, -50, 0, 40, 100, 200, 400, 700, 1000, 1200]
      const lower = paramLevel - paramWidth / 2
      const upper = paramLevel + paramWidth / 2

      // Displayed 8-bit intensity mapping in [0, 255]
      const mappedIntensity = huValues.map((hu) => {
        if (hu <= lower) return 0
        if (hu >= upper) return 255
        return ((hu - lower) / paramWidth) * 255
      })

      // Linear benchmark without windowing (raw rescaling -1024..+3071)
      const rawRescaling = huValues.map((hu) => Math.max(0, Math.min(255, ((hu + 1000) / 2200) * 255)))

      const isBrain = paramLevel >= 30 && paramLevel <= 50 && paramWidth <= 120
      const isBone = paramLevel >= 300 && paramWidth >= 1000
      const isLung = paramLevel <= -500 && paramWidth >= 1200

      const preset = isBrain
        ? 'Brain / Soft Tissue'
        : isBone
          ? 'Bone / Skeletal'
          : isLung
            ? 'Lung Parenchyma'
            : 'Custom Radiologic Window'

      return {
        x: huValues,
        y: mappedIntensity,
        baseline: rawRescaling,
        stat1: `Window [L=${paramLevel}, W=${paramWidth}] HU`,
        stat2: `Visible Range: [${lower.toFixed(0)}, ${upper.toFixed(0)}] HU`,
        stat3: `Preset Tuning: ${preset}`,
        warning:
          paramWidth < 150
            ? 'Ultra-narrow window maximizes soft-tissue local contrast (e.g. stroke infarct detection) but completely saturates adjacent densities'
            : 'Broad dynamic window spans diverse tissue densities while attenuating fine low-contrast contrast gradients',
        axisXLabel: 'CT Radiodensity in Hounsfield Units (HU: Air=-1000, Water=0, Bone=+1000)',
        axisYLabel: 'Discretized Display Luminance Intensity (8-bit Quantization [0, 255])',
        xRange: [-1050, 1250] as [number, number],
        yRange: [0, 260] as [number, number],
      }
    } else if (imagingMode === 'multispectral') {
      // Sentinel-2 Bands: B1 (443nm), B2 (490nm), B3 (560nm), B4 (665nm), B5 (705nm), B8 (842nm), B11 (1610nm), B12 (2190nm)
      const wavelengths = [443, 490, 560, 665, 705, 842, 1610, 2190]
      // Reflectance curves (%): Vegetation (chlorophyll red absorption, strong NIR plateau)
      const vegetation = [8, 10, 15, 6, 35, 52, 28, 14]
      // Water reflectance (attenuates rapidly past green/red into NIR)
      const water = [15, 12, 7, 2, 1, 0.5, 0.2, 0.1]

      // Selected band via paramLevel mapped to index
      const bandIndex = Math.min(7, Math.max(0, Math.round(((paramLevel + 600) / 1400) * 7)))
      const currWave = wavelengths[bandIndex]
      const vegRefl = vegetation[bandIndex]
      const watRefl = water[bandIndex]
      const ndvi = (52 - 6) / (52 + 6) // (NIR - Red) / (NIR + Red) = 46/58 = 0.79

      return {
        x: wavelengths,
        y: vegetation,
        baseline: water,
        stat1: `Band λ = ${currWave} nm (B${bandIndex + 1})`,
        stat2: `Vegetation Refl: ${vegRefl}% | Water: ${watRefl}%`,
        stat3: `Canopy NDVI: +${ndvi.toFixed(2)} (High Biomass)`,
        warning:
          currWave >= 800
            ? 'Near-Infrared (NIR) spectrum penetrates atmosphere with minimal scattering; massive chlorophyll reflectance separates biomass from soil and water'
            : 'Visible spectrum bands (Red/Green/Blue) absorb heavily in photosynthetic pigments; atmospheric Rayleigh scattering requires BOA atmospheric correction',
        axisXLabel: 'Spectral Center Wavelength (Nanometers nm)',
        axisYLabel: 'Top-of-Canopy Surface Reflectance Factor (%)',
        xRange: [400, 2300] as [number, number],
        yRange: [0, 60] as [number, number],
      }
    } else if (imagingMode === 'wsi_pyramid') {
      // Pyramid Downsample factors: 1x (40x mag), 2x (20x), 4x (10x), 8x (5x), 16x (2.5x), 32x (1.25x)
      const zoomDownsamples = [1, 2, 4, 8, 16, 32]
      // Tile count for 100k x 100k image with 512x512 patches
      const patchSize = Math.max(256, Math.min(1024, paramWidth))
      const tileCounts = zoomDownsamples.map((d) => Math.pow(100000 / (d * patchSize), 2))
      // Total gigabytes uncompressed
      const uncompressedGb = zoomDownsamples.map((d) => (Math.pow(100000 / d, 2) * 3) / (1024 * 1024 * 1024))

      const downsample = Math.min(32, Math.max(1, Math.pow(2, Math.round(((paramLevel + 600) / 1400) * 5))))
      const currTiles = Math.round(Math.pow(100000 / (downsample * patchSize), 2))
      const currGb = (Math.pow(100000 / downsample, 2) * 3) / (1024 * 1024 * 1024)

      return {
        x: zoomDownsamples,
        y: tileCounts,
        baseline: uncompressedGb.map((g) => g * 1000), // scaled to plot
        stat1: `Pyramid Level: ${downsample}x Downsample`,
        stat2: `Patches (${patchSize}px): ${currTiles.toLocaleString()}`,
        stat3: `Layer Memory: ${currGb.toFixed(2)} GB`,
        warning:
          downsample === 1
            ? 'Level 0 resolution (0.25 µm/voxel) yields >30,000 tiles per slide; requires tissue thresholding (Otsu) to avoid processing empty glass background'
            : 'Higher pyramid levels provide global slide context (tumor microenvironment staging) while discarding sub-cellular nuclear chromatin textures',
        axisXLabel: 'Multi-Resolution Pyramid Downsample Factor (1x=40x Optical Mag)',
        axisYLabel: 'Total Grid Patches Generated Across Whole Slide Image',
        xRange: [0.5, 34] as [number, number],
        yRange: [0, Math.max(...tileCounts) * 1.1] as [number, number],
      }
    } else {
      // Distance range Z from 0.5 m to 10 m
      const distances = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
      // Quadratic stereo/structured light depth error: sigma_Z = (Z^2 / (f * B)) * sigma_d
      // Baseline B = 0.075 m, f = 500 px, disparity uncertainty 0.2 px
      const depthErrorsMm = distances.map((z) => ((z * z) / (500 * 0.075)) * 0.2 * 1000)
      // Percentage of invalid / lost return pixels due to specular reflection or inverse square falloff
      const invalidReturnsPct = distances.map((z) => Math.min(85, Math.pow(z / 10, 2.2) * 85))

      const currZ = Math.min(10, Math.max(0.5, ((paramLevel + 600) / 1400) * 9.5 + 0.5))
      const currErrMm = ((currZ * currZ) / (500 * 0.075)) * 0.2 * 1000
      const currInvalid = Math.min(85, Math.pow(currZ / 10, 2.2) * 85)

      return {
        x: distances,
        y: depthErrorsMm,
        baseline: invalidReturnsPct.map((p) => p * 1.2),
        stat1: `Target Distance: ${currZ.toFixed(1)} m`,
        stat2: `Depth Uncertainty: ±${currErrMm.toFixed(1)} mm`,
        stat3: `Missing Pixel Rate: ${currInvalid.toFixed(1)}%`,
        warning:
          currZ > 5.0
            ? 'Quadratic disparity error σ_Z ∝ Z² and 1/Z² optical power decay cause catastrophic depth degradation at long ranges'
            : 'Sub-centimeter range precision maintained; active illumination signal-to-noise ratio is well within calibrated sensor bounds',
        axisXLabel: 'Target Standoff Distance Z (Meters)',
        axisYLabel: 'Range Uncertainty σ_Z (Millimeters mm)',
        xRange: [0.3, 10.5] as [number, number],
        yRange: [0, Math.max(...depthErrorsMm) * 1.15] as [number, number],
      }
    }
  }, [imagingMode, paramLevel, paramWidth])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Specialised Imaging Data: Modality Formats, Dynamic Ranges, and Physical Scaling"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Acquisition Setting" value={results.stat1} />
          <Readout label="Metric Measurement" value={results.stat2} />
          <Readout label="Pipeline State" value={results.stat3} />
          <Readout label="Physical Insight" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Modality Sensor Response Function">
        <Curve
          name={
            imagingMode === 'ct_windowing'
              ? 'Raw Unwindowed Linear Remapping'
              : imagingMode === 'multispectral'
                ? 'Clear Deep Water Reflectance Signature (%)'
                : imagingMode === 'wsi_pyramid'
                  ? 'Raw Pyramid Layer Storage (MB x 10)'
                  : 'Missing / Specular Dropout Pixels (% x 1.2)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            imagingMode === 'ct_windowing'
              ? 'Windowed Display Luminance [L, W]'
              : imagingMode === 'multispectral'
                ? 'Photosynthetic Canopy Reflectance Spectrum (%)'
                : imagingMode === 'wsi_pyramid'
                  ? 'Extractable Image Patches Across Slide'
                  : 'Depth Range Uncertainty σ_Z (mm)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Spectral / Spatial Samples" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
