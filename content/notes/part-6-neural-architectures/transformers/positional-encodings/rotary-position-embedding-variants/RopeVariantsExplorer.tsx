import { useMemo, useState } from 'react'
import {
  Area,
  ControlRow,
  Curve,
  Figure,
  Plot,
  Plots,
  Raster,
  Readout,
  Select,
  Slider,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const VIEW_OPTIONS = [
  { value: 'xpos', label: '1D: xPos exponential decay vs standard RoPE' },
  { value: '2d', label: '2D: Axial vs Mixed RoPE spatial kernels' },
]

const SCHEME_OPTIONS = [
  { value: 'axial', label: 'Axial 2D RoPE (separable x and y channels)' },
  { value: 'mixed', label: 'Mixed 2D RoPE (directional orientation vectors)' },
]

export function RopeVariantsExplorer() {
  const [view, setView] = useState<'xpos' | '2d'>('xpos')

  // xPos parameters
  const [base, setBase] = useState(10000)
  const [gamma, setGamma] = useState(0.4)
  const [headDim, setHeadDim] = useState(64)
  const [pairIdx, setPairIdx] = useState(0)
  const [maxDist, setMaxDist] = useState(512)

  // 2D RoPE parameters
  const [scheme2d, setScheme2d] = useState<'axial' | 'mixed'>('axial')
  const [gridSize, setGridSize] = useState(15) // grid from -gridSize to +gridSize
  const [base2d, setBase2d] = useState(100)

  // ── 1. xPos computations ───────────────────────────────────────────────────
  const numPairs = Math.floor(headDim / 2)
  const safePairIdx = Math.min(pairIdx, numPairs - 1)

  const freqs = useMemo(
    () => Float64Array.from({ length: numPairs }, (_, i) => base ** ((-2 * i) / headDim)),
    [base, headDim, numPairs],
  )

  const zetas = useMemo(
    () => Float64Array.from({ length: numPairs }, (_, i) => (i / numPairs + gamma) / (1 + gamma)),
    [numPairs, gamma],
  )

  const thetaFocal = freqs[safePairIdx]
  const zetaFocal = zetas[safePairIdx]

  const xposCurves = useMemo(() => {
    const steps = 300
    const xs = Float64Array.from({ length: steps }, (_, s) => (s * maxDist) / (steps - 1))

    // Focal pair single channel
    const ropeSingle = Float64Array.from(xs, (d) => Math.cos(d * thetaFocal))
    const xposSingle = Float64Array.from(xs, (d) => (zetaFocal ** d) * Math.cos(d * thetaFocal))
    const envUpper = Float64Array.from(xs, (d) => zetaFocal ** d)
    const envLower = Float64Array.from(xs, (d) => -(zetaFocal ** d))

    // Aggregate expected kernel across all pairs: sum cos(d * theta_i) / numPairs
    const ropeAgg = Float64Array.from(xs, (d) => {
      let sum = 0
      for (let i = 0; i < numPairs; i++) sum += Math.cos(d * freqs[i])
      return sum / numPairs
    })

    const xposAgg = Float64Array.from(xs, (d) => {
      let sum = 0
      for (let i = 0; i < numPairs; i++) sum += (zetas[i] ** d) * Math.cos(d * freqs[i])
      return sum / numPairs
    })

    return { xs, ropeSingle, xposSingle, envUpper, envLower, ropeAgg, xposAgg }
  }, [maxDist, thetaFocal, zetaFocal, freqs, zetas, numPairs])

  // ── 2. 2D RoPE computations ─────────────────────────────────────────────────
  const numPairs2D = 32
  const freqs2D = useMemo(
    () => Float64Array.from({ length: numPairs2D }, (_, i) => base2d ** ((-2 * (i % (numPairs2D / 2))) / (numPairs2D / 2))),
    [base2d, numPairs2D],
  )

  const angles2D = useMemo(
    () => Float64Array.from({ length: numPairs2D }, (_, i) => (Math.PI * i) / numPairs2D),
    [numPairs2D],
  )

  const gridCoords = useMemo(() => {
    const coords: number[] = []
    for (let g = -gridSize; g <= gridSize; g++) coords.push(g)
    return coords
  }, [gridSize])

  const rasterData = useMemo(() => {
    const z: number[][] = []
    const half = Math.floor(numPairs2D / 2)

    for (const dy of gridCoords) {
      const row: number[] = []
      for (const dx of gridCoords) {
        let score = 0
        if (scheme2d === 'axial') {
          // Half dimensions rotate by dx, half by dy
          for (let i = 0; i < half; i++) {
            score += Math.cos(dx * freqs2D[i])
            score += Math.cos(dy * freqs2D[i])
          }
        } else {
          // Mixed: each pair has direction vector (cos alpha_i, sin alpha_i)
          for (let i = 0; i < numPairs2D; i++) {
            const thX = freqs2D[i] * Math.cos(angles2D[i])
            const thY = freqs2D[i] * Math.sin(angles2D[i])
            score += Math.cos(dx * thX + dy * thY)
          }
        }
        row.push(score / numPairs2D)
      }
      z.push(row)
    }
    return z
  }, [gridCoords, scheme2d, numPairs2D, freqs2D, angles2D])

  // 1D cross-sections: on-axis vs diagonal
  const crossSections = useMemo(() => {
    const rs = Float64Array.from({ length: gridSize + 1 }, (_, r) => r)
    const half = Math.floor(numPairs2D / 2)

    const onAxis = Float64Array.from(rs, (r) => {
      let score = 0
      if (scheme2d === 'axial') {
        for (let i = 0; i < half; i++) score += Math.cos(r * freqs2D[i]) + 1
      } else {
        for (let i = 0; i < numPairs2D; i++) {
          const thX = freqs2D[i] * Math.cos(angles2D[i])
          score += Math.cos(r * thX)
        }
      }
      return score / numPairs2D
    })

    const diagonal = Float64Array.from(rs, (r) => {
      const coord = r / Math.SQRT2
      let score = 0
      if (scheme2d === 'axial') {
        for (let i = 0; i < half; i++) score += 2 * Math.cos(coord * freqs2D[i])
      } else {
        for (let i = 0; i < numPairs2D; i++) {
          const thX = freqs2D[i] * Math.cos(angles2D[i])
          const thY = freqs2D[i] * Math.sin(angles2D[i])
          score += Math.cos(coord * (thX + thY))
        }
      }
      return score / numPairs2D
    })

    return { rs, onAxis, diagonal }
  }, [gridSize, scheme2d, numPairs2D, freqs2D, angles2D])

  // Axes
  const distAxis = useAxis({ label: 'relative token distance d = |m − n|', range: [0, maxDist], key: maxDist })
  const singleScoreAxis = useAxis({ label: 'pair contribution', range: [-1.05, 1.05] })
  const aggScoreAxis = useAxis({ label: 'mean dot-product kernel K(d)', range: [-0.3, 1.05] })

  const spatialXAxis = useAxis({ label: 'horizontal displacement Δx (patches)', range: [-gridSize, gridSize] })
  const spatialYAxis = useAxis({ label: 'vertical displacement Δy (patches)', range: [-gridSize, gridSize] })
  const radDistAxis = useAxis({ label: 'radial displacement r (patches)', range: [0, gridSize] })
  const radScoreAxis = useAxis({ label: 'kernel response K(r)', range: [-0.4, 1.05] })

  return (
    <Figure
      title="Rotary position embedding variants: xPos and 2D spatial RoPE"
      purpose="Visualizes how rotary positional variants overcome standard RoPE limitations: xPos multiplies rotation by an exponential envelope to damp long-range oscillations and enable length extrapolation, while 2D Mixed RoPE eliminates the directional cross-artifacts of 2D Axial RoPE."
      controls={
        <>
          <ControlRow label="Explorer perspective">
            <Select
              label="Variant family"
              value={view}
              onChange={(v) => setView(v as 'xpos' | '2d')}
              options={VIEW_OPTIONS}
            />
            {view === 'xpos' ? (
              <Select
                label="Head dimension"
                value={String(headDim)}
                onChange={(v) => setHeadDim(Number(v))}
                options={[
                  { value: '32', label: 'd_head = 32 (16 pairs)' },
                  { value: '64', label: 'd_head = 64 (32 pairs)' },
                  { value: '128', label: 'd_head = 128 (64 pairs)' },
                ]}
              />
            ) : (
              <Select
                label="2D RoPE formulation"
                value={scheme2d}
                onChange={(v) => setScheme2d(v as 'axial' | 'mixed')}
                options={SCHEME_OPTIONS}
              />
            )}
          </ControlRow>
          {view === 'xpos' ? (
            <ControlRow label="xPos tuning">
              <Slider
                label="Focal pair index i"
                value={safePairIdx}
                onChange={setPairIdx}
                min={0}
                max={numPairs - 1}
                step={1}
              />
              <Slider
                label="Decay constant γ"
                value={gamma}
                onChange={setGamma}
                min={0.1}
                max={1.5}
                step={0.05}
              />
              <Slider
                label="Base frequency b"
                value={base}
                onChange={setBase}
                min={1000}
                max={50000}
                step={1000}
              />
              <Slider
                label="Sequence horizon"
                value={maxDist}
                onChange={setMaxDist}
                min={128}
                max={1024}
                step={64}
              />
            </ControlRow>
          ) : (
            <ControlRow label="2D grid geometry">
              <Slider
                label="Patch radius"
                value={gridSize}
                onChange={setGridSize}
                min={8}
                max={25}
                step={1}
              />
              <Slider
                label="2D frequency base"
                value={base2d}
                onChange={setBase2d}
                min={20}
                max={500}
                step={20}
              />
            </ControlRow>
          )}
        </>
      }
      readouts={
        view === 'xpos'
          ? {
              [`focal pair #${safePairIdx} (of ${numPairs})`]: (
                <>
                  <Readout label="frequency θ_i" value={fmt(thetaFocal, 4)} />
                  <Readout label="wavelength λ" value={`${fmt((2 * Math.PI) / thetaFocal, 1)} tokens`} />
                  <Readout label="xPos scale ζ_i" value={fmt(zetaFocal, 4)} />
                  <Readout label="half-life (50% decay)" value={`${fmt(Math.log(0.5) / Math.log(zetaFocal), 1)} tokens`} />
                </>
              ),
              extrapolation: (
                <>
                  <Readout
                    label="RoPE tail variance"
                    value={safePairIdx < numPairs / 3 ? 'high (undamped ripples)' : 'low (slow drift)'}
                  />
                  <Readout
                    label={`xPos envelope at d=${maxDist}`}
                    value={fmt(zetaFocal ** maxDist, 5)}
                  />
                </>
              ),
            }
          : {
              'receptive field comparison': (
                <>
                  <Readout label="scheme" value={scheme2d === 'axial' ? 'Axial (separable)' : 'Mixed (omnidirectional)'} />
                  <Readout label="on-axis response at r=5" value={fmt(crossSections.onAxis[Math.min(5, gridSize)])} />
                  <Readout label="diagonal response at r=5" value={fmt(crossSections.diagonal[Math.min(5, gridSize)])} />
                  <Readout
                    label="diagonal-to-axial ratio"
                    value={`${fmt(
                      (100 * crossSections.diagonal[Math.min(5, gridSize)]) /
                        Math.max(1e-4, crossSections.onAxis[Math.min(5, gridSize)]),
                    )}%`}
                  />
                </>
              ),
            }
      }
      caption={
        view === 'xpos'
          ? `xPos dampens high-frequency positional oscillations. Left: Pair #${safePairIdx} contribution. Standard RoPE (blue curve) oscillates endlessly with unit amplitude; xPos (red curve) is bound by the exponential envelope ±ζ_i^d (shaded region), where high-frequency pairs decay in tens of tokens while slow pairs persist over hundreds. Right: Expected dot-product kernel averaged over all ${numPairs} pairs. RoPE retains volatile pseudo-random fluctuations at large token separations, causing out-of-distribution attention collapse; xPos smoothly decays to 0, providing length extrapolation.`
          : `2D RoPE comparison for vision and multimodal tokens. Left: 2D spatial attention response K(Δx, Δy). Axial RoPE creates pronounced horizontal and vertical cross-artifacts because x and y coordinates rotate independent coordinate halves, penalizing diagonal neighbours. Mixed RoPE rotates channels along diverse radial vectors, yielding an isotropic receptive field. Right: Cross-section response comparing on-axis displacement (Δx=r, Δy=0) against diagonal displacement (Δx=Δy=r/√2).`
      }
    >
      {view === 'xpos' ? (
        <Plots cols={2}>
          <Plot x={distAxis} y={singleScoreAxis} title={`pair #${safePairIdx}: rotation & decay envelope`}>
            <Area
              name="xPos decay envelope"
              x={xposCurves.xs}
              y={xposCurves.envUpper}
              base={xposCurves.envLower}
              opacity={0.1}
              line={false}
            />
            <Curve name="RoPE cos(d·θ_i)" x={xposCurves.xs} y={xposCurves.ropeSingle} muted thin />
            <Curve name="xPos ζ^d cos(d·θ_i)" x={xposCurves.xs} y={xposCurves.xposSingle} emphasis />
            <Curve name="+ζ^d upper bound" x={xposCurves.xs} y={xposCurves.envUpper} dashed thin />
            <Curve name="−ζ^d lower bound" x={xposCurves.xs} y={xposCurves.envLower} dashed thin />
          </Plot>
          <Plot x={distAxis} y={aggScoreAxis} title="aggregate attention kernel K(d) over all pairs">
            <Curve name="zero level" x={[0, maxDist]} y={[0, 0]} muted dashed thin />
            <Curve name="RoPE mean kernel" x={xposCurves.xs} y={xposCurves.ropeAgg} muted />
            <Curve name="xPos mean kernel" x={xposCurves.xs} y={xposCurves.xposAgg} emphasis />
          </Plot>
        </Plots>
      ) : (
        <Plots cols={2}>
          <Plot
            x={spatialXAxis}
            y={spatialYAxis}
            title={`${scheme2d === 'axial' ? 'Axial' : 'Mixed'} 2D spatial attention kernel`}
          >
            <Raster
              x={gridCoords}
              y={gridCoords}
              z={rasterData}
              scale="diverging"
              range={[-0.4, 1.0]}
              valueLabel="kernel K(Δx, Δy)"
            />
          </Plot>
          <Plot x={radDistAxis} y={radScoreAxis} title="directional cross-sections: on-axis vs diagonal">
            <Curve name="zero response" x={[0, gridSize]} y={[0, 0]} muted dashed thin />
            <Curve name="on-axis response (Δy = 0)" x={crossSections.rs} y={crossSections.onAxis} emphasis />
            <Curve
              name="diagonal response (Δx = Δy)"
              x={crossSections.rs}
              y={crossSections.diagonal}
              dashed
            />
          </Plot>
        </Plots>
      )}
    </Figure>
  )
}
