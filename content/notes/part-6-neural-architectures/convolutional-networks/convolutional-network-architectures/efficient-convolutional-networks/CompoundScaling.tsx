import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { networkCost } from 'aifn-compute/nn/functional'
import { efficientnet, EFFICIENTNET_SCALES } from '../_shared/architectures.ts'

const PHI_MAX = 6
const STEP = 0.25
const PHIS = Array.from({ length: PHI_MAX / STEP + 1 }, (_, i) => i * STEP)

type Point = {
  phi: number
  depth: number
  width: number
  resolution: number
  params: number
  macs: number
  layers: number
}

function scaled(depth: number, width: number, resolution: number, phi: number): Point {
  const r = Math.round(224 * resolution)
  const m = efficientnet(width, depth, r)
  const c = networkCost(m.stages, m.input)
  return { phi, depth, width, resolution: r, params: c.params / 1e6, macs: c.multiplies / 1e9, layers: c.depth }
}

/** Single-dimension paths: each doubles the multiply-adds per unit of φ by scaling one dimension alone. */
const COARSE = PHIS.filter((_, i) => i % 2 === 0)
const SINGLE = {
  depth: COARSE.map((p) => scaled(2 ** p, 1, 1, p)),
  width: COARSE.map((p) => scaled(1, 2 ** (p / 2), 1, p)),
  resolution: COARSE.map((p) => scaled(1, 1, 2 ** (p / 2), p)),
}
const OFFICIAL = Object.entries(EFFICIENTNET_SCALES).map(([name, [w, d, r]]) => {
  const m = efficientnet(w, d, r)
  const c = networkCost(m.stages, m.input)
  return { name, params: c.params / 1e6, macs: c.multiplies / 1e9 }
})

const xs = (path: Point[]) => path.map((p) => p.macs)
const ys = (path: Point[]) => path.map((p) => p.params)

/**
 * EfficientNet's compound scaling: depth α^φ, width β^φ and resolution γ^φ from the B0 baseline, against scaling one
 * dimension alone, each network counted layer by layer.
 */
export function CompoundScaling() {
  const state = useFigureState({
    alpha: float(1.2, { min: 1, max: 2, step: 0.01, label: 'depth base α' }),
    beta: float(1.1, { min: 1, max: 1.6, step: 0.01, label: 'width base β' }),
    gamma: float(1.15, { min: 1, max: 1.6, step: 0.01, label: 'resolution base γ' }),
    phi: float(3, { min: 0, max: PHI_MAX, step: STEP, label: 'compound coefficient φ', onChart: true }),
  })
  const { alpha, beta, gamma } = state
  const compound = useMemo(() => PHIS.map((p) => scaled(alpha ** p, beta ** p, gamma ** p, p)), [alpha, beta, gamma])
  const index = Math.min(PHIS.length - 1, Math.max(0, Math.round(state.phi / STEP)))
  const at = compound[index]
  const product = alpha * beta ** 2 * gamma ** 2

  const nearest = ([mx, py]: [number, number]) => {
    let best = 0
    let bestD = Infinity
    compound.forEach((p, i) => {
      const d = Math.log(p.macs / mx) ** 2 + Math.log(p.params / py) ** 2
      if (d < bestD) {
        bestD = d
        best = i
      }
    })
    return PHIS[best]
  }

  const x = useAxis({ label: 'multiply-adds per image (billions)', log: true })
  const y = useAxis({ label: 'parameters (millions)', log: true })
  return (
    <Figure
      title="Compound scaling from EfficientNet-B0"
      state={state}
      defaultSize="L"
      caption="Each curve starts at EfficientNet-B0 and scales it, network by network, from φ = 0 to 6: deeper only (repeats × 2^φ), wider only (channels × 2^(φ/2)), higher resolution only (input side × 2^(φ/2)), and the compound rule (α^φ, β^φ, γ^φ), every point counted layer by layer. The single-dimension paths double the multiply-adds per unit of φ and spend them differently: resolution adds no parameters, while depth and width add parameters about in proportion to the cost. The compound path sits between them, because part of its budget goes to resolution. Grey points are the published B0 to B7. Drag the ink point along the compound curve to set φ; change α, β and γ to move the curve, and watch α·β²·γ², the factor by which each unit of φ multiplies the cost."
      readouts={{
        [`at φ = ${formatNumber(at.phi)}`]: (
          <>
            <Readout label="depth × α^φ" value={formatNumber(at.depth)} />
            <Readout label="width × β^φ" value={formatNumber(at.width)} />
            <Readout label="input side" value={`${at.resolution} px`} />
            <Readout label="weight layers" value={at.layers} />
            <Readout label="parameters" value={`${formatNumber(at.params)} M`} />
            <Readout label="multiply-adds" value={`${formatNumber(at.macs)} G`} />
          </>
        ),
        rule: (
          <>
            <Readout label="α·β²·γ²" value={formatNumber(product)} />
            <Readout
              label="predicted cost, B0 × (α·β²·γ²)^φ"
              value={`${formatNumber(compound[0].macs * product ** at.phi)} G`}
            />
          </>
        ),
      }}
    >
      <Plot x={x} y={y} height={340}>
        <Curve name="deeper only" x={xs(SINGLE.depth)} y={ys(SINGLE.depth)} slot={0} />
        <Curve name="wider only" x={xs(SINGLE.width)} y={ys(SINGLE.width)} slot={1} />
        <Curve name="higher resolution only" x={xs(SINGLE.resolution)} y={ys(SINGLE.resolution)} slot={2} />
        <Curve name="compound" x={xs(compound)} y={ys(compound)} emphasis showPoints />
        <Points
          name="EfficientNet B0-B7"
          x={OFFICIAL.map((o) => o.macs)}
          y={OFFICIAL.map((o) => o.params)}
          labels={OFFICIAL.map((o) => o.name)}
          muted
        />
        <Handle
          kind="point"
          at={[at.macs, at.params]}
          label={`φ = ${formatNumber(at.phi)}`}
          onDrag={(p) => state.set('phi', nearest(p))}
        />
      </Plot>
    </Figure>
  )
}
