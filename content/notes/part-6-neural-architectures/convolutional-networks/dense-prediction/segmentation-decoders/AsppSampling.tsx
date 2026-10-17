import { useContext, useMemo, type ReactNode } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  FrameContext,
  Handle,
  Plot,
  Points,
  Readout,
  row,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { dilatedTapCoverage, receptiveField } from 'aifn-compute/nn/functional'

/** Input side in pixels (DeepLab's 513 × 513 crops). */
const SIDE = 513
const PANEL = 340

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

const square = (x0: number, y0: number, x1: number, y1: number) => [
  { from: [x0, y0] as [number, number], to: [x1, y0] as [number, number] },
  { from: [x1, y0] as [number, number], to: [x1, y1] as [number, number] },
  { from: [x1, y1] as [number, number], to: [x0, y1] as [number, number] },
  { from: [x0, y1] as [number, number], to: [x0, y0] as [number, number] },
]

/**
 * The taps of ASPP's three 3 × 3 atrous branches at one position of the feature map, drawn in input pixels, with the
 * share of map positions whose taps all land inside the map as the rate grows.
 */
export function AsppSampling() {
  const state = useFigureState({
    outputStride: choice([16, 8], 16, { label: 'output stride' }),
    rates: row('atrous rates', {
      a: slider(1, 48, 6, { step: 1, label: 'branch 1' }),
      b: slider(1, 48, 12, { step: 1, label: 'branch 2' }),
      c: slider(1, 48, 18, { step: 1, label: 'branch 3' }),
    }),
    x: slider(0, SIDE - 1, 256, { onChart: true }),
    y: slider(0, SIDE - 1, 256, { onChart: true }),
  })
  const s = state.outputStride
  const n = Math.floor((SIDE - 1) / s) + 1
  const { a, b, c } = state.rates

  // The feature-map cell under the handle, and each branch's nine taps in input pixels.
  const ci = Math.min(n - 1, Math.round(state.x / s))
  const cj = Math.min(n - 1, Math.round(state.y / s))
  const branches = useMemo(
    () =>
      [a, b, c].map((r) => {
        const inside: { x: number[]; y: number[] } = { x: [], y: [] }
        const outside: { x: number[]; y: number[] } = { x: [], y: [] }
        for (const du of [-1, 0, 1])
          for (const dv of [-1, 0, 1]) {
            const i = ci + du * r
            const j = cj + dv * r
            const target = i >= 0 && i < n && j >= 0 && j < n ? inside : outside
            target.x.push(i * s)
            target.y.push(j * s)
          }
        const span = receptiveField([
          { kernel: s, stride: s },
          { kernel: 3, dilation: r },
        ]).at(-1)!.size
        return { r, inside, outside, span }
      }),
    [a, b, c, ci, cj, n, s],
  )

  // Share of positions with 9, 4 and 1 taps inside, for every rate from 1 to n.
  const sweep = useMemo(() => {
    const r = Array.from({ length: n }, (_, i) => i + 1)
    const shares = r.map((d) => dilatedTapCoverage(n, 3, d))
    return { r, nine: shares.map((c) => c[9]), four: shares.map((c) => c[4]), one: shares.map((c) => c[1]) }
  }, [n])
  const nineAt = (r: number) => (r <= n ? sweep.nine[r - 1] : 0)

  const mapOutline = useMemo(() => square(-s / 2, -s / 2, (n - 0.5) * s, (n - 0.5) * s), [n, s])
  const x = useAxis({ label: 'input column (pixels)', range: [-400, 912] })
  const y = useAxis({ label: 'input row (pixels)', range: [-400, 912], inverse: true, equal: x })
  const rx = useAxis({ label: 'atrous rate', range: [0, 66] })
  const ry = useAxis({ label: 'share of map positions', range: [0, 1] })
  return (
    <Figure
      title="Atrous spatial pyramid pooling on a finite map"
      state={state}
      caption="Left: a 513 × 513 input (outlined) and its feature map at the chosen output stride, 33 × 33 cells at stride 16 or 65 × 65 at stride 8. Drag the ink point to pick a cell; each ASPP branch is a 3 × 3 convolution whose taps sit a rate of cells apart, drawn in its colour where they land on the map and faded where they land on zero padding. Right: the share of map positions at which all nine taps (ink), only four, or only one land inside the map, for every rate. At stride 16, rate 18 never has all nine taps inside: its 3 × 3 kernel acts as a 2 × 2 or smaller filter everywhere, which is why ASPP adds a global-pooling branch. Halving the stride to 8 and doubling the rates keeps the spans in pixels and the shares about the same."
      readouts={{
        'at the chosen cell': (
          <>
            <Readout label="feature map" value={`${n} × ${n} cells; cell (${ci}, ${cj})`} />
            {branches.map((br, k) => (
              <Readout
                key={k}
                label={`branch ${k + 1}, rate ${br.r}`}
                value={`${br.inside.x.length} of 9 taps inside`}
              />
            ))}
          </>
        ),
        'over the whole map': (
          <>
            {branches.map((br, k) => (
              <Readout
                key={k}
                label={`rate ${br.r}`}
                value={`all 9 taps inside at ${formatNumber(nineAt(br.r))} of positions; spans ${br.span} pixels`}
              />
            ))}
          </>
        ),
      }}
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[3fr_2fr]">
        <FixedHeight height={PANEL}>
          <Plot x={x} y={y} ariaLabel="Taps of three atrous branches over the input image and its feature map">
            <Segments name="input and feature map" segments={mapOutline} muted width={1.5} />
            {branches.map((br, k) =>
              br.outside.x.length > 0 ? (
                <Points key={`o${k}`} id={`out${k}`} name={`rate ${br.r}, on padding`} {...br.outside} slot={k} thin />
              ) : null,
            )}
            {branches.map((br, k) => (
              <Points key={`i${k}`} id={`in${k}`} name={`rate ${br.r}, inside`} {...br.inside} slot={k} size={9} />
            ))}
            <Handle {...state.handle(['x', 'y'], { label: 'cell' })} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={rx} y={ry} ariaLabel="Share of positions with nine, four and one valid taps against the rate">
            <Curve name="all 9 taps inside" x={sweep.r} y={sweep.nine} emphasis />
            <Curve name="4 taps inside" x={sweep.r} y={sweep.four} muted />
            <Curve name="1 tap inside" x={sweep.r} y={sweep.one} muted dashed />
            {branches.map((br, k) => (
              <Points key={k} id={`rate${k}`} name={`rate ${br.r}`} x={[br.r]} y={[nineAt(br.r)]} slot={k} size={9} />
            ))}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
