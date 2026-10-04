import { useMemo, useState } from 'react'
import { normal, stream } from 'aifn/foundation/random'
import { toFlat, unwrap, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { flashAttentionSteps, scaledDotProductAttention } from 'aifn/nn/attention'
import {
  Bars,
  ControlGroup,
  ControlRow,
  Curve,
  Figure,
  Player,
  Plot,
  Plots,
  Raster,
  Readout,
  row,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const FT = 12
const raw = (v: unknown) => unwrap(v as Tensor) as Tensor
const range = (n: number) => Array.from({ length: n }, (_, i) => i)

export function FlashAttentionTiling() {
  const state = useFigureState({
    tiles: row('Tiles', {
      queryBlock: slider(1, 6, 4, { label: 'queries per tile (B_r)', step: 1 }),
      keyBlock: slider(1, 6, 3, { label: 'keys per tile (B_c)', step: 1 }),
      causal: setting(true, 'causal mask'),
    }),
  })
  const { queryBlock, keyBlock, causal } = state.tiles

  const qkv = useMemo(() => {
    const draw = (name: string, d: number) => normal(stream(`flash-${name}`), 0, 1.2, { shape: [FT, d] }) as Tensor
    return { q: draw('q', 4), k: draw('k', 4), v: draw('v', 2) }
  }, [])

  const exact = useMemo(
    () => toFlat(raw(scaledDotProductAttention(qkv.q, qkv.k, qkv.v, { causal }).output)),
    [qkv, causal],
  )

  const tr = useMemo(
    () =>
      trace(flashAttentionSteps(qkv.q, qkv.k, qkv.v, { queryBlock, keyBlock, causal }), undefined, 1000, {
        keep: 'all',
      }),
    [qkv, queryBlock, keyBlock, causal],
  )

  const [position, setPosition] = useState(0)
  const pos = Math.min(position, tr.steps.length - 1)
  const at = tr.steps[pos]

  const order = useMemo(() => {
    const z = range(FT).map(() => range(FT).map(() => NaN))
    at.tiles.forEach((tile, n) => {
      if (n >= at.t) return
      for (let i = tile.queries[0]; i < tile.queries[1]; i++) {
        for (let j = tile.keys[0]; j < tile.keys[1]; j++) {
          if (!causal || j <= i) z[i][j] = n + 1
        }
      }
    })
    return z.reverse()
  }, [at, causal])

  const box = useMemo(() => {
    if (!at.tile) return { x: [], y: [] }
    const [q0, q1] = at.tile.queries
    const [k0, k1] = at.tile.keys
    const top = FT - 1 - q0 + 0.5
    const bottom = FT - 1 - (q1 - 1) - 0.5
    return { x: [k0 - 0.5, k1 - 0.5, k1 - 0.5, k0 - 0.5, k0 - 0.5], y: [bottom, bottom, top, top, bottom] }
  }, [at])

  const errors = useMemo(() => {
    const a = toFlat(at.accumulator)
    const l = toFlat(at.normaliser)
    return range(FT).map((i) =>
      l[i] > 0 ? Math.max(...[0, 1].map((c) => Math.abs(a[i * 2 + c] / l[i] - exact[i * 2 + c]))) : NaN,
    )
  }, [at, exact])

  const worst = Math.max(...errors.map((e) => (Number.isFinite(e) ? e : 0)))

  const kx = useAxis({ label: 'key index j' })
  const qy = useAxis({ label: 'query index i (top: 0)', equal: kx })
  const ex = useAxis({ label: 'query index i' })
  const ey = useAxis({ label: '|a_i/ℓ_i − exact output|', log: true, range: [1e-17, 10] })

  return (
    <Figure
      title="Tiled attention with online softmax"
      purpose="FlashAttention never materialises the full N × N attention matrix in HBM. Instead, it streams tiles into fast SRAM, maintaining running max and normalisers per query, and matches exact attention within machine precision."
      defaultSize="L"
      state={state}
      controls={
        <ControlGroup title="Playback" collapsible={false}>
          <ControlRow label="Tile step">
            <Player
              value={pos}
              onChange={setPosition}
              count={tr.steps.length}
              label="tile"
              format={(k) => `${k} of ${tr.steps.length - 1}`}
            />
          </ControlRow>
        </ControlGroup>
      }
      readouts={{
        tiling: (
          <>
            <Readout label="tiles completed" value={`${at.t} of ${at.tiles.length}`} />
            <Readout label="skipped (causal mask)" value={String(at.skipped)} />
            <Readout
              label="current tile"
              value={at.tile ? `[${at.tile.queries.join('..')}, ${at.tile.keys.join('..')}]` : 'done'}
            />
            <Readout label="max error so far" value={worst > 0 ? worst.toExponential(2) : '0.00'} />
          </>
        ),
      }}
      caption="Step through the FlashAttention execution. Left: The N × N attention score grid coloured by the step number at which each block is loaded and merged into SRAM (unshaded tiles have not yet been evaluated; lower right triangle is omitted when causal masking is active). The outline highlights the current block. Right: The absolute discrepancy between the online accumulator ratio a_i / ℓ_i and standard scaled dot-product attention. Each query row drops to machine precision once its final block has been processed."
    >
      <Plots cols={2}>
        <Plot x={kx} y={qy}>
          <Raster x={range(FT)} y={range(FT)} z={order} valueLabel="tile step" />
          <Curve name="current tile" x={box.x} y={box.y} emphasis live />
        </Plot>
        <Plot x={ex} y={ey}>
          <Bars
            name="error"
            x={range(FT)}
            y={errors.map((e) => (Number.isFinite(e) ? Math.max(e, 1e-17) : NaN))}
            slot={0}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
