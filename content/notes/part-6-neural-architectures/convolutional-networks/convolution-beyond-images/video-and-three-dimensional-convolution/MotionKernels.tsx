import { useMemo } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { conv } from 'aifn-compute/foundation/convolution'
import { fromData, toArray, type Tensor } from 'aifn-compute/foundation/tensor'
import { FixedHeight } from '../_shared/FixedHeight'

/** Frames, rows and columns of the clip. */
const T = 10
const H = 16
const W = 24
/** The row whose space-time slice is drawn. */
const ROW = 8
const SPEEDS = [-2, -1, 0, 1, 2]

type Kernel = 'edge' | 'inflated' | 'difference' | 'tuned'
const KERNELS: { value: Kernel; label: string }[] = [
  { value: 'edge', label: '2-D edge on one frame (k_t = 1)' },
  { value: 'inflated', label: 'inflated edge: the 2-D edge ÷ 3 on 3 frames' },
  { value: 'difference', label: 'temporal difference: next frame − previous' },
  { value: 'tuned', label: 'space-time diagonal: tuned to +1 pixel per frame' },
]

/**
 * A 3 × 3 × 3 kernel, indexed [frame τ][row r][column c]. The layer cross-correlates, so output frame t reads input
 * frames t, t + 1, t + 2 and output pixel (i, j) reads rows i − 1 … i + 1 and columns j − 1 … j + 1.
 */
function kernel(kind: Kernel): number[][][] {
  const edge = [1, 0, -1]
  return [0, 1, 2].map((tau) =>
    [0, 1, 2].map(() =>
      [0, 1, 2].map((c) => {
        switch (kind) {
          case 'edge':
            return tau === 1 ? edge[c] : 0
          case 'inflated':
            return edge[c] / 3
          case 'difference':
            return tau === 2 ? 1 / 9 : tau === 0 ? -1 / 9 : 0
          default:
            // tuned: a column moving one step per frame, mean removed
            return ((c === tau ? 1 : 0) - 1 / 3) / 9
        }
      }),
    ),
  )
}

/** A bright vertical bar two pixels wide whose left edge is at column 11 + v(t − 5) in frame t. */
function clip(v: number): Float64Array {
  const x = new Float64Array(T * H * W)
  for (let t = 0; t < T; t++) {
    const left = 11 + v * (t - 5)
    for (let i = 0; i < H; i++) {
      for (const j of [left, left + 1]) if (j >= 0 && j < W) x[(t * H + i) * W + j] = 1
    }
  }
  return x
}

/** The 3-D convolution of the clip, valid in time (T − 2 output frames) and same-padded in space. */
function respond(v: number, kind: Kernel): number[][][] {
  const x = fromData(clip(v), [1, 1, T, H, W])
  const w = fromData(Float64Array.from(kernel(kind).flat(2)), [1, 1, 3, 3, 3])
  const y = conv(x, w, {
    padding: [
      [0, 0],
      [1, 1],
      [1, 1],
    ],
    flip: false,
    layout: 'ncdhw',
  }) as Tensor
  return (toArray(y) as number[][][][][])[0][0]
}

const energy = (y: number[][][]) => y.flat(2).reduce((a, v) => a + v * v, 0)

/**
 * A bar moving across a short clip, filtered by four 3-D kernels. A kernel whose taps lie along a slanted line in space
 * and time responds to motion at the matching speed; a kernel that is the same on every frame does not see motion.
 */
export function MotionKernels() {
  const state = useFigureState({
    kernel: choice<Kernel>(KERNELS, 'tuned', { label: 'kernel' }),
    speed: int(1, { ge: -2, le: 2, label: 'bar speed v (pixels per frame)' }),
    frame: int(5, { ge: 1, le: T - 2, label: 'frame', onChart: true }),
  })
  const { kernel: kind, speed, frame } = state

  const input = useMemo(() => {
    const x = clip(speed)
    return Array.from({ length: T }, (_, t) =>
      Array.from({ length: H }, (_, i) => Array.from(x.subarray((t * H + i) * W, (t * H + i + 1) * W))),
    )
  }, [speed])
  const output = useMemo(() => respond(speed, kind), [speed, kind])
  const tuning = useMemo(() => SPEEDS.map((v) => energy(respond(v, kind))), [kind])

  const cols = useMemo(() => Array.from({ length: W }, (_, j) => j), [])
  const rows = useMemo(() => Array.from({ length: H }, (_, i) => i), [])
  const inTimes = useMemo(() => Array.from({ length: T }, (_, t) => t), [])
  // Output frame t' reads input frames t', t' + 1, t' + 2, so it is drawn at the middle one.
  const outTimes = useMemo(() => Array.from({ length: T - 2 }, (_, t) => t + 1), [])
  const outFrame = output[frame - 1]

  const fx = useAxis({ label: 'column', range: [-0.5, W - 0.5] })
  const fy = useAxis({ label: 'row', range: [-0.5, H - 0.5], integer: true, equal: fx })
  const sx = useAxis({ label: 'column', range: [-0.5, W - 0.5] })
  const sy = useAxis({ label: 'frame', range: [-0.5, T - 0.5], integer: true })
  const vx = useAxis({ label: 'bar speed v (pixels per frame)', range: [-2.5, 2.5], integer: true })
  const vy = useAxis({ label: 'response energy Σ y²', range: [0, undefined], key: kind })
  return (
    <Figure
      title="Motion is orientation in space and time"
      state={state}
      defaultSize="L"
      caption="A bright bar two pixels wide crosses a clip of 10 frames of 16 × 24 pixels at v pixels per frame. Top: one frame of the clip and the same frame of the output of a 3 × 3 × 3 convolution, which cross-correlates the clip with the kernel over frames, rows and columns (valid in time, so frames 1 to 8 have outputs). Middle: the space-time slice through row 8, one row of pixels per frame, where a moving bar is a slanted stripe; drag the frame line on the left slice to choose the frame shown at the top. Bottom: the energy of the whole output for every speed. The 2-D edge kernel responds to the bar's two edges with the same energy at every speed: one frame carries no motion. Its inflated copy gives the same output on a still clip and averages the bar over three frames when it moves, so its energy falls to a third at 1 pixel per frame and a ninth at 2. The temporal difference gives nothing on a still clip and grows with speed in either direction. The space-time diagonal kernel, whose taps follow a line moving one column per frame, gives nothing on a still clip and responds more than three times as strongly at v = +1 as at v = −1: it detects direction. At 2 pixels per frame it responds equally to both directions, because three frames are too few to tell fast motions apart."
      readouts={
        <>
          <Readout label="response energy at this speed" value={formatNumber(energy(output))} />
          <Readout label="at v = 0" value={formatNumber(tuning[2])} />
          <Readout label="at v = +1 and v = −1" value={`${formatNumber(tuning[3])} and ${formatNumber(tuning[1])}`} />
          <Readout label="weights" value="3 × 3 × 3 = 27" />
        </>
      }
    >
      <FixedHeight height={190}>
        <div className="grid gap-4 md:grid-cols-2">
          <Plot x={fx} y={fy} title={`input, frame ${frame}`} ariaLabel="One frame of the clip">
            <Raster x={cols} y={rows} z={input[frame]} scale="sequential" colorBar={false} valueLabel="brightness" />
          </Plot>
          <Plot x={fx} y={fy} title={`output, frame ${frame}`} ariaLabel="The same frame of the convolution output">
            <Raster x={cols} y={rows} z={outFrame} scale="diverging" valueLabel="response" />
          </Plot>
        </div>
      </FixedHeight>
      <FixedHeight height={190}>
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <Plot x={sx} y={sy} title="input, row 8 in every frame" ariaLabel="Space-time slice of the clip">
            <Raster
              x={cols}
              y={inTimes}
              z={input.map((f) => f[ROW])}
              scale="sequential"
              colorBar={false}
              valueLabel="brightness"
            />
            <Handle
              kind="y"
              at={frame}
              onDrag={(y) => state.set('frame', Math.min(T - 2, Math.max(1, Math.round(y))))}
              label="frame shown"
            />
          </Plot>
          <Plot x={sx} y={sy} title="output, row 8 in every frame" ariaLabel="Space-time slice of the output">
            <Raster x={cols} y={outTimes} z={output.map((f) => f[ROW])} scale="diverging" valueLabel="response" />
            <Annotation y={frame} text="frame shown" dashed />
          </Plot>
        </div>
      </FixedHeight>
      <FixedHeight height={170}>
        <div className="mt-3">
          <Plot x={vx} y={vy} ariaLabel="Response energy against bar speed">
            <Curve name="response energy" x={SPEEDS} y={tuning} slot={0} showPoints />
            <Points name="this speed" x={[speed]} y={[energy(output)]} emphasis size={9} />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
