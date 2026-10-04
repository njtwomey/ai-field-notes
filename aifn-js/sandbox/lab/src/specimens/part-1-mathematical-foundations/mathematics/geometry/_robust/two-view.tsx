/**
 * "Two-view geometry and RANSAC": synthetic correspondences between two cameras (aifn-methods `twoViewScene`) with a
 * chosen share of outliers; core `ransac` fits a homography (plane scene) or a fundamental matrix (scene in depth) from
 * minimal samples, and the player walks through the samples it draws.
 */
import { useMemo } from 'react'
import { fundamentalProblem, homographyProblem, twoViewScene } from 'aifn-methods/vision/two-view'
import {
  applyHomography,
  epipolarLines,
  fundamentalMatrix,
  homography,
  sampsonDistance,
  transferError,
} from 'aifn/numerics/geometry'
import { ransac, type RansacState } from 'aifn/numerics/robust'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat, toRows, type Matrix } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { Player, usePlayhead } from '@lab/controls'
import { Figure } from '@lab/layout'
import { choice, int, row, slider, useFigureState } from '@lab/state'
import { Annotation, Curve, Handle, Plot, Plots, Points, Readout, formatNumber, useAxis } from '@lab/viz'

const fmt = (v: number) => formatNumber(v)
const MAX_SAMPLES = 400

/** The part of the line ax + by + c = 0 inside [0, size]², in plot coordinates (x, −y). */
function clipLine([a, b, c]: number[], size: number) {
  const pts: [number, number][] = []
  if (Math.abs(b) > Math.abs(a)) for (const x of [0, size]) pts.push([x, -(a * x + c) / b])
  else for (const y of [0, size]) pts.push([-(b * y + c) / a, y])
  return { x: pts.map((p) => p[0]), y: pts.map((p) => -p[1]) }
}

export function TwoViewRansacSpecimen() {
  const state = useFigureState({
    scene: row('1 · scene', {
      model: choice(
        [
          { value: 'homography', label: 'plane: homography (4-point samples)' },
          { value: 'fundamental', label: 'depth: fundamental matrix (8-point samples)' },
        ],
        'homography',
        { label: 'model' },
      ),
      outliers: slider(0, 0.7, 0.4, { label: 'outlier fraction', step: 0.01 }),
      noise: slider(0, 3, 0.5, { label: 'pixel noise σ', step: 0.05 }),
      seed: int(0, { ge: 0, le: 9999, label: 'seed' }),
    }),
    fit: row('2 · RANSAC', {
      threshold: slider(0.5, 10, 3, { label: 'inlier threshold (px)', step: 0.1 }),
      confidence: slider(0.5, 0.999, 0.99, { label: 'confidence p', step: 0.001 }),
    }),
    px: slider(0, 400, 120, { onChart: true, label: 'probe x' }),
    py: slider(0, 400, 150, { onChart: true, label: 'probe y' }),
  })
  const { model, outliers, noise, seed } = state.scene
  const { threshold, confidence } = state.fit
  const isH = model === 'homography'
  const scene = useMemo(
    () =>
      twoViewScene(stream(`two-view-${seed}`), {
        kind: isH ? 'plane' : 'depth',
        outlierFraction: outliers,
        noise,
        count: isH ? 80 : 120,
      }),
    [isH, outliers, noise, seed],
  )
  const run = useMemo(() => {
    const problem = isH ? homographyProblem(scene.x1, scene.x2) : fundamentalProblem(scene.x1, scene.x2)
    const tr = trace(ransac<Matrix>(problem, { threshold, confidence }), undefined, MAX_SAMPLES - 1, {
      stream: stream(`two-view-ransac-${seed}`),
    })
    return tr.steps as readonly RansacState<Matrix>[]
  }, [scene, isH, threshold, confidence, seed])
  const [i, setI] = usePlayhead(run.length)
  const s = run[i]
  const x1 = useMemo(() => toRows(scene.x1), [scene])
  const x2 = useMemo(() => toRows(scene.x2), [scene])
  const truthIn = scene.outlier.map((o) => 1 - o)
  // Refit on the consensus set at this step, as `ransacFit` would.
  const refit = useMemo(() => {
    const idx = s.inliers.flatMap((v, k) => (v ? [k] : []))
    const pick = (rows: number[][]) => fromData(Float64Array.from(idx.flatMap((k) => rows[k])), [idx.length, 2])
    try {
      if (isH ? idx.length < 4 : idx.length < 8) return null
      return isH ? homography(pick(x1), pick(x2)) : fundamentalMatrix(pick(x1), pick(x2))
    } catch {
      return null
    }
  }, [s, isH, x1, x2])
  const refitInliers = useMemo(() => {
    if (!refit) return NaN
    const r = isH
      ? toFlat(transferError(refit, scene.x1, scene.x2))
      : toFlat(sampsonDistance(refit, scene.x1, scene.x2)).map(Math.sqrt)
    return r.filter((v) => v <= threshold).length
  }, [refit, isH, scene, threshold])
  const tp = s.inliers.filter((v, k) => v && truthIn[k]).length
  const fp = s.inliers.filter((v, k) => v && !truthIn[k]).length
  const nIn = truthIn.filter(Boolean).length
  const probe: [number, number] = [state.px, state.py]
  const probeOut = useMemo(() => {
    const p = [[probe[0], probe[1]]]
    if (isH) {
      const mapped = (H: Matrix | null) => (H ? toRows(applyHomography(H, p))[0] : null)
      return { est: mapped(refit), truth: mapped(scene.H), lines: null }
    }
    const line = (F: Matrix | null) => (F ? clipLine(toRows(epipolarLines(F, p))[0], scene.size) : null)
    return { est: null, truth: null, lines: { est: line(refit), truth: line(scene.F) } }
  }, [probe[0], probe[1], isH, refit, scene]) // eslint-disable-line react-hooks/exhaustive-deps
  const split = (rows: number[][], mask: number[], want: number) => ({
    x: rows.filter((_, k) => mask[k] === want).map((r) => r[0]),
    y: rows.filter((_, k) => mask[k] === want).map((r) => -r[1]),
  })
  const in2 = split(x2, s.inliers, 1)
  const out2 = split(x2, s.inliers, 0)
  const sample2 = { x: s.sample.map((k) => x2[k][0]), y: s.sample.map((k) => -x2[k][1]) }
  const sample1 = { x: s.sample.map((k) => x1[k][0]), y: s.sample.map((k) => -x1[k][1]) }
  const t1 = split(x1, truthIn, 1)
  const o1 = split(x1, truthIn, 0)
  const steps = run.map((_, k) => k + 1)
  // The adaptive count at the end of the run, where it stopped (fixed while the player moves, so the axis holds).
  const needed = run[run.length - 1].required
  const ix = useAxis({ label: 'x (px)', range: [0, scene.size], zoom: false })
  const iy = useAxis({
    label: 'y (px)',
    format: (v) => formatNumber(-v + 0),
    range: [-scene.size, 0],
    equal: ix,
    zoom: false,
  })
  // Image 2 gets its own pair of axes (same ranges): sharing image 1's left it without y ticks at another scale.
  const ix2 = useAxis({ label: 'x (px)', range: [0, scene.size], zoom: false })
  const iy2 = useAxis({
    label: 'y (px)',
    format: (v) => formatNumber(-v + 0),
    range: [-scene.size, 0],
    equal: ix2,
    zoom: false,
  })
  const sx = useAxis({ label: 'sample', integer: true, key: `${model}${seed}` })
  const sy = useAxis({ label: 'inliers', hold: 'union', key: `${model}${seed}` })
  return (
    <Figure
      title="Two-view geometry and RANSAC"
      purpose="RANSAC fits a model to minimal random samples and keeps the one most data agree with; it stops once an all-inlier sample has been drawn with the chosen confidence, which takes more samples the more outliers there are and the bigger the sample."
      defaultSize="XL"
      state={state}
      controls={
        <Player
          value={i}
          onChange={setI}
          count={run.length}
          format={(k) => `sample ${k + 1}`}
          label="3 · samples drawn"
        />
      }
      readouts={{
        'at this sample': (
          <>
            <Readout label="candidate inliers" value={s.candidateInliers} />
            <Readout label="best inliers" value={s.inlierCount} />
            <Readout
              label="inliers after refitting on them"
              value={Number.isFinite(refitInliers) ? refitInliers : '—'}
            />
            <Readout label="true / false inliers found" value={`${tp} of ${nIn} / ${fp}`} />
            <Readout label="samples needed (adaptive)" value={Number.isFinite(s.required) ? s.required : '∞'} />
            <Readout label="stopped" value={s.terminated ? 'yes' : 'no'} />
          </>
        ),
      }}
      caption={`Left: image 1, true inliers (first colour) and the injected outliers (second), with the current minimal sample ringed in ink. Middle: image 2, coloured by the best consensus set so far (first colour: inliers within ${fmt(threshold)} px, grey: rejected). Drag the probe on image 1: ${isH ? 'its image under the homography refitted on the consensus set (first colour) and under the true one (ink) appear on image 2' : 'its epipolar line under the fundamental matrix refitted on the consensus set (solid) and the true one (dashed) appear on image 2'}. Right: the candidate's inlier count at each sample (dots), the best so far (line) and, dashed, the adaptive number of samples needed at the end of the run (where it stopped). Raise the outlier fraction and watch the required count grow, faster for eight-point samples than four-point ones. The player starts at the first sample.`}
    >
      <Plots cols={3} widths={[1, 1, 1.1]}>
        <Plot x={ix} y={iy} title="image 1">
          <Points name="true inliers" x={t1.x} y={t1.y} slot={0} size={5} />
          <Points name="outliers" x={o1.x} y={o1.y} slot={1} size={5} />
          <Points name="sample" x={sample1.x} y={sample1.y} emphasis shape={1} size={11} live />
          <Points name="probe" x={[probe[0]]} y={[-probe[1]]} slot={3} size={9} live />
          <Handle
            kind="point"
            label="probe"
            at={[probe[0], -probe[1]]}
            onDrag={([x, y]) => {
              state.set('px', Math.max(0, Math.min(scene.size, x)))
              state.set('py', Math.max(0, Math.min(scene.size, -y)))
            }}
          />
        </Plot>
        <Plot x={ix2} y={iy2} title="image 2">
          <Points name="consensus" x={in2.x} y={in2.y} slot={0} size={5} />
          <Points name="rejected" x={out2.x} y={out2.y} muted size={5} />
          <Points name="sample" x={sample2.x} y={sample2.y} emphasis shape={1} size={11} live />
          {probeOut.est && (
            <Points name="probe, estimated H" x={[probeOut.est[0]]} y={[-probeOut.est[1]]} slot={3} size={9} />
          )}
          {probeOut.truth && (
            <Points
              name="probe, true H"
              x={[probeOut.truth[0]]}
              y={[-probeOut.truth[1]]}
              emphasis
              shape={1}
              size={13}
            />
          )}
          {probeOut.lines?.est && (
            <Curve name="epipolar line (estimated F)" x={probeOut.lines.est.x} y={probeOut.lines.est.y} slot={3} />
          )}
          {probeOut.lines?.truth && (
            <Curve
              name="epipolar line (true F)"
              x={probeOut.lines.truth.x}
              y={probeOut.lines.truth.y}
              emphasis
              dashed
            />
          )}
        </Plot>
        <Plot x={sx} y={sy} title="consensus">
          <Points name="candidate" x={steps} y={run.map((r) => r.candidateInliers)} slot={2} size={4} />
          <Curve name="best so far" x={steps} y={run.map((r) => r.inlierCount)} slot={0} />
          {Number.isFinite(needed) && needed <= MAX_SAMPLES && <Annotation x={needed} dashed text="stop" />}
          <Points name="now" x={[i + 1]} y={[s.inlierCount]} emphasis live />
        </Plot>
      </Plots>
    </Figure>
  )
}
