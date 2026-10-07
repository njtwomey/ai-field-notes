import { useMemo, useRef, useState } from 'react'
import { choice, Figure, formatNumber, int, Readout, setting, useElementSize, useFigureState } from 'aifn-render'
import { chrome, seriesColor, useTheme } from 'aifn-render/design'
import { drawMap, drawView, prepareCanvas } from '../_shared/render'
import { useGame } from '../_shared/useGame'
import { castFan, rayOffsets, WORLDS, worldById, type Hit, type Pose } from '../_shared/world'

const MATERIAL_NAMES = ['', 'stone', 'brick', 'wood', 'metal']

export function RaycasterGame() {
  const state = useFigureState({
    world: choice(
      WORLDS.map((w) => ({ value: w.id, label: w.label })),
      'columns',
      { label: 'map' },
    ),
    fov: int(66, { ge: 30, le: 140, suggestions: [45, 66, 90, 120], label: 'field of view (°)' }),
    rays: int(240, { ge: 2, le: 480, suggestions: [8, 24, 80, 240, 480], label: 'rays (screen columns)' }),
    look: choice(
      [
        { value: 'texture', label: 'textured walls' },
        { value: 'depth', label: 'depth only' },
      ],
      'texture',
      { label: 'walls' },
    ),
    fisheye: setting(false, { label: 'fish-eye (Euclidean distance)' }),
  })
  const world = worldById(state.world)
  const fov = (state.fov * Math.PI) / 180
  const offsets = useMemo(() => rayOffsets(state.rays, fov), [state.rays, fov])
  const mode = useTheme().resolved
  const dark = mode === 'dark'
  const colours = chrome(mode)
  const rayColour = seriesColor(mode, 3)

  const [viewBox, viewSize] = useElementSize<HTMLDivElement>()
  const [mapBox, mapSize] = useElementSize<HTMLDivElement>()
  const viewCanvas = useRef<HTMLCanvasElement>(null)
  const mapCanvas = useRef<HTMLCanvasElement>(null)
  const [info, setInfo] = useState<{ pose: Pose; centre: Hit } | null>(null)
  const lastInfo = useRef(0)

  const W = viewSize.width
  const H = Math.round(W * 0.62)
  const M = Math.min(mapSize.width, 420)

  const { areaProps, focused, mapPointer } = useGame(world, (pose) => {
    const hits = castFan(world, pose, offsets)
    const distances = hits.map((h) => h.distance)
    if (viewCanvas.current && W > 0) {
      const ctx = prepareCanvas(viewCanvas.current, W, H)
      drawView(ctx, W, H, distances, hits, { fov, offsets, mode: state.look, fisheye: state.fisheye, dark })
    }
    if (mapCanvas.current && M > 0) {
      const ctx = prepareCanvas(mapCanvas.current, M, M)
      drawMap(ctx, M, world, pose, [{ distances, colour: rayColour, lines: true }], {
        offsets,
        dark,
        ink: colours.ink,
        grid: colours.grid,
        surface: colours.surface,
      })
    }
    // Readouts at about 8 Hz, so walking does not re-render React every frame.
    const now = performance.now()
    if (now - lastInfo.current > 120) {
      lastInfo.current = now
      setInfo({ pose: { ...pose }, centre: castFan(world, pose, Float64Array.of(0))[0] })
    }
  })

  const deg = info ? ((((info.pose.theta * 180) / Math.PI) % 360) + 360) % 360 : 0
  return (
    <Figure
      title="A ray-casting renderer"
      state={state}
      defaultSize="XL"
      caption="Left: the first-person view. Each screen column is one ray from the eye through the projection plane; the column's wall slice is f / p pixels tall, where p is the hit's distance perpendicular to the projection plane and f = (W / 2) / tan(fov / 2). Faces met across a horizontal grid line are drawn darker, as in Wolfenstein 3D. Right: the same scene from above, with the camera (black), its rays and their hits (yellow). Click the view, then walk with the arrow keys or W A S D (Q and E also turn, Shift runs). Press on the map to move the camera to that cell and drag to turn it. Reduce the rays to see the view as a few wide slices; turn on fish-eye to use the Euclidean distance, which bends straight walls."
      readouts={
        <>
          <Readout
            label="position (x, y)"
            value={info ? `${formatNumber(info.pose.x)}, ${formatNumber(info.pose.y)}` : '–'}
          />
          <Readout label="heading" value={info ? `${deg.toFixed(0)}°` : '–'} />
          <Readout label="centre ray distance" value={info ? formatNumber(info.centre.distance) : '–'} />
          <Readout label="centre ray hits" value={info ? MATERIAL_NAMES[info.centre.material] : '–'} />
        </>
      }
    >
      <div
        {...areaProps}
        className="grid gap-4 rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring md:grid-cols-[3fr_2fr]"
      >
        <div ref={viewBox} className="relative min-w-0">
          <canvas ref={viewCanvas} style={{ width: W, height: H }} className="block rounded-md" />
          {!focused && (
            <div className="pointer-events-none absolute inset-x-0 top-2 text-center text-xs text-white/85">
              Click here, then use the arrow keys or W A S D
            </div>
          )}
        </div>
        <div ref={mapBox} className="min-w-0">
          <canvas
            ref={mapCanvas}
            style={{ width: M, height: M, touchAction: 'none' }}
            className="mx-auto block cursor-crosshair rounded-md"
            {...mapPointer(M / Math.max(world.width, world.height))}
          />
        </div>
      </div>
    </Figure>
  )
}
