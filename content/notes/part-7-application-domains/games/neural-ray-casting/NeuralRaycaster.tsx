import { useContext, useEffect, useMemo, useRef, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  FrameContext,
  int,
  Plot,
  Readout,
  useAxis,
  useElementSize,
  useFigureState,
} from 'aifn-render'
import { chrome, seriesColor, useTheme } from 'aifn-render/design'
import { Button } from '@/components/ui/button'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { ravel, type Params } from 'aifn-compute/foundation/pytree'
import { drawMap, drawView, prepareCanvas } from '../_shared/render'
import { useGame } from '../_shared/useGame'
import { castFan, rayOffsets, WORLDS, worldById } from '../_shared/world'
import { initialParams, network, predict, type ModelOptions } from './neural'
import { loadModel, modelUrl, runRays, TRAINED } from './onnx'
import type { InferenceSession } from 'onnxruntime-web/wasm'
import type { Checkpoint, FromWorker, ToWorker } from './train.worker'

const FOV = (66 * Math.PI) / 180
const training = (v: Readonly<Record<string, unknown>>) => v.mode === 'train'

/** The model classes, each with its trained models in the manifest's order. */
const CLASSES = [...new Set(TRAINED.map((m) => m.family))].map((family) => {
  const models = [...new Map(TRAINED.filter((m) => m.family === family).map((m) => [m.name, m])).values()]
  return { family, label: models[0].familyLabel, models }
})
const available = (name: string, world: string) => TRAINED.some((m) => m.name === name && m.world === world)

/** One single-choice row of buttons; options that cannot be chosen are greyed out. */
function ButtonRow({
  label,
  groups,
  value,
  onChange,
}: {
  label: string
  groups: { label?: string; options: { value: string; label: string; disabled?: boolean }[] }[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div className="flex gap-3 text-xs">
      <span className="w-10 shrink-0 pt-1.5 text-muted-foreground">{label}</span>
      <div className="flex min-w-0 flex-col gap-1.5">
        {groups.map((g, i) => (
          <div key={i} className="flex flex-wrap items-center gap-x-2 gap-y-1">
            {g.label && <span className="w-56 shrink-0 text-muted-foreground">{g.label}</span>}
            <ToggleGroup
              size="sm"
              variant="outline"
              spacing={0}
              value={g.options.some((o) => o.value === value) ? [value] : []}
              onValueChange={(v: unknown[]) => v.length && onChange(String(v[0]))}
              className="flex-wrap"
            >
              {g.options.map((o) => (
                <ToggleGroupItem key={o.value} value={o.value} disabled={o.disabled} className="px-2.5 text-xs">
                  {o.label}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          </div>
        ))}
      </div>
    </div>
  )
}

export function NeuralRaycaster() {
  const state = useFigureState({
    mode: choice(
      [
        { value: 'use', label: 'use a trained model (ONNX)' },
        { value: 'train', label: 'train a network here' },
      ],
      'use',
      { label: 'mode' },
    ),
    world: choice(
      WORLDS.map((w) => ({ value: w.id, label: w.label })),
      'tiny-maze',
      { label: 'map', when: training },
    ),
    form: choice(
      [
        { value: 'ray', label: 'one ray: (x, y, φ) → d' },
        { value: 'fan', label: 'whole view: (x, y, θ) → R distances' },
      ],
      'ray',
      { label: 'network', when: training },
    ),
    target: choice(
      [
        { value: 'log', label: 'log distance' },
        { value: 'distance', label: 'distance' },
      ],
      'log',
      { label: 'output', when: training },
    ),
    encoding: choice(
      [
        { value: 'fourier', label: 'Fourier features' },
        { value: 'raw', label: 'raw (x, y, cos, sin)' },
      ],
      'fourier',
      { label: 'input encoding', when: training },
    ),
    octaves: int(6, {
      ge: 1,
      le: 10,
      suggestions: [2, 4, 6, 8],
      label: 'octaves L',
      when: (v) => v.mode === 'train' && v.encoding === 'fourier',
    }),
    width: int(64, { ge: 8, le: 256, suggestions: [32, 64, 128], label: 'hidden width', when: training }),
    depth: int(2, { ge: 1, le: 5, suggestions: [1, 2, 3], label: 'hidden layers', when: training }),
    learningRate: float(0.01, {
      gt: 0,
      le: 0.1,
      suggestions: [0.001, 0.003, 0.01],
      label: 'learning rate',
      when: training,
    }),
    batchSize: int(1024, {
      ge: 16,
      le: 4096,
      suggestions: [64, 256, 1024],
      label: 'batch size (rays)',
      when: training,
    }),
    rays: int(48, { ge: 4, le: 128, suggestions: [16, 32, 48, 96], label: 'rays R' }),
    every: int(100, {
      ge: 1,
      le: 10000,
      suggestions: [1, 10, 100, 1000],
      label: 'update the network view every (steps)',
      when: training,
    }),
  })
  const using = state.mode === 'use'
  // A trained model is chosen first, then one of the maps it was trained on.
  const [modelName, setModelName] = useState('walls')
  const [useWorld, setUseWorld] = useState('small-maze')
  const pickModel = (name: string) => {
    setModelName(name)
    if (!available(name, useWorld)) setUseWorld(TRAINED.find((m) => m.name === name)?.world ?? useWorld)
  }
  const trained = using ? TRAINED.find((m) => m.name === modelName && m.world === useWorld) : undefined
  const world = worldById(using ? useWorld : state.world)
  const offsets = useMemo(() => rayOffsets(state.rays, FOV), [state.rays])
  const options: ModelOptions = useMemo(
    () => ({
      form: state.form,
      target: state.target,
      rays: state.rays,
      fov: FOV,
      encoding: state.encoding,
      octaves: state.octaves,
      width: state.width,
      depth: state.depth,
      learningRate: state.learningRate,
      batchSize: state.batchSize,
    }),
    [
      state.form,
      state.target,
      state.rays,
      state.encoding,
      state.octaves,
      state.width,
      state.depth,
      state.learningRate,
      state.batchSize,
    ],
  )
  // The page keeps its own copy of the network to draw the network's view; the weights arrive from the worker.
  const net = useMemo(() => network(options), [options])
  const start = useMemo(() => initialParams(world, options), [world, options])
  const unravel = useMemo(() => ravel(start).unravel, [start])
  const params = useRef<Params[]>(start)

  const [running, setRunning] = useState(false)
  const [history, setHistory] = useState<Checkpoint[]>([])
  const [steps, setSteps] = useState(0)
  const [rate, setRate] = useState(0)

  // Training runs in a worker. Each model gets a new generation, so messages from a replaced model are dropped.
  const worker = useRef<Worker | null>(null)
  const generation = useRef(0)
  const unravelRef = useRef(unravel)
  const send = (m: ToWorker) => worker.current?.postMessage(m)
  useEffect(() => {
    const w = new Worker(new URL('./train.worker.ts', import.meta.url), { type: 'module' })
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data
      if (m.generation !== generation.current) return
      params.current = unravelRef.current(m.params)
      setSteps(m.t)
      if (m.t > 0) setRate(m.stepsPerSecond)
      const c = m.checkpoint
      if (c) setHistory((h) => [...h.filter((p) => p.t < c.t), c])
    }
    worker.current = w
    return () => w.terminate()
  }, [])

  const reset = () => {
    generation.current += 1
    unravelRef.current = unravel
    params.current = start
    setRunning(false)
    setHistory([])
    setSteps(0)
    setRate(0)
    send({ type: 'run', on: false })
    send({ type: 'init', generation: generation.current, world: world.id, options })
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(reset, [world, options, unravel, start])
  useEffect(() => send({ type: 'run', on: running }), [running])
  useEffect(() => send({ type: 'every', steps: state.every }), [state.every])
  useEffect(() => {
    if (using) setRunning(false)
  }, [using])

  // A trained model runs in ONNX Runtime Web; its latest view is drawn while the next one is computed.
  const [session, setSession] = useState<InferenceSession | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const onnxView = useRef<{ distances: Float32Array } | null>(null)
  const busy = useRef(false)
  useEffect(() => {
    setSession(null)
    setLoadError(null)
    onnxView.current = null
    if (!using) return
    if (!trained) return
    let live = true
    loadModel(modelUrl(trained)).then(
      (s) => live && setSession(s),
      (e: unknown) => live && setLoadError(e instanceof Error ? e.message : String(e)),
    )
    return () => {
      live = false
    }
  }, [using, trained])
  const [viewScore, setViewScore] = useState<number | null>(null)
  const lastScore = useRef(0)

  const mode = useTheme().resolved
  const dark = mode === 'dark'
  const colours = chrome(mode)
  const truthColour = seriesColor(mode, 3)
  const netColour = seriesColor(mode, 0)

  const [viewBox, viewSize] = useElementSize<HTMLDivElement>()
  const [mapBox, mapSize] = useElementSize<HTMLDivElement>()
  const truthCanvas = useRef<HTMLCanvasElement>(null)
  const netCanvas = useRef<HTMLCanvasElement>(null)
  const mapCanvas = useRef<HTMLCanvasElement>(null)
  const W = viewSize.width
  const H = Math.round(W * 0.42)
  const M = Math.min(mapSize.width, 2 * H + 28)

  const { areaProps, focused, mapPointer } = useGame(world, (pose) => {
    const truth = castFan(world, pose, offsets)
    const truthD = truth.map((h) => h.distance)
    let netD: ArrayLike<number> | null
    if (using) {
      if (session && !busy.current) {
        busy.current = true
        const rays = new Float32Array(3 * offsets.length)
        offsets.forEach((o, i) => rays.set([pose.x, pose.y, pose.theta + o], 3 * i))
        runRays(session, rays)
          .then((d) => (onnxView.current = { distances: d }))
          .catch((e: unknown) => setLoadError(e instanceof Error ? e.message : String(e)))
          .finally(() => (busy.current = false))
      }
      const v = onnxView.current?.distances
      netD = v && v.length === offsets.length ? v : null
    } else netD = predict(net, world, params.current, pose, offsets, options)
    // The share of this view's rays within 10 % of the truth, a few times a second.
    const now = performance.now()
    if (netD && now - lastScore.current > 250) {
      lastScore.current = now
      const n = truthD.filter((t, i) => Math.abs(netD[i] - t) / t < 0.1).length
      setViewScore(n / truthD.length)
    }
    const view = { fov: FOV, offsets, mode: 'depth' as const, dark }
    if (W > 0) {
      if (truthCanvas.current) drawView(prepareCanvas(truthCanvas.current, W, H), W, H, truthD, truth, view)
      if (netCanvas.current) {
        const ctx = prepareCanvas(netCanvas.current, W, H)
        if (netD) drawView(ctx, W, H, netD, null, view)
        else ctx.clearRect(0, 0, W, H)
      }
    }
    if (mapCanvas.current && M > 0) {
      drawMap(
        prepareCanvas(mapCanvas.current, M, M),
        M,
        world,
        pose,
        netD
          ? [
              { distances: truthD, colour: truthColour },
              { distances: netD, colour: netColour, lines: true },
            ]
          : [{ distances: truthD, colour: truthColour }],
        { offsets, dark, ink: colours.ink, grid: colours.grid, surface: colours.surface },
      )
    }
  })

  const tAxis = useAxis({ label: 'training step' })
  const eAxis = useAxis({ label: 'RMS error (cells)', log: true })
  const curves = useMemo(
    () => ({
      t: history.map((c) => c.t),
      train: history.map((c) => c.train),
      test: history.map((c) => c.test),
    }),
    [history],
  )
  const last = history.at(-1)
  // A chart inside a Figure takes its height from the frame; this one is a short strip under the views.
  const frame = useContext(FrameContext)

  return (
    <Figure
      title="A network that sees the world"
      state={state}
      defaultSize="XL"
      caption="The figure opens on models trained offline (the Code tab) and run in the browser with ONNX Runtime Web: pick a model, then one of the maps it was trained on; maps it was not trained on are greyed out. The wall-line classifiers pick the grid line each ray stops on and compute its distance exactly, so their walls are straight and their edges sharp. Switch the mode to train a network in the page. Top left: the ray caster's view, each column coloured by its distance alone. Bottom left: the network's view, the same columns drawn from the distances the network predicts at the camera's pose. Right: the map with the true hits (yellow) and the network's rays and hits (blue). Below: the root-mean-square error of the predicted distances, over the whole field of view, at 200 training poses and 200 held-out poses. Training runs in a background thread at full speed; the network's view, its rays on the map and the readouts take the newest weights every N steps (update the network view every), and the errors are measured at most every 100 steps, since one measurement costs about fifteen training steps. Press Train, then walk with the arrow keys or W A S D after clicking the views, or press and drag on the map. Early on the network predicts a smooth average room; the columns' edges, which are jumps in distance, appear last. Compare raw inputs with Fourier features, and log distance with distance."
      readouts={
        using ? (
          <>
            <Readout label="model" value={trained ? `${trained.familyLabel}, ${trained.label}` : '–'} />
            {trained && (
              <>
                <Readout label="parameters" value={trained.params.toLocaleString()} />
                <Readout
                  label="held-out rays within 10 % (offline test)"
                  value={`${(100 * trained.within10).toFixed(1)}%`}
                />
                <Readout
                  label="held-out median relative error"
                  value={`${(100 * trained.medianRelative).toFixed(1)}%`}
                />
              </>
            )}
            <Readout
              label="this view: rays within 10 %"
              value={viewScore === null ? '–' : `${(100 * viewScore).toFixed(0)}%`}
            />
          </>
        ) : (
          <>
            <Readout
              label="this view: rays within 10 %"
              value={viewScore === null ? '–' : `${(100 * viewScore).toFixed(0)}%`}
            />
            <Readout label="steps" value={String(steps)} />
            <Readout label="training RMS error" value={last ? formatNumber(last.train) : '–'} />
            <Readout label="held-out RMS error" value={last ? formatNumber(last.test) : '–'} />
            <Readout
              label="held-out median relative error"
              value={last ? `${(100 * last.relative).toFixed(1)}%` : '–'}
            />
            <Readout label="held-out rays within 10 %" value={last ? `${(100 * last.within10).toFixed(0)}%` : '–'} />
            <Readout label="steps per second" value={rate ? rate.toFixed(0) : '–'} />
          </>
        )
      }
    >
      <div className="flex flex-col gap-3">
        {using ? (
          <div className="flex flex-col gap-2">
            <ButtonRow
              label="model"
              value={modelName}
              onChange={pickModel}
              groups={CLASSES.map((c) => ({
                label: c.label,
                options: c.models.map((m) => {
                  // Models trained on one map only say so, rather than leaving the reader to find the other maps greyed.
                  const maps = TRAINED.filter((t) => t.name === m.name)
                  const only =
                    maps.length === 1
                      ? ` (${worldById(maps[0].world)
                          .label.replace(/ \(.*\)$/, '')
                          .toLowerCase()} only)`
                      : ''
                  return { value: m.name, label: m.label + only }
                }),
              }))}
            />
            <ButtonRow
              label="map"
              value={useWorld}
              onChange={setUseWorld}
              groups={[
                {
                  options: WORLDS.map((w) => ({
                    value: w.id,
                    label: w.label,
                    disabled: !available(modelName, w.id),
                  })),
                },
              ]}
            />
            <span className="text-xs text-muted-foreground">
              {loadError
                ? `Could not run the model: ${loadError}`
                : session
                  ? 'Running in ONNX Runtime Web. Input rays [N, 3] = (x, y, φ); output distance [N].'
                  : 'Loading the model…'}
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={() => setRunning((r) => !r)}>
              {running ? 'Pause' : 'Train'}
            </Button>
            <Button size="sm" variant="outline" onClick={reset}>
              Reset
            </Button>
            <span className="text-xs text-muted-foreground">step {steps}</span>
          </div>
        )}
        <div
          {...areaProps}
          className="grid gap-4 rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring md:grid-cols-[3fr_2fr]"
        >
          <div ref={viewBox} className="relative flex min-w-0 flex-col gap-2">
            <div className="relative">
              <canvas ref={truthCanvas} style={{ width: W, height: H }} className="block rounded-md" />
              <span className="absolute top-1.5 left-2 text-xs text-white/85">ray caster</span>
            </div>
            <div className="relative">
              <canvas ref={netCanvas} style={{ width: W, height: H }} className="block rounded-md" />
              <span className="absolute top-1.5 left-2 text-xs text-white/85">network</span>
            </div>
            {!focused && (
              <div className="pointer-events-none absolute inset-x-0 top-1.5 text-center text-xs text-white/85">
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
        {!using && (
          <FrameContext.Provider value={{ ...frame, height: 200 }}>
            <Plot x={tAxis} y={eAxis}>
              <Curve name="training poses" x={curves.t} y={curves.train} slot={1} width={1.5} />
              <Curve name="held-out poses" x={curves.t} y={curves.test} slot={0} width={1.5} />
            </Plot>
          </FrameContext.Provider>
        )}
      </div>
    </Figure>
  )
}
