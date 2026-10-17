import { useMemo } from 'react'
import { Bars, Figure, float, formatNumber, int, Plot, Readout, row, useAxis, useFigureState } from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { shapeTextureDataset, type TextureName } from 'aifn-methods/data/synthetic'
import { FixedHeight, ImageTile, useTrainedCnn } from '../_shared/figure-helpers'
import {
  argmax,
  imageAt,
  logits,
  SHAPE_CLASSES,
  SIZE,
  smallCnn,
  trainingSet,
  type TrainSetup,
} from '../_shared/small-cnn'

/** Texture i is the one paired with shape i in training: horizontal stripes with discs, and so on. */
const TEXTURES: readonly TextureName[] = ['horizontal', 'vertical', 'checker']
const TEXTURE_NAMES = ['horizontal stripes', 'vertical stripes', 'checks']
const net = smallCnn(SHAPE_CLASSES.length)

const conflict = shapeTextureDataset(stream(22), { perClass: 50, pairing: 'conflict', textures: TEXTURES })
const matched = shapeTextureDataset(stream(21), { perClass: 30, textures: TEXTURES })
const silhouettes = shapeTextureDataset(stream(23), { perClass: 30, textures: ['solid', 'solid', 'solid'] })
/** A few cue-conflict images to show, two of each shape. */
const SHOWN = [0, 1, 50, 51, 100, 101]
const SHOWN_TRAIN = [0, 1, 40, 41, 80, 81]

/**
 * A small network trained on shapes whose texture matches their class, then tested on cue-conflict images with the
 * shape of one class and the texture of another. Restyling a share of the training images (drawing their texture at
 * random) removes texture as a reliable cue and moves the decisions from texture to shape.
 */
export function CueConflict() {
  const state = useFigureState({
    training: row('train the network', {
      restyle: float(0, { ge: 0, le: 1, suggestions: [0, 0.25, 0.5, 1], label: 'restyled share ρ' }),
      steps: int(120, { ge: 10, le: 600, suggestions: [60, 120, 240], label: 'Adam steps' }),
      rate: float(0.02, { gt: 0, le: 0.5, suggestions: [0.005, 0.01, 0.02, 0.05], label: 'learning rate' }),
    }),
  })
  const { restyle, steps, rate } = state.training
  const setup: TrainSetup = useMemo(
    () => ({
      shapes: SHAPE_CLASSES,
      textures: TEXTURES,
      restyle,
      perClass: 40,
      steps,
      learningRate: rate,
      batchSize: 30,
      seed: 1,
    }),
    [restyle, steps, rate],
  )
  const progress = useTrainedCnn(setup)
  const params = progress.params
  const train = useMemo(() => trainingSet(setup), [setup])

  const result = useMemo(() => {
    if (!params) return null
    const accuracy = (d: typeof matched) => {
      const pred = logits(net, params, d.x).map(argmax)
      return pred.filter((p, i) => p === d.shape[i]).length / pred.length
    }
    const pred = logits(net, params, conflict.x).map(argmax)
    let shape = 0
    let texture = 0
    pred.forEach((p, i) => {
      if (p === conflict.shape[i]) shape++
      else if (p === conflict.texture[i]) texture++
    })
    const n = pred.length
    return {
      pred,
      shape: shape / n,
      texture: texture / n,
      other: (n - shape - texture) / n,
      bias: shape + texture > 0 ? shape / (shape + texture) : NaN,
      matched: accuracy(matched),
      silhouettes: accuracy(silhouettes),
    }
  }, [params])

  const decisionAxis = useAxis({ categories: ['shape class', 'texture class', 'neither'] })
  const shareAxis = useAxis({ label: 'share of cue-conflict decisions', range: [0, 1] })
  return (
    <Figure
      title="Shape or texture on cue-conflict images"
      state={state}
      caption={
        "The network of the chapter (two 3 × 3 convolutions, global average pooling, a linear layer) is trained in the page on 16 × 16 shapes filled with a texture: discs with horizontal stripes, squares with vertical stripes, crosses with checks. Restyled training images (share ρ) get a texture drawn at random. Left: training images. Middle: cue-conflict test images, each with the shape of one class and the texture of another, labelled with the network's answer. Right: the decisions on 150 cue-conflict images. At ρ = 0 both cues predict the label perfectly and the network answers by texture; raise ρ and its answers move to the shape."
      }
      readouts={
        <>
          <Readout
            label="training"
            value={progress.done ? `${steps} steps done` : `step ${progress.losses.length} of ${steps}`}
          />
          <Readout label="shape bias, shape ÷ (shape + texture)" value={result ? formatNumber(result.bias) : '—'} />
          <Readout label="accuracy, texture matches shape" value={result ? formatNumber(result.matched) : '—'} />
          <Readout label="accuracy, plain silhouettes" value={result ? formatNumber(result.silhouettes) : '—'} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_1fr_1fr]">
        <div>
          <div className="mb-1 text-xs text-muted-foreground">training images</div>
          <div className="grid grid-cols-3 gap-2">
            {SHOWN_TRAIN.map((i) => (
              <ImageTile
                key={i}
                values={imageAt(train.x, i)}
                width={SIZE}
                height={SIZE}
                range={[0, 1]}
                title={SHAPE_CLASSES[train.shape[i]]}
                ariaLabel={`Training image of a ${SHAPE_CLASSES[train.shape[i]]}`}
              />
            ))}
          </div>
        </div>
        <div>
          <div className="mb-1 text-xs text-muted-foreground">cue-conflict images: shape / texture → answer</div>
          <div className="grid grid-cols-3 gap-2">
            {SHOWN.map((i) => {
              const p = result ? result.pred[i] : -1
              const verdict =
                p < 0 ? '' : p === conflict.shape[i] ? 'shape' : p === conflict.texture[i] ? 'texture' : 'other'
              return (
                <div key={i}>
                  <ImageTile
                    values={imageAt(conflict.x, i)}
                    width={SIZE}
                    height={SIZE}
                    range={[0, 1]}
                    title={p < 0 ? '' : `→ ${SHAPE_CLASSES[p]}`}
                    ariaLabel={`Cue-conflict image: ${SHAPE_CLASSES[conflict.shape[i]]} with ${TEXTURE_NAMES[conflict.texture[i]]}`}
                  />
                  <div className="text-[10px] leading-tight text-muted-foreground">
                    {SHAPE_CLASSES[conflict.shape[i]]} / {TEXTURE_NAMES[conflict.texture[i]]}
                    {verdict && ` (${verdict})`}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
        <FixedHeight height={240}>
          <Plot x={decisionAxis} y={shareAxis} ariaLabel="Shares of shape, texture and other decisions">
            <Bars
              name="share of decisions"
              x={[0, 1, 2]}
              y={result ? [result.shape, result.texture, result.other] : [0, 0, 0]}
            />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
