import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { LABEL_COUNTS, POINTS, REPEATS, runStudy, type Problem } from './pretraining-study'

/**
 * Pretrain on all inputs without labels, fine-tune a head on the few labels, distil into a student: learning curves
 * against supervised training and self-training from scratch, on data where the unlabelled structure matches the
 * classes and on data where it does not.
 */
export function PretrainFineTuneDistil() {
  const fig = useFigureState({
    problem: choice(
      [
        { value: 'circles', label: 'two circles (clusters are the classes)' },
        { value: 'moons', label: 'two moons (clusters are the classes)' },
        { value: 'across', label: 'two blobs, class cuts across both' },
      ],
      'circles',
      { label: 'data' },
    ),
    seed: int(1, { min: 1, max: 999, label: 'data seed' }),
  })
  const { problem, seed } = fig
  const study = useComputed(() => runStudy(problem as Problem, seed), [problem, seed], { mode: 'release' })
  const { rows, embedding, y } = study.value

  const xa = useAxis({ label: 'labels per class', log: true, range: [0.8, 20] })
  const ya = useAxis({ label: 'accuracy', range: [0.5, 1.01] })
  const xe = useAxis({ label: 'embedding coordinate 1' })
  const ye = useAxis({ label: 'embedding coordinate 2' })
  const m = LABEL_COUNTS
  const byClass = [0, 1].map((k) => embedding.filter((_, i) => y[i] === k))

  return (
    <Figure
      title="Pretrain, fine-tune, distil"
      state={fig}
      caption={`${POINTS} unlabelled points with 1 to 16 labels per class, averaged over ${REPEATS} draws of the labels. Supervised: logistic regression on 40 random Fourier features of the labelled points. Self-training: the same model refitted three times on its confident predictions. Pretrain and fine-tune: a two-dimensional representation learned from all ${POINTS} inputs without labels (Laplacian eigenmaps, the spectral embedding that the spectral contrastive loss recovers), then a logistic head trained on the labelled points. Distilled: the fine-tuned model labels every point and a fresh random-feature student learns those labels from the raw inputs; it is scored on 400 new points, which the transductive embedding cannot embed. Right: the learned representation, coloured by the true class it never saw. On circles and moons the representation separates the classes, and one or two labels per class reach accuracy that supervised training needs many more labels for. On two blobs whose class cuts across both, the representation separates the blobs, not the classes, and pretraining loses to supervised training as labels grow.`}
      readouts={
        <>
          <Readout label="supervised, 2 labels per class" value={formatNumber(rows[1].supervised)} />
          <Readout label="pretrained and fine-tuned, 2 labels" value={formatNumber(rows[1].fineTuned)} />
          <Readout label="distilled student on new points, 2 labels" value={formatNumber(rows[1].distilledTest)} />
          <Readout
            label={`supervised, ${m[m.length - 1]} labels`}
            value={formatNumber(rows[rows.length - 1].supervised)}
          />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={xa} y={ya} title="accuracy on the unlabelled points">
          <Curve name="supervised" x={m} y={rows.map((r) => r.supervised)} slot={3} showPoints stale={study.stale} />
          <Curve
            name="self-training from scratch"
            x={m}
            y={rows.map((r) => r.selfTraining)}
            slot={4}
            showPoints
            stale={study.stale}
          />
          <Curve
            name="pretrained, fine-tuned"
            x={m}
            y={rows.map((r) => r.fineTuned)}
            slot={2}
            showPoints
            stale={study.stale}
          />
          <Curve
            name="distilled student, new points"
            x={m}
            y={rows.map((r) => r.distilledTest)}
            slot={2}
            dashed
            stale={study.stale}
          />
        </Plot>
        <Plot x={xe} y={ye} title="representation learned without labels">
          {byClass.map((pts, k) =>
            pts.length > 0 ? (
              <Points
                key={k}
                name={`class ${k}`}
                x={pts.map((p) => p[0])}
                y={pts.map((p) => p[1])}
                slot={k}
                thin
                stale={study.stale}
              />
            ) : null,
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
