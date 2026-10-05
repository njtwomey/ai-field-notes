import { useMemo } from 'react'
import { iris } from 'aifn-methods/data/real'
import { recipe, type Dataset } from 'aifn-methods/data'
import { gaussians, shuffleDataset, withNuisanceFeatures } from 'aifn-methods/data/synthetic'
import { child, stream } from 'aifn-compute/foundation/random'
import { Figure } from 'aifn-render/layout'
import { choice, useFigureState } from 'aifn-render/state'
import { AndrewsCurvesPanel, PairPlotPanel, ParallelCoordinatesPanel } from '@lab/views'

const IRIS = iris()

type Choice = { label: string; make: () => Dataset; purpose: string }

const PAIR_DATA: Record<'iris' | 'blobs' | 'wide', Choice> = {
  iris: {
    label: 'Iris (4 measurements)',
    make: () => IRIS,
    purpose:
      'Which measurements separate the three Iris species: petal length and width split setosa off completely, and sepal width barely separates anything.',
  },
  blobs: {
    label: 'blobs + 3 nuisance features',
    make: () =>
      recipe({
        base: 'blobs',
        seed: 3,
        knobs: { n: 450, centers: 3 },
        modifiers: [{ op: 'withNuisanceFeatures', params: { count: 3 } }, { op: 'shuffleDataset' }],
      }),
    purpose:
      'Three blobs in x1 and x2 plus three pure-noise features: only the x1–x2 panels separate the classes, and every panel with a noise feature shows the classes on top of each other.',
  },
  wide: {
    label: '8-D Gaussians + 8 nuisance (16 features)',
    // Means in 8 dimensions are not a recipe knob, so the generator and modifiers are called directly.
    make: () => {
      const s = stream(5)
      const d = gaussians(child(s, 'base'), { n: 1500, means: MEANS_8D })
      return shuffleDataset(child(s, 'shuffle'), withNuisanceFeatures(child(s, 'nuisance'), d, { count: 8 }))
    },
    purpose:
      'Sixteen features are too many panels: pick up to eight. Only x1–x4 carry class information; compare a pair of them with a pair of noise features.',
  },
}

/** Three Gaussian classes in 8 dimensions that differ in x1–x4 only; x5–x8 have the same distribution in every class. */
const MEANS_8D = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [2.2, 1.6, 0, 0, 0, 0, 0, 0],
  [0.4, 2.4, -2, 1.2, 0, 0, 0, 0],
]

export function PairPlotSpecimen() {
  const state = useFigureState({
    which: choice(
      Object.entries(PAIR_DATA).map(([value, c]) => ({ value: value as keyof typeof PAIR_DATA, label: c.label })),
      'iris',
      { label: 'dataset' },
    ),
  })
  const which = state.which
  const data = useMemo(() => PAIR_DATA[which].make(), [which])
  return (
    <Figure
      id="pair-plot"
      title="Pair plot"
      purpose={PAIR_DATA[which].purpose}
      state={state}
      defaultSize="L"
      hoverReadout={false}
      caption="Per-class histograms (or KDE curves) on the diagonal, scatters below, correlations or scatters above. Drag a rectangle in any scatter to select the rows inside it; they keep their colour in every panel and the rest fade. A click without a drag clears it. Hover a point to mark the same row everywhere."
    >
      <PairPlotPanel key={which} data={data} features={which === 'wide' ? [0, 1, 2, 3, 8, 9] : undefined} />
    </Figure>
  )
}

const PARALLEL_DATA: Record<'iris' | 'wide', Choice> = {
  iris: {
    label: 'Iris (4 measurements)',
    make: () => IRIS,
    purpose:
      'Each Iris flower as one line across its four measurements: the setosa lines run low on the petal axes and high on sepal width, apart from the other two species.',
  },
  wide: PAIR_DATA.wide,
}

export function ParallelCoordinatesSpecimen() {
  const state = useFigureState({
    which: choice(
      Object.entries(PARALLEL_DATA).map(([value, c]) => ({
        value: value as keyof typeof PARALLEL_DATA,
        label: c.label,
      })),
      'wide',
      { label: 'dataset' },
    ),
  })
  const which = state.which
  const data = useMemo(() => PARALLEL_DATA[which].make(), [which])
  return (
    <>
      <Figure
        id="parallel-coordinates"
        title="Parallel coordinates"
        purpose={
          which === 'wide'
            ? 'Three classes that differ in x1–x4 only: the class colours pull apart on the first four axes and mix on the other twelve.'
            : PARALLEL_DATA[which].purpose
        }
        state={state}
        defaultSize="XL"
        hoverReadout={false}
        caption={`${data.x.shape[0]} rows in ${data.x.shape[1]} features, one line per row coloured by class, with the class medians bold. Drag along an axis to brush an interval and follow a class across the others (drag again to add one; click the axis outside it to clear). Drag a chip in the header to move its axis; its arrow flips the axis.`}
      >
        <ParallelCoordinatesPanel key={which} data={data} medians />
      </Figure>
      <Figure
        id="andrews-curves"
        title="Andrews curves"
        purpose="Each standardised row becomes the curve x₁/√2 + x₂ sin t + x₃ cos t + x₄ sin 2t; curves are as far apart (in L²) as the rows are, so the Iris species form separate bands, setosa most clearly."
        defaultSize="M"
        caption="Thin lines are rows; bold lines are class means. Hover reads the class means at t."
      >
        <AndrewsCurvesPanel data={IRIS} />
      </Figure>
    </>
  )
}
