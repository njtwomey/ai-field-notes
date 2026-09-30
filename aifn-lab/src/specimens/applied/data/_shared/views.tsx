import { useMemo, useState } from 'react'
import { iris } from 'aifn-applied/data/real'
import { recipe } from 'aifn-applied/data/synthetic'
import { type Dataset } from 'aifn-applied/data'
import { Select } from '@lab/controls'
import { AndrewsCurvesView, PairPlotView, ParallelCoordinatesView } from '@lab/views'

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
    make: () => recipe({ base: 'blobs', seed: 3, n: 450, options: { centers: 3 }, nuisance: 3, shuffle: true }),
    purpose:
      'Three blobs in x1 and x2 plus three pure-noise features: only the x1–x2 panels separate the classes, and every panel with a noise feature shows the classes on top of each other.',
  },
  wide: {
    label: '8-D Gaussians + 8 nuisance (16 features)',
    make: () =>
      recipe({ base: 'gaussians', seed: 5, n: 1500, options: { means: MEANS_8D }, nuisance: 8, shuffle: true }),
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
  const [which, setWhich] = useState<keyof typeof PAIR_DATA>('iris')
  const choice = PAIR_DATA[which]
  const data = useMemo(() => choice.make(), [choice])
  return (
    <PairPlotView
      key={which}
      id="pair-plot"
      data={data}
      title="Pair plot"
      description={choice.purpose}
      defaultSize="L"
      features={which === 'wide' ? [0, 1, 2, 3, 8, 9] : undefined}
      controls={
        <Select
          label="dataset"
          value={which}
          onChange={setWhich}
          options={Object.entries(PAIR_DATA).map(([value, c]) => ({
            value: value as keyof typeof PAIR_DATA,
            label: c.label,
          }))}
        />
      }
    />
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
  const [which, setWhich] = useState<keyof typeof PARALLEL_DATA>('wide')
  const choice = PARALLEL_DATA[which]
  const data = useMemo(() => choice.make(), [choice])
  return (
    <>
      <ParallelCoordinatesView
        key={which}
        id="parallel-coordinates"
        data={data}
        title="Parallel coordinates"
        description={
          which === 'wide'
            ? '1500 rows in 16 features, three classes that differ in x1–x4 only: the class colours pull apart on the first four axes and mix on the rest. Brush an axis to follow a class across the others.'
            : choice.purpose
        }
        medians
        defaultSize="XL"
        controls={
          <Select
            label="dataset"
            value={which}
            onChange={setWhich}
            options={Object.entries(PARALLEL_DATA).map(([value, c]) => ({
              value: value as keyof typeof PARALLEL_DATA,
              label: c.label,
            }))}
          />
        }
      />
      <AndrewsCurvesView
        id="andrews-curves"
        data={IRIS}
        title="Andrews curves"
        description="Each standardised Iris flower as the curve x₁/√2 + x₂ sin t + x₃ cos t + x₄ sin 2t: curves are as far apart (in L²) as the rows are, so the species form separate bands, setosa most clearly."
        defaultSize="M"
      />
    </>
  )
}
