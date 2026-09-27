import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamChoice, ParamSlider, Readout, formatNumber, useParam } from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { countLeaves, fitTree, predictTree, type Criterion } from '../_shared/trees'

const N_TRAIN = 300
const N_TEST = 2000
const BOX = 3
const GRID = linspace(-BOX, BOX, 61)
const RANGE: [number, number] = [0, 1]
const FLIP = 0.1

/** Class 1 inside a disc of radius 1.8, class 0 outside, with 10% of labels flipped. */
function sample(n: number, seed: number) {
  const g = rng(seed)
  const X: number[][] = []
  const y: number[] = []
  for (let i = 0; i < n; i++) {
    const p = [BOX * (2 * g.uniform() - 1), BOX * (2 * g.uniform() - 1)]
    const inside = p[0] * p[0] + p[1] * p[1] < 1.8 * 1.8 ? 1 : 0
    X.push(p)
    y.push(g.uniform() < FLIP ? 1 - inside : inside)
  }
  return { X, y }
}

const CRITERIA = [
  { value: 'gini', label: 'Gini' },
  { value: 'entropy', label: 'entropy' },
] as const

/** A classification tree on 2D data: each leaf is an axis-aligned rectangle shaded by its fraction of class 1. */
export function TreePartition() {
  const depth = useParam(3, { min: 0, max: 12, step: 1 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const [criterion, setCriterion] = useState<Exclude<Criterion, 'squared'>>('gini')

  const train = useMemo(() => sample(N_TRAIN, seed.value), [seed.value])
  const test = useMemo(() => sample(N_TEST, 1000 + seed.value), [seed.value])

  const r = useMemo(() => {
    const tree = fitTree(train.X, train.y, { maxDepth: depth.value, criterion })
    const accuracy = (d: { X: number[][]; y: number[] }) =>
      d.X.reduce((acc, x, i) => acc + ((predictTree(tree, x) > 0.5 ? 1 : 0) === d.y[i] ? 1 : 0), 0) / d.y.length
    const z = GRID.map((b) => GRID.map((a) => predictTree(tree, [a, b])))
    return { z, leaves: countLeaves(tree), trainAcc: accuracy(train), testAcc: accuracy(test) }
  }, [train, test, depth.value, criterion])

  const overlay = useMemo(
    () => [
      {
        name: 'points',
        type: 'scatter' as const,
        x: train.X.map((p) => p[0]),
        y: train.X.map((p) => p[1]),
        group: train.y,
        groupNames: ['class 0', 'class 1'],
      },
    ],
    [train],
  )

  return (
    <Interactive
      title="Recursive partitioning of the plane"
      caption="Class 1 lies inside a circle of radius 1.8 and class 0 outside it, with 10% of labels flipped at random. Each split cuts one rectangle in two along one axis, so every leaf is a rectangle, shaded by the fraction of its training points in class 1. Depth 0 is a single leaf. Shallow trees approximate the circle with a coarse staircase; deep trees carve out single noisy points, and training accuracy climbs towards 1 while test accuracy, measured on 2,000 fresh points, falls. Test accuracy usually peaks between depths 4 and 6, at about 0.83. No classifier can exceed 0.9 on average, the fraction of labels left unflipped."
      controls={
        <>
          <ParamSlider label="maximum depth" param={depth} format={(v) => String(v)} withArrows />
          <ParamChoice label="impurity" value={criterion} onChange={setCriterion} options={CRITERIA} />
          <ParamSlider label="data seed" param={seed} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="leaves" value={String(r.leaves)} />
          <Readout label="training accuracy" value={formatNumber(r.trainAcc)} />
          <Readout label="test accuracy" value={formatNumber(r.testAcc)} />
        </>
      }
    >
      <Heatmap
        x={GRID}
        y={GRID}
        z={r.z}
        scale="diverging"
        range={RANGE}
        xLabel="x₁"
        yLabel="x₂"
        valueLabel="leaf fraction of class 1"
        overlay={overlay}
        height={420}
      />
    </Interactive>
  )
}
