/** Small real datasets, embedded: Iris, Old Faithful and Anscombe's quartet. */

import { ANSCOMBE_DATA, FAITHFUL_DATA, IRIS_DATA } from './embedded'
import { labels, matrix, vector, type Dataset } from '../types'

/**
 * Fisher's Iris data: 150 flowers (50 each of setosa, versicolor and virginica) with four measurements in cm. x is
 * 150 × 4, y the species (0, 1, 2).
 */
export function iris(): Dataset {
  return {
    kind: 'dataset',
    x: matrix(Float64Array.from(IRIS_DATA), 150, 4),
    y: labels(Array.from({ length: 150 }, (_, i) => Math.floor(i / 50))),
    meta: {
      name: 'Iris',
      description: 'Fisher’s Iris data: sepal and petal length and width (cm) of 150 flowers of three species.',
      task: 'classification',
      featureNames: ['sepal length', 'sepal width', 'petal length', 'petal width'],
      labelNames: ['setosa', 'versicolor', 'virginica'],
      source:
        'Fisher (1936), "The use of multiple measurements in taxonomic problems", Annals of Eugenics 7(2); UCI copy',
      url: 'https://archive.ics.uci.edu/dataset/53/iris',
    },
  }
}

/**
 * Old Faithful geyser data: 272 eruptions with duration and waiting time to the next eruption (minutes). x is 272 × 2
 * (eruption, waiting); there are no labels, but the data form two clear clusters.
 */
export function oldFaithful(): Dataset {
  return {
    kind: 'dataset',
    x: matrix(Float64Array.from(FAITHFUL_DATA), 272, 2),
    meta: {
      name: 'Old Faithful',
      description:
        'Eruption duration and waiting time to the next eruption (minutes) for 272 eruptions of Old Faithful.',
      task: 'clustering',
      featureNames: ['eruption duration', 'waiting time'],
      source: 'Härdle (1991), Smoothing Techniques with Implementation in S; R datasets::faithful',
      url: 'https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/faithful.html',
    },
  }
}

/**
 * Anscombe's quartet: four sets of eleven (x, y) points with the same means, variances, correlation (0.816) and
 * least-squares line (y = 3 + 0.5x), yet very different shapes. Returns the four sets, each with x 11 × 1 and y.
 */
export function anscombe(): Dataset[] {
  const numerals = ['I', 'II', 'III', 'IV']
  return numerals.map((numeral, k) => ({
    kind: 'dataset',
    x: matrix(Float64Array.from(ANSCOMBE_DATA.x[k]), 11, 1),
    y: vector(ANSCOMBE_DATA.y[k]),
    meta: {
      name: `Anscombe ${numeral}`,
      description: `Set ${numeral} of Anscombe's quartet: eleven points sharing the summary statistics of the other three sets.`,
      task: 'regression',
      featureNames: ['x'],
      targetName: 'y',
      source: 'Anscombe (1973), "Graphs in statistical analysis", The American Statistician 27(1)',
    },
  }))
}
