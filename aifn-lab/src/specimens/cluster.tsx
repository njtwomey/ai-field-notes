import type { Specimen } from '../specimen'
import {
  DbscanSpecimen,
  DendrogramSpecimen,
  KMeansSpecimen,
  MixtureSpecimen,
  SpectralSpecimen,
} from './_cluster/figures'

export const specimens: Specimen[] = [
  {
    module: 'cluster',
    title: 'k-means steps from draggable centroids',
    description:
      "kmeansSteps (Lloyd's algorithm) from starting centroids placed by hand, against the best of k-means++ restarts.",
    tags: ['kmeans', 'kmeansSteps', 'Lloyd', 'k-means++', 'trace'],
    render: () => <KMeansSpecimen />,
  },
  {
    module: 'cluster',
    title: 'Gaussian mixture EM',
    description:
      'gaussianMixtureSteps with full, diagonal or spherical covariances: ellipses, responsibilities and the log-likelihood per step.',
    tags: ['gaussianMixture', 'EM', 'responsibilities', 'covarianceEllipse'],
    render: () => <MixtureSpecimen />,
  },
  {
    module: 'cluster',
    title: 'Dendrograms and cuts',
    description:
      'linkage (single, complete, average, Ward) as a merge tree in DendrogramView, cut at a height with cutTree.',
    tags: ['agglomerative', 'linkage', 'DendrogramView', 'cutTree', 'TreeView'],
    render: () => <DendrogramSpecimen />,
  },
  {
    module: 'cluster',
    title: 'DBSCAN and OPTICS',
    description: "dbscan's core, border and noise points at a draggable ε on optics' reachability plot.",
    tags: ['dbscan', 'optics', 'density'],
    render: () => <DbscanSpecimen />,
  },
  {
    module: 'cluster',
    title: 'Spectral clustering',
    description: 'spectralClustering on nested circles, against k-means, with the affinity lengthscale as the control.',
    tags: ['spectralClustering', 'affinityMatrix', 'kmeans'],
    render: () => <SpectralSpecimen />,
  },
]
