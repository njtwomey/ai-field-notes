import type { Specimen } from '../../specimen'
import { CocktailPartySpecimen, FactorAnalysisSpecimen, NmfGlyphsSpecimen } from './_factorisation/figures'
import { JohnsonLindenstraussSpecimen } from './_factorisation/projections'

export const specimens: Specimen[] = [
  {
    module: 'numerics/factorisation',
    title: 'Matrix factorisations: NMF, ICA, factor analysis',
    description:
      'FastICA unmixing a cocktail party of three sources where PCA leaves mixtures; NMF recovering the strokes of synthetic glyphs as parts, against PCA’s signed components, by multiplicative updates or HALS; factor analysis against probabilistic PCA when noise differs by feature, with EM played step by step.',
    tags: [
      'NMF',
      'non-negative matrix factorisation',
      'HALS',
      'ICA',
      'FastICA',
      'cocktail party',
      'factor analysis',
      'probabilistic PCA',
      'EM',
    ],
    render: () => (
      <>
        <CocktailPartySpecimen />
        <NmfGlyphsSpecimen />
        <FactorAnalysisSpecimen />
      </>
    ),
  },
  {
    module: 'numerics/factorisation',
    title: 'Random projections and Johnson–Lindenstrauss',
    description:
      'Points projected by Gaussian, sparse and Achlioptas random matrices: the spread of pairwise distance ratios against the target dimension, against the distortion the Johnson–Lindenstrauss lemma guarantees, and the lemma’s dimension against the number of points.',
    tags: ['random projection', 'Johnson–Lindenstrauss', 'sparse projection', 'Achlioptas', 'distortion'],
    render: () => <JohnsonLindenstraussSpecimen />,
  },
]
