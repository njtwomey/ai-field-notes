import React, { useState } from 'react'

type ModelFamily = 'linear' | 'distance' | 'trees' | 'neural'

interface FamilyDetail {
  id: ModelFamily
  name: string
  examples: string
  scaling: { status: 'Mandatory' | 'Critical' | 'Invariant' | 'Mandatory'; desc: string }
  imputation: { status: 'Mandatory' | 'Mandatory' | 'Native' | 'Mandatory'; desc: string }
  categoricals: {
    status: 'Mandatory (OHE/Target)' | 'Mandatory (OHE/Metric)' | 'Native / Ordinal' | 'Embeddings / OHE'
    desc: string
  }
  outliers: { status: 'High Sensitivity' | 'High Sensitivity' | 'Robust' | 'Moderate Sensitivity'; desc: string }
  interactions: {
    status: 'Manual (Poly/Spline)' | 'Implicit via Kernel' | 'Automatic (Depth >= 2)' | 'Automatic via Layers'
    desc: string
  }
}

export const PreprocessingMatrixExplorer: React.FC = () => {
  const [selectedFamily, setSelectedFamily] = useState<ModelFamily>('linear')
  const [testScenario, setTestScenario] = useState<'unscaled' | 'missing' | 'outliers' | 'nonlinear'>('unscaled')

  const families: Record<ModelFamily, FamilyDetail> = {
    linear: {
      id: 'linear',
      name: 'Linear Models',
      examples: 'Logistic Regression, Ridge, Lasso, ElasticNet, Linear SVM',
      scaling: {
        status: 'Mandatory',
        desc: 'Unscaled features distort L1/L2 penalties; larger scale features receive artificially lower regularization, and gradient descent oscillates severely.',
      },
      imputation: {
        status: 'Mandatory',
        desc: 'Linear dot products w^T x cannot evaluate NaN values. Requires median/mean imputer, iterative imputer, or missing indicator flags.',
      },
      categoricals: {
        status: 'Mandatory (OHE/Target)',
        desc: 'Must be converted to numerical vectors. One-Hot (with drop="first" if unregularized) or Target Encoding with empirical Bayes smoothing.',
      },
      outliers: {
        status: 'High Sensitivity',
        desc: 'Squared loss or hinge loss is pulled heavily by high-leverage points; requires Winsorization, RobustScaler, or Huber loss.',
      },
      interactions: {
        status: 'Manual (Poly/Spline)',
        desc: 'Model cannot learn non-linear boundaries or multiplicative interactions x1*x2 unless manually engineered into the feature matrix.',
      },
    },
    distance: {
      id: 'distance',
      name: 'Distance & Kernel Methods',
      examples: 'k-Nearest Neighbors, SVM (RBF/poly), k-Means, PCA',
      scaling: {
        status: 'Critical',
        desc: 'Euclidean distance ||u - v|| is mathematically dominated by the coordinate with the largest numerical spread. Unscaled features render small-variance features irrelevant.',
      },
      imputation: {
        status: 'Mandatory',
        desc: 'Metric distances fail if any coordinate is null. Requires k-NN imputer or median imputation.',
      },
      categoricals: {
        status: 'Mandatory (OHE/Metric)',
        desc: 'Arbitrary integer ordinal encodings create fake spatial proximity (e.g. Red=1, Green=2, Blue=3 assumes Green is midway between Red and Blue). Requires One-Hot or metric embeddings.',
      },
      outliers: {
        status: 'High Sensitivity',
        desc: 'Extreme outliers warp centroid computations in k-Means and isolate single outlier query points in k-NN.',
      },
      interactions: {
        status: 'Implicit via Kernel',
        desc: 'RBF/Polynomial kernels map inputs to infinite/higher-dimensional Hilbert spaces, capturing smooth interactions without explicit feature expansion.',
      },
    },
    trees: {
      id: 'trees',
      name: 'Tree Ensembles',
      examples: 'Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost',
      scaling: {
        status: 'Invariant',
        desc: 'Split criteria (Gini/Entropy/MSE) evaluate strictly monotonic order statistics (x_j <= theta). Any monotonic transform preserves split points and purity gains identically.',
      },
      imputation: {
        status: 'Native',
        desc: 'Modern boosting libraries (XGBoost, LightGBM) learn optimal default split directions per node for missing values during training.',
      },
      categoricals: {
        status: 'Native / Ordinal',
        desc: 'LightGBM sorts histogram bins by target ratio; CatBoost computes online target statistics on permutations. Simple integer encodings work without metric distortion.',
      },
      outliers: {
        status: 'Robust',
        desc: 'Outliers occupy the outermost rank in sorted candidate splits; an extreme value cannot distort split thresholds for the rest of the distribution.',
      },
      interactions: {
        status: 'Automatic (Depth >= 2)',
        desc: 'Hierarchical tree branching naturally models multiplicative feature interactions: a child node split condition depends implicitly on its parent split condition.',
      },
    },
    neural: {
      id: 'neural',
      name: 'Neural Networks & Deep Tabular',
      examples: 'MLP, TabNet, FT-Transformer, SAINT, ResNet-Tabular',
      scaling: {
        status: 'Mandatory',
        desc: 'Unscaled inputs destabilize gradient backpropagation, saturate activation functions (sigmoid/tanh), and cause ill-conditioned Hessian eigenvalues.',
      },
      imputation: {
        status: 'Mandatory',
        desc: 'Forward passes fail on NaN. Standard practice uses median/mean imputation combined with a binary missingness indicator column, or learned missing tokens.',
      },
      categoricals: {
        status: 'Embeddings / OHE',
        desc: 'Learned entity embeddings project categorical tokens into dense d-dimensional continuous vectors, capturing semantic clustering through backpropagation.',
      },
      outliers: {
        status: 'Moderate Sensitivity',
        desc: 'Extreme inputs can trigger exploding activations or dead ReLUs. Quantile transformations or log1p scaling provide stabilization.',
      },
      interactions: {
        status: 'Automatic via Layers',
        desc: 'Multi-layer non-linear compositions f(W2 * relu(W1 x + b1) + b2) approximate complex high-order feature interactions automatically.',
      },
    },
  }

  const current = families[selectedFamily]

  // Scenario simulations
  const scenarioImpacts = {
    unscaled: {
      title: 'Scenario: Unscaled Inputs (Income: $10k–$150k vs Age: 18–75)',
      impacts: {
        linear: {
          penalty: '-32% Test Accuracy',
          reason: 'L2 penalty suppresses Age weight to ~0; SGD oscillates uncontrollably along Income axis.',
        },
        distance: {
          penalty: '-45% Test Accuracy',
          reason: 'Distance ||x1 - x2|| is 99.9% dominated by Income difference; Age coordinate is ignored.',
        },
        trees: {
          penalty: '0% Impact (Exact Match)',
          reason: 'Threshold split at Income <= $64k is identical whether Income is unscaled or z-scored.',
        },
        neural: {
          penalty: '-28% Test Accuracy',
          reason: 'Early layers experience exploding gradients on Income and zero gradients on Age.',
        },
      },
    },
    missing: {
      title: 'Scenario: Missing Values (15% Missing at Random in 2 columns)',
      impacts: {
        linear: {
          penalty: 'Runtime Crash / Drop Rows',
          reason: 'Matrix multiplication X @ W cannot compute float operations containing NaN.',
        },
        distance: { penalty: 'Runtime Crash', reason: 'Euclidean distance fails on incomplete vectors.' },
        trees: {
          penalty: 'Native Routing (<1% diff)',
          reason: 'LightGBM sends NaNs to the child node that minimizes loss.',
        },
        neural: {
          penalty: 'Runtime Crash (without mask)',
          reason: 'NaN propagates through tensor dot products and turns all downstream activations into NaN.',
        },
      },
    },
    outliers: {
      title: 'Scenario: Extreme Outlier (Sensor Spike: x_i = 10,000 vs Normal: [0, 50])',
      impacts: {
        linear: {
          penalty: 'Severe Slope Distortion',
          reason: 'High leverage point rotates ordinary least squares regression line toward the spike.',
        },
        distance: {
          penalty: 'Cluster Distortion',
          reason: 'k-Means centroid gets pulled across space; k-NN creates isolated query islands.',
        },
        trees: {
          penalty: 'Isolated Single Leaf',
          reason: 'Tree isolates the spike in a single terminal split leaf with zero distortion on other nodes.',
        },
        neural: {
          penalty: 'Activation Saturation',
          reason: 'Produces huge intermediate activation magnitude, causing dead neurons or gradient spikes.',
        },
      },
    },
    nonlinear: {
      title: 'Scenario: Concentric Circular Non-Linear Boundary (x1^2 + x2^2 = r^2)',
      impacts: {
        linear: {
          penalty: 'Fails (~50% Accuracy)',
          reason: 'Cannot separate non-linear circles without explicit polynomial or polar transformation.',
        },
        distance: {
          penalty: 'Solves with RBF Kernel',
          reason: 'RBF distance metric effortlessly segments concentric spheres.',
        },
        trees: {
          penalty: 'Stepwise Staircase Approx',
          reason: 'Approximates circle via orthogonal axis-aligned bounding boxes (requires depth >= 6).',
        },
        neural: {
          penalty: 'Smooth Non-Linear Fit',
          reason: 'Hidden layer with 8-16 ReLU/GELU units easily maps circular boundary.',
        },
      },
    },
  }

  const activeScenario = scenarioImpacts[testScenario]

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">Preprocessing Requirements by Model Family</h3>
        <p className="text-xs text-muted-foreground">
          Compare the mathematical prerequisites of Linear Models, Distance/Kernel Methods, Tree Ensembles, and Neural
          Networks across scaling, imputation, encoding, and outliers.
        </p>
      </div>

      {/* Model Family Tabs */}
      <div className="mb-5 grid grid-cols-2 gap-2 sm:grid-cols-4">
        {(Object.keys(families) as ModelFamily[]).map((fam) => {
          const item = families[fam]
          const isSelected = selectedFamily === fam
          return (
            <button
              key={fam}
              onClick={() => setSelectedFamily(fam)}
              className={`rounded-lg border p-3 text-left transition-all ${
                isSelected
                  ? 'border-primary bg-primary font-semibold text-primary-foreground shadow-xs'
                  : 'border-border bg-muted/40 text-muted-foreground hover:bg-muted'
              }`}
            >
              <div className="text-xs font-bold">{item.name}</div>
              <div
                className={`mt-0.5 truncate text-[10px] ${isSelected ? 'text-primary-foreground/80' : 'text-muted-foreground'}`}
              >
                {fam === 'linear' && 'Ridge / Lasso / LR'}
                {fam === 'distance' && 'k-NN / SVM / k-Means'}
                {fam === 'trees' && 'XGB / LGBM / CatBoost'}
                {fam === 'neural' && 'MLP / TabNet / ResNet'}
              </div>
            </button>
          )
        })}
      </div>

      {/* Selected Family Feature Grid */}
      <div className="mb-5 rounded-lg border border-border bg-card/60 p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-border pb-3">
          <div>
            <span className="text-sm font-bold text-foreground">{current.name}</span>
            <span className="mt-0.5 block font-mono text-xs text-muted-foreground">{current.examples}</span>
          </div>
          <span className="rounded-full bg-primary/10 px-2.5 py-1 text-[10px] font-bold tracking-wider text-primary uppercase">
            Family Specification
          </span>
        </div>

        <div className="grid grid-cols-1 gap-3 text-xs md:grid-cols-2">
          {/* Scaling */}
          <div className="rounded-lg border border-border/80 bg-muted/20 p-3">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="font-semibold text-foreground">1. Feature Scaling</span>
              <span
                className={`rounded px-2 py-0.5 text-[10px] font-bold ${
                  current.scaling.status === 'Invariant'
                    ? 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400'
                    : 'bg-rose-500/20 text-rose-600 dark:text-rose-400'
                }`}
              >
                {current.scaling.status}
              </span>
            </div>
            <p className="text-[11px] leading-relaxed text-muted-foreground">{current.scaling.desc}</p>
          </div>

          {/* Imputation */}
          <div className="rounded-lg border border-border/80 bg-muted/20 p-3">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="font-semibold text-foreground">2. Missing Value Imputation</span>
              <span
                className={`rounded px-2 py-0.5 text-[10px] font-bold ${
                  current.imputation.status === 'Native'
                    ? 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400'
                    : 'bg-rose-500/20 text-rose-600 dark:text-rose-400'
                }`}
              >
                {current.imputation.status}
              </span>
            </div>
            <p className="text-[11px] leading-relaxed text-muted-foreground">{current.imputation.desc}</p>
          </div>

          {/* Categoricals */}
          <div className="rounded-lg border border-border/80 bg-muted/20 p-3">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="font-semibold text-foreground">3. Categorical Encoding</span>
              <span className="rounded bg-amber-500/20 px-2 py-0.5 text-[10px] font-bold text-amber-600 dark:text-amber-400">
                {current.categoricals.status}
              </span>
            </div>
            <p className="text-[11px] leading-relaxed text-muted-foreground">{current.categoricals.desc}</p>
          </div>

          {/* Outliers */}
          <div className="rounded-lg border border-border/80 bg-muted/20 p-3">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="font-semibold text-foreground">4. Outlier Sensitivity</span>
              <span
                className={`rounded px-2 py-0.5 text-[10px] font-bold ${
                  current.outliers.status === 'Robust'
                    ? 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400'
                    : 'bg-rose-500/20 text-rose-600 dark:text-rose-400'
                }`}
              >
                {current.outliers.status}
              </span>
            </div>
            <p className="text-[11px] leading-relaxed text-muted-foreground">{current.outliers.desc}</p>
          </div>
        </div>
      </div>

      {/* Interactive Scenario Stress-Tester */}
      <div className="rounded-lg border border-border bg-card/60 p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <span className="text-xs font-semibold tracking-wider text-foreground uppercase">
            Stress Test: What Happens If You Skip Preprocessing?
          </span>
          <div className="flex items-center gap-1.5">
            {[
              { id: 'unscaled', label: 'Unscaled Features' },
              { id: 'missing', label: 'Missing Values' },
              { id: 'outliers', label: 'Sensor Outlier' },
              { id: 'nonlinear', label: 'Circular Boundary' },
            ].map((sc) => (
              <button
                key={sc.id}
                onClick={() => setTestScenario(sc.id as any)}
                className={`rounded px-2.5 py-1 text-[11px] font-medium transition-colors ${
                  testScenario === sc.id
                    ? 'bg-foreground font-bold text-background'
                    : 'bg-muted/40 text-muted-foreground hover:bg-muted'
                }`}
              >
                {sc.label}
              </button>
            ))}
          </div>
        </div>

        <div className="mb-3 rounded border border-border/70 bg-muted/30 p-2.5 text-xs font-medium text-foreground">
          {activeScenario.title}
        </div>

        <div className="grid grid-cols-1 gap-2.5 text-xs sm:grid-cols-4">
          {(['linear', 'distance', 'trees', 'neural'] as ModelFamily[]).map((fam) => {
            const sc = activeScenario.impacts[fam]
            const isHighlight = selectedFamily === fam
            return (
              <div
                key={fam}
                className={`rounded-lg border p-3 transition-all ${
                  isHighlight ? 'border-primary bg-primary/5 shadow-xs ring-1 ring-primary/30' : 'border-border bg-card'
                }`}
              >
                <div className="mb-1 flex items-center justify-between">
                  <span className="font-bold text-foreground capitalize">{fam}</span>
                  {isHighlight && (
                    <span className="py-0.2 rounded bg-primary px-1.5 text-[9px] font-semibold text-primary-foreground">
                      Active
                    </span>
                  )}
                </div>
                <div className="mb-1 font-mono text-xs font-bold text-rose-500">{sc.penalty}</div>
                <p className="text-[10px] leading-normal text-muted-foreground">{sc.reason}</p>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
