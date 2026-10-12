import React, { useState } from 'react'

interface StageState {
  id: string
  name: string
  type: 'imputer' | 'scaler' | 'encoder' | 'estimator'
  columns: string[]
  fittedParams: Record<string, string>
  inputShape: string
  outputShape: string
  leakageRisk: string
}

export const PipelineDAGVisualizer: React.FC = () => {
  const [activeStage, setActiveStage] = useState<string>('num_scaler')
  const [mode, setMode] = useState<'train' | 'inference' | 'leakage'>('train')
  const [selectedColumn, setSelectedColumn] = useState<'age' | 'income' | 'device' | 'state'>('income')

  const stages: Record<string, StageState> = {
    num_imputer: {
      id: 'num_imputer',
      name: 'SimpleImputer(strategy="median")',
      type: 'imputer',
      columns: ['age', 'income', 'credit_score'],
      fittedParams: {
        'age.median_': '38.5 years',
        'income.median_': '$64,200',
        'credit_score.median_': '710 pts',
        n_features_in_: '3',
      },
      inputShape: '(N, 3) with ~4% NaNs',
      outputShape: '(N, 3) dense, 0 NaNs',
      leakageRisk: 'If computed over test set, test medians pull training inputs towards future distribution.',
    },
    num_scaler: {
      id: 'num_scaler',
      name: 'StandardScaler()',
      type: 'scaler',
      columns: ['age', 'income', 'credit_score'],
      fittedParams: {
        mean_: '[39.2, 65800, 708.4]',
        'scale_ (std)': '[12.4, 28400, 68.2]',
        var_: '[153.7, 8.06e8, 4651.2]',
      },
      inputShape: '(N, 3)',
      outputShape: '(N, 3) z-scores: N(0, 1)',
      leakageRisk: 'Fitting mean/std globally normalizes away test distribution shift, creating false optimism.',
    },
    cat_imputer: {
      id: 'cat_imputer',
      name: 'SimpleImputer(fill_value="missing")',
      type: 'imputer',
      columns: ['device', 'state'],
      fittedParams: {
        fill_value: '"__missing__"',
        missing_values: 'np.nan',
      },
      inputShape: '(N, 2) strings with nulls',
      outputShape: '(N, 2) strings, all nulls mapped',
      leakageRisk: 'Low leakage risk, but must ensure __missing__ is indexed as a known level.',
    },
    cat_encoder: {
      id: 'cat_encoder',
      name: 'OneHotEncoder(handle_unknown="ignore")',
      type: 'encoder',
      columns: ['device', 'state'],
      fittedParams: {
        'device.categories_': "['desktop', 'mobile', 'tablet'] (3 cols)",
        'state.categories_': "['CA', 'NY', 'TX', 'FL', '__missing__'] (5 cols)",
        drop: 'None',
        total_ohe_cols: '8 columns',
      },
      inputShape: '(N, 2)',
      outputShape: '(N, 8) sparse/dense binary',
      leakageRisk:
        'Fitting on test data includes test-only rare categories, changing column count and breaking deployed models.',
    },
    estimator: {
      id: 'estimator',
      name: 'LogisticRegression(C=1.0)',
      type: 'estimator',
      columns: ['All Transformed Features (11 total)'],
      fittedParams: {
        coef_: 'shape (1, 11) weights',
        intercept_: '-0.428',
        classes_: '[0, 1]',
        n_iter_: '24',
      },
      inputShape: '(N, 11)',
      outputShape: '(N,) probability vector p(y=1|x)',
      leakageRisk: 'Training estimator on leaked representations yields overfitted coefficients.',
    },
  }

  const current = stages[activeStage] || stages.num_scaler

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">
          Encapsulated Preprocessing Pipeline &amp; ColumnTransformer Architecture
        </h3>
        <p className="text-xs text-muted-foreground">
          Transforms are stateful estimators that learn statistics exclusively from training folds. Inspect per-column
          routing, internal fitted states, and leakage safeguards.
        </p>
      </div>

      {/* Mode Controls */}
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-1.5 rounded-lg border border-border bg-muted/60 p-1">
          <button
            onClick={() => setMode('train')}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              mode === 'train'
                ? 'bg-primary text-primary-foreground shadow-xs'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            1. Train Phase (Fit &amp; Transform)
          </button>
          <button
            onClick={() => setMode('inference')}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              mode === 'inference'
                ? 'bg-primary text-primary-foreground shadow-xs'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            2. Inference Phase (Transform Only)
          </button>
          <button
            onClick={() => setMode('leakage')}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              mode === 'leakage' ? 'bg-rose-600 text-white shadow-xs' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            3. Leakage Risk Visualizer
          </button>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span className="text-muted-foreground">Trace Column:</span>
          {(['income', 'age', 'device', 'state'] as const).map((col) => (
            <button
              key={col}
              onClick={() => setSelectedColumn(col)}
              className={`rounded border px-2 py-1 font-mono text-[11px] transition-colors ${
                selectedColumn === col
                  ? 'border-foreground/30 bg-muted font-semibold text-foreground'
                  : 'border-border text-muted-foreground hover:text-foreground'
              }`}
            >
              {col}
            </button>
          ))}
        </div>
      </div>

      {/* DAG Flow Chart */}
      <div className="mb-5 overflow-x-auto rounded-lg border border-border bg-card/60 p-4">
        <div className="min-w-[620px]">
          {/* Input schema banner */}
          <div className="mb-4 flex items-center justify-between rounded border border-border/80 bg-muted/30 p-2.5 font-mono text-xs">
            <span className="font-sans text-[11px] font-medium tracking-wider text-muted-foreground uppercase">
              Raw Input Batch
            </span>
            <div className="flex items-center gap-3">
              <span
                className={`rounded px-2 py-0.5 text-[11px] ${['income', 'age'].includes(selectedColumn) ? 'bg-blue-500/20 font-bold text-blue-500' : 'text-muted-foreground'}`}
              >
                Numeric: [age, income, credit_score]
              </span>
              <span
                className={`rounded px-2 py-0.5 text-[11px] ${['device', 'state'].includes(selectedColumn) ? 'bg-amber-500/20 font-bold text-amber-500' : 'text-muted-foreground'}`}
              >
                Categorical: [device, state]
              </span>
            </div>
          </div>

          {/* DAG Pipeline Graph */}
          <div className="relative grid grid-cols-4 items-center gap-3">
            {/* Numeric Branch Column */}
            <div className="space-y-3">
              <div className="px-1 text-[10px] font-bold tracking-wider text-blue-500 uppercase">Numeric Pipeline</div>
              <button
                onClick={() => setActiveStage('num_imputer')}
                className={`w-full rounded-lg border p-3 text-left transition-all ${
                  activeStage === 'num_imputer'
                    ? 'border-blue-500 bg-blue-500/10 shadow-sm'
                    : 'border-border bg-card hover:bg-muted/30'
                }`}
              >
                <div className="text-xs font-semibold text-foreground">MedianImputer</div>
                <div className="mt-1 text-[10px] text-muted-foreground">Imputes NaN &rarr; median_</div>
                <span className="mt-2 inline-block rounded bg-muted px-1.5 py-0.5 font-mono text-[9px]">
                  State: 3 medians
                </span>
              </button>

              <button
                onClick={() => setActiveStage('num_scaler')}
                className={`w-full rounded-lg border p-3 text-left transition-all ${
                  activeStage === 'num_scaler'
                    ? 'border-blue-500 bg-blue-500/10 shadow-sm'
                    : 'border-border bg-card hover:bg-muted/30'
                }`}
              >
                <div className="text-xs font-semibold text-foreground">StandardScaler</div>
                <div className="mt-1 text-[10px] text-muted-foreground">Standardizes to (x - &mu;) / &sigma;</div>
                <span className="mt-2 inline-block rounded bg-muted px-1.5 py-0.5 font-mono text-[9px]">
                  State: &mu;[3], &sigma;[3]
                </span>
              </button>
            </div>

            {/* Categorical Branch Column */}
            <div className="space-y-3">
              <div className="px-1 text-[10px] font-bold tracking-wider text-amber-500 uppercase">
                Categorical Pipeline
              </div>
              <button
                onClick={() => setActiveStage('cat_imputer')}
                className={`w-full rounded-lg border p-3 text-left transition-all ${
                  activeStage === 'cat_imputer'
                    ? 'border-amber-500 bg-amber-500/10 shadow-sm'
                    : 'border-border bg-card hover:bg-muted/30'
                }`}
              >
                <div className="text-xs font-semibold text-foreground">ConstantImputer</div>
                <div className="mt-1 text-[10px] text-muted-foreground">Replaces null &rarr; "__missing__"</div>
                <span className="mt-2 inline-block rounded bg-muted px-1.5 py-0.5 font-mono text-[9px]">
                  State: fill_value
                </span>
              </button>

              <button
                onClick={() => setActiveStage('cat_encoder')}
                className={`w-full rounded-lg border p-3 text-left transition-all ${
                  activeStage === 'cat_encoder'
                    ? 'border-amber-500 bg-amber-500/10 shadow-sm'
                    : 'border-border bg-card hover:bg-muted/30'
                }`}
              >
                <div className="text-xs font-semibold text-foreground">OneHotEncoder</div>
                <div className="mt-1 text-[10px] text-muted-foreground">Fixed vocabulary per column</div>
                <span className="mt-2 inline-block rounded bg-muted px-1.5 py-0.5 font-mono text-[9px]">
                  State: 8 categories
                </span>
              </button>
            </div>

            {/* Feature Union Concatenation */}
            <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border bg-muted/20 p-3 text-center">
              <div className="text-xs font-semibold text-foreground">Feature Union</div>
              <p className="mt-1 text-[10px] text-muted-foreground">np.hstack([3 num, 8 cat])</p>
              <div className="mt-2 font-mono text-xs font-bold text-primary">11 Dimensions</div>
              <span className="mt-1 text-[9px] text-muted-foreground">
                Preserves column order identically at inference
              </span>
            </div>

            {/* Final Estimator */}
            <div>
              <div className="mb-3 px-1 text-[10px] font-bold tracking-wider text-emerald-500 uppercase">
                Supervised Model
              </div>
              <button
                onClick={() => setActiveStage('estimator')}
                className={`w-full rounded-lg border p-3 text-left transition-all ${
                  activeStage === 'estimator'
                    ? 'border-emerald-500 bg-emerald-500/10 shadow-sm'
                    : 'border-border bg-card hover:bg-muted/30'
                }`}
              >
                <div className="text-xs font-semibold text-foreground">LogisticRegression</div>
                <div className="mt-1 text-[10px] text-muted-foreground">Estimator fit on final matrix</div>
                <span className="mt-2 inline-block rounded bg-muted px-1.5 py-0.5 font-mono text-[9px]">
                  State: weights[11], bias
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Stage Detail Inspector */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* Left: Fitted State Parameters */}
        <div className="rounded-lg border border-border bg-card/40 p-4">
          <div className="mb-3 flex items-center justify-between">
            <span className="font-mono text-xs font-bold text-foreground">{current.name}</span>
            <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-medium text-muted-foreground uppercase">
              {current.type}
            </span>
          </div>

          <div className="mb-3 space-y-2 text-xs">
            <div className="flex justify-between border-b border-border/50 py-1">
              <span className="text-muted-foreground">Input Tensor:</span>
              <span className="font-mono font-medium text-foreground">{current.inputShape}</span>
            </div>
            <div className="flex justify-between border-b border-border/50 py-1">
              <span className="text-muted-foreground">Output Tensor:</span>
              <span className="font-mono font-medium text-foreground">{current.outputShape}</span>
            </div>
          </div>

          <div className="mt-3">
            <span className="mb-2 block text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
              Learned State Dictionary (Persisted on Disk)
            </span>
            <div className="space-y-1 rounded bg-muted/40 p-2.5 font-mono text-xs">
              {Object.entries(current.fittedParams).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-primary">{k}:</span>
                  <span className="text-foreground">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Operational Behavior under Active Mode */}
        <div className="flex flex-col justify-between rounded-lg border border-border bg-card/40 p-4">
          <div>
            <div className="mb-2 flex items-center gap-2 text-xs font-bold text-foreground">
              <span>Lifecycle Behavior:</span>
              <span className="rounded bg-primary/20 px-2 py-0.5 text-[10px] font-semibold text-primary">
                {mode === 'train'
                  ? 'Training (fit_transform)'
                  : mode === 'inference'
                    ? 'Serving (transform)'
                    : 'Data Leakage Threat'}
              </span>
            </div>

            {mode === 'train' && (
              <div className="space-y-2 text-xs leading-relaxed text-muted-foreground">
                <p>
                  1. The pipeline executes <code className="font-mono text-foreground">fit()</code> strictly on the
                  training partition <em>D</em>
                  <sub>train</sub>.
                </p>
                <p>
                  2. All empirical sample statistics (medians, means, variances, vocabulary indices) are computed from
                  training rows only.
                </p>
                <p>
                  3. The internal parameters are frozen into estimator attributes (suffixed with a trailing underscore
                  in scikit-learn convention, e.g. <code className="font-mono text-foreground">mean_</code>).
                </p>
              </div>
            )}

            {mode === 'inference' && (
              <div className="space-y-2 text-xs leading-relaxed text-muted-foreground">
                <p>
                  1. During deployment or test validation, <strong className="text-foreground">never call fit()</strong>
                  . The pipeline invokes only <code className="font-mono text-foreground">transform()</code>.
                </p>
                <p>
                  2. Single incoming JSON payloads (e.g.,{' '}
                  <code className="font-mono text-foreground">&#123;income: 82000, state: "WY"&#125;</code>) are mapped
                  using the frozen training statistics.
                </p>
                <p>
                  3. If a novel category appears ("WY"),{' '}
                  <code className="font-mono text-foreground">handle_unknown="ignore"</code> encodes it as an all-zero
                  vector, preserving exact input dimensionality without throwing runtime exceptions.
                </p>
              </div>
            )}

            {mode === 'leakage' && (
              <div className="space-y-2 rounded border border-rose-500/20 bg-rose-500/10 p-3 text-xs text-card-foreground">
                <span className="block font-semibold text-rose-600 dark:text-rose-400">
                  Critical Trap: Preprocessing Before Cross-Validation
                </span>
                <p className="text-[11px] leading-relaxed text-muted-foreground">{current.leakageRisk}</p>
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  <strong>The Rule:</strong> If preprocessing is performed on the whole dataset before{' '}
                  <code className="font-mono text-foreground">cross_val_score</code> or train/test splitting, test rows
                  bleed into the scaler/imputer, corrupting performance estimates.
                </p>
              </div>
            )}
          </div>

          <div className="mt-4 flex items-center justify-between border-t border-border pt-3 text-[11px] text-muted-foreground">
            <span>Serialization Format: Joblib / ONNX / Skops</span>
            <span className="font-mono text-primary">pickle.dump(pipeline)</span>
          </div>
        </div>
      </div>
    </div>
  )
}
