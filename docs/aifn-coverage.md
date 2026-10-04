# aifn coverage of the field notes

Survey of 2026-10-01. Read-only: no code changed. It asks what aifn (core `aifn`, applications `aifn-methods`) and the
lab still lack in order to give every note an interactive exposition. Sources: the 1,223 `content/notes/**/index.mdx`
files, `aifn-js/generated/catalog.json` (721 entries), `aifn-js/modules.json`, the aifn source tree, the lab
(`aifn-lab/src`) and `.scratch/aifn/refinement.md`. Working log: `.scratch/aifn/progress/coverage-survey.md`.

**Coverage** counts a note only when a registered catalog entry lists it in `notes`. Widgets beside notes are given as
context only. They show which topics call for a figure. They are not a model for aifn's design.

## Headline

- **261 of 1,223 notes (21%) are linked by a catalog entry.** Another 14 slugs are mentioned in aifn source without a
  registry link, so 275 (22%) is the loose upper bound.
- The links are lopsided. Metrics (83%) and probability distributions (95%) account for 75 of the 261. Nine top-level
  topics, with 156 notes between them, have **no linked note at all**.
- **Much of the gap is registration debt, not missing code.** About 260 unlinked notes sit in branches where aifn
  already has working code with no registry entries or no `notes` links. Examples: `timeseries` (43 exports, 0
  entries), `probability/stats` (100, 0), `nn/layers` (32, 0), `systems` (39, 1), `generative/diffusion` (34, 0),
  `dynamics/fields` (22, 0), the `inference/*` application models (49, 0), `foundation/autodiff` (33 exports, 1 entry,
  0 notes) and `numerics/linalg` (14 entries, 1 note).
- The rest is real capability gaps. The largest are classical hypothesis testing, the neural sequence and transformer
  stack, deep generative models, text processing, anomaly detection, ANN indexes and quantisation, and
  time-series similarity search.

## Coverage by top-level topic

In `taxonomy.yaml` order. "Widget notes" counts notes that have a `.tsx` file beside them (context only).

| Topic                        |     Notes | aifn-linked |       % | Widget notes |
| ---------------------------- | --------: | ----------: | ------: | -----------: |
| maths                        |       128 |          25 |     19% |           68 |
| probability                  |        60 |          11 |     18% |           41 |
| probability-distributions    |        22 |          21 |     95% |            9 |
| statistics                   |        45 |           0 |      0% |           26 |
| probabilistic-inference      |        59 |          18 |     30% |           48 |
| signal-processing            |        79 |          10 |     12% |           63 |
| control-theory               |        25 |           1 |      4% |           12 |
| learning-foundations         |        17 |           3 |     17% |            5 |
| losses                       |        10 |           4 |     40% |            6 |
| metrics                      |        65 |          54 |     83% |           19 |
| supervised-learning          |       131 |          54 |     41% |           76 |
| semi-supervised-learning     |         2 |           0 |      0% |            2 |
| weak-supervision             |        18 |           0 |      0% |            8 |
| unsupervised-learning        |        61 |          13 |     21% |           31 |
| online-learning              |        22 |           0 |      0% |            5 |
| transfer-learning            |        20 |           1 |      5% |            6 |
| neural-networks              |        26 |           7 |     26% |           10 |
| convolutional-networks       |         4 |           0 |      0% |            3 |
| sequence-models              |        15 |           0 |      0% |            7 |
| graph-neural-networks        |         3 |           0 |      0% |            1 |
| transformers                 |        35 |           0 |      0% |           12 |
| large-language-models        |        11 |           0 |      0% |            6 |
| generative-models            |        17 |           0 |      0% |            8 |
| representation-learning      |         9 |           1 |     11% |            4 |
| time-series                  |        66 |           7 |     10% |           35 |
| anomaly-detection            |        24 |           1 |      4% |            6 |
| recommendation-and-retrieval |        71 |           9 |     12% |           26 |
| natural-language-processing  |        29 |           0 |      0% |            5 |
| computer-vision              |        17 |           0 |      0% |           10 |
| decision-making              |        25 |           3 |     12% |           15 |
| reinforcement-learning       |        19 |          10 |     52% |            9 |
| online-experimentation       |        41 |           6 |     14% |           27 |
| trustworthy-machine-learning |        24 |           1 |      4% |            9 |
| engineering                  |        16 |           1 |      6% |           14 |
| case-studies                 |         7 |           0 |      0% |            0 |
| **Total**                    | **1,223** |     **261** | **21%** |      **632** |

### Weakest branches (≥ 8 notes, 0% linked)

`unsupervised-learning/topic-models` (30), `natural-language-processing/text-representation` (19),
`probabilistic-inference/skill-rating` (14), `probabilistic-inference/model-based-machine-learning` (14),
`signal-processing/audio` (13), `time-series/neural-forecasting` (12), `probability/theory` (12),
`recommendation-and-retrieval/pinsage-family` (11), `time-series/similarity-search` (11),
`statistics/hypothesis-testing` (9), `sequence-models/efficient-sequence-models` (9), `maths/calculus-of-variations`
(9), `recommendation-and-retrieval/debiasing` (8), `anomaly-detection/deep-one-class-methods` (8),
`neural-networks/energy-based-networks` (8), `transformers/positional-encodings` (8),
`signal-processing/spectral-estimation` (8), `signal-processing/signals-and-systems` (8). `maths/calculus` has 29 notes
and 1 link.

## Item 0: register and link the code that already exists

This item comes before any new capability. It is the cheapest way to raise coverage. Each row lists unlinked notes in
the branch, out of the branch's notes, and the existing aifn code that serves them.

| Branch group                                                                                 | Unlinked notes | Existing code with no entries or links                                                                                                                                                                                                                                                 |
| -------------------------------------------------------------------------------------------- | -------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Signal processing (Fourier, filters, spectral, time-frequency, multirate, systems, wavelets) |          52/60 | `foundation/fourier` (dft, dct), `signal/spectral` (stft, welch, periodogram, multitaper), `signal/time-frequency` (hilbert, cqt), `signal/multirate`, `signal/filters` (butter, cheby, firwin), `signal/decompositions` (emd, eemd, ceemdan), `signal/statistical` (yuleWalker, burg) |
| Linear algebra, calculus, autodiff, numerical methods                                        |          52/58 | `numerics/linalg` (82 exports, 14 entries, 1 note), `foundation/autodiff`, `foundation/trace`, `numerics/quadrature`                                                                                                                                                                   |
| Inference applications (LDA, TrueSkill, HMM/CRF, Ising, EP clutter, BPM)                     |          38/43 | `inference/topic-models`, `rating-models`, `sequence-models`, `lattice-models`, `classifier-models`, `mixture-models`, `conjugate-models`: 0 entries in total                                                                                                                          |
| nn layers (attention, RNN/GRU/LSTM cells, conv, norms, dropout)                              |          26/30 | `nn/layers`: 32 exports, 0 entries                                                                                                                                                                                                                                                     |
| Time series (ARMA, SARIMA, GARCH, ETS, STL, ACF/PACF, state-space EM)                        |          24/30 | `timeseries`: 43 exports including `armaFitSteps`, `garchFitSteps`, `sarimaFitSteps`; 0 entries                                                                                                                                                                                        |
| ODEs, PDEs, SDEs, stochastic processes                                                       |          22/27 | `dynamics/fields`, `applications/dynamics/pde`, `dynamics/sde`                                                                                                                                                                                                                         |
| Statistics resampling, information theory, sampling, KDE                                     |          18/24 | `probability/stats` (bootstrap, permutationTest, kde, ecdf), `probability/information`, `probability/samplers`: 0 entries                                                                                                                                                              |
| Control (system modelling, state-space control, classical, stability)                        |          16/17 | `systems` (stateSpace, bode, margins, placePoles, controllability), `dynamics/control` (lqr, pid)                                                                                                                                                                                      |
| Multi-armed bandits                                                                          |           8/11 | `gym/agents/bandits` (exp3, klUcb, thompson)                                                                                                                                                                                                                                           |
| Diffusion models and SDE notes                                                               |        5/5 + 5 | `generative/diffusion`: 34 exports, 0 entries (ddpm, ddim, probability flow, reverse SDE, Tweedie)                                                                                                                                                                                     |

Size **M** (spread over many modules; mostly registry metadata, `notes` lists and a step `Algorithm` form where one is
missing). It reaches roughly 260 notes. Realistically 150–200 of them become linked, because some notes in these
branches need the new capabilities listed below.

## Top missing capabilities

Ranked by the number of notes each would unlock. That count is the notes whose central computation needs the
capability. Sizes: S (under an hour), M (a few hours), L (a day or more).

|   # | Capability                                                                                                                                                                                                                                                                                                                                                 | Notes | Size | Placement                                                                                   | Example slugs                                                                                                                                                                                                                   |
| --: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----: | :--: | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   1 | **Classical hypothesis tests and intervals.** One test protocol (statistic, null distribution, p-value, CI, effect size). Includes t (one-sample, paired, pooled, Welch), z, exact binomial, χ²/G, Ljung–Box, Grubbs, multiple testing (Bonferroni, Holm, BH), power and sample size, delta method, CUPED.                                                 |   ~38 |  M   | core `probability/tests` (new; uses `numerics/special`)                                     | `welch-t-test`, `binomial-test`, `p-value`, `multiple-testing`, `sample-size-planning`, `a-b-test-power`, `ratio-metrics-and-delta-method`, `controlled-experiments-using-pre-experiment-data`, `ljung-box-test`, `grubbs-test` |
|   2 | **Attention and positional-encoding family.** Sinusoidal, learned, RoPE (+ variants), ALiBi, T5 bias, NoPE; MQA/GQA, MLA; causal and sliding-window masks; QK-norm and soft-capping; pre/post-norm, RMSNorm, gated FFN (SwiGLU); KV cache accounting; a small decoder block.                                                                               |   ~32 |  M   | core `nn/layers` + `nn/positional` (new); apps `neural/architectures`                       | `rotary-position-embedding`, `alibi`, `multi-query-and-grouped-query-attention`, `pre-norm-and-post-norm`, `gated-feed-forward-layers`, `key-value-cache`, `attention-sinks`                                                    |
|   3 | **Deep generative models on 2-D toys.** Autoencoder, VAE (β, conditional, VQ), GAN with optimal discriminator, coupling flows (RealNVP from existing bijectors), EBM with Langevin and CD, RBM, Boltzmann machine, Hopfield (classical and modern), DBN, score matching.                                                                                   |   ~26 |  L   | apps `generative/{autoencoders,adversarial,flows,energy}`                                   | `variational-autoencoder`, `generative-adversarial-network`, `normalising-flow`, `energy-based-models`, `restricted-boltzmann-machine`, `hopfield-network`, `modern-hopfield-network`                                           |
|   4 | **Online learning and expert advice.** Hedge, weighted majority, aggregating algorithm, fixed share, OGD, FTRL, ONS, online-to-batch; regret traces as step Algorithms.                                                                                                                                                                                    |   ~22 |  M   | core `optim/online` (new)                                                                   | `hedge-and-exponential-weights`, `follow-the-regularised-leader`, `online-gradient-descent`, `tracking-the-best-expert`, `online-newton-step-and-exp-concavity`                                                                 |
|   5 | **Text processing pipeline.** Normalise, tokenise, stem (Porter), stop words, word and character n-grams, BPE/WordPiece/Unigram training as step Algorithms, bag of words, TF-IDF/BM25 variants, feature hashing, co-occurrence + PPMI + SVD, skip-gram with negative sampling, GloVe; a small built-in corpus.                                            |   ~23 |  M   | apps `text` (new area) + `data/real` corpus                                                 | `byte-pair-encoding`, `term-frequency-inverse-document-frequency`, `co-occurrence-matrices-and-pointwise-mutual-information`, `stemming`, `word-embeddings`, `n-gram-language-model`                                            |
|   6 | **Weak supervision and label noise.** Majority vote, Dawid–Skene EM, Snorkel label model, PU (Elkan–Noto, nnPU), LLP (proportion loss, mean map, InvCal), MIL (attention pooling, mi-SVM), complementary and partial labels; noise-rate estimation (confident learning), robust losses, forward correction; crowd/LF generators.                           |   ~24 |  M   | apps `learning/weak-supervision` (new); generators in `data/synthetic`                      | `dawid-skene-model`, `snorkel`, `positive-unlabelled-learning`, `learning-from-label-proportions`, `multiple-instance-learning`, `estimating-noise-rates`                                                                       |
|   7 | **Anomaly detection family.** Isolation forest, LOF, k-NN score, one-class SVM and SVDD (QP exists), Mahalanobis, EVT peaks-over-threshold (GPD), PCA reconstruction, deep SVDD and autoencoder scores (needs #3), detector ensembles.                                                                                                                     |   ~22 |  M   | apps `unsupervised/anomaly` (new)                                                           | `isolation-forest`, `local-outlier-factor`, `one-class-support-vector-machine`, `support-vector-data-description`, `extreme-value-theory-for-anomalies`, `deep-support-vector-data-description`                                 |
|   8 | **Recommender models.** Neighbourhood CF, MF (SGD/ALS), weighted/implicit MF, FM/FFM/FwFM, wide-and-deep, DeepFM, two-tower with sampled softmax, NCF, SASRec/BERT4Rec (needs #2), popularity bias and feedback-loop simulators. `ratings`/`clickLog` and the retrieval losses already exist.                                                              |   ~22 | M–L  | apps `retrieval/recommenders` (new)                                                         | `matrix-factorisation-for-recommendation`, `factorisation-machines`, `two-tower-model`, `neighbourhood-collaborative-filtering`, `feedback-loops-in-recommendation`, `sasrec`                                                   |
|   9 | **Recurrent and state-space sequence models.** BPTT with gradient-norm traces, seq2seq + Bahdanau attention, SSM discretisation (ZOH, bilinear), S4 kernel, HiPPO, selective scan (Mamba), parallel scan (Hillis–Steele, Blelloch) as Algorithms, linear attention as an RNN, RWKV/retention.                                                              |   ~20 |  M   | core `nn/layers` (cells exist) + `nn/sequence` (new)                                        | `backpropagation-through-time`, `structured-state-space-models`, `mamba`, `parallel-scan`, `linear-attention`, `sequence-to-sequence`                                                                                           |
|  10 | **Learning-theory and limit-theorem exposition.** Bias–variance over resampled training sets, double descent (min-norm random features), shattering/VC, empirical Rademacher complexity, Hoeffding/Chernoff/McDiarmid bounds against simulated tails, CLT/LLN standardised sums, AIC/BIC, effective degrees of freedom.                                    |   ~20 |  M   | core `learning/validate` + `probability/stats`                                              | `bias-variance-decomposition`, `double-descent`, `rademacher-complexity`, `vapnik-chervonenkis-dimension`, `hoeffdings-inequality`, `central-limit-theorem`, `information-criteria`                                             |
|  11 | **Spectral estimation and audio analysis extras.** Lomb–Scargle, MUSIC/ESPRIT, coherence and cross-spectra, Wigner–Ville and Cohen's class, reassignment and synchrosqueezing, matched and Wiener filters, LMS/RLS, cepstrum, YIN pitch, spectral flux onsets with DP beat tracking, NMF separation, spectral subtraction, gammatone, chroma, SpecAugment. |   ~20 |  M   | core `signal/spectral`, `signal/time-frequency`, `signal/statistical`; apps `signals/audio` | `lomb-scargle-periodogram`, `subspace-frequency-estimation`, `wigner-ville-distribution`, `pitch-estimation`, `onset-detection-and-beat-tracking`, `audio-source-separation`                                                    |
|  12 | **Trustworthy-ML methods.** KernelSHAP and TreeSHAP, LIME, integrated gradients (autodiff exists), counterfactual explanations, fairness mitigation (reweighing, equalised-odds post-processing), DP mechanisms with composition and DP-SGD clipping, membership inference, FGSM/PGD, randomised smoothing.                                                |   ~20 |  M   | apps `trust` (new area)                                                                     | `feature-attribution-methods`, `shapley-additive-explanations-for-trees`, `differential-privacy`, `adversarial-examples`, `certified-robustness`, `bias-mitigation`                                                             |
|  13 | **Transfer, continual and meta-learning.** DANN (gradient reversal), MMD/CORAL alignment, label-shift EM and BBSE, EWC, replay, MAML on sine regression, prototypical networks, test-time adaptation (entropy minimisation, BN statistics).                                                                                                                |   ~20 |  L   | apps `learning/transfer` (new); needs nn training                                           | `domain-adversarial-training`, `label-shift-and-target-shift`, `regularisation-based-continual-learning`, `meta-learning`, `test-time-adaptation`                                                                               |
|  14 | **Time-series similarity and classification.** z-normalised distance profile (MASS by FFT), matrix profile (STOMP/SCRIMP), motifs, discords, chains, snippets, FLUSS segmentation, AB-joins, streaming updates; DTW with Sakoe–Chiba window and LB_Keogh; PAA/SAX/iSAX; shapelet search and learned shapelets.                                             |   ~17 |  M   | apps `timeseries/similarity` (new)                                                          | `matrix-profile`, `distance-profile-and-mass`, `dynamic-time-warping`, `symbolic-aggregate-approximation`, `shapelets`, `time-series-chains`                                                                                    |
|  15 | **Approximate nearest neighbours and quantisation.** k-d and ball trees, LSH (hyperplane, p-stable, MinHash, S-curve), IVF, PQ/OPQ/additive/residual, scalar and binary codes, HNSW and other graph indexes, ScaNN anisotropic loss, MIPS reductions, recall@k benchmarking.                                                                               |   ~17 | M–L  | apps `retrieval/ann` (new)                                                                  | `product-quantisation`, `hnsw`, `locality-sensitive-hashing`, `inverted-file-index`, `scann-and-anisotropic-quantisation`, `maximum-inner-product-search`                                                                       |
|  16 | **Counterfactual evaluation and logged-bandit estimators.** IPS, clipped and self-normalised IPS, DM, DR, slate (pseudo-inverse) estimators, propensity estimation, interleaving, unbiased LTR (position-bias EM); a logged-data generator.                                                                                                                |   ~17 |  M   | apps `evaluation/counterfactual` (new)                                                      | `off-policy-evaluation`, `counterfactual-evaluation-of-recommenders`, `slate-off-policy-evaluation`, `estimating-item-propensities`, `interleaving`, `unbiased-learning-to-rank`                                                |
|  17 | **Dimensionality reduction and covariance extras.** Factor analysis (EM), PPCA, FastICA, NMF (multiplicative updates; also serves audio and topics), CCA, random projections (JL), SOM, diffusion maps, LTSA/Hessian eigenmaps, PaCMAP/TriMAP, graphical lasso, Ledoit–Wolf shrinkage, gap statistic.                                                      |   ~16 |  M   | apps `unsupervised/embedding/*`, `unsupervised/covariance` (new)                            | `factor-analysis`, `independent-component-analysis`, `non-negative-matrix-factorisation`, `canonical-correlation-analysis`, `self-organising-maps`, `graphical-lasso`                                                           |
|  18 | **Calibration maps and conformal prediction.** PAV isotonic as a step Algorithm, temperature, beta and Dirichlet calibration, histogram binning, split and conformalised-quantile conformal, APS sets, quantile/distribution calibration, risk–coverage curves, reject option. Platt and the ECE family exist.                                             |   ~16 | S–M  | core `learning/calibration` (new)                                                           | `isotonic-calibration`, `temperature-scaling`, `beta-calibration`, `conformal-prediction`, `selective-classification`, `quantile-calibration`                                                                                   |
|  19 | **Skill rating, paired comparison and IRT.** Elo, Glicko/Glicko-2, Bradley–Terry and Plackett–Luce MLE (MM), IRT 1–4PL fitting (JML/MML) and item information, polytomous IRT, TrueSkill Through Time. TrueSkill already exists and needs linking.                                                                                                         |   ~16 | S–M  | apps `inference/rating-models`                                                              | `elo-rating`, `glicko-and-glicko-2`, `plackett-luce-model`, `item-response-theory`, `paired-comparison-models`, `trueskill-through-time`                                                                                        |
|  20 | **Graph neural networks and graph recommenders.** GCN, GAT, MPNN, GraphSAGE sampling, LightGCN propagation, Pixie random walks, harmonic label propagation; `graph/matrices` and `graph/propagation` exist.                                                                                                                                                |   ~16 |  M   | core `nn/graph` (new) on `graph/*`; apps `retrieval/graph`                                  | `graph-convolutional-network`, `graph-attention-network`, `lightgcn`, `graphsage`, `pixie-random-walk`, `label-propagation`                                                                                                     |
|  21 | **Topic-model family beyond LDA.** pLSA (EM), LSA, CTM, DTM, HDP (Chinese restaurant franchise), hLDA (nCRP), labelled and supervised LDA, author-topic, biterm, ProdLDA/NVDM/ETM (needs #3), coherence and perplexity; needs #5 for real text.                                                                                                            |   ~15 | M–L  | apps `inference/topic-models`                                                               | `probabilistic-latent-semantic-analysis`, `hierarchical-dirichlet-process`, `dynamic-topic-model`, `correlated-topic-model`, `topic-model-evaluation`                                                                           |
|  22 | **Control design and identification.** MPC (QP per step; `optim/programming` exists), LQG (compose Kalman + LQR), root locus, Nyquist plot, Routh–Hurwitz table, lead-lag design, sliding mode, feedback linearisation, MRAC, ARX/PEM, subspace identification (N4SID), Pontryagin shooting and HJB by DP.                                                 |   ~15 |  M   | core `systems`, `dynamics/control`; apps `dynamics/control`                                 | `model-predictive-control`, `root-locus`, `nyquist-stability-criterion`, `routh-hurwitz-criterion`, `subspace-identification`, `pontryagin-maximum-principle`                                                                   |
|  23 | **Computer vision.** Canny, Harris and DoG features, Hough, morphology, Gaussian/Laplacian pyramids, seam carving; pinhole camera, homography by DLT + RANSAC, eight-point epipolar geometry, Zhang calibration, triangulation; NMS. `vision/filters` has blur, Sobel and LoG only.                                                                        |   ~16 |  M   | apps `vision/{features,geometry}`                                                           | `edge-detection`, `hough-transform`, `homography`, `epipolar-geometry-and-stereo`, `camera-calibration`, `morphological-operations`                                                                                             |
|  24 | **Survival analysis.** Kaplan–Meier with Greenwood, Nelson–Aalen, log-rank, Cox PH by partial-likelihood Newton (ties: Breslow/Efron), Weibull/log-normal AFT; censored-data generator.                                                                                                                                                                    |     7 |  M   | apps `learning/survival` (new)                                                              | `kaplan-meier-estimator`, `cox-proportional-hazards`, `log-rank-test`, `nelson-aalen-estimator`, `parametric-survival-models`                                                                                                   |
|  25 | **Sequential testing and change detection.** Group-sequential boundaries with alpha spending, SPRT/mSPRT, e-values and confidence sequences, CUSUM with ARL. BOCPD exists.                                                                                                                                                                                 |     7 | S–M  | core `probability/tests` (with #1)                                                          | `group-sequential-designs`, `mixture-sequential-probability-ratio-test`, `e-values-and-confidence-sequences`, `peeking-and-optional-stopping`, `cumulative-sum-control-chart`                                                   |

Smaller items, each under 12 notes:

- **LM decoding** (~7, S–M, apps `neural/language-models`): greedy, beam, top-k, top-p, temperature, speculative
  decoding, KN-smoothed n-grams. Slugs: `decoding-strategies`, `speculative-decoding`.
- **Imbalanced learning** (~10, S, apps `learning/preprocessing`): SMOTE, Borderline-SMOTE, ADASYN, over- and
  under-sampling, Saerens prior-shift EM, threshold moving.
- **RL policy methods** (~9, M, apps `gym/agents`): REINFORCE with baseline, actor–critic, PPO, DDPG, RLHF reward model,
  CQL, RL off-policy evaluation.
- **Engineering calculators** (~15, S–M): int8 quantisation, magnitude pruning, distillation, fp16/bf16 rounding,
  memory footprint, MoE routing and load-balancing loss, flash-attention tiling with online softmax.
- **Causal and experimental design** (~10, M): back-door adjustment, IPW and matching, blocking, factorial ANOVA,
  cluster randomisation, switchback.
- **MBML gates and models** (14, M, core `inference/model`): a gate (switch/mixture) factor, then Matchbox, AdPredictor,
  CBCC, reviewer calibration and murder mystery as registered model specs.
- **Markov-chain toolkit** (~5, S, core `probability`): stationary distribution, absorption and hitting probabilities,
  mixing. Slugs: `markov-chain`, `gamblers-ruin`.
- **Splines and OT extras** (~10, S): Bézier/de Casteljau, Catmull–Rom, NURBS, curvature; EMD by LP, unbalanced and
  partial Sinkhorn, Wasserstein barycentres.

## Thin areas

1. **Registration debt.** 64 source modules have exports but 0 catalog entries. About 45 of them hold exposition code; the rest are infrastructure such as `errors`, `registry` and `pytree` (see Item 0). Some registered modules
   link no notes: `foundation/tensor` (54 entries) and `numerics/special` (54) link none. Most links sit in
   `learning/metrics` (116 entries, 37 notes).
2. **No reference fixtures** for `timeseries`, which is checked against statsmodels only in its own SARIMA test, and
   that test is flaky (refinement "Test flakiness"). The same is true of `learning/trees-and-ensembles` (sklearn),
   `kernel-methods`, `neighbours`, `preprocessing`, the `inference/*` application models, `signals/audio` (librosa is
   not installed), `vision/filters`, `graph/*` (networkx), `dynamics/sde`, `dynamics/fields`, and `signal/wavelets`
   (pywt, awaiting `uv add` approval).
3. **No step-through `Algorithm` form** for several procedures that notes walk through: Gram–Schmidt, QR/Householder,
   Jacobi and Gauss–Seidel, conjugate gradient (`iterative-linear-solvers`), LDA collapsed Gibbs, PAV, the Lanczos and
   eigen iterations, the EMD sifting loop, and k-means/EM for GMM. Only 21 modules export `*Steps`. Six are in
   `clustering` and none in `numerics/linalg`.
4. **`nn` is thin for exposition.** Its 32 layer exports have 0 entries. Only `nn.json` serves as a torch fixture.
   There is no gradient-norm or activation-statistics trace for `vanishing-and-exploding-gradients` or
   `weight-initialisation`, and `nn/training` has one registered entry.
5. **`probability/stats` has no tests.** It has bootstrap, permutation and KS but no t, z, χ² or binomial test, and no
   CI/effect-size protocol. So `statistics` sits at 0% even though its building blocks exist.
6. **Open items from `refinement.md`** that still affect exposition: Laplace GP-classification evidence has no
   gradient (Nelder–Mead fit); the group A programming/KL/distribution entries are not yet marked `stable`; DWT and
   Sinkhorn are unchecked against pywt and POT; drag latency is above 32 ms for GPC, Kalman and HMC. One entry is
   stale: "No EP GP classifier" is already done (`learning/gaussian-processes/classification-ep.ts`).

## Missing lab components

The lab registers 13 views: `chains/diagnostics`, `computation-graph/backprop`, `cross-validation/folds`,
`curve/unit-square`, `dataset/{andrews,pairs,parallel,scatter}`, `distribution/density`, `linkage/dendrogram`,
`params/table`, `tensor/values` and `trace/series`. Plot layers include Raster, Contours, Vectors and Histogram, and a
force-directed diagram layout exists. Recurring needs with no view:

| Component                                                                                                     | Needed by (examples)                                                                 |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Time-frequency view (spectrogram/scalogram/WVD over a waveform, with audio playback)                          | audio (13), time-frequency (10), spectral estimation                                 |
| Image and image-grid view (input → filter → output panels, patch grids, reconstructions)                      | CV (17), VAE/GAN samples, `singular-value-decomposition` low-rank image, conv layers |
| Attention / token view (token strip, attention matrix, heads, cache contents)                                 | transformers (35), LLM (11), BPE/tokenisation                                        |
| Network graph view with a layout and node values (propagation, message passing, kNN/HNSW layers)              | GNN, graph recommenders, HNSW, label propagation, MRF                                |
| 3-D surface and 3-D scatter                                                                                   | loss surfaces, `hessian`, `gradient`, Swiss roll, `multiple-integrals`               |
| Sequence alignment and DP-matrix view (cost matrix with warping path)                                         | DTW, edit distance, beat tracking, Viterbi                                           |
| Step-function / survival view (censor ticks, CI bands, at-risk table)                                         | survival (7), Kaplan–Meier, ECDF, isotonic/PAV                                       |
| Hypothesis-test view (null density, rejection region, observed statistic, power) and forest plot of intervals | statistics (45), online experimentation (41)                                         |
| Sankey / flow view                                                                                            | multi-stage recommendation, transport plans, Snorkel pipeline                        |
| Systems views (pole–zero plane, Nyquist, root locus, Bode linked to a pole drag)                              | control (25), filters, z-transform                                                   |
| Matrix-profile view (series above its profile with motif/discord markers)                                     | similarity search (11)                                                               |
| Regret / online trace view (cumulative loss per expert, weights over time)                                    | online learning (22), bandits                                                        |
| Animated distribution evolution (density over time as a player: Fokker–Planck, diffusion, heat)               | stochastic calculus (10), diffusion models, heat equation                            |

## Suggested order of work

1. **Item 0: registration and links** (M). Add entries and `notes` for `timeseries`, `probability/stats` and
   `information`, `systems`, `dynamics/fields`, `nn/layers`, `generative/diffusion`, the `inference/*` application
   models, the signal modules and `numerics/linalg`. Add `*Steps` forms where a note walks through the procedure. This
   step alone could take coverage from 21% to about 35%.
2. **Hypothesis tests + sequential testing** (#1, #25), with the hypothesis-test view. This opens statistics and
   online experimentation, 86 notes at 0–14%.
3. **Attention/positional + sequence models + decoding** (#2, #9, LM decoding), with the attention/token view. This
   opens transformers, sequence models and LLMs, 61 notes at 0%.
4. **Text pipeline** (#5), then the topic-model family (#21). The text pipeline is also a prerequisite for NLP,
   word embeddings and real-text topic models.
5. **Calibration/conformal, imbalanced learning and skill rating** (#18, #19, imbalanced). Each is small, with high
   reuse.
6. **Anomaly detection, time-series similarity, ANN/quantisation** (#7, #14, #15), with the matrix-profile and
   network views.
7. **Online learning and counterfactual evaluation** (#4, #16), with the regret view; then recommender models (#8)
   and GNNs (#20).
8. **Deep generative models** (#3), with the image-grid view; then trustworthy ML (#12) and transfer learning (#13),
   which reuse its training loops.
9. **Spectral/audio extras, control design, CV, survival, dimensionality reduction** (#11, #22, #23, #24, #17), with
   the time-frequency, systems, image and survival views.
10. **Thin-area work** runs alongside: fixtures for `timeseries`, trees and the inference applications first.
