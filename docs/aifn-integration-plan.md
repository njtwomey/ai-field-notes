# aifn integration plan: core versus applications for the coverage gaps

Status: **plan (2026-10-01)**. Source: `docs/aifn-coverage.md` (the gaps), `docs/aifn-architecture.md` (the rule).

## 1. The rule, restated

**Core** holds what passes all of C1–C5, or C6:

- **C1** its interface names no model, problem or dataset (a tokeniser, a test, a scan, a loss; not "Matchbox" or
  "two-tower");
- **C2** at least two areas (or core modules) use it;
- **C3** it is testable against a reference or a law;
- **C4** it is Tensor-native;
- **C5** it does not change when a note or figure does;
- **C6** other core code depends on it.

**Applications** hold named models, data generators, simulators and teaching set-ups: things a note is _about_ rather
than things notes are _built from_. An application is promoted to core when C1–C5 hold.

Rules of thumb used below:

- **Building blocks are core; named assemblies are applications.** A GRU cell, attention, a scan, a tokeniser, an
  index, an estimator of a quantity are core. LSTM language models, SASRec, Dawid–Skene, Isolation Forest are
  applications built from them.
- **Estimators of a statistical quantity are core** (Kaplan–Meier, IPS, a t-test, conformal quantiles): their interface
  is data in, an estimate out, no model named.
- **Simulators and demonstrations are applications** (double descent, bias–variance resampling, feedback loops).
- **When a model family has one canonical fitting algorithm that is itself general, the algorithm is core and the model
  is an application** (NMF multiplicative updates core; topic models apps).

## 2. Placement of every gap

| Gap (coverage #)                                   | Core                                                                                                                                                                                                                                                       | Applications                                                                                                                                                 |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Text processing (#5)                               | **new family `text`**: normalisation, word/char tokenisers, n-grams, vocabularies, BPE/WordPiece/Unigram training as step Algorithms, Porter stemmer, stop words, bag of words, TF-IDF/BM25 weightings, feature hashing, co-occurrence + PPMI              | word2vec/GloVe training set-ups, n-gram language models (KN smoothing), corpora                                                                              |
| Hypothesis tests (#1) and sequential testing (#25) | **new `probability/tests`**: a test protocol (statistic, null, p-value, CI, effect size); t/z/binomial/χ²/G/Ljung–Box/Grubbs/KS; multiple testing; SPRT/mSPRT, alpha spending, confidence sequences, e-values, CUSUM; Kaplan–Meier, Nelson–Aalen, log-rank | experiment simulators (A/B, peeking, switchback)                                                                                                             |
| Attention and positions (#2)                       | **`nn/attention`**: scaled dot-product, multi-head, MQA/GQA/MLA, masks (causal, sliding), sinusoidal/learned/RoPE/ALiBi/T5 bias, KV cache, RMSNorm, SwiGLU, transformer block                                                                              | a small GPT on toy text, attention-pattern demos                                                                                                             |
| Sequence models (#9)                               | **`nn/sequence`**: BPTT already exists (`unrollRecurrent`, LSTM/GRU cells); add SSM discretisation (ZOH, bilinear), S4 kernel, selective scan, linear attention; **`foundation/tensor` associative scan** (Hillis–Steele, Blelloch) as a primitive         | seq2seq toy translation, Mamba demo, gradient-norm studies                                                                                                   |
| LM decoding                                        | **`nn/decoding`**: greedy, beam, top-k, top-p, temperature, speculative decoding over any logits function                                                                                                                                                  | n-gram LMs and decoding demos                                                                                                                                |
| Online learning (#4)                               | **new `optim/online`**: Hedge, weighted majority, fixed share, OGD, FTRL, ONS, online-to-batch, regret traces                                                                                                                                              | expert-advice games, adversarial sequences                                                                                                                   |
| Counterfactual evaluation (#16)                    | **new `learning/off-policy`**: IPS, clipped/self-normalised IPS, DM, DR, slate estimators, propensity estimation                                                                                                                                           | logged-bandit generators, interleaving experiments                                                                                                           |
| Calibration and conformal (#18)                    | **`learning/calibration`**: PAV isotonic (step Algorithm), temperature/beta/Dirichlet maps, histogram binning; **`learning/conformal`**: split, CQR, APS                                                                                                   | —                                                                                                                                                            |
| Nearest neighbours and quantisation (#15)          | **new `numerics/neighbours`**: k-d and ball trees, LSH families, NN-descent (moved from applications), IVF, PQ/OPQ, HNSW, recall metrics                                                                                                                   | retrieval demos                                                                                                                                              |
| Graph learning (#20), label propagation            | **`graph/propagation`** (exists): harmonic label propagation, label spreading; **`nn/graph`**: GCN, GAT, MPNN, GraphSAGE layers                                                                                                                            | LightGCN, Pixie, citation-graph set-ups                                                                                                                      |
| Weak supervision (#6)                              | losses only, in `learning/losses`: PU (uPU, nnPU), LLP proportion loss, complementary-label loss                                                                                                                                                           | **`learning/weak-supervision`**: majority vote, Dawid–Skene, Snorkel label model, Elkan–Noto, MIL models, noise-rate estimation                              |
| Recommenders (#8)                                  | sampled softmax and in-batch negatives in `learning/losses` (if not there)                                                                                                                                                                                 | **`retrieval/recommenders`**: neighbourhood CF, MF (SGD/ALS), implicit MF, FM/FFM, wide-and-deep, DeepFM, two-tower, NCF, SASRec, popularity-bias simulators |
| Anomaly detection (#7)                             | GPD and EVT peaks-over-threshold in `probability/distributions`/`stats`                                                                                                                                                                                    | **`unsupervised/anomaly`**: isolation forest, LOF, k-NN score, one-class SVM/SVDD, Mahalanobis, PCA reconstruction, ensembles                                |
| Time-series similarity (#14)                       | **`signal/similarity`**: MASS distance profile, matrix profile (STOMP/SCRIMP), DTW with Sakoe–Chiba and LB_Keogh, PAA/SAX                                                                                                                                  | motif, discord, segmentation demos                                                                                                                           |
| Spectral and audio (#11)                           | **`signal`**: Lomb–Scargle, MUSIC/ESPRIT, coherence, Wigner–Ville, reassignment, Wiener/matched filters, cepstrum, YIN                                                                                                                                     | audio demos                                                                                                                                                  |
| Control (#22)                                      | **`systems`**: MPC, LQG, root locus, Nyquist, Routh–Hurwitz, ARX/PEM, subspace identification                                                                                                                                                              | controller demos                                                                                                                                             |
| Vision (#23)                                       | **`signal/image`**: Canny, Harris, DoG, Hough, morphology, pyramids; **`numerics/geometry`**: homography DLT, eight-point, triangulation; **`numerics/robust`**: RANSAC                                                                                    | seam carving, calibration demos                                                                                                                              |
| Dimensionality reduction (#17)                     | **`numerics/factorisation`**: NMF updates, random projections (JL), CCA                                                                                                                                                                                    | factor analysis, PPCA, FastICA, SOM, diffusion maps, PaCMAP models                                                                                           |
| Deep generative (#3)                               | coupling bijectors in `probability/bijectors`; Langevin and contrastive-divergence samplers in `inference/stochastic`                                                                                                                                      | **`generative`**: autoencoder, VAE family, GAN, RealNVP, EBM, RBM, Hopfield, DBN on 2-D toys                                                                 |
| Trustworthy ML (#12)                               | **`learning/explain`**: KernelSHAP, TreeSHAP, LIME, integrated gradients; **`probability/privacy`**: DP mechanisms, composition, clipping                                                                                                                  | fairness mitigation, membership-inference demos                                                                                                              |
| Transfer and meta (#13)                            | —                                                                                                                                                                                                                                                          | **`learning/transfer`**: DANN, MMD/CORAL, label shift, EWC, replay, MAML, prototypical nets                                                                  |
| Learning theory (#10)                              | —                                                                                                                                                                                                                                                          | **`theory`**: bias–variance, double descent, VC/shattering, Rademacher, concentration-bound simulations, CLT/LLN                                             |
| Topic models (#21)                                 | NMF (above)                                                                                                                                                                                                                                                | **`inference/topic-models`**: pLSA, LSA, CTM, DTM, HDP, hLDA, labelled/supervised LDA, ProdLDA                                                               |
| Skill rating (#19)                                 | —                                                                                                                                                                                                                                                          | **`inference/rating`**: Elo, Glicko, Bradley–Terry, Plackett–Luce, IRT                                                                                       |
| Survival (#24)                                     | Kaplan–Meier, Nelson–Aalen, log-rank (in `probability/tests`)                                                                                                                                                                                              | Cox PH, AFT models, censored-data generators                                                                                                                 |
| Markov chains                                      | **`probability/markov`**: stationary distribution, absorption, hitting times, mixing                                                                                                                                                                       | gambler's ruin and other chain demos                                                                                                                         |
| Imbalanced learning                                | —                                                                                                                                                                                                                                                          | SMOTE family in `learning/preprocessing`                                                                                                                     |
| RL policy methods                                  | —                                                                                                                                                                                                                                                          | `gym/agents`: REINFORCE with baseline, actor–critic, PPO, DDPG, CQL                                                                                          |
| MBML gates                                         | gate factor in `inference/model`                                                                                                                                                                                                                           | Matchbox, AdPredictor and other named models                                                                                                                 |
| Engineering calculators                            | quantisation and rounding in `nn/quantise`; MoE routing in `nn/layers`                                                                                                                                                                                     | memory and throughput calculators                                                                                                                            |

## 3. Waves

Each wave ends green (`make check`). Agents: at most 2 heavy processes each; no parallel sweeps.

**Wave 1: foundations that unlock the most notes**

1. **Registration and links** (coverage item 0): register the ~64 unregistered modules' exports with `notes` links;
   add `*Steps` forms where a note walks through a procedure (Gram–Schmidt, QR, CG, PAV, LDA's collapsed Gibbs).
2. **`text`** (new core family).
3. **`probability/tests`** (new core module), with the hypothesis-test lab view.
4. **`nn/attention`, `nn/sequence`, `nn/decoding`** and the associative scan, with the attention/token lab view.

**Wave 2: shared machinery**

`optim/online`, `learning/off-policy`, `learning/calibration` and `learning/conformal`, `numerics/neighbours`,
`graph/propagation` label propagation and `nn/graph`, `probability/markov`, `signal/similarity`.

**Wave 3: applications on top**

Recommenders, anomaly detection, weak supervision, topic models, deep generative toys, learning theory, transfer, skill
rating, survival models, RL policy agents, imbalanced learning, MBML models. Each registered with notes and, where it
earns one, a lab page or showcase.

**Wave 4: specialist core**

Spectral and audio extras, control design, vision, dimensionality-reduction factorisations, explainability and privacy,
quantisation.

**Alongside every wave:** reference fixtures for the thin areas (`timeseries` first, then trees and the inference
applications), and the open items of `.scratch/aifn/refinement.md`.
