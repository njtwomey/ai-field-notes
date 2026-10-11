# Part III (Data): plan

Status 2026-10-11: the taxonomy for Part III is in `content/taxonomy.yaml`, with 285 notes: 5 full notes, about 29
being drafted, and the rest placeholders (`status: stub`) holding a title, a summary and planned headings. A second
round (section 7) added three subjects (shift and drift; privacy, bias and security; data operations), document and
chemistry-and-biology chapters, routine exploration, test-set practice, labels and about 150 further placeholders.

## 1. What Part III owns

Part III owns **data handling**: how data are typed, sampled, documented, explored, cleaned, transformed, encoded,
featurised, split, labelled, augmented and selected, for every modality and domain; the **datasets** of every field;
and the **environments**.

Other parts keep what they already own, and Part III links to them rather than duplicating:

| Topic                                             | Stays in                          |
| ------------------------------------------------- | --------------------------------- |
| Spectral and audio features (STFT, mel, MFCC)     | Part II, signal processing        |
| Tokenisation, bag of words, TF-IDF, embeddings    | Part VII, NLP text representation |
| Image filtering, edges, pyramids                  | Part VII, computer vision         |
| Stationarity, differencing, decomposition, TS CV  | Part VII, time series             |
| Class imbalance, SMOTE, resampling                | Part V, imbalanced classes        |
| Weak supervision, crowdsourcing, label noise      | Part V and Part VIII              |
| Dataset shift, covariate and label shift          | Part V, domain adaptation         |
| Bootstrap, permutation tests, experimental design | Part I, statistics                |
| Drift detection, monitoring                       | Part VIII, deployment             |
| Cohen's and Fleiss' kappa                         | Part IV, metrics                  |

The test for where a note goes: if it is about turning raw data into model inputs, or about the data themselves, it is
Part III; if it is about a signal-processing or modelling method that happens to consume data, it stays where it is.

Candidate moves into Part III (URLs never break): `data-augmentation` (now under feedforward training) to
`data-centric-learning/augmentation-and-synthetic-data/augmentation-principles`; `spectrogram-augmentation` to
`data-by-modality/audio-data/audio-augmentation`; `feature-hashing-for-text` could stay in NLP with a link from
`encoding/categorical-encoding`. Owner's call.

## 2. Structure

Eleven subjects (section 7 lists the three added in the second round). Each subject has chapters; most chapters have sections. A section becomes a folder of notes; a
section's introducing note becomes its `index`.

1. **Data foundations**: types and scales; data setups (rows, groups, panels, sequences, bags, pairs, streams,
   censored records); sampling and selection bias; provenance, documentation and licensing; formats, storage and
   pipelines.
2. **Exploration and discovery**: profiling; visual exploration; dependence discovery; pattern mining; causal
   discovery; dataset search and data discovery.
3. **Data preparation**: data quality (missing data, errors and outliers, deduplication and record linkage,
   validation and contracts); scaling and transforms; encoding (categorical, numeric, cyclical and temporal,
   positional and coordinate, sets and multi-valued, missingness, targets); splitting and leakage (split strategies,
   kinds of leakage, detecting leakage).
4. **Features**: feature engineering (construction, temporal, relational, automated discovery, feature stores);
   feature selection (filter, wrapper, embedded, stability and inference).
5. **Data by modality**: tabular and relational; time-series and sequence (sampling and alignment, windowing,
   normalisation, features, forecasting setups, augmentation, event sequences, wearables); audio and speech; image;
   video; text and corpora; graph and network (with molecular data); geospatial; three-dimensional; multimodal and
   paired (synchronisation, cross-modal pairing, missing modalities, curation, fusion layouts).
6. **Data by domain**: clinical and health; financial; user interaction; scientific and simulation.
7. **Data-centric learning**: labelling and annotation; augmentation and synthetic data; data selection and
   valuation.
8. **Datasets** (by area, see section 4) and **Environments** (eight sections, including offline RL datasets and
   agent environments).

Two axes cross in subjects 3 and 5. Subject 3 holds the general technique, written for tables first (standardisation,
one-hot encoding, grouped splits). Subject 5 holds what changes for a modality (per-channel image normalisation,
instance normalisation of series, clip sampling). A modality note links back to the general note and never restates
it.

## 3. Planned notes

Slugs are proposals. Bold marks the introducing (index) note of its section. ✓ = exists. ⚙ = the engine already has
the method (`aifn-methods/learning/preprocessing` or `aifn-methods/data`), so the note can have a live figure now.

### 3.1 Data foundations

- Types and scales: **measurement-scales** (nominal, ordinal, interval, ratio, and the statistics each permits);
  compositional-data; counts-rates-and-exposure; circular-data.
- Data setups: **data-setups** (overview table of setups and the split each needs); grouped-and-clustered-data;
  panel-data; streams-and-online-data; multi-instance-and-set-data (link to MIL); censored-data (link to survival).
- Sampling and bias: **data-generating-process**; selection-bias; survivorship-bias; sampling-designs-for-ml
  (stratified, cluster, case-control); simpsons-paradox.
- Provenance: **dataset-documentation**; dataset-licences-and-consent; dataset-versioning;
  deprecated-and-withdrawn-datasets.
- Formats and pipelines: **data-formats** (CSV, Parquet, Arrow, HDF5, Zarr); sharded-record-formats (TFRecord,
  WebDataset); data-loading-pipelines (prefetch, shuffling buffers, streaming).

### 3.2 Exploration and discovery

- Profiling: **data-profiling**; missingness-patterns.
- Visual: **exploratory-data-analysis**; pair-plots-and-scatterplot-matrices; embedding-plots-for-exploration (link
  to t-SNE/UMAP).
- Dependence: **dependence-measures** (Pearson ✓ elsewhere, Spearman, MI); maximal-information-coefficient;
  distance-correlation; hilbert-schmidt-independence-criterion.
- Pattern mining: **pattern-mining** (Apriori, FP-growth); sequential-pattern-mining;
  subgroup-discovery.
- Causal discovery: **causal-discovery**; pc-algorithm; greedy-equivalence-search; lingam; notears.
- Dataset discovery: **dataset-search**; joinable-and-unionable-table-discovery.

### 3.3 Data preparation

- Missing data: **missing-data-and-imputation** ✓⚙; multiple-imputation-by-chained-equations;
  k-nearest-neighbour-imputation; missingness-in-time-series (link).
- Errors and outliers: **outliers-and-gross-errors**; winsorising-and-clipping.
- Deduplication: **deduplication-and-record-linkage** (Fellegi-Sunter); near-duplicate-detection (MinHash, SimHash; link to LSH);
  entity-resolution.
- Validation: **data-validation** (schemas, expectations); training-serving-skew.
- Scaling: **feature-scaling** ✓⚙; quantile-and-rank-transforms; per-sample-normalisation.
- Power transforms: **power-transforms** ⚙ (Box-Cox, Yeo-Johnson); log-and-logit-transforms;
  variance-stabilising-transforms (Anscombe, arcsine).
- Whitening: **whitening-and-decorrelation** ⚙ (PCA and ZCA whitening).
- Encoding: **categorical-encoding** ✓⚙; target-encoding ⚙ (with cross-fitting); high-cardinality-categoricals;
  entity-embeddings; **numeric-encoding** (binning ✓ in discretisation, piecewise-linear, periodic encodings);
  discretisation (equal-width, quantile, MDLP); **cyclical-encoding** (sine-cosine of hour, weekday, angle);
  **positional-and-coordinate-encoding** (links to RoPE ✓, Fourier features ✓; geohash, H3); **multi-valued-field-encoding**;
  **encoding-missingness**; **encoding-targets** (one-hot targets, label smoothing, ordinal targets, hierarchical and
  multi-label targets, transformed regression targets and back-transformation bias).
- Splitting and leakage: **train-validation-test-splits**; stratified-splits; grouped-splits; temporal-splits (link
  to TS CV ✓); spatial-blocking; **data-leakage** ✓; target-leakage; preprocessing-leakage ⚙ (`fitTransform`);
  temporal-look-ahead; duplicate-and-group-leakage; benchmark-contamination (link to text); **detecting-leakage**.

### 3.4 Features

- Construction: **feature-construction**; interaction-features; polynomial-features ⚙ (link to basis expansions ✓);
  group-aggregate-features.
- Temporal: **date-and-time-features**; holiday-and-event-features; time-since-event-features.
- Relational: **relational-feature-aggregation**; deep-feature-synthesis.
- Automated: **automated-feature-engineering** (DFS, OpenFE, genetic programming); learned-versus-engineered-features.
- Feature stores: **feature-stores**; point-in-time-joins.
- Selection: **feature-selection** ✓; mutual-information-feature-selection; minimum-redundancy-maximum-relevance;
  recursive-feature-elimination; lasso-selection (link); permutation-importance (link to attribution ✓);
  stability-selection; boruta; model-x-knockoffs.

### 3.5 Data by modality

- Tabular: **tabular-data** (what sets tables apart: mixed types, heterogeneous scales, no spatial structure);
  multi-table-schemas; panel-data-preparation; numeric-embeddings-for-tabular-models.
- Time series: **time-series-data-pipeline** (raw stream to model input); irregular-sampling; resampling-and-
  aggregation; multi-sensor-synchronisation; **time-series-windowing** (window length, stride, overlap, label per window,
  leakage between overlapping windows); event-based-segmentation; sessionisation; **series-normalisation** (global,
  per-series, per-window, reversible instance normalisation); **time-series-feature-extraction**; catch22-and-feature-
  libraries; random-convolutional-kernels (ROCKET); **forecasting-setups** (look-back, horizon, recursive, direct,
  multi-output; global and local models; static, known-future and observed covariates); **time-series-augmentation**;
  event-sequences; **wearable-sensor-data** (calibration, gravity removal, orientation, sensor fusion, the
  activity-recognition chain).
- Audio: **digital-audio**; resampling-audio; loudness-normalisation; voice-activity-detection;
  **audio-augmentation** (noise at set SNR, room impulse responses, speed and pitch perturbation; link to
  spectrogram augmentation ✓); forced-alignment.
- Images: **digital-images**; colour-spaces; **image-preprocessing** (resizing, interpolation, crops, per-channel
  normalisation); tiling-large-images; **image-augmentation**; mixup-and-cutmix; learned-augmentation-policies
  (AutoAugment, RandAugment); test-time-augmentation; **annotation-formats** (boxes, masks, keypoints; COCO, VOC,
  YOLO); medical-image-formats (DICOM, NIfTI); multispectral-images.
- Video: **digital-video**; **clip-sampling**; optical-flow (link); video-augmentation.
- Text: **text-cleaning**; language-identification; **corpus-construction** (crawl, filter, dedupe, scrub); quality-
  filtering; near-duplicate-removal; pii-removal; **benchmark-contamination**; **instruction-and-preference-data**;
  text-augmentation (back-translation, EDA).
- Graphs: **graph-construction** (k-NN and similarity graphs); **graph-features** (degree, centrality, Laplacian and
  random-walk positional encodings); **graph-splits** (transductive, inductive, link-prediction splits, negative
  sampling); neighbour-sampling; **molecular-representations** (SMILES, fingerprints, graphs, conformers).
- Geospatial: **coordinate-reference-systems**; vector-and-raster-data; spatial-indexes (geohash, H3, S2);
  spatial-autocorrelation-and-spatial-cross-validation; **remote-sensing-imagery** (bands, indices, cloud masks).
- 3D: **three-dimensional-data** (point clouds, meshes, voxels, depth); point-cloud-sampling-and-normalisation.
- Multimodal: **multimodal-data** (overview of the issues below); **temporal-synchronisation** (clock offsets,
  rates, latency; cross-correlation alignment; audio-video and sensor-label alignment); **cross-modal-pairing**
  (pair types, weak and noisy pairs, grounding); **missing-modalities**; modality-imbalance; **paired-data-curation**
  (CLIP-score filtering, concept balancing, dedup; link to CLIP score ✓); **fusion-data-layouts** (early,
  intermediate, late fusion as data arrangements; link to Part VII multimodal learning).

### 3.6 Data by domain

- Clinical: **clinical-and-health-data** (codes, cohorts, irregular visits); immortal-time-bias.
- Financial: **financial-data** (returns, adjustments, order books); look-ahead-and-survivorship-bias.
- Interaction: **interaction-logs** (impressions, implicit feedback, exposure bias; links to recsys debiasing ✓).
- Scientific: **scientific-data** (units, uncertainties, provenance); simulation-data.

### 3.7 Data-centric learning

- Labelling: **labelling-and-annotation** (guidelines, schemas, pilot rounds); annotator-agreement (link to kappa ✓);
  adjudication; labelling-strategies (link to active learning ✓, weak supervision ✓).
- Augmentation: **data-augmentation** (move candidate); consistency-and-invariance.
- Synthetic data: **synthetic-data**; synthetic-tabular-data (copulas, CTGAN); llm-generated-data and model collapse;
  simulation-and-domain-randomisation; privacy-preserving-synthetic-data (link to DP ✓).
- Selection and valuation: **data-valuation** (Data Shapley); influence-functions (link to leverage and influence ✓);
  coresets; data-pruning; dataset-distillation; curriculum-learning; data-mixtures; data-scaling-laws.

## 4. Datasets: one file per area now, one file per dataset later

The structure is two levels: area categories (`datasets/vision-datasets/image-classification-datasets`, …) and, later,
one note per dataset inside its area folder. Each area's introducing note (`<area>` with `index: true`) is the "file per
area": a catalogue table of that area's datasets with one paragraph on what the area's benchmarks measure and how they
have changed.

**Recommendation: keep dataset facts in a registry, `content/datasets.yaml`, like `references.yaml`.** One entry per
dataset, validated by `check:content`:

```yaml
mnist:
  name: MNIST
  area: image-classification-datasets # a datasets/* category id
  modality: [image]
  tasks: [classification]
  size: { examples: 70000, train: 60000, test: 10000, classes: 10, shape: [28, 28] }
  year: 1998
  paper: lecun1998 # key in references.yaml
  homepage: https://...
  download: https://... # or loaders: { torchvision: MNIST, huggingface: ylecun/mnist, openml: 554 }
  licence: CC BY-SA 3.0
  status: active # active | superseded | withdrawn
  caveats: Saturated; test error below 0.3% for most modern models.
  note: mnist # optional per-dataset note, later
  engine: digits # optional aifn-methods dataset key, when the engine bundles a copy or a generator
```

- Area notes render `<DatasetTable area="image-classification-datasets" />` from the registry, so adding a dataset is
  one YAML entry, and the table, the counts and a site-wide dataset index all follow.
- A per-dataset note (later, `kind: dataset`, a new kind with a template: description, collection, splits, known
  problems, typical results, uses) attaches with `dataset: <key>` in frontmatter and shows the registry facts in a
  header card. Datasets without a note still appear in tables and the index.
- URLs and licences are checked when entered, as references are.

Areas (34) and seed lists:

- Classic: Iris, Anscombe, Old Faithful, Galton heights, Titanic, airline passengers, Mauna Loa CO₂, sunspots,
  Canadian lynx, British coal-mining disasters (several already in `aifn-methods/data/real`).
- Toy and synthetic: moons, circles, blobs, XOR, spirals, Swiss roll, S-curve, Friedman #1 (all in
  `aifn-methods/data/synthetic`).
- Tabular: UCI classics, Adult, Covertype, Higgs, California housing, Ames housing, OpenML-CC18, Grinsztajn
  benchmark, TabArena.
- Fairness: Adult, COMPAS, German Credit, ACS (folktables), Law School.
- Causal and uplift: LaLonde, IHDP, Twins, ACIC, Criteo uplift, Hillstrom.
- Finance and economics: FRED, Kenneth French data library, FI-2010 limit order book.
- Image classification: MNIST, Fashion-MNIST, CIFAR-10/100, SVHN, ImageNet, Places365, iNaturalist.
- Detection and segmentation: PASCAL VOC, COCO, Cityscapes, ADE20K, LVIS, Open Images, SA-1B.
- Faces and humans: LFW, CelebA, FFHQ, MPII Human Pose, Human3.6M.
- Video: UCF101, HMDB51, Kinetics, Something-Something v2, AVA, EPIC-Kitchens, Ego4D.
- 3D and driving: ModelNet, ShapeNet, ScanNet, KITTI, nuScenes, Waymo Open.
- Image-text and multimodal: COCO Captions, Flickr30k, Conceptual Captions, LAION-5B, DataComp, VQA v2, Visual Genome.
- Speech: TIMIT, LibriSpeech, Common Voice, VoxCeleb, Speech Commands, LJSpeech.
- Sound and music: AudioSet, ESC-50, UrbanSound8K, FSD50K, GTZAN, MAESTRO, MusicNet, MUSDB18.
- NLP benchmarks: 20 Newsgroups, IMDB, SST, AG News, Penn Treebank, CoNLL-2003, SQuAD, SNLI, MNLI, GLUE, SuperGLUE,
  WMT.
- LM corpora: Common Crawl, C4, The Pile, RefinedWeb, Dolma, FineWeb.
- LM benchmarks: MMLU, HellaSwag, GSM8K, MATH, BIG-bench, GPQA, HumanEval, SWE-bench.
- Code: The Stack, CodeSearchNet, APPS, MBPP.
- Time-series classification: UCR and UEA archives.
- Forecasting: M1 to M6, Monash archive, ETT, Electricity, Traffic, Weather, ILI, LOTSA, GIFT-Eval.
- Time-series anomaly: NAB, Yahoo S5, SMD, SWaT, UCR anomaly archive, TSB-UAD.
- Activity recognition: UCI HAR, WISDM, PAMAP2, Opportunity, SPHERE, Capture-24, MHEALTH.
- Physiological signals: MIT-BIH, PTB-XL, PhysioNet challenges, Sleep-EDF, CHB-MIT, TUH EEG.
- Medical imaging: ChestX-ray14, CheXpert, MIMIC-CXR, BraTS, LIDC-IDRI, CAMELYON, ISIC.
- Clinical records: MIMIC-III and IV, eICU, UK Biobank, Synthea.
- Protein and genomics: PDB, CASP, UniProt, 1000 Genomes, TCGA, single-cell atlases.
- Graph benchmarks: Zachary karate club, Cora, CiteSeer, PubMed, OGB, TUDataset, PPI.
- Knowledge graphs: FB15k-237, WN18RR, YAGO, Wikidata.
- Recommender: MovieLens, Netflix Prize, Amazon reviews, Yelp, Last.fm, Criteo, MIND, KuaiRec.
- Retrieval: TREC, MS MARCO, Natural Questions, BEIR, MTEB.
- Remote sensing: EuroSAT, BigEarthNet, SpaceNet, xView, fMoW.
- Weather and climate: ERA5, WeatherBench, CMIP6, CAMELS.
- Chemistry and materials: QM9, ZINC, MoleculeNet, Materials Project, OC20.
- Physics and astronomy: PDEBench, The Well, LHC open data, Galaxy Zoo, SDSS.

Environments add sections for arcade games (ALE), procedurally generated (Procgen), multi-agent (PettingZoo), agent
environments (WebArena, ALFWorld) and offline RL datasets (D4RL, Minari, RL Unplugged).

## 5. Engine

Already in the engine with no or partial notes: power transforms, whitening, target encoding with cross-fitting,
spline and polynomial features, `fitTransform` (leakage-safe fitting), the 95-entry `datasetRegistry`, and the real
datasets in `data/real`. These notes can have live figures now.

Missing, to sketch for the owner when the notes are written (not before): sliding windows with stride and label
policies; resampling and alignment of irregular series; lag and rolling features; per-series and reversible instance
normalisation; time-series augmentations; discretisers (quantile, MDLP); grouped, temporal and spatial splitters with a
leakage check; near-duplicate detection (MinHash); adversarial validation; cyclical and periodic numeric encodings.

## 6. Order of work

1. Taxonomy placeholders (done).
2. Owner decisions: the dataset registry (section 4), the `dataset` kind, and the candidate moves (section 1).
3. Introducing notes for each section of subjects 1 to 5, as short `status: stub` notes, so every chapter has an entry
   point and the reading order fills in.
4. The high-value chapters in full, in this order: splitting and leakage; encoding; time-series windowing and
   forecasting setups; multimodal alignment; scaling and transforms; feature engineering.
5. Area catalogue notes from the registry, then per-dataset notes for the most used datasets.

## 7. Second round: gaps filled

Added from a review of the outline (all as placeholders):

- **Foundations:** unit of observation; target population; measurement error and proxies; censoring and coarsening;
  instrumentation and logging; data acquisition; observational and logged data; selective labels; feedback loops;
  learning curves and sample size.
- **Exploration:** a routine-checks section (reading raw examples, target exploration, temporal profiling, slice
  analysis, baseline probes, exploration hygiene), projections, clustering and embedding-based exploration, corpus
  statistics.
- **Preparation:** data quality dimensions; cleaning and evaluation; missingness mechanisms, deletion, single,
  multiple and learned imputation, MNAR sensitivity, missing values at prediction time; outlier detection methods;
  constraint-based repair; parsing and type inference; value standardisation; near-duplicate detection; quantile and
  rank transforms; row normalisation; variance-stabilising transforms; compositional data; retransformation bias;
  one-hot and contrasts, mean target encoding, count encoding, feature hashing, category embeddings, high cardinality,
  identifier features; structured target encodings; sequence length and chunking; sample weights, imbalanced
  regression, long-tailed recognition; preprocessing pipelines and preprocessing by model family; temporal splits,
  matching the split to deployment, stable split assignment, point-in-time correctness, leakage case studies,
  adversarial validation; test-set design, reuse and out-of-distribution test sets.
- **Features:** domain-knowledge features, feature crosses, hand-built descriptors, pretrained embeddings as features,
  dimensionality reduction as preprocessing, importance measures, selection inside cross-validation.
- **Shift and drift** (new subject): an overview linking Part V and VIII, subpopulation shift and shortcuts, training
  window and freshness.
- **Modalities:** table structure (tidy tables, semi-structured data, text fields), time-to-event data, tabular
  augmentation; time-series gaps, look-ahead bias, intermittent series, physiological signals; transcript
  normalisation, symbolic music; image quality screening and dataset bias; video annotation; multilingual text,
  retrieval corpora, code data, dialogue and agent traces, span annotation; a **Documents** chapter (parsing, OCR and
  layout, log parsing); knowledge-graph data, heterogeneous and temporal graphs, graph augmentation; trajectories,
  areal aggregation; a **Chemistry and biology** chapter (molecular representations moved in, biological sequences,
  omics, scaffold splits); modality tokenisation.
- **Domains:** surveys, fraud, advertising, retail demand, industrial maintenance, search relevance, user-generated
  content, driving and robotics.
- **Data-centric learning:** label definition, annotator agreement and aggregation, annotation operations,
  model-generated labels, delayed and partial labels, self-supervised data design, evaluating synthetic data; an
  error-analysis chapter (error analysis, training dynamics, hard-example mining, targeted collection).
- **Privacy, bias and security** (new subject): identifiability, PII redaction, k-anonymity, training-data
  extraction, data rights and deletion; dataset bias, protected attributes and proxies, content filtering; data
  poisoning.
- **Data operations** (new subject): formats, storage and loading (moved from foundations), loading and shuffling,
  large-scale processing; versioning and lineage, schema evolution, batch and stream processing; data tests, data
  monitoring; training-serving skew, feature freshness.
- **Datasets:** survival, robot learning, document understanding, instruction and dialogue, multilingual, mobility;
  a datasets-for-data-problems chapter (distribution shift, noisy labels, anomaly detection, known dataset defects).

Not added, because other Parts hold them (Part III notes link there): class imbalance, resampling, SMOTE, class
weights, threshold moving and prior correction (Part V); label noise and finding label errors (Part VIII); weak
supervision, crowdsourcing, Dawid-Skene, positive-unlabelled learning, self-training and active learning (Part V);
dataset shift, covariate and label shift, concept drift and continual learning (Part V); drift detection and
monitoring, differential privacy, federated learning and membership inference (Part VIII); tokenisation, bag of words
and TF-IDF (Part VII); spectrograms, mel features and MFCCs (Part II); stationarity, decomposition, change points and
hierarchical reconciliation (Part VII); leverage and influence; multicollinearity.
