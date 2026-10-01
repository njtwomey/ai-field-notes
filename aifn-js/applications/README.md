# aifn-applied (aifn-js/applications)

Applications of the aifn core: named models, problems, environments, datasets and worked examples. What makes code an
application rather than core, the import rules and the promotion rule are in `../README.md`; the areas, their groups,
their order and the topics they serve are in `../modules.json` (and the generated "Areas" table in `../README.md`).

## Layout

- `src/<area>/index.ts` is an area's public surface, imported as `aifn-applied/<area>` (e.g. `aifn-applied/learning`).
  Areas nest groups and modules (`src/learning/generalised/ordinal/`), imported by path
  (`aifn-applied/learning/generalised/ordinal`). A group's root files are its shared layer; siblings share only
  through the parent.
- `test/<area>/…/*.test.ts` mirrors the source tree. Root files: `test/names.test.ts` (the name-collision lint across
  both packages, `make aifn-names`), `test/registry.ts` and `test/model-fixtures.ts` (the model protocol test fits
  every registered model on tiny data), `test/protocol.ts`.
- `test/fixtures/<area>/…json` holds golden values written by the Python scripts in `test/fixtures/gen/` (run by
  `make fixtures`, which covers both packages); `test/fixtures.ts` loads them.
- Applications import any core node through `aifn/<family>/<module>`, their ancestors' shared files by relative path,
  and other areas only down the area DAG. No React, no DOM.

## Registries

Applications register their named variants with core's `define`/`entries` (`aifn/foundation/registry`), as core does,
and `make catalog` collects them into `../generated/catalog.json` with addresses prefixed `applied/`.

| Kind          | Table                                                                                             | Count | Kind-specific info                                                                                                           |
| ------------- | ------------------------------------------------------------------------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------- |
| `model`       | `learningModelRegistry` (`aifn-applied/learning`), `unsupervisedModelRegistry` (`…/unsupervised`) | 66    | `task`, declared `capabilities` (`decide`, `scores`, `predictive`, `expect`, `transform`, …), `hyper: Space`, `transductive` |
| `dataset`     | `datasetRegistry` (`aifn-applied/data`), `fontDatasetRegistry` (`aifn-applied/data/real/fonts`)   | 36    | `task`, `knobs: Space`, `truth` (a known generating model), `output`                                                         |
| `modifier`    | `modifierRegistry` (`aifn-applied/data`)                                                          | 9     | `params: Space`, `needs: 'labels'` where it applies                                                                          |
| `environment` | `environmentRegistry` (`aifn-applied/data`)                                                       | 7     | `family` (`bandit` or `mdp`), `params: Space`                                                                                |
| `objective`   | `objectiveRegistry` (`aifn-applied/data`): optimisation test surfaces                             | 5     | `params: Space`, `dim`, `truth` (known minimisers)                                                                           |
| `log-density` | `logDensityRegistry` (`aifn-applied/data`): MCMC and VI targets                                   | 5     | `params: Space`, `dim`, `truth` (a known reference)                                                                          |

Models are built with core's `defineModel` (`aifn/learning/estimators`); the protocol test checks each one's
capabilities and `transductive` flag against its declaration. Datasets and modifiers are replayed by **recipes**: a
recipe names a base dataset and a list of modifiers with their parameters; `recipe` (in `aifn-applied/data`) builds
it, `meta.recipe` holds the normalised recipe that rebuilds the data, `meta.ignored` lists the knobs and steps that did
not apply, and `recipeSpace` is the `Space` of every base's knobs (for the lab's controls).

## Running

`make test` runs both packages; one area with `npx vitest run --config aifn-js/applications/vitest.config.ts
test/learning`. `make catalog` rebuilds the catalog after a registry change, and `make catalog-check` (in `make
check`) fails when it is stale or a `notes`, `glossary` or `cite` key does not exist.

## Style

The module contract of `../core/README.md` applies: readable before clever, cited, typed, numerical failure reported.
Applications are read by learners as the worked form of an idea, so they favour a plain loop over a clever one.
