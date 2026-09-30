# aifn-applied (aifn-js/applications)

Applications of the aifn core: named models, problems, environments, datasets and worked examples. What makes code an
application rather than core, the import rules and the promotion rule are in `../README.md`; the areas, their order
and the topics they serve are in `../modules.json`.

## Layout

- `src/<area>/index.ts` is an area's public surface, imported as `aifn-applied/<area>` (e.g. `aifn-applied/learning`).
- Until phase 1, an area keeps each former module as a subfolder (`src/learning/classify/`), re-exported whole from
  the area index. Code inside an area reaches a sibling subfolder as `aifn-applied/<area>/<sub>`.
- `test/<module>.test.ts` holds vitest tests (`make test`); `test/fixtures/<module>.json` holds golden values from
  `test/fixtures/gen/<module>.py`, written by `make fixtures`.
- Applications import any core module through `aifn/<module>` and other areas only down the area DAG. No React, no DOM.

## Style

The module contract of `../core/README.md` applies: readable before clever, cited, typed, numerical failure reported.
Applications are read by learners as the worked form of an idea, so they favour a plain loop over a clever one.
