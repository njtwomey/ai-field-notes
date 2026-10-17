import type { Triple } from 'aifn-compute/logic/induction'

/**
 * A small knowledge graph of eight people, four cities and two countries. It is incomplete on purpose: Fay has no
 * home and no citizenship, Gus has no citizenship, and Fay's marriage is recorded in one direction only. Dia is a French
 * citizen living in Rome, and Hal lives in Milan, away from his parents.
 */
export const PEOPLE_AND_PLACES: readonly Triple[] = [
  ['ann', 'marriedTo', 'ben'],
  ['ben', 'marriedTo', 'ann'],
  ['cal', 'marriedTo', 'dia'],
  ['dia', 'marriedTo', 'cal'],
  ['eli', 'marriedTo', 'fay'],
  ['ann', 'hasChild', 'gus'],
  ['ben', 'hasChild', 'gus'],
  ['cal', 'hasChild', 'hal'],
  ['dia', 'hasChild', 'hal'],
  ['ann', 'livesIn', 'paris'],
  ['ben', 'livesIn', 'paris'],
  ['cal', 'livesIn', 'rome'],
  ['dia', 'livesIn', 'rome'],
  ['eli', 'livesIn', 'lyon'],
  ['gus', 'livesIn', 'paris'],
  ['hal', 'livesIn', 'milan'],
  ['paris', 'locatedIn', 'france'],
  ['lyon', 'locatedIn', 'france'],
  ['rome', 'locatedIn', 'italy'],
  ['milan', 'locatedIn', 'italy'],
  ['ann', 'citizenOf', 'france'],
  ['ben', 'citizenOf', 'france'],
  ['cal', 'citizenOf', 'italy'],
  ['dia', 'citizenOf', 'france'],
  ['eli', 'citizenOf', 'france'],
  ['hal', 'citizenOf', 'italy'],
]
