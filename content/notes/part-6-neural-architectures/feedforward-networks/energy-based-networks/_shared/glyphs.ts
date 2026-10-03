/**
 * Capital letters as ±1 patterns on a 10 × 10 grid: a 5 × 5 pixel font, each pixel drawn as a 2 × 2 block. Letters
 * overlap heavily (O, D and U share most pixels), which is what makes them hard for a classical Hopfield network.
 */

export const GLYPH_SIZE = 10

const FONT: Record<string, string> = {
  A: '.###. #...# ##### #...# #...#',
  B: '####. #...# ####. #...# ####.',
  C: '.#### #.... #.... #.... .####',
  D: '####. #...# #...# #...# ####.',
  E: '##### #.... ####. #.... #####',
  F: '##### #.... ####. #.... #....',
  G: '.#### #.... #..## #...# .###.',
  H: '#...# #...# ##### #...# #...#',
  I: '##### ..#.. ..#.. ..#.. #####',
  J: '##### ...#. ...#. #..#. .##..',
  K: '#...# #..#. ###.. #..#. #...#',
  L: '#.... #.... #.... #.... #####',
  M: '#...# ##.## #.#.# #...# #...#',
  N: '#...# ##..# #.#.# #..## #...#',
  O: '.###. #...# #...# #...# .###.',
  P: '####. #...# ####. #.... #....',
  Q: '.###. #...# #.#.# #..#. .##.#',
  R: '####. #...# ####. #..#. #...#',
  S: '.#### #.... .###. ....# ####.',
  T: '##### ..#.. ..#.. ..#.. ..#..',
  U: '#...# #...# #...# #...# .###.',
  V: '#...# #...# #...# .#.#. ..#..',
  W: '#...# #...# #.#.# ##.## #...#',
  X: '#...# .#.#. ..#.. .#.#. #...#',
  Y: '#...# .#.#. ..#.. ..#.. ..#..',
  Z: '##### ...#. ..#.. .#... #####',
}

export const LETTERS = Object.keys(FONT)

const cache = new Map<string, Int8Array>()

/** The letter as a ±1 pattern of length 100, row-major: +1 is ink, −1 is background. */
export function glyph(letter: string): Int8Array {
  const hit = cache.get(letter)
  if (hit) return hit
  const rows = FONT[letter].split(' ')
  const out = new Int8Array(GLYPH_SIZE * GLYPH_SIZE)
  for (let r = 0; r < GLYPH_SIZE; r++)
    for (let c = 0; c < GLYPH_SIZE; c++) out[r * GLYPH_SIZE + c] = rows[r >> 1][c >> 1] === '#' ? 1 : -1
  cache.set(letter, out)
  return out
}
