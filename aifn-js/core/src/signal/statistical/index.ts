/**
 * `aifn/signal/statistical`: statistical signal processing. Autoregressive (AR) estimation by the Yule–Walker
 * equations (`yuleWalker`, through `aifn/numerics/linalg`'s `levinsonDurbin`) and by Burg's method (`burg`).
 */

export { burg, yuleWalker, type AutoregressiveFit } from './autoregression'
