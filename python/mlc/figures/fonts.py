"""Glyph outlines of real fonts, in dense correspondence, for the manifold-of-fonts note.

The fonts come from the Google Fonts repository (SIL Open Font License or Apache 2.0), pinned to one commit and
checked by sha256. Binaries are downloaded into ``.cache/fonts/`` at the repository root, which git ignores. Variable
fonts are instanced at several weights, widths and italics with ``fontTools.varLib.instancer``, and overlapping
contours are merged so that every glyph is a set of simple closed outlines.

Correspondence, per character and per contour:

1. Flatten the Bézier outline to a polyline; scale every font to the same cap height.
2. Orient outer contours counter-clockwise and counters (holes) clockwise; order counters by matching their centroids
   to a reference font's.
3. Resample densely by arc length and pick a canonical start: the reference's lower-left point, and for every other
   font the cyclic shift that best matches the reference.
4. Align groupwise by dynamic time warping against a template, coarse to fine: the curves are smoothed along their
   length, matched on position and unit tangent with a penalty on uneven stretching, then matched again with less
   smoothing inside a band around the coarse path. The template is re-estimated as the mean warped outline and the
   alignment repeated. Italics are de-slanted for matching only.
5. Choose the output samples along the template: one at every place where several fonts have a corner, the rest
   spread along the outline, denser on curves. Each font's own corner then moves onto its corner sample, so that
   stems, serifs and vertices land on the same samples in every font.

Fonts whose contour counts for a character differ from the majority are dropped and listed in the output.
"""

import hashlib
import itertools
import math
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from fontTools.ttLib.removeOverlaps import removeOverlaps
from fontTools.varLib import instancer
from pydantic import Field

from mlc.core.contracts import Strict
from mlc.core.figures import figure
from mlc.core.paths import REPO_ROOT

REPO = "https://raw.githubusercontent.com/google/fonts"
COMMIT = "23e54b51ddffbc7713c583748e3bd86f62b1fa4a"
CACHE = REPO_ROOT / ".cache" / "fonts"

CAP_HEIGHT = 700
"""Every font is scaled so that the top of its "H" is at this height, in output units."""

DISPLAY = "GPLVM"
TRAIN = "ABCDEFGHIJLMNOPRSTUVXYZ"
"""Capitals used to fit the manifold. K, Q and W are left out: their contour counts or shapes vary between designs
(a split K, a free Q tail, a crossed W)."""

type FontClass = Literal["sans", "humanist", "serif", "didone", "slab", "mono"]
CLASSES: tuple[FontClass, ...] = ("sans", "humanist", "serif", "didone", "slab", "mono")

FILES: dict[str, str] = {
    "ofl/roboto/Roboto[wdth,wght].ttf": "d7598e12c5dbef095ff8272cfc55da0250bd07fbdecbac8a530b9b277872a134",
    "ofl/roboto/Roboto-Italic[wdth,wght].ttf": "9725a847af6b460ffca162ae66d20dad48b01876137947180b42d7dcd7887182",
    "ofl/inter/Inter[opsz,wght].ttf": "29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031",
    "ofl/montserrat/Montserrat[wght].ttf": "0f7b311b2f3279e4eef9b2f968bcdbab6e28f4daeb1f049f4f278a902bcd82f7",
    "ofl/montserrat/Montserrat-Italic[wght].ttf": "51607f316bc020e59f03cbf51543eecffbea501c0b31d73e5b82927c5cca442c",
    "ofl/opensans/OpenSans[wdth,wght].ttf": "36643644f318a812aab2d2ed3bb98f8cf0872527f835fe9398d95fe6b9adb878",
    "ofl/opensans/OpenSans-Italic[wdth,wght].ttf": "fe269381e992f32e135801740998544d6235061e37c93ec067ad2be3edd5b17b",
    "ofl/sourcesans3/SourceSans3[wght].ttf": "042fe2cc0b933e328410d7acbd0aa6a1873dca5aef81875f4bc214b08825c7b9",
    "ofl/firasans/FiraSans-Regular.ttf": "c29556a2719bf613ef3d5e070e40d903a8965d9c081beca1375dc1e6e0f93c23",
    "ofl/firasans/FiraSans-Bold.ttf": "a4d8e149ecdd4874a0726eb0af894488b3b31c423d6b0017c8f415ed1b795b45",
    "ofl/firasans/FiraSans-Italic.ttf": "a4204f9a8029c1c55daa5385ac95936f74abea3e232770f1fb2862a19d1dfa13",
    "ofl/ebgaramond/EBGaramond[wght].ttf": "ef9512f92f6d579e5dc75af59a5a4b1b8b47d2eda89e00b954d44520e5369027",
    "ofl/sourceserif4/SourceSerif4[opsz,wght].ttf": "97b2d4da6e3cb494b5a1e66ae176914d852ccabef49e0c02c0df25f3e39aca0b",
    "ofl/sourceserif4/SourceSerif4-Italic[opsz,wght].ttf": (
        "15fbc7e4679489a501998c3669272637a6646388ef7e4bd77eebb5bf967a1f42"
    ),
    "ofl/librebaskerville/LibreBaskerville[wght].ttf": (
        "05a95421961341c5b2556285e8415df9db27dab4f4abe22b446b3c6a8b916c5d"
    ),
    "ofl/librebaskerville/LibreBaskerville-Italic[wght].ttf": (
        "223959683dc73ec4437bd61fabaa4b3f22209e22855ffd3aee36ba61a5116e97"
    ),
    "ofl/crimsonpro/CrimsonPro[wght].ttf": "16aa9fb7300a93637da51fac03a071b2ff08b6bbf65f99c794c25f040b58af6a",
    "ofl/lora/Lora[wght].ttf": "822a6621ccbe8d97d20ac88c1c41f5615c9c2c202eaa75f272cd452aac6475a7",
    "ofl/lora/Lora-Italic[wght].ttf": "22d8d8854b53807aa664ca34f2031a9ed57a1d0dea296b8b96cdd3aad937a2b3",
    "ofl/playfairdisplay/PlayfairDisplay[wght].ttf": "c40f2293766a503bc70cce9e512ef844a4ccb7cbcde792fe2ea31d191917d8d6",
    "ofl/playfairdisplay/PlayfairDisplay-Italic[wght].ttf": (
        "a5e26dc5e2e77fb2803a0bf02fd4f81ee136ec8dea863ccdb0c59a263b21378b"
    ),
    "ofl/bodonimoda/BodoniModa[opsz,wght].ttf": "550f5e34ee0a828d7941b1fe9bc58b34e5260d3f33a61532e6d0a0114e79a5cf",
    "ofl/abrilfatface/AbrilFatface-Regular.ttf": "5971d4a3758a922a9fedc7f6fb825a96341a2e718c45a4b2c9a6b417c8c4dbe9",
    "ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf": "8cc3643535edf039aa5d95440a8542735e9197e4f4b8d9303e980fefbf5ab616",
    "ofl/dmserifdisplay/DMSerifDisplay-Italic.ttf": "df74c0ac387baeaeb0fe4f2324e1668e6a3ed8c09cd9796fe162c71753e19e45",
    "apache/robotoslab/RobotoSlab[wght].ttf": "786ae192477447d33c6672c3055fba7cbfe45184c9a79e77a14f15716ca05b16",
    "ofl/arvo/Arvo-Regular.ttf": "f41bd41471ec2db7140351bdde614da5341524503598ff7fe79f3c89c13b605e",
    "ofl/arvo/Arvo-Bold.ttf": "6239b2edee762db0ab99343137c9ba15ae81fc843da2e76a0a395781748cc21f",
    "ofl/zillaslab/ZillaSlab-Regular.ttf": "41a4626844da9216b031308d4423045c765fe4231d99862d85d7d74509d37703",
    "ofl/zillaslab/ZillaSlab-Bold.ttf": "4ec3a04a4eef37074b42ef542e4d874e13646668cfe65256e0bf100441cf8719",
    "ofl/bitter/Bitter[wght].ttf": "ef2b9a711fb02f1e5823b34da1b7450e0fc76793b7d733a8b41006e24916d4a7",
    "ofl/robotomono/RobotoMono[wght].ttf": "66a80e79d17e4c7cabd162e2916578a4cc08fd19eef6e2a643305eae9c567b2b",
    "ofl/jetbrainsmono/JetBrainsMono[wght].ttf": "48715a42ec242c21e9f02692891e147d022299a52e48d5e413e1a942193ffeda",
    "ofl/ibmplexmono/IBMPlexMono-Regular.ttf": "6a3412f058c7d8dfd9170c41e85ade48e5156ecb89356110ca57a0a27734af46",
    "ofl/ibmplexmono/IBMPlexMono-Bold.ttf": "ac27abd6450a64dd94467580a02fe6235156d5b92f2926ebbc8e7489df64e0be",
    "ofl/sourcecodepro/SourceCodePro[wght].ttf": "b400fc584e10aff25d0e775ce181b4fc1c5ea1b5dc37b81aeb2084375b945790",
    "ofl/courierprime/CourierPrime-Regular.ttf": "72f793376f8e2841656bf21d77a5de010f2929bd6956a22ee848ad0c7eb978af",
    "ofl/courierprime/CourierPrime-Bold.ttf": "ff1f38786c849d1c41fa8e447960abdb2bd75fdfb0cfcdeb524fad65a5af3638",
}


@dataclass(frozen=True)
class Spec:
    family: str
    cls: FontClass
    file: str
    weight: int = 400
    width: int = 100
    italic: bool = False
    axes: dict[str, float] = field(default_factory=dict[str, float])
    """Extra axis settings, e.g. optical size. ``wght`` and ``wdth`` come from ``weight`` and ``width``."""


def _family(
    family: str, cls: FontClass, stem: str, weights: list[int], italics: tuple[int, ...] = (), **axes: float
) -> list[Spec]:
    """Instances of one variable font from ``ofl/<family>/``, upright and italic."""
    upright = f"ofl/{family.lower().replace(' ', '')}/{stem}"
    italic = upright.replace("[", "-Italic[")
    return [Spec(family, cls, upright, w, axes=axes) for w in weights] + [
        Spec(family, cls, italic, w, italic=True, axes=axes) for w in italics
    ]


SPECS: list[Spec] = [
    *_family("Roboto", "sans", "Roboto[wdth,wght].ttf", [300, 500, 900], (400,)),
    Spec("Roboto", "sans", "ofl/roboto/Roboto[wdth,wght].ttf", 400, width=75),
    *_family("Inter", "sans", "Inter[opsz,wght].ttf", [400, 800]),
    *_family("Montserrat", "sans", "Montserrat[wght].ttf", [300, 700], (500,)),
    *_family("Open Sans", "humanist", "OpenSans[wdth,wght].ttf", [300, 700], (400,)),
    Spec("Open Sans", "humanist", "ofl/opensans/OpenSans[wdth,wght].ttf", 600, width=75),
    *_family("Source Sans 3", "humanist", "SourceSans3[wght].ttf", [300, 600, 900]),
    Spec("Fira Sans", "humanist", "ofl/firasans/FiraSans-Regular.ttf", 400),
    Spec("Fira Sans", "humanist", "ofl/firasans/FiraSans-Bold.ttf", 700),
    Spec("Fira Sans", "humanist", "ofl/firasans/FiraSans-Italic.ttf", 400, italic=True),
    *_family("EB Garamond", "serif", "EBGaramond[wght].ttf", [400]),
    *_family("Source Serif 4", "serif", "SourceSerif4[opsz,wght].ttf", [300, 500, 800], (400,), opsz=20),
    *_family("Libre Baskerville", "serif", "LibreBaskerville[wght].ttf", [400, 700], (400,)),
    *_family("Crimson Pro", "serif", "CrimsonPro[wght].ttf", [300, 600, 900]),
    *_family("Lora", "serif", "Lora[wght].ttf", [400, 700], (500,)),
    *_family(
        "Playfair Display",
        "didone",
        "PlayfairDisplay[wght].ttf",
        [400, 700, 900],
        (
            400,
            900,
        ),
    ),
    *_family("Bodoni Moda", "didone", "BodoniModa[opsz,wght].ttf", [400, 700, 900], opsz=28),
    Spec("Abril Fatface", "didone", "ofl/abrilfatface/AbrilFatface-Regular.ttf", 400),
    Spec("DM Serif Display", "didone", "ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf", 400),
    Spec("DM Serif Display", "didone", "ofl/dmserifdisplay/DMSerifDisplay-Italic.ttf", 400, italic=True),
    Spec("Roboto Slab", "slab", "apache/robotoslab/RobotoSlab[wght].ttf", 200),
    Spec("Roboto Slab", "slab", "apache/robotoslab/RobotoSlab[wght].ttf", 400),
    Spec("Roboto Slab", "slab", "apache/robotoslab/RobotoSlab[wght].ttf", 700),
    Spec("Roboto Slab", "slab", "apache/robotoslab/RobotoSlab[wght].ttf", 900),
    Spec("Arvo", "slab", "ofl/arvo/Arvo-Regular.ttf", 400),
    Spec("Arvo", "slab", "ofl/arvo/Arvo-Bold.ttf", 700),
    Spec("Zilla Slab", "slab", "ofl/zillaslab/ZillaSlab-Regular.ttf", 400),
    Spec("Zilla Slab", "slab", "ofl/zillaslab/ZillaSlab-Bold.ttf", 700),
    *_family("Bitter", "slab", "Bitter[wght].ttf", [300, 600, 900]),
    *_family("Roboto Mono", "mono", "RobotoMono[wght].ttf", [300, 500, 700]),
    *_family("JetBrains Mono", "mono", "JetBrainsMono[wght].ttf", [200, 800]),
    Spec("IBM Plex Mono", "mono", "ofl/ibmplexmono/IBMPlexMono-Regular.ttf", 400),
    Spec("IBM Plex Mono", "mono", "ofl/ibmplexmono/IBMPlexMono-Bold.ttf", 700),
    *_family("Source Code Pro", "mono", "SourceCodePro[wght].ttf", [300, 900]),
    Spec("Courier Prime", "mono", "ofl/courierprime/CourierPrime-Regular.ttf", 400),
    Spec("Courier Prime", "mono", "ofl/courierprime/CourierPrime-Bold.ttf", 700),
]

WEIGHT_NAMES = {
    100: "Thin",
    200: "ExtraLight",
    300: "Light",
    400: "Regular",
    500: "Medium",
    600: "SemiBold",
    700: "Bold",
    800: "ExtraBold",
    900: "Black",
}


def _style(s: Spec) -> str:
    words = [] if s.width == 100 else ["Condensed"]
    if s.weight != 400 or not s.italic:
        words.append(WEIGHT_NAMES[s.weight])
    if s.italic:
        words.append("Italic")
    return " ".join(words)


# --------------------------------------------------------------------------------------------------------------------
# Font files and outlines
# --------------------------------------------------------------------------------------------------------------------


def _fetch(path: str) -> Path:
    target = CACHE / Path(path).name
    if not target.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        url = f"{REPO}/{COMMIT}/{urllib.parse.quote(path)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != FILES[path]:
        target.unlink()
        raise ValueError(f"{path}: sha256 {digest} does not match the pinned {FILES[path]}")
    return target


def _load(s: Spec) -> TTFont:
    font = TTFont(_fetch(s.file))
    if "fvar" in font:
        location = {a.axisTag: a.defaultValue for a in font["fvar"].axes}  # pyright: ignore[reportAttributeAccessIssue]
        location.update(s.axes)
        if "wght" in location:
            location["wght"] = s.weight
        if "wdth" in location:
            location["wdth"] = s.width
        return instancer.instantiateVariableFont(font, location, overlap=instancer.OverlapMode.REMOVE)
    removeOverlaps(font)
    return font


class _FlattenPen(BasePen):
    """Records closed polylines, splitting quadratic and cubic Béziers into short line segments."""

    STEPS = 16

    def __init__(self, glyph_set: object) -> None:
        super().__init__(glyph_set)  # pyright: ignore[reportArgumentType]
        self.contours: list[list[tuple[float, float]]] = []
        self._current: list[tuple[float, float]] = []

    def _moveTo(self, pt: tuple[float, float]) -> None:
        self._current = [pt]

    def _lineTo(self, pt: tuple[float, float]) -> None:
        self._current.append(pt)

    def _curveToOne(self, pt1: tuple[float, float], pt2: tuple[float, float], pt3: tuple[float, float]) -> None:
        p0 = np.array(self._current[-1])
        p1, p2, p3 = np.array(pt1), np.array(pt2), np.array(pt3)
        for t in np.linspace(0, 1, self.STEPS + 1)[1:]:
            u = 1 - t
            p = u**3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t**3 * p3
            self._current.append((float(p[0]), float(p[1])))

    def _qCurveToOne(self, pt1: tuple[float, float], pt2: tuple[float, float]) -> None:
        p0 = np.array(self._current[-1])
        p1, p2 = np.array(pt1), np.array(pt2)
        for t in np.linspace(0, 1, self.STEPS + 1)[1:]:
            u = 1 - t
            p = u * u * p0 + 2 * u * t * p1 + t * t * p2
            self._current.append((float(p[0]), float(p[1])))

    def _closePath(self) -> None:
        self.contours.append(self._current)
        self._current = []

    _endPath = _closePath


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _inside(point: np.ndarray, poly: np.ndarray) -> bool:
    x, y = point
    xi, yi = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
    return bool(np.count_nonzero(crosses) % 2)


@dataclass
class Outline:
    """One glyph of one font: closed polylines (outer contours counter-clockwise, counters clockwise) in output
    units, the advance width, and the font's italic angle in degrees (negative leans right)."""

    outers: list[np.ndarray]
    holes: list[np.ndarray]
    advance: float
    slant: float


def _outline(font: TTFont, char: str, scale: float, slant: float) -> Outline:
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    assert cmap is not None, "every font here has a Unicode cmap"
    name = cmap[ord(char)]
    pen = _FlattenPen(glyph_set)
    glyph_set[name].draw(pen)
    polys: list[np.ndarray] = []
    for c in pen.contours:
        p = np.array(c, dtype=float) * scale
        keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]
        p = p[keep]
        if np.linalg.norm(p[0] - p[-1]) < 1e-6:
            p = p[:-1]
        if len(p) >= 3 and abs(_signed_area(p)) > 1.0:
            polys.append(p)
    outers: list[np.ndarray] = []
    holes: list[np.ndarray] = []
    for i, p in enumerate(polys):
        depth = sum(_inside(p[0], q) for j, q in enumerate(polys) if j != i)
        hole = depth % 2 == 1
        if (_signed_area(p) > 0) == hole:
            p = p[::-1]
        (holes if hole else outers).append(p)
    advance = float(font["hmtx"][name][0]) * scale  # pyright: ignore[reportIndexIssue]
    return Outline(outers, holes, advance, slant)


def _outlines(s: Spec) -> dict[str, Outline]:
    font = _load(s)
    slant = float(font["post"].italicAngle)  # pyright: ignore[reportAttributeAccessIssue]
    probe = _outline(font, "H", 1.0, slant)
    cap = max(float(p[:, 1].max()) for p in probe.outers)
    scale = CAP_HEIGHT / cap
    return {c: _outline(font, c, scale, slant) for c in TRAIN}


# --------------------------------------------------------------------------------------------------------------------
# Correspondence
# --------------------------------------------------------------------------------------------------------------------

DENSE = 400
"""Samples per contour for matching."""


class Curve:
    """A closed polyline parameterised by arc-length fraction t in [0, 1), measured from a chosen start."""

    def __init__(self, points: np.ndarray, start: float = 0.0) -> None:
        self.points = points
        seg = np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1)
        self.length = float(seg.sum())
        self.cum = np.r_[0.0, np.cumsum(seg)] / self.length
        self.start = start

    def at(self, t: np.ndarray) -> np.ndarray:
        u = (np.asarray(t) + self.start) % 1.0
        closed = np.vstack([self.points, self.points[:1]])
        return np.stack([np.interp(u, self.cum, closed[:, 0]), np.interp(u, self.cum, closed[:, 1])], axis=-1)

    def dense(self, n: int = DENSE) -> np.ndarray:
        return self.at(np.arange(n) / n)

    def corners(self, threshold_deg: float = 28.0) -> tuple[np.ndarray, np.ndarray]:
        """Vertices where the outline turns by more than the threshold, as (t from the start, turning angle)."""
        p = self.points
        a = p - np.roll(p, 1, axis=0)
        b = np.roll(p, -1, axis=0) - p
        turn = np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(axis=1)))
        idx = np.nonzero(turn > math.radians(threshold_deg))[0]
        return (self.cum[idx] - self.start) % 1.0, turn[idx]


def _features(points: np.ndarray, sigma: float, w_pos: float = 2.0) -> np.ndarray:
    """Position and unit tangent of a closed, uniformly sampled curve smoothed by a circular Gaussian of width sigma
    (a fraction of its length)."""
    n = len(points)
    k = np.fft.fftfreq(n)
    smooth = np.real(
        np.fft.ifft(np.fft.fft(points, axis=0) * np.exp(-2 * (math.pi * k * sigma * n) ** 2)[:, None], axis=0)
    )
    tangent = np.roll(smooth, -1, axis=0) - np.roll(smooth, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12
    return np.hstack([w_pos * smooth, tangent])


def _dtw(a: np.ndarray, b: np.ndarray, penalty: float, band: np.ndarray | None = None) -> np.ndarray:
    """Dynamic time warping of feature sequences a (template) and b, both starting and ending together. Diagonal steps
    cost the feature distance; horizontal and vertical steps also pay ``penalty``, which resists uneven stretching.
    ``band`` optionally gives, per row of a, the centre column of an allowed window. Returns, for each row of a, the
    mean matched column of b."""
    n, m = len(a), len(b)
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    if band is not None:
        cols = np.arange(m)[None, :]
        cost = np.where(np.abs(cols - band[:, None]) <= 12, cost, 1e6)
    acc = np.empty((n, m))
    step = cost + penalty
    acc[0] = cost[0, 0] + np.r_[0.0, np.cumsum(step[0, 1:])]
    for i in range(1, n):
        prev = acc[i - 1]
        entry = cost[i] + np.minimum(np.r_[np.inf, prev[:-1]], prev + penalty)
        s = np.cumsum(step[i])
        acc[i] = s + np.minimum.accumulate(entry - s)
    # Backtrack.
    i, j = n - 1, m - 1
    sums = np.zeros(n)
    counts = np.zeros(n)
    while True:
        sums[i] += j
        counts[i] += 1
        if i == 0 and j == 0:
            break
        options = []
        if i > 0 and j > 0:
            options.append((acc[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            options.append((acc[i - 1, j] + penalty, i - 1, j))
        if j > 0:
            options.append((acc[i, j - 1] + penalty, i, j - 1))
        _, i, j = min(options)
    return sums / counts


def _deslant(p: np.ndarray, slant: float, centre: np.ndarray) -> np.ndarray:
    q = p - centre
    return np.stack([q[:, 0] + q[:, 1] * math.tan(math.radians(slant)), q[:, 1]], axis=1) / CAP_HEIGHT


def _best_shift(ref: np.ndarray, x: np.ndarray) -> int:
    """Cyclic shift s minimising sum ||ref[i] - x[i + s]||², by FFT cross-correlation of complex coordinates."""
    za = ref[:, 0] + 1j * ref[:, 1]
    zb = x[:, 0] + 1j * x[:, 1]
    corr = np.fft.ifft(np.conj(np.fft.fft(za)) * np.fft.fft(zb))
    return int(np.argmax(corr.real))


def _correspond(curves: list[Curve], feats_of: list[np.ndarray], samples: int, ref: int) -> list[np.ndarray]:
    """Put one contour of every font into correspondence. ``feats_of[f]`` is font f's dense curve in de-slanted,
    centred matching coordinates (same sampling as ``curves[f].dense()``). Returns each font's output points."""
    nf = len(curves)
    # Canonical start: the reference's lower-left point; every other font takes its best cyclic shift.
    r = feats_of[ref]
    start = int(np.argmin(r[:, 1] + 0.3 * r[:, 0]))
    shift = [0] * nf
    shift[ref] = start
    base = np.roll(r, -start, axis=0)
    for f in range(nf):
        if f != ref:
            shift[f] = _best_shift(base, feats_of[f])
    dense = [np.roll(feats_of[f], -shift[f], axis=0) for f in range(nf)]
    for f in range(nf):
        curves[f].start = (curves[f].start + shift[f] / DENSE) % 1.0
    template = dense[ref]
    phi = [np.arange(DENSE, dtype=float) for _ in range(nf)]
    for _ in range(3):
        coarse_t = _features(template, 0.03)
        fine_t = _features(template, 0.006)
        for f in range(nf):
            coarse = _dtw(coarse_t, _features(dense[f], 0.03), penalty=0.02)
            phi[f] = _dtw(fine_t, _features(dense[f], 0.006), penalty=0.01, band=coarse)
        warped = np.mean(
            [
                np.stack([np.interp(phi[f], np.arange(DENSE + 1), np.r_[d[:, k], d[0, k]]) for k in (0, 1)], axis=1)
                for f, d in enumerate(dense)
            ],
            axis=0,
        )
        # Resample the mean uniformly so the next pass matches against an evenly sampled template.
        template = Curve(warped).dense()
    return _sample(curves, phi, samples)


def _sample(curves: list[Curve], phi: list[np.ndarray], samples: int) -> list[np.ndarray]:
    """Choose output samples along the template and place them on every font's outline.

    Corners come first. Each font's corners (vertices turning by more than 28°) are mapped to the template through its
    warping ``phi``; a template position where at least three fonts have a corner becomes a corner sample. The other
    samples fill the gaps between corners, denser along curves than along straight edges, so that a corner never
    gathers a cluster of samples that different fonts would place on different sides of it. Then, in every font,
    each corner sample moves to that font's own sharpest vertex nearby. Only corner samples move, and every font
    moves the same sample to the same feature, so corners stay in correspondence and neighbouring samples never swap
    order.
    """
    nf = len(curves)
    grid = np.arange(DENSE + 1)

    # Turning mass and corner votes per template position.
    turning = np.zeros(DENSE)
    votes = np.zeros(DENSE)
    for f in range(nf):
        t, turn = curves[f].corners(0.0)
        idx = np.rint(np.interp(t * DENSE, np.r_[phi[f], DENSE], grid)).astype(int) % DENSE
        sharp = turn > math.radians(28)
        # Corners get a sample of their own; only the gradual turning of curves attracts extra samples.
        np.add.at(turning, idx[~sharp], turn[~sharp])
        np.add.at(votes, idx[sharp], 1.0)

    def circular(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        r = len(kernel) // 2
        return np.convolve(np.r_[x[-r:], x, x[:r]], kernel, mode="same")[r:-r]

    # A corner is a local maximum of the number of fonts with a vertex within ±1 template sample, at least three,
    # kept only if no stronger corner lies within two samples.
    votes = circular(votes, np.ones(3))
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    turning = circular(turning, kernel / kernel.sum())
    candidates = [
        i for i in range(DENSE) if votes[i] >= 3 and votes[i] >= votes[i - 1] and votes[i] > votes[(i + 1) % DENSE]
    ]
    chosen: list[int] = []
    for i in sorted(candidates, key=lambda i: -votes[i]):
        if len(chosen) < 0.6 * samples and all(min(abs(i - j), DENSE - abs(i - j)) > 2 for j in chosen):
            chosen.append(i)
    corners = np.sort(np.array(chosen, dtype=float))
    # Fill the gaps between corners with the rest of the budget, in proportion to each gap's density mass.
    density = 1.0 / DENSE + turning / (turning.sum() + 1e-12)
    cdf = np.r_[0.0, np.cumsum(density)]
    total = float(cdf[-1])

    def mass(u: float) -> float:
        """Density mass from template position 0 to u, for u in [0, 2 DENSE)."""
        return float(np.interp(u, grid, cdf)) if u <= DENSE else total + float(np.interp(u - DENSE, grid, cdf))

    anchors = corners if len(corners) else np.array([0.0])
    ends = np.r_[anchors[1:], anchors[0] + DENSE]
    gap_mass = np.array([mass(e) - mass(a) for a, e in zip(anchors, ends, strict=True)])
    free = samples - len(corners)
    share = gap_mass / gap_mass.sum() * free
    counts = np.floor(share).astype(int)
    for k in np.argsort(-(share - counts))[: free - counts.sum()]:
        counts[k] += 1
    u_list: list[float] = []
    is_corner: list[bool] = []
    for a, e, n in zip(anchors, ends, counts, strict=True):
        if len(corners):
            u_list.append(a)
            is_corner.append(True)
        m0, m1 = mass(a), mass(e)
        for j in range(1, n + 1):
            target = m0 + (m1 - m0) * j / (n + 1)
            u_list.append(float(np.interp(target % total, cdf, grid)))
            is_corner.append(False)
    u = np.array(u_list) % DENSE
    order = np.argsort(u)
    u = u[order]
    corner_flags = np.array(is_corner)[order]

    out: list[np.ndarray] = []
    for f in range(nf):
        q = np.interp(u, grid, np.r_[phi[f], DENSE]) / DENSE
        t, turn = curves[f].corners(20.0)
        for k in np.nonzero(corner_flags)[0]:
            prev = q[k - 1] if k > 0 else q[-1] - 1.0
            nxt = q[k + 1] if k + 1 < len(q) else q[0] + 1.0
            lo, hi = (prev + q[k]) / 2, (q[k] + nxt) / 2
            gap = (t - q[k] + 0.5) % 1.0 - 0.5
            near = np.nonzero((q[k] + gap > lo) & (q[k] + gap < hi))[0]
            if len(near):
                best = near[np.argmax(turn[near])]
                q[k] = q[k] + gap[best]
        out.append(curves[f].at(q))
    return out


# --------------------------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------------------------


class FontInfo(Strict):
    """One training font: a family instanced at one weight, width and posture."""

    family: str
    style: str = Field(description="Style label, e.g. 'Bold Italic'.")
    cls: FontClass = Field(description="Design class, for colouring the latent space.")
    weight: int = Field(description="CSS weight, 100 to 900.")
    width: int = Field(description="Width as a percentage of normal.")
    italic: bool


class GlyphLayout(Strict):
    """Where one character sits in every font vector."""

    char: str
    contours: list[int] = Field(description="Samples per contour: the outer contour first, then counters.")
    offset: int = Field(description="Index in the vector of the first coordinate (x of sample 0 of contour 0).")
    advance: int = Field(description="Index in the vector of the advance width.")


class DroppedFont(Strict):
    font: str
    reason: str


class FontManifoldData(Strict):
    """Glyph outlines of real fonts in dense correspondence: one integer vector per font."""

    source: str = Field(description="Pinned repository commit the fonts come from.")
    cap_height: int
    display: str = Field(description="The word the figure sets.")
    fonts: list[FontInfo]
    glyphs: list[GlyphLayout]
    vectors: list[list[int]] = Field(
        description="One row per font: x, y of every sample of every contour of every glyph, then advance widths."
    )
    handle: int = Field(description="Sample index (into the display word's P, outer contour) of the foot-serif tip.")
    anchor: int = Field(
        description="Sample index on the left edge of the P's stem, held fixed while the serif is dragged."
    )
    dropped: list[DroppedFont]


def _samples(char: str, contour: int, perimeter: float) -> int:
    step = 20.0 if char in DISPLAY else 52.0
    if contour > 0:
        step *= 1.4
    return int(min(max(round(perimeter / step), 24), 260))


def _glyph(glyph: list[Outline], char: str, ref: int) -> tuple[list[int], list[list[np.ndarray]]]:
    """Corresponds one character across fonts: samples per contour, and per font its contours' output points."""
    centres = [np.vstack(g.outers).mean(axis=0) for g in glyph]
    # Counters: order by centroid, matched to the reference's by the best permutation.
    ref_holes = [_deslant(h, glyph[ref].slant, centres[ref]).mean(axis=0) for h in glyph[ref].holes]
    ordered: list[list[np.ndarray]] = []
    for g, centre in zip(glyph, centres, strict=True):
        hole_c = [_deslant(h, g.slant, centre).mean(axis=0) for h in g.holes]
        best = min(
            itertools.permutations(range(len(g.holes))),
            key=lambda perm: sum(float(np.sum((hole_c[p] - ref_holes[k]) ** 2)) for k, p in enumerate(perm)),
        )
        ordered.append([*g.outers, *[g.holes[p] for p in best]])
    counts: list[int] = []
    per_font: list[list[np.ndarray]] = [[] for _ in glyph]
    for k in range(len(ordered[0])):
        curves = [Curve(o[k]) for o in ordered]
        feats = [_deslant(cv.dense(), g.slant, ctr) for cv, g, ctr in zip(curves, glyph, centres, strict=True)]
        n = _samples(char, k, float(np.median([cv.length for cv in curves])))
        for f, p in enumerate(_correspond(curves, feats, n, ref)):
            per_font[f].append(p)
        counts.append(n)
    return counts, per_font


@figure("manifold-of-fonts/glyphs", title="Capital letters of 60-odd real font instances, in dense correspondence")
def glyphs() -> FontManifoldData:
    outlines = [_outlines(s) for s in SPECS]
    names = [f"{s.family} {_style(s)}" for s in SPECS]
    # Topology: majority number of outer contours and counters for each character.
    keep = list(range(len(SPECS)))
    dropped: list[DroppedFont] = []
    for c in TRAIN:
        shapes = [(len(outlines[i][c].outers), len(outlines[i][c].holes)) for i in keep]
        mode = max(set(shapes), key=shapes.count)
        for i, shape in zip(list(keep), shapes, strict=True):
            if shape != mode:
                keep.remove(i)
                dropped.append(
                    DroppedFont(font=names[i], reason=f"“{c}” has {shape[0] + shape[1]} contours, not {sum(mode)}")
                )
    ref = next(i for i in keep if SPECS[i].family == "Roboto" and SPECS[i].weight == 500)
    ref_pos = keep.index(ref)
    columns: list[list[np.ndarray]] = [[] for _ in keep]
    layouts: list[GlyphLayout] = []
    offset = 0
    handle = anchor = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_glyph, [[outlines[i][c] for i in keep] for c in TRAIN], TRAIN, [ref_pos] * len(TRAIN)))
    for c, (counts, per_font) in zip(TRAIN, results, strict=True):
        for f, contours in enumerate(per_font):
            columns[f].extend(contours)
        if c == "P":
            handle, anchor = _p_handle([p[0] for p in per_font], [SPECS[i] for i in keep])
        layouts.append(GlyphLayout(char=c, contours=counts, offset=offset, advance=0))
        offset += 2 * sum(counts)
    advances = [[round(outlines[i][c].advance) for c in TRAIN] for i in keep]
    layouts = [g.model_copy(update={"advance": offset + j}) for j, g in enumerate(layouts)]
    vectors = [
        [*np.rint(np.vstack(cols)).astype(int).ravel().tolist(), *adv]
        for cols, adv in zip(columns, advances, strict=True)
    ]
    return FontManifoldData(
        source=f"github.com/google/fonts@{COMMIT[:12]}",
        cap_height=CAP_HEIGHT,
        display=DISPLAY,
        fonts=[
            FontInfo(
                family=SPECS[i].family,
                style=_style(SPECS[i]),
                cls=SPECS[i].cls,
                weight=SPECS[i].weight,
                width=SPECS[i].width,
                italic=SPECS[i].italic,
            )
            for i in keep
        ],
        glyphs=layouts,
        vectors=vectors,
        handle=handle,
        anchor=anchor,
        dropped=dropped,
    )


def _p_handle(points: list[np.ndarray], specs: list[Spec]) -> tuple[int, int]:
    """The P's foot-serif tip is the sample that, averaged over the upright serif fonts, lies furthest right on the
    baseline. The anchor is the sample on the left edge of the stem nearest a height of 200, which is the same place in
    every font and keeps the letter in position while the serif is dragged."""
    serif = [p for p, s in zip(points, specs, strict=True) if s.cls in ("serif", "slab", "didone") and not s.italic]
    mean = np.mean(serif, axis=0)
    low = np.nonzero(mean[:, 1] < 15)[0]
    handle = int(low[np.argmax(mean[low, 0])])
    everyone = np.mean(points, axis=0)
    band = np.nonzero(np.abs(everyone[:, 1] - 200) < 60)[0]
    anchor = int(band[np.argmin(everyone[band, 0] + 0.2 * np.abs(everyone[band, 1] - 200))])
    return handle, anchor
