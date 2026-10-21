"""
sn_cutouts.py
=============

Build a "finder chart" style figure for a transient (e.g. a supernova):
a wide main/base colour image (left in whatever orientation your own
mosaicking/reprojection pipeline produced -- NOT forced to true North-up),
with fixed-size (angular), fixed-position cutout insets from several
different instruments (NIRCam, HST, MIRI, ...) lined up along the bottom.
Each inset is *reprojected* (via `reproject.reproject_interp`, same tool
you used to build the base mosaic) onto a small WCS that shares the base
image's own orientation, so every panel's compass points the same way.
Under each inset sits a log-wavelength bar showing where that instrument's
filters sit across the full UV-to-mid-IR range. A title and filter legend
for the main image sit at the top, overlaid on the base image, along with
its own compass in the upper-right corner.

Design notes
------------
- The base image is plotted exactly as given -- if it's the output of
  `find_optimal_celestial_wcs(..., auto_rotate=True)` it will be *close*
  to North-up but not exactly, and that's fine; we read its true
  orientation straight from `base_wcs` rather than assuming 90 deg.
- Each inset's *own* RGB channels (already stretched/composited, one array
  per instrument) are reprojected with `reproject_interp` onto a freshly
  built small WCS: centred on the target, at that instrument's own native
  pixel scale, but rotated so its North vector points at the *same
  absolute angle* as the base image's North vector. That's what makes
  every compass -- base and insets alike -- agree with each other, even
  though the base image itself isn't perfectly North-up.
- The scale bar stays a plain horizontal bar (its length is still exact,
  since each inset's reprojected WCS has an isotropic pixel scale); with
  only a few degrees of typical residual tilt this is a normal, standard
  simplification and keeps every panel's bar visually consistent.
- The wavelength bar is a static illustration (a log-lambda axis with a
  gradient background and this instrument's 2 boundary filters marked) --
  no throughput curves yet, per your "for now" note; the function is
  written so a real transmission-curve overlay can drop in later without
  changing its interface.

Swap in your own RGB arrays + WCS objects (and your `find_optimal_celestial_wcs`
mosaic for the base) in the `__main__` block at the bottom.
"""

import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Circle, Rectangle, ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits

from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

# ----------------------------------------------------------------------
# LaTeX text rendering -- only turned on if a working LaTeX install is
# found (tested, not just detected), so the script doesn't crash on
# machines without one.
# ----------------------------------------------------------------------
def _latex_actually_works():
    if shutil.which('latex') is None:
        return False
    try:
        fig_test = plt.figure()
        fig_test.text(0.5, 0.5, r'\textbf{test}', usetex=True)
        fig_test.canvas.draw()
        plt.close(fig_test)
        return True
    except Exception:
        plt.close('all')
        return False


USE_TEX = _latex_actually_works()
if USE_TEX:
    plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def _bf(s):
    """Bold-wrap text: \\textbf{...} under usetex, a plain string otherwise
    (bold is instead applied via fontweight='bold' at the call site)."""
    return r'\textbf{%s}' % s if USE_TEX else s


inches_per_pt = 1.0 / 72.27
text_width_pt = 508.0          # \the\textwidth in your LaTeX document
fig_width_full = text_width_pt * inches_per_pt * 1.5


# ----------------------------------------------------------------------
# Core geometry helpers
# ----------------------------------------------------------------------

def zoom_pixel_limits(wcs, center, size):
    """Pixel-space (xlim, ylim) bounding a square sky box of angular width
    `size` centred on `center`, in `wcs`'s own native pixel orientation."""
    half = size.to(u.deg) / 2.0
    cos_dec = np.cos(center.dec.to(u.rad).value)
    dra = half / cos_dec

    corners = SkyCoord(
        ra=[center.ra - dra, center.ra + dra, center.ra - dra, center.ra + dra],
        dec=[center.dec - half, center.dec - half, center.dec + half, center.dec + half],
    )
    px, py = wcs.world_to_pixel(corners)
    xlim = (float(np.max(px)), float(np.min(px)))
    ylim = (float(np.min(py)), float(np.max(py)))
    return xlim, ylim


def _sky_direction_vector(wcs, center, position_angle_deg, length=1 * u.arcsec):
    """Pixel-space vector (dx, dy) for a `length` step from `center` in sky
    direction `position_angle_deg` (0 = North, 90 = East), for this `wcs`."""
    pt = center.directional_offset_by(position_angle_deg * u.deg, length)
    x0, y0 = wcs.world_to_pixel(center)
    x1, y1 = wcs.world_to_pixel(pt)
    return np.array([x1 - x0, y1 - y0])


def wcs_north_angle(wcs, center):
    """Absolute angle (deg, standard atan2(y,x) convention) of this WCS's
    own North direction, in its own pixel space."""
    v = _sky_direction_vector(wcs, center, 0)
    return float(np.degrees(np.arctan2(v[1], v[0])))


# ----------------------------------------------------------------------
# Reproject each inset to share the base image's own orientation
# ----------------------------------------------------------------------

def reproject_to_orientation(image, wcs, center, size, target_angle_deg, oversample=1.6):
    """
    Reproject an RGB (ny, nx, 3) cutout from its native `wcs` onto a new
    WCS that is centred on `center`, at (approximately) this instrument's
    own native pixel scale, but rotated so its North vector points at the
    absolute angle `target_angle_deg` (the same convention as
    `wcs_north_angle`'s return value) -- e.g. pass the *base image's own*// 
    `wcs_north_angle(base_wcs, center)` so every inset shares the base
    image's orientation exactly, tilt and all.

    Uses `reproject.reproject_interp` per RGB channel -- proper WCS-based
    resampling (handles each instrument's full WCS/distortion solution),
    the same tool used to build the base mosaic, rather than just an
    affine rotation of the already-projected display array.

    Returns
    -------
    reprojected : ndarray, arcsec_per_pixel : float, target_wcs : WCS
    """
    scale_arcsec = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0
    npix = int(np.ceil((size.to(u.arcsec).value * oversample) / scale_arcsec))
    npix += npix % 2  # keep even, for a clean centre pixel

    target_wcs = WCS(naxis=2)
    target_wcs.wcs.crpix = [npix / 2 + 0.5, npix / 2 + 0.5]
    target_wcs.wcs.crval = [center.ra.deg, center.dec.deg]
    target_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    pa = np.radians(90.0 - target_angle_deg)
    scale = scale_arcsec / 3600.0
    target_wcs.wcs.cd = scale * np.array([[-np.cos(pa), np.sin(pa)],
                                           [np.sin(pa), np.cos(pa)]])

    shape_out = (npix, npix)
    channels = []
    for k in range(image.shape[2]):
        arr, _ = reproject_interp((image[..., k], wcs), target_wcs,
                                   shape_out=shape_out, order='bilinear')
        channels.append(np.nan_to_num(arr))
    reprojected = np.clip(np.dstack(channels), 0, 1)

    return reprojected, scale_arcsec, target_wcs


# ----------------------------------------------------------------------
# Compass (WCS-based -- works for any orientation: base panel or a
# reprojected inset alike)
# ----------------------------------------------------------------------

def add_compass(ax, wcs, anchor_pixel, arrow_length_arcsec, color='white',
                 fontsize=13, label_pad=1.35, lw=2.0):
    """Draw North/East arrows on `ax`, anchored at `anchor_pixel = (x, y)`
    in that panel's own pixel coordinates, using this panel's own `wcs`
    (so it's correct for any orientation, tilted or not)."""
    ax_x, ax_y = anchor_pixel
    origin_world = wcs.pixel_to_world(ax_x, ax_y)

    east_world = origin_world.directional_offset_by(90 * u.deg, arrow_length_arcsec * u.arcsec)
    north_world = origin_world.directional_offset_by(0 * u.deg, arrow_length_arcsec * u.arcsec)
    ex, ey = wcs.world_to_pixel(east_world)
    nx, ny = wcs.world_to_pixel(north_world)

    stroke = [PathEffects.withStroke(linewidth=1.5, foreground='black')]
    for tip in [(ex, ey), (nx, ny)]:
        ax.annotate('', xy=tip, xytext=(ax_x, ax_y), annotation_clip=True,
                    arrowprops=dict(arrowstyle='->', lw=lw, color=color))

    ax.text(ax_x + (ex - ax_x) * label_pad, ax_y + (ey - ax_y) * label_pad, r'\textbf{E}',
            color=color, fontsize=fontsize, ha='center', va='center',
            clip_on=True, path_effects=stroke)
    ax.text(ax_x + (nx - ax_x) * label_pad, ax_y + (ny - ax_y) * label_pad, r'\textbf{N}',
            color=color, fontsize=fontsize, ha='center', va='center',
            clip_on=True, path_effects=stroke)


def add_scalebar_aligned(ax, xlim, ylim, length_px, label, color='white',
                          fontsize=13, lw=1.5, bottom_frac=0.10, cap_frac=0.35, text_up = 0.01,
                          side='center', side_frac=0.05):
    """Horizontal scale bar with small end caps near the bottom of the
    view. Each inset's reprojected WCS has an isotropic pixel scale, so
    this stays numerically exact even though it isn't drawn exactly
    East-West once the panel shares the base image's residual tilt.

    `side` places the bar horizontally: 'center' (default, unchanged
    behaviour), or 'left'/'right' to instead anchor it `side_frac` of the
    view's width in from that edge."""
    x0, x1 = sorted(xlim)
    y0, y1 = sorted(ylim)
    if side == 'left':
        bar_x0 = x0 + side_frac * (x1 - x0)
        bar_x1 = bar_x0 + length_px
        cx = (bar_x0 + bar_x1) / 2.0
    elif side == 'right':
        bar_x1 = x1 - side_frac * (x1 - x0)
        bar_x0 = bar_x1 - length_px
        cx = (bar_x0 + bar_x1) / 2.0
    else:
        cx = (x0 + x1) / 2.0
        bar_x0, bar_x1 = cx - length_px / 2.0, cx + length_px / 2.0
    y = y0 + bottom_frac * (y1 - y0)

    cap_h = max(cap_frac * 0.05 * (y1 - y0) * 4, 0.012 * (y1 - y0))

    ax.plot([bar_x0, bar_x1], [y, y], color=color, lw=lw, solid_capstyle='butt',
            clip_on=True, zorder=6)
    for xe in (bar_x0, bar_x1):
        ax.plot([xe, xe], [y - cap_h / 2, y + cap_h / 2], color=color, lw=lw,
                solid_capstyle='butt', clip_on=True, zorder=6)

    ax.text(cx, y - cap_h / 2 + text_up * (y1 - y0), label, color=color,
            fontsize=fontsize, ha='center', va='top', clip_on=True,
            path_effects=[PathEffects.withStroke(linewidth=1.1, foreground='black')]) # ZZZ


# ----------------------------------------------------------------------
# Multi-colour text (filter-combination legends)
# ----------------------------------------------------------------------

def draw_multicolor_text(target, x, y, pieces, fontsize=11, ha='left', va='top',
                          transform=None, stroke_lw=0.8, weight='bold', zorder=10):
    """Draw a run of text where each piece has its own colour, laid out
    left-to-right using actual rendered widths; `ha` can left/right/centre
    align the whole run at `(x, y)`."""
    fig = target if isinstance(target, plt.Figure) else target.figure
    transform = transform or (target.transAxes if hasattr(target, 'transAxes') else fig.transFigure)
    stroke = [PathEffects.withStroke(linewidth=stroke_lw, foreground='black')]

    def _make(px, txt, color):
        t = target.text(px, y, txt, color=color, fontsize=fontsize, ha='left', va=va,
                         transform=transform, fontweight=weight, zorder=zorder)
        t.set_path_effects(stroke)
        return t

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    widths, temp = [], []
    for txt, color in pieces:
        t = _make(0.0, txt, color)
        bbox = t.get_window_extent(renderer=renderer)
        (x0d, _), (x1d, _) = transform.inverted().transform(bbox)
        widths.append(x1d - x0d)
        temp.append(t)
    for t in temp:
        t.remove()

    total_width = sum(widths)
    start_x = x - total_width / 2 if ha == 'center' else (x - total_width if ha == 'right' else x)

    cursor = start_x
    artists = []
    for (txt, color), w in zip(pieces, widths):
        artists.append(_make(cursor, txt, color))
        cursor += w
    return artists

def draw_multicolor_text_cutouts(target, x, y, pieces, fontsize=11, ha='left', va='bottom',
                          transform=None, stroke_lw=0.8, weight='bold', zorder=10,
                          line_spacing=1.15):
    """Draw each piece on its own line, colour-coded, stacked in a column
    anchored at `(x, y)` -- e.g. the bottom-left corner of a cutout, with
    `va='bottom'` so the stack grows upward from `y`. `ha` sets the
    horizontal alignment (e.g. 'left') shared by every line."""
    fig = target if isinstance(target, plt.Figure) else target.figure
    transform = transform or (target.transAxes if hasattr(target, 'transAxes') else fig.transFigure)
    stroke = [PathEffects.withStroke(linewidth=stroke_lw, foreground='black')]

    def _make(px, py, txt, color):
        t = target.text(px, py, txt, color=color, fontsize=fontsize, ha=ha, va='bottom',
                         transform=transform, fontweight=weight, zorder=zorder)
        t.set_path_effects(stroke)
        return t

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Measure one line's rendered height (in transform-data units) so
    # successive rows can be stacked with consistent spacing.
    probe = _make(0.0, 0.0, 'Ag', 'white')
    bbox = probe.get_window_extent(renderer=renderer)
    (_, y0d), (_, y1d) = transform.inverted().transform(bbox)
    probe.remove()
    line_height = (y1d - y0d) * line_spacing

    # pieces are given top-to-bottom (R, G, B); if anchored at the bottom
    # we stack upward, so the last piece should end up closest to `y`.
    rows = list(reversed(pieces)) if va == 'bottom' else pieces
    cursor = y
    artists = []
    for txt, color in rows:
        artists.append(_make(x, cursor, txt, color))
        cursor += line_height
    return artists

CHANNEL_COLORS = {'R': 'red', 'G': 'limegreen', 'B': 'deepskyblue'}


def filter_legend_pieces(filters, separator=True):
    """`filters`: dict with keys 'R','G','B' -> filter-combo string.
    Set `separator=False` to get one piece per channel with no ' / '
    joiner in between (e.g. for a stacked/column layout)."""
    order = ['R', 'G', 'B']
    pieces = []
    for i, k in enumerate(order):
        pieces.append((filters[k], CHANNEL_COLORS[k]))
        if separator and i < len(order) - 1:
            pieces.append((' / ', 'white'))
    return pieces


# ----------------------------------------------------------------------
# Wavelength-coverage bar
# ----------------------------------------------------------------------

SPECTRUM_CMAP = LinearSegmentedColormap.from_list(
    'spectrum', ['#5c0a0a', '#ff2b1e', '#ff9a00', '#ffe135',
                 '#3cb043', '#0077ff', '#5b0aa3'])


def _lam_to_frac(lam_um, lam_min_um, lam_max_um):
    """Log-wavelength position, 0 at `lam_max_um` (left/red end) to 1 at
    `lam_min_um` (right/blue end) -- i.e. decreasing wavelength left to right."""
    return (np.log10(lam_max_um) - np.log10(lam_um)) / (np.log10(lam_max_um) - np.log10(lam_min_um))


def _fmt_wavelength(lam_um):
    return f'{lam_um * 1000:.0f} nm' if lam_um < 1 else f'{lam_um:.2f}'.rstrip('0').rstrip('.') + r' $\mu$m'


def add_wavelength_bar(fig, left, width, bottom, height, filters_um,
                        lam_min_um=0.25, lam_max_um=25.0, band_color='white',
                        fontsize=12, end_label_fontsize=13):
    """
    Draw a log-wavelength coverage bar directly below an inset, spanning
    the same `left`/`width` (figure fraction) as that inset, from
    `lam_max_um` (red end, left) to `lam_min_um` (blue end, right), with
    a stylised spectrum-colour gradient background.

    `filters_um`: list of (wavelength_um, filter_name) for this
    instrument's boundary filters, e.g.
    ``[(0.70, 'F070W'), (4.44, 'F444W')]`` -- each gets a tick + label, and
    the span between the shortest and longest is highlighted. (A future
    version can overlay real normalised transmission curves here instead
    of/alongside the flat highlight band.)
    """
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    band_y0, band_y1 = 0.30, 0.62
    gradient = np.linspace(0, 1, 400).reshape(1, -1)
    ax.imshow(gradient, extent=(0, 1, band_y0, band_y1), aspect='auto',
              cmap=SPECTRUM_CMAP, zorder=1, clip_on=True)

    lam_values = sorted(w for w, _ in filters_um)
    f_lo, f_hi = _lam_to_frac(lam_values[-1], lam_min_um, lam_max_um), \
        _lam_to_frac(lam_values[0], lam_min_um, lam_max_um)
    ax.add_patch(Rectangle((f_lo, band_y0), f_hi - f_lo, band_y1 - band_y0,
                            facecolor='white', edgecolor='none', alpha=0.28, zorder=2))
    ax.add_patch(Rectangle((0, band_y0), 1, band_y1 - band_y0, fill=False,
                            edgecolor=band_color, lw=1.0, zorder=3))

    stroke = [PathEffects.withStroke(linewidth=1.3, foreground='black')]
    for lam_um, name in filters_um:
        f = _lam_to_frac(lam_um, lam_min_um, lam_max_um)
        ax.plot([f, f], [band_y0 - 0.05, band_y1 + 0.05], color=band_color, lw=1.6,
                clip_on=False, zorder=4)
        t = ax.text(f, band_y1 + 0.12, f'{name}\n{_fmt_wavelength(lam_um)}',
                     color=band_color, fontsize=fontsize, ha='center', va='bottom',
                     linespacing=1.15, zorder=5, clip_on=False)
        t.set_path_effects(stroke)

    end_stroke = [PathEffects.withStroke(linewidth=1.1, foreground='black')]
    t0 = ax.text(0.0, band_y0 - 0.10, _fmt_wavelength(lam_max_um), color=band_color,
                  fontsize=end_label_fontsize, ha='left', va='top', clip_on=False, zorder=5)
    t0.set_path_effects(end_stroke)
    t1 = ax.text(1.0, band_y0 - 0.10, _fmt_wavelength(lam_min_um), color=band_color,
                  fontsize=end_label_fontsize, ha='right', va='top', clip_on=False, zorder=5)
    t1.set_path_effects(end_stroke)

    return ax


def add_wavelength_bar_combined(fig, left, width, bottom, height, filters_by_instrument,
                                 instrument_colors, lam_min_um=0.25, lam_max_um=25.0,
                                 band_edges_um=(0.4, 0.7, 2.5),
                                 band_names= (r'\textbf{UV}', r'\textbf{Optical}', r'\textbf{NIR}', r'\textbf{MIR}'),
                                 min_tier_gap=0.045, fontsize=12, band_fontsize=13,
                                 end_label_fontsize=12):
    """
    One wide log-wavelength coverage bar spanning several insets at once,
    from `lam_max_um` (red end, left) to `lam_min_um` (blue end, right),
    with every filter from every instrument marked (colour-coded by
    `instrument_colors`), and the MIR/NIR/optical/UV regime names written
    underneath at the boundaries given by `band_edges_um` (three internal
    edges, e.g. ``(2.5, 0.7, 0.4)`` splits the full range into four bands).

    `filters_by_instrument`: dict instrument name -> list of
    (wavelength_um, filter_name) for *every* filter to mark (not just the
    two boundary ones) -- e.g.
    ``{'NIRCam': [(3.56, 'F356W'), (4.44, 'F444W'), ...], ...}``.

    Labels are drawn rotated 90 deg above the bar and greedily assigned to
    one of two height tiers (whichever keeps `min_tier_gap` of separation
    from the previous label in that tier) so closely spaced filters (e.g.
    NIRCam's six) don't collide.
    """
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    band_y0, band_y1 = 0.40, 0.62
    gradient = np.linspace(1, 0, 600).reshape(1, -1)
    ax.imshow(gradient, extent=(0, 1, band_y0, band_y1), aspect='auto',
              cmap=SPECTRUM_CMAP, zorder=1, clip_on=True)
    ax.add_patch(Rectangle((0, band_y0), 1, band_y1 - band_y0, fill=False,
                            edgecolor='white', lw=1.2, zorder=3))

    # -- regime (MIR/NIR/optical/UV) dividers + labels, below the bar --
    edges_frac = [_lam_to_frac(e, lam_max_um, lam_min_um) for e in band_edges_um]
    all_edges = [0.0] + edges_frac + [1.0]
    for e in edges_frac:
        ax.plot([e, e], [band_y0 - 0.10, band_y1 + 0.02], color='white', lw=1.0,
                 zorder=4, clip_on=False, alpha=0.85)

    band_stroke = [PathEffects.withStroke(linewidth=1.6, foreground='black')]
    for i, name in enumerate(band_names):
        c = (all_edges[i] + all_edges[i + 1]) / 2
        t = ax.text(c, band_y0 - 0.14, name, color='white', fontsize=band_fontsize,
                     ha='center', va='top', zorder=5, clip_on=False,
                     fontweight='bold' if not USE_TEX else 'normal')
        t.set_path_effects(band_stroke)

    end_stroke = [PathEffects.withStroke(linewidth=0.6, foreground='black')]
    t0 = ax.text(0.0, band_y0 - 0.34, fr"\textbf{{{_fmt_wavelength(lam_min_um)}}}", color='white',
                  fontsize=end_label_fontsize, ha='left', va='top', clip_on=False, zorder=5)
    t0.set_path_effects(end_stroke)
    t1 = ax.text(1.0, band_y0 - 0.34, fr"\textbf{{{_fmt_wavelength(lam_max_um)}}}", color='white',
                  fontsize=end_label_fontsize, ha='right', va='top', clip_on=False, zorder=5)
    t1.set_path_effects(end_stroke)

    # -- every filter, ticked + labelled above the bar, staggered in 2 tiers --
    all_filters = []
    for instr, flist in filters_by_instrument.items():
        color = instrument_colors.get(instr, 'white')
        for lam_um, name in flist:
            all_filters.append((_lam_to_frac(lam_um, lam_max_um, lam_min_um), lam_um, name, color))
    all_filters.sort(key=lambda x: x[0])

    tier_last_frac = [-1.0, -1.0]
    tier_h = [0.13, 0.30]
    label_stroke = [PathEffects.withStroke(linewidth=0.8, foreground='black')]
    for f, lam_um, name, color in all_filters:
        tier = 0 # if f - tier_last_frac[0] >= min_tier_gap #else 1
        tier_last_frac[tier] = f

        # ax.plot([f, f], [band_y1, band_y1 + tier_h[tier]], color=color, lw=1.3,
        #         clip_on=False, zorder=4)
        # print(fr"\textbf{{{name}}}", band_y1 + tier_h[tier] - 0.02)
        t = ax.text(f, band_y1 + tier_h[tier] - 0.15, fr"\textbf{{{name}}}", color=color, fontsize=fontsize,
                     ha='left', va='center', rotation=90, zorder=5, clip_on=False) # ZZZ
        t.set_path_effects(label_stroke)

    return ax


# ----------------------------------------------------------------------
# One instrument inset panel
# ----------------------------------------------------------------------

def plot_cutout_panel(ax, image, wcs, center, size, target_angle_deg, label=None,
                       filters=None, compass_arcsec=None, scalebar_arcsec=None,
                       target_marker_radius_arcsec=0.5, label_corner='upper right',
                       compass_corner='upper left', label_color='white',
                       label_fontsize=15, filter_fontsize=12, show_target_marker=True,
                       edgebox_color='white'):
    """
    Plot one instrument's cutout, reprojected so its orientation matches
    `target_angle_deg` (pass `wcs_north_angle(base_wcs, center)` to match
    the base panel), zoomed to a `size`-wide box, with a compass, a scale
    bar, an instrument `label`, and (if `filters` given) a colour-coded
    filter-combination legend.
    """
    reproj, scale_arcsec, target_wcs = reproject_to_orientation(image, wcs, center, size, target_angle_deg)
    ax.imshow(reproj, origin='lower')

    cx, cy = target_wcs.world_to_pixel(center)
    px = 1.0 / scale_arcsec  # isotropic pixels-per-arcsec, by construction

    size_arcsec = size.to(u.arcsec).value
    half = size_arcsec * px / 2.0
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(edgebox_color)
        s.set_linewidth(1.5)

    if compass_arcsec is None:
        compass_arcsec = size_arcsec / 8
    if scalebar_arcsec is None:
        scalebar_arcsec = size_arcsec / 4

    corner_frac = 0.20
    x0, x1 = xlim
    y0, y1 = ylim
    x_side, y_side = compass_corner.split()
    if x_side in ('upper', 'lower'):
        x_side, y_side = y_side, x_side
    anchor_x = x0 + corner_frac * (x1 - x0) if x_side == 'left' else x1 - corner_frac * (x1 - x0)
    anchor_y = y1 - 1.7 * corner_frac * (y1 - y0) + 0.12 * (y1 - y0) if y_side == 'upper' else y0 + corner_frac * (y1 - y0)

    add_compass(ax, target_wcs, (anchor_x, anchor_y), compass_arcsec, color=label_color)
    add_scalebar_aligned(ax, xlim, ylim, scalebar_arcsec * px, label=fr"$\textbf{{{scalebar_arcsec:.1f}}}''$",
                          color=label_color, side='right')

    if show_target_marker:
        r_px = target_marker_radius_arcsec * px
        ax.add_patch(Circle((cx, cy), r_px, edgecolor='white', facecolor='none', lw=1.6))

    lx_side, ly_side = label_corner.split()
    if lx_side in ('upper', 'lower'):
        lx_side, ly_side = ly_side, lx_side
    label_x = 0.05 if lx_side == 'left' else 0.95
    label_ha = 'left' if lx_side == 'left' else 'right'
    label_y = 0.06 if ly_side == 'lower' else 0.94
    label_va = 'bottom' if ly_side == 'lower' else 'top'

    if label is not None:
        t = ax.text(label_x, label_y, _bf(label), transform=ax.transAxes, color=label_color,
                     fontsize=label_fontsize, ha=label_ha, va=label_va,
                     fontweight='bold' if not USE_TEX else 'normal')
        t.set_path_effects([PathEffects.withStroke(linewidth=2.2, foreground='black')])

    if filters is not None:
        draw_multicolor_text_cutouts(ax, 0.03, 0.06, filter_legend_pieces(filters, separator=False),
                                     fontsize=filter_fontsize, ha='left', va='bottom',
                                     transform=ax.transAxes, stroke_lw=0.1, weight='bold', zorder=10)

    return xlim, ylim, target_wcs


# ----------------------------------------------------------------------
# Base-panel <-> inset connector (kept available; not called by default)
# ----------------------------------------------------------------------

def draw_zoom_indicator(base_ax, base_wcs, center, size, inset_ax,
                         box_color='white', line_color='white', lw=1.8,
                         corners=('upper left', 'lower left'), draw_connectors=False):
    """Rectangle on `base_ax` marking the cutout region; optionally (off
    by default) connector lines to `inset_ax` in display coordinates."""
    xlim, ylim = zoom_pixel_limits(base_wcs, center, size)
    x0, x1 = sorted(xlim)
    y0, y1 = sorted(ylim)

    base_ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                 edgecolor=box_color, lw=lw, zorder=6))

    if not draw_connectors:
        return

    corner_xy = {'lower left': (x0, y0), 'lower right': (x1, y0),
                 'upper left': (x0, y1), 'upper right': (x1, y1)}
    corner_frac = {'lower left': (0, 0), 'lower right': (1, 0),
                   'upper left': (0, 1), 'upper right': (1, 1)}
    for key in corners:
        con = ConnectionPatch(xyA=corner_xy[key], coordsA=base_ax.transData,
                               xyB=corner_frac[key], coordsB=inset_ax.transAxes,
                               color=line_color, lw=lw, zorder=5)
        base_ax.figure.add_artist(con)


# ----------------------------------------------------------------------
# Title + filter-legend + compass overlay on the main/base panel
# ----------------------------------------------------------------------

def add_main_title(base_ax, sn_name, main_instrument, main_filters,
                    title_fontsize=22, subtitle_fontsize=20, legend_fontsize=20):
    """Overlay 'SN <name>' at the top of the base panel, with
    '<Instrument> combinations' and its colour-coded filter legend
    immediately below, all white with a black outline."""
    stroke = [PathEffects.withStroke(linewidth=2.5, foreground='black')]

    t1 = base_ax.text(0.5, 0.975, _bf(f'SN {sn_name}'), transform=base_ax.transAxes,
                       color='white', fontsize=title_fontsize, ha='center', va='top',
                       fontweight='bold' if not USE_TEX else 'normal', zorder=10)
    t1.set_path_effects(stroke)

    t2 = base_ax.text(0.5, 0.935, _bf(f'{main_instrument} combinations'),
                       transform=base_ax.transAxes, color='white', fontsize=subtitle_fontsize,
                       ha='center', va='top', fontweight='bold' if not USE_TEX else 'normal',
                       zorder=10)
    t2.set_path_effects(stroke)

    draw_multicolor_text(base_ax, 0.5, 0.895, filter_legend_pieces(main_filters),
                          fontsize=legend_fontsize, ha='center', va='top',
                          transform=base_ax.transAxes)


def add_base_compass(base_ax, base_wcs, anchor_data_xy, arrow_length_arcsec,
                      color='white', fontsize=15):
    """Compass in the upper-right of the base panel, using `base_wcs`
    directly -- so if the base mosaic is only *close* to North-up (e.g.
    `find_optimal_celestial_wcs(..., auto_rotate=True)` leftover tilt),
    the arrows correctly show that residual tilt rather than assuming 90 deg."""
    add_compass(base_ax, base_wcs, anchor_data_xy, arrow_length_arcsec,
                color=color, fontsize=fontsize, lw=2.4)

def rotate_image(image, wcs_in):
    # Base mosaic: built exactly like your snippet, e.g.
    colour_cube = []
    for i in range(3):
        wcs_out, shape_out = find_optimal_celestial_wcs((image[:, :, i], wcs_in), frame='icrs', auto_rotate=True)
        array_north_up, footprint = reproject_interp((image[:, :, i], wcs_in), wcs_out, shape_out=image[:, :, i].shape)
        colour_cube.append(array_north_up)
    colour_cube = np.dstack(colour_cube)
    return colour_cube, wcs_out

# ----------------------------------------------------------------------
# Example usage / self-test
# ----------------------------------------------------------------------

if __name__ == '__main__':
    
    BASE_IMAGE = 'NIRCam'  # 'HST' or 'MIRI' also work, but NIRCam is the main one for this SN

    sn_coord = SkyCoord(ra=166.45643462, dec=-0.0348496976029, unit='deg')

    hst_wcs_frame = '/Users/zalane/Desktop/HST_2024aecx/HST/hst_17502_18_wfc3_uvis_f275w_ifab18/hst_17502_18_wfc3_uvis_f275w_ifab18_drc.fits'
    nircam_wcs_frame = '/Users/zalane/Desktop/JWST_2024aecx/MAST/JWST/jw09258-o014_t002_nircam_clear-f070w_i2d.fits'
    miri_wcs_frame = '/Users/zalane/Desktop/JWST_2024aecx/MAST/JWST/jw09258013001_04103_00001_mirimage_o013_crf.fits'

    with fits.open(hst_wcs_frame) as hdul:
        hst_wcs = WCS(hdul[1].header)
    with fits.open(nircam_wcs_frame) as hdul:
        nircam_wcs = WCS(hdul[1].header)
    with fits.open(miri_wcs_frame) as hdul:
        miri_wcs = WCS(hdul[1].header)

    nircam_image = np.load('/Users/zalane/Python_Scripts/SN2024aecx/colour_image.npy')
    miri_image = np.load('/Users/zalane/Python_Scripts/SN2024aecx/colour_image_miri.npy')
    hst_image = np.load('/Users/zalane/Python_Scripts/SN2024aecx/colour_image_hst.npy')
    
    
    if BASE_IMAGE.lower() == 'nircam':
        base_image, base_wcs = rotate_image(nircam_image, nircam_wcs)
    elif BASE_IMAGE.lower() == 'hst':
        base_image, base_wcs = rotate_image(hst_image, hst_wcs)
    elif BASE_IMAGE.lower() == 'miri':
        base_image, base_wcs = rotate_image(miri_image, miri_wcs)
    else:
        raise ValueError(f"Unknown BASE_IMAGE '{BASE_IMAGE}' -- must be 'NIRCam', 'HST', or 'MIRI'")

    instruments = {'MIRI': (miri_image, miri_wcs),
                   'NIRCam': (nircam_image, nircam_wcs),
                   'HST': (hst_image, hst_wcs),}

    FILTER_COMBOS = {'NIRCam': {'R': r'\textbf{F356W+F444W}', 'G': r'\textbf{F115W+F150W}', 'B': r'\textbf{F070W+F090W}'},
                     'HST': {'R': r'\textbf{F814W}', 'G': r'\textbf{F438W+F555W}', 'B': r'\textbf{F275W+F336W}'},
                     'MIRI': {'R': r'\textbf{F2100W}', 'G': r'\textbf{F1500W}', 'B': r'\textbf{F770W}'},}

    # Boundary (shortest, longest) effective wavelengths per instrument, um --
    # kept for reference / the per-panel bar function if you use it instead.
    FILTER_WAVELENGTHS_UM = {'NIRCam': [(0.70, 'F070W'), (4.44, 'F444W')],
                             'HST': [(0.275, 'F275W'), (0.814, 'F814W')],
                             'MIRI': [(7.70, 'F770W'), (21.0, 'F2100W')],}

    # Every filter used in each instrument's colour combination, um.
    ALL_FILTER_WAVELENGTHS_UM = {'NIRCam': [(3.56, 'F356W'), (4.44, 'F444W'), (1.15, 'F115W'),
                                            (1.50, 'F150W'), (0.70, 'F070W'), (0.90, 'F090W')],
                                 'HST': [(0.814, 'F814W'), (0.438, 'F438W'), (0.555, 'F555W'),
                                         (0.275, 'F275W'), (0.336, 'F336W')],
                                 'MIRI': [(21.0, 'F2100W'), (15.0, 'F1500W'), (7.7, 'F770W')],}

    zoom_box_colours = {'MIRI': 'red', 'NIRCam': 'limegreen', 'HST': 'deepskyblue'}

    cutout_size = 8 * u.arcsec
    sn_name = '2024aecx'
    main_instrument = 'NIRCam'

    phi_base = wcs_north_angle(base_wcs, sn_coord)
    
    
    # nircam_colour_cube = []

    # for i in range(3):
    #     wcs_out, shape_out = find_optimal_celestial_wcs((nircam_image[:,:,i], nircam_wcs), frame='icrs', auto_rotate=True)

    #     array_north_up, footprint = reproject_interp((nircam_image[:,:,i], nircam_wcs), wcs_out, shape_out=shape_out)

    #     header_north = wcs_out.to_header()

    #     nircam_colour_cube.append(array_north_up)

    # nircam_colour_cube = np.dstack(nircam_colour_cube)
    
    sn_x, sn_y = base_wcs.all_world2pix(sn_coord.ra.deg, sn_coord.dec.deg, 0)
    

    fig = plt.figure(figsize=(fig_width_full, fig_width_full))
    base_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    base_ax.imshow(base_image, origin='lower')
    scale_arcsec_base = float(np.mean(proj_plane_pixel_scales(base_wcs))) * 3600.0
    
    r_px = 0.5 * (1.0 / scale_arcsec_base)  # 0.5 arcsec radius in pixels
    base_ax.add_patch(Circle((sn_x, sn_y), r_px, edgecolor='white', facecolor='none', lw=1.6))
    base_ax.set_xlim(100, base_image.shape[1] - 100)
    base_ax.set_ylim(100, base_image.shape[0] - 100)
    base_ax.set_xticks([])
    base_ax.set_yticks([])
    for s in base_ax.spines.values():
        s.set_visible(False)

    add_main_title(base_ax, sn_name, main_instrument, FILTER_COMBOS[main_instrument])

    # Base compass, upper-right, sized as a fraction of the visible field of view.
    visible_w_arcsec = (base_ax.get_xlim()[1] - base_ax.get_xlim()[0]) * scale_arcsec_base
    base_compass_arcsec = visible_w_arcsec * 0.07
    bx0, bx1 = base_ax.get_xlim()
    by0, by1 = base_ax.get_ylim()
    add_base_compass(base_ax, base_wcs, (bx1 - 0.89 * (bx1 - bx0), by1 - 0.11 * (by1 - by0)),
                      base_compass_arcsec) # ZZZ
    
    # add_scalebar_aligned(base_ax, base_ax.get_xlim(), base_ax.get_ylim(), 120, r"$\textbf{2}'$", color='white',
    #                       fontsize=16, lw=1.5, bottom_frac=0.45, cap_frac=0.1)
    
    px = 1.0 / scale_arcsec_base
    scalebar_arcsec_main = 20
    
    add_scalebar_aligned(base_ax, base_ax.get_xlim(), base_ax.get_ylim(), 
                         scalebar_arcsec_main * px, label=fr"$\textbf{{{scalebar_arcsec_main}}}''$",
                         color='white', fontsize=16, lw=1.5, bottom_frac=0.45, cap_frac=0.1, text_up = -0.007)

    inset_w = inset_h = 0.30
    bottom = 0.12       # raised, to leave room for the combined wavelength bar below
    wave_gap = 0.015
    wave_height = 0.09
    positions = {'MIRI': 0.98 - inset_w,
                 'NIRCam': 0.5 - inset_w / 2,
                 'HST': 0.02,}

    for name, colour in zoom_box_colours.items():
        img, wcs = instruments[name]
        iax = fig.add_axes([positions[name], bottom - 0.01, inset_w, inset_h])
        plot_cutout_panel(iax, img, wcs, sn_coord, cutout_size, phi_base, label=name,
                           filters=FILTER_COMBOS[name], edgebox_color=colour)
        draw_zoom_indicator(base_ax, base_wcs, sn_coord, cutout_size, iax, box_color='white')

    wave_left = positions['HST']
    wave_width = (positions['MIRI'] + inset_w) - positions['HST']
    add_wavelength_bar_combined(fig, wave_left, wave_width, bottom - wave_gap - wave_height,
                                wave_height, ALL_FILTER_WAVELENGTHS_UM, zoom_box_colours)

    fig.savefig('sn_finder_chart.png', dpi=200)
    fig.savefig('sn_finder_chart.pdf')
    plt.close(fig)

    print('OK: wrote sn_finder_chart.png / .pdf')