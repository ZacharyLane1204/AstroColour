"""
three_panel_figure.py
---------------------
Figure of total size [fig_width_full, fig_height_full]:

    +--------------------------------------+
    |  top panel  (fig_width_full x H/2)   |
    +----------------------------+---------+
    |  spectra (0.7 W x H/2)     |  image  |
    |                            | (0.3 W) |
    +----------------------------+---------+

Bottom-right is a colour image zoomed to sn_x, sn_y +/- 250 px, decorated with
the helpers from sn_cutouts.py (scale bar, filter colours, target circle,
compass).
"""
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

# helpers from your module (importing it also sets the usetex / serif rc params)
from sn_cutouts import (USE_TEX, fig_width_full, add_compass, add_scalebar_aligned,
                        draw_multicolor_text_cutouts, filter_legend_pieces)

# fig_height_full = fig_width_full * 1.0   # <-- set this to whatever you use
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# plt.rc('font', size=21)
# plt.rc('xtick', labelsize=21)
# plt.rc('ytick', labelsize=21)

fig_width_pt = 244.0  # Get this from LaTeX using \the\columnwidth
text_width_pt = 508.0 # Get this from LaTeX using \the\textwidth

inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt*1.5 # width in inches
fig_width_full = text_width_pt*inches_per_pt*1.5  # 17
fig_height =fig_width*golden_mean # height in inches
fig_size = [fig_width,fig_height] #(9,5.5) #(9, 4.5)
fig_height_full = fig_width_full*golden_mean

fig_size_full = [fig_width_full,fig_height_full] #(9,5.5) #(9, 4.5)


EXP_DATE = '2024-12-16'   # explosion date (YYYY-MM-DD), used by spectra + timeline


# ----------------------------------------------------------------------
# Bottom-left: spectra (your code, moved onto an explicit Axes)
# ----------------------------------------------------------------------
def plot_spectra(ax, miri_mrs_files, nirspec_files,
                 mrs_dates=('2025-01-17', '2025-06-01', '2025-12-07', '2025-12-28'),
                 exp_date=EXP_DATE, wv_choice=1.8,
                 colors=('b', 'g', 'r', 'k'),
                 places=(0.01, 0.007, 0.0022, 0.0003),
                 z=None, xlim=(1.5, 17.5), ylim=(-0.011, 0.0125), yticks=None,
                 insets=(), inset_ylim=None,
                 inset_bottom=0.11, inset_height=0.27, inset_width=None,
                 inset_pad=0.02, inset_gap=0.03,
                 inset_fontsize=7, inset_title_fontsize=9):
    """Spectra plot with optional zoomed insets along the bottom.

    z      : redshift. If given, every wavelength is divided by (1 + z) and the
             plot is in the REST frame (xlim, wv_choice and the inset ranges
             are then rest-frame values too). None = leave wavelengths as-is.
    xlim / ylim / yticks : limits (and optionally tick positions) of the main axes.

    insets : iterable of (wave_min, wave_max, title) or
             (wave_min, wave_max, title, opts), where opts is a dict that can
             override per inset: 'ylim', 'width', 'height', 'bottom'.
             e.g. [(3.9, 4.2, 'A'), (4.2, 5.2, 'B', {'ylim': (0, 0.01), 'width': 0.4})]
    inset_ylim    : default y-limits for all insets (None = autoscale to data).
    inset_width   : default width of each inset, as a fraction of the main axes
                    width (None = insets share the free width equally; any
                    width given here or per-inset is honoured first).
    inset_height / inset_bottom : default size / position as a fraction of the
                    main axes height.
    inset_pad / inset_gap : left/right padding and gap between insets (axes width
                    fraction).
    """
    d1 = datetime.strptime(exp_date, "%Y-%m-%d")
    fac = 1.0 / (1.0 + z) if z is not None else 1.0     # observed -> rest
    curves = []          # (wavelength, flux_with_offset, colour) for reuse in insets

    for i, file in enumerate(miri_mrs_files):
        miri_df = pd.read_csv(file, comment='#', sep=r'\s+')
        c = colors[i % len(colors)]
        wl = miri_df['WAVELENGTH'].to_numpy() * fac
        y = (miri_df['FLUX'] - 0.002 * i + 0.006).to_numpy()
        ax.plot(wl, y, color=c, linewidth=1.2)
        curves.append((wl, y, c))

        d2 = datetime.strptime(mrs_dates[i], "%Y-%m-%d")
        day_difference = abs((d2 - d1).days)
        ax.text(wv_choice, places[i % len(places)],
                f'{mrs_dates[i]} \n (+{day_difference} days)', color=c)

    for i, file in enumerate(nirspec_files):
        if file is None:
            continue
        nirspec_df = pd.read_csv(file, comment='#', sep=r'\s+')
        c = colors[i % len(colors)]
        wl = nirspec_df['wavelength'].to_numpy() * fac
        y = (nirspec_df['flux'] - 0.002 * i + 0.006).to_numpy()
        ax.plot(wl, y, color=c, linewidth=1.2)
        curves.append((wl, y, c))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    elif insets:   # keep ticks off the empty band reserved for the insets
        ax.set_yticks([v for v in np.arange(0.0, ylim[1], 0.002) if v >= ylim[0]])

    ax.axvspan(0.6 * fac, 5.3 * fac, color='C3', alpha=0.2, lw=0)    # NIRSpec range
    ax.axvspan(4.9 * fac, 27.3 * fac, color='C1', alpha=0.2, lw=0)   # MIRI MRS range

    ax.set_xlabel((r'Rest Wavelength [$\mu$m]' if z is not None
                   else r'Wavelength [$\mu$m]'))
    ax.set_ylabel('Flux [Arbitrary Units]')
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_locator(MultipleLocator(0.002))

    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(2))

    # ---- zoomed insets ------------------------------------------------
    specs = []
    for item in insets:
        lo, hi, title = item[:3]
        opts = item[3] if len(item) > 3 else {}
        specs.append((lo, hi, title, opts))

    n = len(specs)
    if n:
        widths = [o.get('width', inset_width) for *_, o in specs]
        given = sum(w for w in widths if w is not None)
        n_auto = sum(w is None for w in widths)
        auto_w = ((1 - 2 * inset_pad - (n - 1) * inset_gap - given) / n_auto
                  if n_auto else 0)
        widths = [auto_w if w is None else w for w in widths]

        x0 = inset_pad
        for (lo, hi, title, opts), w_in in zip(specs, widths):
            h_in = opts.get('height', inset_height)
            b_in = opts.get('bottom', inset_bottom)
            iax = ax.inset_axes([x0, b_in, w_in, h_in])
            x0 += w_in + inset_gap
            iax.set_facecolor('white')

            ymin, ymax = np.inf, -np.inf
            for wl, fl, c in curves:
                m = (wl >= lo) & (wl <= hi)
                if not m.any():
                    continue
                iax.plot(wl[m], fl[m], color=c, linewidth=1.0)
                ymin, ymax = min(ymin, np.nanmin(fl[m])), max(ymax, np.nanmax(fl[m]))

            yl = opts.get('ylim', inset_ylim)
            if yl is not None:
                iax.set_ylim(*yl)
            elif np.isfinite(ymin):
                pad = 0.08 * (ymax - ymin)
                iax.set_ylim(ymin - pad, ymax + pad)

            iax.set_xlim(lo, hi)
            iax.set_yticks([])
            iax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            iax.tick_params(axis='x', labelsize=inset_fontsize, length=2, pad=1)
            iax.set_title(title, fontsize=inset_title_fontsize, pad=2)


def plot_timeline(ax, windows=(), t_min=0, t_max=1200, base_color='0.85',
                  epochs=(), epoch_color='k', alpha=0.9, label_fontsize=12,
                  shaft_frac=0.55, head_frac=0.06, arrow_frac=0.4,
                  obs_dates=(), exp_date=EXP_DATE, obs_fontsize=9,
                  label_width_frac=0.075, label_step=0.10, label_gap=0.02,
                  label_height=0.09, cmap=None, cmap_range=(0.0, 1.0),
                  label_color=None):
    """Right-pointing arrow spanning t_min..t_max days, with no axes furniture.

    windows    : iterable of (start, end, color) or (start, end, color, label);
                 each span is painted inside the arrow in that colour, e.g.
                 [(0, 200, 'C0', 'Early'), (200, 700, 'C1')].
    epochs     : optional days to mark with thin vertical lines.
    cmap       : matplotlib colormap name (e.g. 'gist_heat') or Colormap.
                 - with `windows`: the windows take progressive colours from the
                   cmap (first window = start of the cmap, last = end); the
                   colour given in each window tuple is ignored. Gaps between
                   windows stay grey.
                 - without `windows`: the whole arrow is a smooth gradient
                   following the cmap from t_min to t_max.
    cmap_range : (lo, hi) portion of the colormap to use, e.g. (0, 0.85) to
                 avoid a near-white end.
    label_color: window label colour; None = black or white, whichever
                 contrasts with the window's colour.
    obs_dates  : observation dates as 'YYYY-MM-DD' strings, or
                 ('YYYY-MM-DD', colour) tuples. Each is converted to days since
                 `exp_date`, marked across the arrow, and labelled with its date
                 and day count. All labels sit above the arrow, hugging it, and
                 are staggered to different heights only where neighbours would
                 overlap (flag-pole style: pole on the left, text to its right).
    label_width_frac : estimated label width as a fraction of the time range
                 (raise it if labels still collide, e.g. bigger font).
    label_step / label_gap / label_height : vertical spacing between stagger
                 levels, gap above the arrow, and pole length above a label's
                 baseline (axes-height fractions).
    arrow_frac : overall arrow thickness (head height) as a fraction of the
                 axes height, centred vertically (0.5 = half the panel).
    shaft_frac : shaft thickness as a fraction of the head height.
    head_frac  : arrow-head length as a fraction of the time range.
    """
    span = t_max - t_min
    head_x = t_max - head_frac * span
    top, bot = 0.5 + arrow_frac / 2, 0.5 - arrow_frac / 2
    y0, y1 = 0.5 - shaft_frac * arrow_frac / 2, 0.5 + shaft_frac * arrow_frac / 2
    outline = [(t_min, y0), (head_x, y0), (head_x, bot), (t_max, 0.5),
               (head_x, top), (head_x, y1), (t_min, y1)]

    base = Polygon(outline, closed=True, facecolor=base_color,
                   edgecolor='none', zorder=1)
    ax.add_patch(base)

    def _clipped(artist):
        artist.set_clip_path(base)      # keep everything inside the arrow
        return artist

    cm = None
    if cmap is not None:
        cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap

    def _gradient(start, end):
        """Colormap gradient over [start, end], indexed by time over the full
        t_min..t_max range, clipped to the arrow."""
        f = lambda t: cmap_range[0] + (cmap_range[1] - cmap_range[0]) * (t - t_min) / span
        im = ax.imshow(np.linspace(f(start), f(end), 256)[None, :],
                       extent=(start, end, 0, 1), cmap=cm, vmin=0, vmax=1,
                       aspect='auto', origin='lower', interpolation='bilinear',
                       zorder=2)
        im.set_clip_path(base)

    if cm is not None and not windows:
        _gradient(t_min, t_max)                     # whole arrow follows the cmap

    n_win = len(windows)
    for k, w in enumerate(windows):
        start, end, color = w[:3]
        label = w[3] if len(w) > 3 else None
        if cm is not None:                 # progressive colour from the cmap
            frac = cmap_range[0] + (cmap_range[1] - cmap_range[0]) * (k + 0.5) / n_win
            color, a_win = cm(frac), 1.0
        else:
            a_win = alpha
        ax.add_patch(_clipped(Rectangle((start, 0), end - start, 1,
                                        facecolor=color, edgecolor='none',
                                        alpha=a_win, zorder=2)))
        if label:
            if label_color is not None:
                lc = label_color
            else:                          # contrast against the window colour
                r, g, b = mpl.colors.to_rgb(color)
                lc = 'k' if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else 'white'
            ax.text((start + end) / 2, 0.5, label, ha='center', va='center',
                    fontsize=label_fontsize, color=lc, zorder=4,
                    fontweight='bold' if not USE_TEX else 'normal')
    for e in epochs:
        ax.add_line(_clipped(ax.axvline(e, ymin=0, ymax=1, color=epoch_color,
                                        lw=1.0, zorder=3)))

    # ---- observation dates -> days since explosion -> markers + labels ----
    d_exp = datetime.strptime(exp_date, "%Y-%m-%d")
    obs = []
    for item in obs_dates:
        date, col = (item, 'k') if isinstance(item, str) else item
        days = (datetime.strptime(date, "%Y-%m-%d") - d_exp).days
        obs.append((days, date, col))
    obs.sort()

    def _edge(x, upper):
        """y of the arrow outline at day x (shaft, or the slanted head)."""
        if x <= head_x:
            return y1 if upper else y0
        half = (arrow_frac / 2) * (t_max - x) / (t_max - head_x)
        return 0.5 + half if upper else 0.5 - half

    obs = [o for o in obs if t_min <= o[0] <= t_max]
    w_days = label_width_frac * span
    # text starts just right of its pole; pulled left if it would run off the end
    x_text = [min(d + 0.006 * span, t_max - w_days) for d, _, _ in obs]

    # stagger: walking right -> left, a label must sit above every later label
    # whose text overlaps it, so poles never cross text
    levels = [0] * len(obs)
    for j in reversed(range(len(obs))):
        overl = [levels[i] for i in range(j + 1, len(obs))
                 if x_text[i] < x_text[j] + w_days]
        levels[j] = 1 + max(overl) if overl else 0

    for (days, date, col), xt, lev in zip(obs, x_text, levels):
        ax.add_line(_clipped(ax.axvline(days, ymin=0, ymax=1, color=col,
                                        lw=1.2, zorder=3)))
        # bottom of level-0 text: just above the arrow outline under the text
        xs = np.linspace(days, min(xt + w_days, t_max), 12)
        y_text = max(_edge(x, True) for x in xs) + label_gap + lev * label_step
        ax.plot([days, days], [_edge(days, True), y_text + label_height],
                color=col, lw=1.0, zorder=3, clip_on=False)
        ax.text(xt, y_text, f'{date}\n(+{days} days)', color=col,
                fontsize=obs_fontsize, ha='left', va='bottom',
                zorder=5, clip_on=False)

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(0, 1)
    ax.axis('off')


# ----------------------------------------------------------------------
# Rainbow filter list -> (text, colour) pieces for the image panel legend
# ----------------------------------------------------------------------
def rainbow_filter_pieces(filter_list, cmap_name='rainbow'):
    """[(filter_name, hex_colour), ...] using a resampled colormap, listed
    top-to-bottom in the order given."""
    cmap = mpl.colormaps[cmap_name].resampled(len(filter_list))
    return [(f, mpl.colors.to_hex(cmap(i))) for i, f in enumerate(filter_list)]


# ----------------------------------------------------------------------
# Bottom-right: colour image + sn_cutouts decorations
# ----------------------------------------------------------------------
def plot_colour_image(ax, image, wcs, sn_x, sn_y, filters=None,
                      half_width_px=200, target_radius_arcsec=0.5,
                      scalebar_arcsec=None, color='white', edgebox_color='white'):

    scale_arcsec = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0
    px_per_arcsec = 1.0 / scale_arcsec

    ax.imshow(image, origin='lower')
    xlim = (sn_x - half_width_px, sn_x + half_width_px)
    ylim = (sn_y - half_width_px, sn_y + half_width_px)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(edgebox_color)
        s.set_linewidth(1.5)

    size_arcsec = 2 * half_width_px * scale_arcsec
    if scalebar_arcsec is None:
        scalebar_arcsec = round(size_arcsec / 4)          # ~quarter of the field

    # target circle
    ax.add_patch(Circle((sn_x, sn_y), target_radius_arcsec * px_per_arcsec,
                        edgecolor='yellow', facecolor='none', lw=1.6))

    # compass (upper-left, same placement rule as plot_cutout_panel)
    w = 2 * half_width_px
    anchor = (xlim[0] + 0.20 * w, ylim[1] - 0.24 * w)
    add_compass(ax, wcs, anchor, size_arcsec / 8, color=color)

    # scale bar (lower right)
    add_scalebar_aligned(ax, xlim, ylim, scalebar_arcsec * px_per_arcsec,
                         label=(fr"$\textbf{{{scalebar_arcsec:.0f}}}''$" if USE_TEX
                                else f"{scalebar_arcsec:.0f}''"),
                         color=color, side='right', cap_frac=0.15, text_up = -0.01,
                         side_frac=0.15)

    # coloured filter legend (lower left)
    if filters is not None:
        pieces = (filter_legend_pieces(filters, separator=False)
                  if isinstance(filters, dict) else list(filters))
        draw_multicolor_text_cutouts(ax, 0.03, 0.04, pieces,
                                     fontsize=9, ha='left', va='bottom',
                                     transform=ax.transAxes, stroke_lw=0.1)
        
    ax.set_title(r"\textbf{SN\,2024aecx}" + "\n" + r"\textbf{2025-06-05}" +"\n" +  r"\textbf{NIRCam}", color='k', fontsize=20)


# ----------------------------------------------------------------------
# Assemble
# ----------------------------------------------------------------------
def make_figure(miri_mrs_files, nirspec_files, image, header, sn_x, sn_y,
                filters=None, spec_margins=(0.14, 0.14, 0.02, 0.03),
                timeline_windows=(), timeline_kw=None, spectra_kw=None,
                z=None, obs_dates=(), exp_date=EXP_DATE, spec_ylim=None,
                timeline_cmap=None, **image_kw):
    """spec_margins = (left, bottom, right, top) as fractions of the spectra
    cell (0.7 W x H/2), leaving room for the axis labels.
    timeline_windows: [(start_day, end_day, colour[, label]), ...]
    obs_dates: ['YYYY-MM-DD', ...] (or (date, colour)) placed on the arrow as
               days since `exp_date`.
    z: redshift; converts all spectra wavelengths to the rest frame.
    spec_ylim: (ymin, ymax) for the spectra axes (shortcut for spectra_kw['ylim']).
    timeline_cmap: colormap (name or object) the arrow's colours follow, e.g. 'gist_heat'.
    spectra_kw: extra kwargs for plot_spectra (insets, ylim, inset_ylim,
               inset_width, inset_height, ...)."""
    W, H = fig_width_full, fig_height_full
    fig = plt.figure(figsize=(W, H))

    # top panel: full width, top half, flush with the figure edges and the
    # bottom row -> 0-1200 day arrow timeline
    ax_top = fig.add_axes([0.0, 0.5, 1.0, 0.5])
    tl_kw = dict(timeline_kw or {})
    if timeline_cmap is not None:
        tl_kw['cmap'] = timeline_cmap
    plot_timeline(ax_top, windows=timeline_windows, obs_dates=obs_dates,
                  exp_date=exp_date, **tl_kw)

    # bottom-left cell: 0.7 W x H/2, with margins for labels
    l, b, r, t = spec_margins
    ax_spec = fig.add_axes([0.7 * l, 0.5 * b, 0.7 * (1 - l - r), 0.5 * (1 - b - t)])
    spectra_kw = dict(spectra_kw or {})
    if spec_ylim is not None:
        spectra_kw['ylim'] = spec_ylim
    plot_spectra(ax_spec, miri_mrs_files, nirspec_files, z=z, exp_date=exp_date,
                 **spectra_kw)

    # bottom-right cell: remaining 0.3 W x H/2, no margins (like the finder chart)
    ax_img = fig.add_axes([0.7, 0.0, 0.3, 0.5])
    plot_colour_image(ax_img, image, header, sn_x, sn_y, filters=filters, **image_kw)

    return fig, (ax_top, ax_spec, ax_img)


if __name__ == '__main__':
    # --- example call: swap in your own objects ---
    # sn_x, sn_y = nircam_wcs.all_world2pix(166.45643462, -0.0348496976029, 0)
    # filter_list = ['F070W', 'F090W', 'F115W', 'F150W', 'F356W', 'F444W']
    # fig, (ax_top, ax_spec, ax_img) = make_figure(
    #     miri_mrs_files, nirspec_files, nircam_image, nircam_wcs, sn_x, sn_y,
    #     filters=rainbow_filter_pieces(filter_list),
    #     z=0.0115,                                        # -> rest-frame spectra
    #     timeline_cmap='gist_heat',   # windows get progressive cmap colours
    #     timeline_windows=[(0, 200, 'C0', 'Early'), (200, 700, 'C1'),
    #                       (700, 900, 'C2'), (900, 1200, 'C3')],
    #     obs_dates=['2025-01-17', '2025-06-01', ('2025-12-07', 'r'), '2025-12-28'],
    #     spectra_kw=dict(
    #         ylim=(-0.011, 0.0125),          # main axes y-limits
    #         inset_height=0.25,              # default inset size
    #         inset_ylim=None,                # default inset y-limits (None=auto)
    #         insets=[(3.9, 4.2, 'Title A'),
    #                 (4.2, 5.2, 'Title B', {'ylim': (0.0, 0.012), 'width': 0.35}),
    #                 (6.5, 8, 'Title C')]))
    # fig.savefig('three_panel.pdf')
    pass