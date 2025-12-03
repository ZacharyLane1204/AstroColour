import numpy as np
from scipy.signal import fftconvolve
from copy import deepcopy

from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from numpy.polynomial import Polynomial as Poly

from scipy.ndimage import map_coordinates

from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift

from scipy.signal.windows import tukey

from AstroColour.psf_analysis import PSF_Analysis, Simple_ePSF

class SpatialPSFMatcher:
    def __init__(self, images, grid=(4,4), epsf_plot=False):
        """
        images : list of 2D arrays
        grid : tuple, number of tiles in x and y for spatially-varying PSF
        """
        self.images = images
        self.N = len(images)
        self.grid = grid
        self.epsf_plot = epsf_plot

        self.fwhms = []
        self.positions = []
        self.epsfs_grid = []

        self._build_all_epsfs()

        self.ref_idx = self._choose_reference()
        self.ref_image = images[self.ref_idx]

        calibrated_images, self.difference_images = self._match_and_subtract()
        
        for i in range(len(calibrated_images)):
            calibrated_images[i] *= np.nanmedian(images[i]) / np.nanmedian(calibrated_images[i])
        
        self.calibrated_images = calibrated_images

    def _build_all_epsfs(self):
        """Compute spatially-varying EPSF grids for all images"""
        nx, ny = self.grid
        self.epsfs_grid = []

        for img in self.images:
            psf_analysis = PSF_Analysis(img, epsf_plot=self.epsf_plot)
            self.fwhms.append(psf_analysis.fwhm_estimate)
            self.positions.append(psf_analysis.positions)

            # Split image into tiles and build EPSF per tile
            tiles = []
            ysize, xsize = img.shape
            dx, dy = xsize//nx, ysize//ny

            for i in range(nx):
                row = []
                for j in range(ny):
                    x0, x1 = i*dx, (i+1)*dx if i< nx-1 else xsize
                    y0, y1 = j*dy, (j+1)*dy if j< ny-1 else ysize
                    sub_img = img[y0:y1, x0:x1]
                    positions_tile = self._filter_positions(psf_analysis.positions, x0, x1, y0, y1)
                    epsf_tile, _ = PSF_Analysis(sub_img)._star_extraction(sub_img, positions_tile)
                    # Take median of stacked stars as EPSF
                    if epsf_tile:
                        tile_kernel = np.nanmedian(np.array([s.data for s in epsf_tile]), axis=0)
                        tile_kernel /= np.nansum(tile_kernel)
                        row.append(tile_kernel)
                    else:
                        row.append(None)
                tiles.append(row)
            self.epsfs_grid.append(tiles)

    def _filter_positions(self, positions, x0, x1, y0, y1):
        """Return positions inside a tile"""
        cond = (
            (positions[:,0] >= x0) & (positions[:,0] < x1) &
            (positions[:,1] >= y0) & (positions[:,1] < y1)
        )
        return positions[cond]

    def _choose_reference(self):
        """Choose the sharpest image as reference"""
        return np.argmin(self.fwhms)  # can add background or S/N metric

    def _interpolate_kernel(self, epsfs_grid, x, y, nx, ny):
        """
        Simple bilinear interpolation between 4 neighboring tile kernels
        """
        x_tile = int(x * nx)
        y_tile = int(y * ny)
        x_tile = np.clip(x_tile, 0, nx-1)
        y_tile = np.clip(y_tile, 0, ny-1)
        kernel = epsfs_grid[x_tile][y_tile]
        return kernel

    def _match_and_subtract(self, n_iter=3, poly_order=3, overlap_frac=0.25, tukey_alpha=0.4,
                            global_pre_align=True, max_dipole_match_distance=6.0):
        """
        Rewritten match & subtract using overlapping-tile convolution helper.
        - global_pre_align: run one global phase_cross_correlation before local dipole steps (recommended)
        - poly_order: polynomial order for dipole warp (3 recommended)
        - overlap_frac: passed to _convolve_with_overlapping_tiles
        """
        calibrated_images = []
        difference_images = []

        nx, ny = self.grid
        ref_image = self.ref_image
        ysize, xsize = ref_image.shape
        dx_tile, dy_tile = xsize//nx, ysize//ny

        # Precompute conv of reference to itself (used as initial conv_ref)
        # Use epsf from reference itself (or simply ref image) for initial dipole detection
        # We will compute final conv after alignment from aligned target
        conv_ref_initial = ref_image.copy()

        for idx, target in enumerate(self.images):
            if idx == self.ref_idx:
                # Convolved reference -> keep as-is for color stacking use
                conv_norm_ref, conv_flux_ref = self.psf_match_dual(ref_image, ref_image, grid=self.grid, epsf_plot=self.epsf_plot)
                calibrated_images.append(conv_flux_ref)
                difference_images.append(np.zeros_like(ref_image))
                continue

            # 0) optional global pre-align to remove large offsets (recommended)
            target_work = target.copy()
            if global_pre_align:
                shift, err, phasediff = phase_cross_correlation(ref_image, target_work, upsample_factor=50)
                # apply global shift to target_work (map_coordinates)
                Y, X = np.indices(target_work.shape)
                coords0 = [Y - shift[0], X - shift[1]]
                target_work = map_coordinates(target_work, coords0, order=3, mode='reflect')

            # 1) iteratively detect dipoles and fit polynomial warp
            for it in range(n_iter):
                # Use a current proxy for convolved reference to find dipoles:
                # At first iteration use conv_ref_initial, later we can use previous conv if desired.
                diff = target_work - conv_ref_initial
                # detect dipoles (we reuse your detect_dipoles but with a slightly larger match radius)
                dipoles_raw = self.detect_dipoles(diff, fwhm=3.0, sigma_thresh=5.0)
                # optionally filter dipoles by significance / distance
                if len(dipoles_raw) == 0:
                    break

                # convert dipoles to numpy list and clamp matches within reasonable radius
                dipoles = []
                for (px, py, dxv, dyv) in dipoles_raw:
                    # ignore very large motions (likely bad matches)
                    if np.hypot(dxv, dyv) <= max_dipole_match_distance:
                        dipoles.append((px, py, dxv, dyv))

                if len(dipoles) < 6:
                    # not enough dipoles to fit polynomial reliably
                    break

                # robust polynomial fit
                poly_dx, poly_dy = self._robust_fit_shift_polynomial(dipoles, order=poly_order)

                # build coords and warp target_work (pix coords)
                Yg, Xg = np.indices(target_work.shape, dtype=float)
                shift_y = poly_dy(Yg, Xg)
                shift_x = poly_dx(Yg, Xg)
                coords = [Yg - shift_y, Xg - shift_x]
                target_work = map_coordinates(target_work, coords, order=3, mode='reflect')

                # update conv_ref_initial optionally by warping the ref or recomputing a coarse conv:
                # we will keep conv_ref_initial fixed during iterations for stability (less resampling).
                # (Alternatively, recompute a light-weight conv_ref_initial here if you prefer.)

            # 2) After final alignment, rebuild EPSFs from aligned target and perform overlapping-tile convolution
            #    psf_match_dual integrated variant: build epsf_grid from target_work, then call overlapping convolver
            # Build epsf_grid from aligned target
            epsf_grid_aligned = []
            psf_analysis_aligned = PSF_Analysis(target_work, epsf_plot=self.epsf_plot)
            positions_aligned = psf_analysis_aligned.positions

            for i in range(nx):
                row = []
                for j in range(ny):
                    x0, x1 = i*dx_tile, (i+1)*dx_tile if i < nx-1 else xsize
                    y0, y1 = j*dy_tile, (j+1)*dy_tile if j < ny-1 else ysize
                    sub_img = target_work[y0:y1, x0:x1]
                    pos_tile = self._filter_positions(positions_aligned, x0, x1, y0, y1)
                    epsf_tile, _ = PSF_Analysis(sub_img)._star_extraction(sub_img, pos_tile)
                    if epsf_tile:
                        tile_kernel = np.nanmedian(np.array([s.data for s in epsf_tile]), axis=0)
                        # apply tukey apodization to kernel (reduce ringing)
                        Wk = self.tukey2d(tile_kernel.shape, alpha=tukey_alpha)
                        tile_kernel *= Wk
                        s = np.nansum(tile_kernel)
                        if s != 0:
                            tile_kernel /= s
                        row.append(tile_kernel)
                    else:
                        row.append(None)
                epsf_grid_aligned.append(row)

            # Now convolve reference with aligned epsf_grid using overlapping tiles helper:
            conv_aligned = self._convolve_with_overlapping_tiles(ref_image, epsf_grid_aligned, grid=self.grid, overlap_frac=overlap_frac)
            conv_flux_preserved = self._convolve_with_overlapping_tiles(ref_image, epsf_grid_aligned, grid=self.grid, overlap_frac=overlap_frac)
            # conv_norm can be approximately conv_aligned (we used normalized kernels above);
            # conv_norm = conv_aligned.copy()
            total_target = np.nansum(target_work)
            total_conv = np.nansum(conv_aligned)
            if total_conv != 0:
                conv_flux = conv_aligned * (total_target / total_conv)
            diff_final = target_work - conv_flux

            calibrated_images.append(conv_flux_preserved)
            difference_images.append(diff_final)

        return calibrated_images, difference_images
    
    def psf_match_dual(self, ref_image, target_image, grid=(4,4), epsf_plot=False):
        """
        Match reference image to target_image using spatially-varying EPSFs.

        Steps:
            1. Build EPSFs per tile from target_image
            2. Convolve reference with EPSFs
            3. Flux-preserve final convolution
            4. Return both normalized and flux-preserved versions

        Parameters
        ----------
        ref_image : 2D array
            Reference image to convolve.
        target_image : 2D array
            Target image from which EPSFs are extracted.
        grid : tuple
            Number of tiles (nx, ny)
        epsf_plot : bool
            Whether to plot EPSFs during extraction.

        Returns
        -------
        conv_norm : 2D array
            Normalized convolution (kernel sum=1)
        conv_flux : 2D array
            Flux-preserved convolution (rescaled to match target_image)
        """

        nx, ny = grid
        ysize, xsize = ref_image.shape
        dx, dy = xsize // nx, ysize // ny

        # Build EPSF grid from target
        psf_analysis = PSF_Analysis(target_image, epsf_plot=epsf_plot)
        positions = psf_analysis.positions

        epsf_grid = []
        for i in range(nx):
            row = []
            for j in range(ny):
                x0, x1 = i*dx, (i+1)*dx if i < nx-1 else xsize
                y0, y1 = j*dy, (j+1)*dy if j < ny-1 else ysize
                sub_img = target_image[y0:y1, x0:x1]
                positions_tile = self._filter_positions(positions, x0, x1, y0, y1)
                epsf_tile, _ = PSF_Analysis(sub_img)._star_extraction(sub_img, positions_tile)
                if epsf_tile:
                    tile_kernel = np.nanmedian(np.array([s.data for s in epsf_tile]), axis=0)
                    tile_kernel /= np.nansum(tile_kernel)
                    row.append(tile_kernel)
                else:
                    row.append(None)
            epsf_grid.append(row)

        # Convolve reference with EPSFs per tile
        conv_norm = np.zeros_like(ref_image)
        conv_flux = np.zeros_like(ref_image)
        for i in range(nx):
            for j in range(ny):
                x0, x1 = i*dx, (i+1)*dx if i < nx-1 else xsize
                y0, y1 = j*dy, (j+1)*dy if j < ny-1 else ysize
                ref_tile = ref_image[y0:y1, x0:x1]
                kernel = epsf_grid[i][j]

                if kernel is None:
                    conv_norm[y0:y1, x0:x1] = ref_tile
                    conv_flux[y0:y1, x0:x1] = ref_tile
                    continue

                conv_tile = fftconvolve(ref_tile, kernel, mode='same')
                conv_tile *= self.tukey2d(conv_tile.shape, alpha=0.5)
                conv_norm[y0:y1, x0:x1] = conv_tile

                # Flux-preserve per tile
                sum_orig = np.nansum(ref_tile)
                sum_conv = np.nansum(conv_tile)
                if sum_conv != 0:
                    conv_flux[y0:y1, x0:x1] = conv_tile * (sum_orig / sum_conv)
                else:
                    conv_flux[y0:y1, x0:x1] = conv_tile

        # Rescale globally to match total target flux
        total_target = np.nansum(target_image)
        total_conv = np.nansum(conv_flux)
        if total_conv != 0:
            conv_flux *= total_target / total_conv

        return conv_norm, conv_flux

    
    def tile_based_alignment(self, ref_image, target_image, grid=(4,4), upsample=200):
        """
        Align target_image to ref_image using local tile-based phase cross-correlation.
        """
        nx, ny = grid
        ysize, xsize = ref_image.shape
        dx, dy = xsize // nx, ysize // ny

        aligned_image = np.zeros_like(ref_image)
        shifts_grid = np.zeros((ny, nx, 2))

        for i in range(ny):
            for j in range(nx):
                y0, y1 = i*dy, (i+1)*dy if i < ny-1 else ysize
                x0, x1 = j*dx, (j+1)*dx if j < nx-1 else xsize

                tile_ref = ref_image[y0:y1, x0:x1]
                tile_tgt = target_image[y0:y1, x0:x1]

                shift, error, _ = phase_cross_correlation(tile_ref, tile_tgt, upsample_factor=upsample)
                shifts_grid[i,j] = shift

                aligned_tile = np.real(fourier_shift(np.fft.fft2(tile_tgt), shift))
                aligned_tile = np.fft.ifft2(aligned_tile)
                aligned_image[y0:y1, x0:x1] = np.real(aligned_tile)

        return aligned_image, shifts_grid


    def detect_dipoles(self, diff_image, fwhm=3.0, sigma_thresh=5.0):
        """
        Detect positive/negative dipoles in a difference image.
        
        Returns
        -------
        dipoles : list of tuples [(x, y, dx, dy), ...] indicating offset vectors
        """
        # Estimate background + std
        mean, median, std = sigma_clipped_stats(diff_image, sigma=3.0)
        
        # Positive sources
        daofind = DAOStarFinder(fwhm=fwhm, threshold=sigma_thresh*std)
        pos_sources = daofind(diff_image - median)
        
        # Negative sources
        neg_sources = daofind(-(diff_image - median))
        
        dipoles = []
        if pos_sources is None or neg_sources is None:
            return dipoles
        
        # Simple nearest-neighbor matching to pair +ve and -ve
        for p in pos_sources:
            px, py = p['xcentroid'], p['ycentroid']
            distances = np.sqrt((neg_sources['xcentroid']-px)**2 + (neg_sources['ycentroid']-py)**2)
            if len(distances) == 0:
                continue
            min_idx = np.argmin(distances)
            if distances[min_idx] < 5.0:  # max expected misalignment in pixels
                nx, ny = neg_sources['xcentroid'][min_idx], neg_sources['ycentroid'][min_idx]
                dipoles.append((px, py, nx-px, ny-py))
        return dipoles
    
    def _robust_fit_shift_polynomial(self, dipoles, order=5, max_iter=3, clip_sigma=3.0):
        """
        Robust polynomial fit with iterative sigma clipping.
        dipoles: list of (x,y,dx,dy)
        returns poly_dx(x,y), poly_dy(x,y) callables
        """
        if len(dipoles) == 0:
            return (lambda X, Y: 0.0), (lambda X, Y: 0.0)

        x = np.array([d[0] for d in dipoles], dtype=float)
        y = np.array([d[1] for d in dipoles], dtype=float)
        dx = np.array([d[2] for d in dipoles], dtype=float)
        dy = np.array([d[3] for d in dipoles], dtype=float)

        def build_A(xv, yv, order):
            cols = []
            for i in range(order+1):
                for j in range(order+1-i):
                    cols.append((xv**i) * (yv**j))
            return np.vstack(cols).T

        mask = np.ones_like(x, dtype=bool)
        for it in range(max_iter):
            A = build_A(x[mask], y[mask], order)
            cx, *_ = np.linalg.lstsq(A, dx[mask], rcond=None)
            cy, *_ = np.linalg.lstsq(A, dy[mask], rcond=None)
            # residuals
            dx_pred = A.dot(cx)
            dy_pred = A.dot(cy)
            res = np.sqrt((dx[mask]-dx_pred)**2 + (dy[mask]-dy_pred)**2)
            med = np.median(res)
            mad = np.median(np.abs(res-med)) + 1e-12
            keep = res <= (med + clip_sigma * (1.4826 * mad))
            # if no change break
            new_mask = mask.copy()
            new_mask[mask] = keep
            if new_mask.sum() == mask.sum():
                break
            mask = new_mask

        # final full coefficient vectors (fill missing coefficients if we clipped)
        A_full = build_A(x, y, order)
        coeff_dx, *_ = np.linalg.lstsq(A_full, dx, rcond=None)
        coeff_dy, *_ = np.linalg.lstsq(A_full, dy, rcond=None)
        coeff_dx = coeff_dx
        coeff_dy = coeff_dy

        def poly_eval_factory(coeff, order):
            def f(X, Y):
                # accepts arrays X,Y and returns same-shape array of shift values
                out = np.zeros_like(X, dtype=float)
                idx = 0
                for i in range(order+1):
                    for j in range(order+1-i):
                        out += coeff[idx] * (X**i) * (Y**j)
                        idx += 1
                return out
            return f

        return poly_eval_factory(coeff_dx, order), poly_eval_factory(coeff_dy, order)

    def tukey2d(self, shape, alpha=0.5):
        win1 = tukey(shape[0], alpha)
        win2 = tukey(shape[1], alpha)
        return np.outer(win1, win2)

    def _convolve_with_overlapping_tiles(self, ref_image, epsf_grid, grid=(4,4), overlap_frac=0.25):
        """
        Convolve ref_image with spatially-varying tile kernels using overlapping tiles + smooth blend.
        Returns a single conv image (and optionally conv_flux if needed).
        epsf_grid: list of list of kernels [nx][ny]
        overlap_frac: fraction of tile size used as half-overlap (0..0.5)
        """
        nx, ny = grid
        ysize, xsize = ref_image.shape

        # tile step and size with overlap
        step_x = int(np.ceil(xsize / nx))
        step_y = int(np.ceil(ysize / ny))
        tile_w = step_x
        tile_h = step_y

        # compute overlap in pixels
        ox = int(np.round(overlap_frac * tile_w))
        oy = int(np.round(overlap_frac * tile_h))

        # create accumulator and weight maps
        acc = np.zeros_like(ref_image, dtype=float)
        wsum = np.zeros_like(ref_image, dtype=float)

        # precompute a smooth 2D weight window (cosine / Tukey-like)
        def make_tile_window(h, w, ox, oy):
            wx = np.ones((w,))
            wy = np.ones((h,))
            # 1D cosine ramps
            if ox > 0:
                rampx = 0.5 * (1 - np.cos(np.pi * np.linspace(0,1,ox)))
                wx[:ox] = rampx
                wx[-ox:] = rampx[::-1]
            if oy > 0:
                rampy = 0.5 * (1 - np.cos(np.pi * np.linspace(0,1,oy)))
                wy[:oy] = rampy
                wy[-oy:] = rampy[::-1]
            W = np.outer(wy, wx)
            return W

        # iterate over tile *centers* to cover image with overlaps
        x_starts = list(range(0, xsize, step_x - ox*2))  # step reduced by overlap
        y_starts = list(range(0, ysize, step_y - oy*2))
        # ensure last tile hits the edge
        if x_starts[-1] + tile_w < xsize:
            x_starts[-1] = max(0, xsize - tile_w)
        if y_starts[-1] + tile_h < ysize:
            y_starts[-1] = max(0, ysize - tile_h)

        for ix, xs in enumerate(x_starts):
            for jy, ys in enumerate(y_starts):
                xe = min(xs + tile_w, xsize)
                ye = min(ys + tile_h, ysize)
                xs_eff = xs
                ys_eff = ys
                tile = ref_image[ys_eff:ye, xs_eff:xe].copy()

                # get kernel for tile: choose nearest kernel in epsf_grid by tile center normalized coords
                # compute normalized tile center in [0,1)
                cx = (xs_eff + (xe-xs_eff)/2) / xsize
                cy = (ys_eff + (ye-ys_eff)/2) / ysize
                kx = int(np.clip(int(cx * nx), 0, nx-1))
                ky = int(np.clip(int(cy * ny), 0, ny-1))
                kernel = epsf_grid[kx][ky]

                # fallback: if None, search neighbors for first non-None
                if kernel is None:
                    found = False
                    for di in range(-2,3):
                        for dj in range(-2,3):
                            ii = kx+di
                            jj = ky+dj
                            if 0 <= ii < nx and 0 <= jj < ny and epsf_grid[ii][jj] is not None:
                                kernel = epsf_grid[ii][jj]
                                found = True
                                break
                        if found:
                            break
                if kernel is None:
                    # no kernel available: use delta (identity)
                    conv_tile = tile
                else:
                    # ensure kernel normalized
                    s = np.nansum(kernel)
                    if s != 0:
                        kernel_use = kernel / s
                    else:
                        kernel_use = kernel.copy()
                    # pad tile if kernel larger than tile to avoid edge effects: reflect pad
                    kh, kw = kernel_use.shape
                    pad_y = max(0, (kh//2) - tile.shape[0]//2)
                    pad_x = max(0, (kw//2) - tile.shape[1]//2)
                    if pad_x>0 or pad_y>0:
                        tile_padded = np.pad(tile, ((pad_y,pad_y),(pad_x,pad_x)), mode='reflect')
                        conv_tile = fftconvolve(tile_padded, kernel_use, mode='same')
                        # crop back
                        cy0 = pad_y; cy1 = cy0 + tile.shape[0]
                        cx0 = pad_x; cx1 = cx0 + tile.shape[1]
                        conv_tile = conv_tile[cy0:cy1, cx0:cx1]
                    else:
                        conv_tile = fftconvolve(tile, kernel_use, mode='same')

                # compute tile weight window same size as conv_tile
                W = make_tile_window(conv_tile.shape[0], conv_tile.shape[1], ox, oy)

                # accumulate
                acc[ys_eff:ye, xs_eff:xe] += conv_tile * W
                wsum[ys_eff:ye, xs_eff:xe] += W

        # avoid divide by zero
        mask = (wsum > 0)
        out = np.zeros_like(ref_image, dtype=float)
        out[mask] = acc[mask] / wsum[mask]
        # for pixels not covered (shouldn't happen) keep ref_image
        out[~mask] = ref_image[~mask]

        return out
