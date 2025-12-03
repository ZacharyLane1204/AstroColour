import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats# , SigmaClip
# from astropy.wcs import WCS
from astropy.nddata import NDData
from astropy.table import Table
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.functional_models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
# from astropy.visualization import simple_norm

from photutils.psf import EPSFBuilder, EPSFStars, SourceGrouper, extract_stars, EPSFModel
from photutils.psf import IntegratedGaussianPRF
from photutils.detection import DAOStarFinder, find_peaks
from photutils.psf import PSFPhotometry, IterativePSFPhotometry, SourceGrouper
from photutils.background import MMMBackground, Background2D, MedianBackground, LocalBackground

from scipy.spatial import cKDTree

from sklearn.cluster import DBSCAN

class PSF_Analysis():
    def __init__(self, image, epsf_plot = False):
        
        self.epsf_plot = epsf_plot
        
        self.full_epsf_pipeline(image)
    
    def full_epsf_pipeline(self, image):
        
        # bkg = self.estimate_background(image)
        image_bkgsub = image #- bkg
        
        epsf_data, epsf, positions = self.create_epsf_kernel(image_bkgsub)
        
        if epsf_data is not None:
            if self.epsf_plot:
                plt.figure()
                plt.imshow(epsf_data)
                plt.show()
        
        # output = self.drizzling_photometry(image_bkgsub, epsf, positions, n_refinements = 3)
        
        # if output is not None:
        #     epsf_data, epsf, positions = output
        
        # if self.epsf_plot:
        #     plt.figure()
        #     plt.imshow(epsf_data)
        #     plt.show()
        
        self.fwhm_estimate = self.estimate_fwhm_from_epsf(epsf_data)

        epsf_data /= np.nansum(epsf_data)
        
        self.epsf_data = epsf_data
        # self.epsf_model = epsf_model
        self.positions = positions
    
    def estimate_background(self, image, box_size=25):
        bkg = Background2D(image, box_size=box_size, filter_size=(3, 3), bkg_estimator=MedianBackground())
        return bkg.background
    
    def position_func(self, data):
    
        nddata = NDData(data=data) 
        mean, median, std = sigma_clipped_stats(data, sigma=3.0) 

        fwhm = 5

        daofind = DAOStarFinder(fwhm=fwhm, threshold=8.*std)  
        sources = daofind(data) 

        sources.sort('flux', reverse=True)
        
        sharp = sources['sharpness']
        round1 = sources['roundness1']
        round2 = sources['roundness2']

        # Define your constraints (tweak these for your data)
        sharp_min, sharp_max = 0.2, 0.95       # Ideal stars are neither too sharp nor too broad
        round1_min, round1_max = -0.5, 0.5    # Closer to 0 means round
        round2_min, round2_max = -0.5, 0.5    # Symmetric roundness in both axes

        mask = (
            (sharp > sharp_min) & (sharp < sharp_max) &
            (round1 > round1_min) & (round1 < round1_max) &
            (round2 > round2_min) & (round2 < round2_max)
        )

        filtered_sources = sources[mask]

        positions = np.zeros((len(filtered_sources), 2))
        positions[:,0] = filtered_sources['xcentroid'].value
        positions[:,1] = filtered_sources['ycentroid'].value
        
        estimated_fwhm = 5
        
        # print(f"Detected {len(positions)} sources, estimated FWHM: {estimated_fwhm}")

        cutout_size = self.cutting_board(filtered_sources, fwhm_guess = estimated_fwhm, 
                                         padding_factor=3.0, safety_margin=True)
        
        # print(f"Using cutout size: {cutout_size}")
        
        # cutout_size = 15

        db = DBSCAN(eps=cutout_size*2, min_samples=2).fit(positions) # Apply DBSCAN clustering

        labels = db.labels_ # Get the labels assigned by DBSCAN (-1 means noise)

        filtered_positions_temp = positions[labels == -1] # Filter out the points that are considered noise
        
        cond = (
            (filtered_positions_temp[:,0] >= cutout_size*2) &
            (filtered_positions_temp[:,0] <= data.shape[0] - cutout_size*2) &
            (filtered_positions_temp[:,1] >= cutout_size*2) &
            (filtered_positions_temp[:,1] <= data.shape[1] - cutout_size*2)
        )
        filtered_positions = filtered_positions_temp[cond]
        
        return filtered_positions, nddata, cutout_size
    
    def cutting_board(self, sources, fwhm_guess=3.0, padding_factor=3.0, safety_margin=True):
        cutout_size = int(np.round(padding_factor * fwhm_guess))
        if cutout_size % 2 == 0:
            cutout_size += 1

        positions = np.vstack([sources['xcentroid'], sources['ycentroid']]).T
        tree = cKDTree(positions)
        distances, _ = tree.query(positions, k=2)  # k=2 because the closest is the point itself
        nearest_neighbor = distances[:, 1]  # Second closest (skip self)

        if safety_margin:
            min_neighbor_dist = np.nanpercentile(nearest_neighbor, 25)
            cutout_size = min(cutout_size, int(min_neighbor_dist / 2))
            if cutout_size % 2 == 0:
                cutout_size += 1

        if cutout_size < 7:
            return 7
        else:
            return cutout_size

    def _star_extraction(self, data, positions=None):
        """
        Extract stars from the image using either given positions or automatically detected stars.

        Parameters
        ----------
        data : 2d array
            The image.
        positions : array_like or None
            Array of shape (N, 2) with (x, y) positions, or None to use DAOStarFinder.
        """
        nddata = NDData(data=data)

        if positions is None:
            # Auto-detect stars with DAOStarFinder
            filtered_positions, _, cutout_size = self.position_func(data)
        else:
            positions = np.array(positions)
            estimated_fwhm = 10
            cutout_size = self.cutting_board_manual(positions, data.shape, fwhm_guess=estimated_fwhm)
            filtered_positions = self._remove_edge_and_cluster_outliers(positions, cutout_size, data.shape)


        # print(f"Using {len(filtered_positions)} stars for EPSF construction. {cutout_size}")
        stars_tbl = Table()
        stars_tbl['x'] = filtered_positions[:, 0]
        stars_tbl['y'] = filtered_positions[:, 1]

        stars = extract_stars(nddata, stars_tbl, size=cutout_size)
        return stars, filtered_positions
    
    def _remove_edge_and_cluster_outliers(self, positions, cutout_size, shape):
        # Filter by DBSCAN
        db = DBSCAN(eps=cutout_size*2, min_samples=2).fit(positions)
        labels = db.labels_
        filtered_positions = positions[labels == -1]

        # Clip edge positions
        cond = (
            (filtered_positions[:, 0] >= cutout_size*2) &
            (filtered_positions[:, 0] <= shape[1] - cutout_size*2) &
            (filtered_positions[:, 1] >= cutout_size*2) &
            (filtered_positions[:, 1] <= shape[0] - cutout_size*2)
        )
        return filtered_positions[cond]

    def cutting_board_manual(self, positions, shape, fwhm_guess=3.0, padding_factor=3.0):
        cutout_size = int(np.round(padding_factor * fwhm_guess))
        if cutout_size % 2 == 0:
            cutout_size += 1

        if len(positions) >= 2:
            tree = cKDTree(positions)
            distances, _ = tree.query(positions, k=2)
            nearest_neighbor = distances[:, 1]
            min_neighbor_dist = np.nanpercentile(nearest_neighbor, 25)
            cutout_size = min(cutout_size, int(min_neighbor_dist / 2))
            if cutout_size % 2 == 0:
                cutout_size += 1

        if cutout_size < 7:
            return 7
        else:
            return cutout_size

    def create_epsf_kernel(self, data, positions=None):
        """
        Build an EPSF model using stars from the input image.

        Parameters
        ----------
        data : 2D ndarray
            The image to extract stars from.
        positions : ndarray or None
            Optionally provide (x, y) coordinates of known sources.

        Returns
        -------
        epsf_data : 2D ndarray
            The EPSF kernel image.
        epsf_model : EPSFModel
            The EPSF model object.
        positions : ndarray
            The positions used.
        """
        stars, positions = self._star_extraction(data, positions=positions)

        threshold = np.nanpercentile(data, 70)

        good_stars = EPSFStars([
            star for star in stars
            if np.all(np.isfinite(star.data)) and
            np.nanmax(star.data) > threshold and
            np.nanmean(star.data) > 0
        ])

        epsf_builder = EPSFBuilder(
            oversampling=1,
            maxiters=20,
            progress_bar=False,
            recentering_maxiters=10,
            smoothing_kernel='quartic'
        )

        try:
            epsf_model, _ = epsf_builder(good_stars)
        except ValueError as e:
            print(f"[Warning] EPSF building failed: {e}")
            return None, None, positions  # gracefully return

        return epsf_model.data, epsf_model, positions

    
    def calculate_fwhm(self, kernel):
        '''
        Calculate the Full Width at Half Maximum (FWHM) of an EPSF kernel in two dimensions.

        Parameters
        ----------
        kernel : 2d array
            EPSF kernel.

        Returns
        -------
        fwhm : float
            FWHM of the EPSF kernel.
        '''
        
        half_max = np.nanmax(kernel) / 2
        # indices = np.where(kernel >= half_max)
        contour = kernel > half_max
        
        y, x = np.where(contour)
        
        fwhm_x = np.ptp(x)
        fwhm_y = np.ptp(y)
        
        return max(fwhm_x, fwhm_y)
    
    def estimate_fwhm_from_epsf(self, epsf_data):
        # Define a 2D Gaussian model
        y, x = np.mgrid[:epsf_data.shape[0], :epsf_data.shape[1]]
        gauss_init = Gaussian2D(
            amplitude=epsf_data.max(),
            x_mean=epsf_data.shape[1] / 2,
            y_mean=epsf_data.shape[0] / 2,
            x_stddev=1,
            y_stddev=1
        )

        fit_p = LevMarLSQFitter()
        with np.errstate(invalid='ignore'):
            gauss_fit = fit_p(gauss_init, x, y, epsf_data)

        fwhm_x = 2*np.sqrt(2*np.log(2)) * gauss_fit.x_stddev.value
        fwhm_y = 2*np.sqrt(2*np.log(2)) * gauss_fit.y_stddev.value
        fwhm_mean = (fwhm_x + fwhm_y) / 2

        return fwhm_mean
    
    def run_iterative_psf_photometry(self, image, epsf_model, positions, aperture_radius=3.5,
                                    max_iter=15, progress_bar=False):
        
        init_table = Table()
        init_table['x_init'] = positions[:, 0]
        init_table['y_init'] = positions[:, 1]

        fitter = LevMarLSQFitter()
        group_maker = SourceGrouper(min_separation=5.0)

        bkg_estimator = LocalBackground(
            bkg_estimator=MMMBackground(),
            inner_radius=5,
            outer_radius=11
        )
        
        mean, median, std = sigma_clipped_stats(image, sigma=3.0) 
        fwhm = 5
        daofind = DAOStarFinder(fwhm=fwhm, threshold=10.*std)

        psf_shape = tuple(np.array(epsf_model.data.shape))

        psf_photometry = IterativePSFPhotometry(
            finder=daofind,
            grouper=group_maker,
            localbkg_estimator=bkg_estimator,
            psf_model=epsf_model,
            fitter=fitter,
            fit_shape=(psf_shape[0], psf_shape[1]),
            aperture_radius=aperture_radius,
            maxiters=max_iter,
            progress_bar=progress_bar
        )
        
        x_margin = psf_shape[1] // 2
        y_margin = psf_shape[0] // 2

        valid = (
            (init_table['x_init'] >= x_margin) &
            (init_table['x_init'] < image.shape[1] - x_margin) &
            (init_table['y_init'] >= y_margin) &
            (init_table['y_init'] < image.shape[0] - y_margin)
        )
        init_table = init_table[valid]

        result = psf_photometry(image, init_params=init_table)
        residual_image = psf_photometry.make_residual_image(data=image, psf_shape=psf_shape, include_localbkg=True)

        return result, residual_image
    
    def drizzling_photometry(self, image, epsf_model, positions, n_refinements=2, alpha=0.3):
        last_successful = (epsf_model.data, epsf_model, positions)
        best_rms = np.inf

        for i in range(n_refinements):
            try:
                result, residual = self.run_iterative_psf_photometry(image, epsf_model, positions)
                current_rms = np.nanstd(residual)

                correction = self.estimate_epsf_correction(residual, positions, epsf_model.data.shape)

                if correction is None:
                    print(f"[Warning] Iteration {i+1}: no correction found, skipping.")
                    continue

                self.last_positions = positions
                best_alpha = self.choose_epsf_alpha(image, residual, correction, epsf_model.data)

                corrected_epsf_data = epsf_model.data + best_alpha * correction
                corrected_epsf_data /= np.nansum(corrected_epsf_data)  # Always normalize
                corrected_epsf_model = EPSFModel(corrected_epsf_data)

                new_positions = np.array([result['x_fit'], result['y_fit']]).T

                if current_rms < best_rms * 0.995:  # Require 0.5% improvement
                    best_rms = current_rms
                    last_successful = (corrected_epsf_data, corrected_epsf_model, new_positions)
                    epsf_model = corrected_epsf_model
                    positions = new_positions
                    print(f"[Info] Iteration {i+1} accepted: residual RMS = {current_rms:.4f}")
                else:
                    print(f"[Info] Iteration {i+1} rejected: RMS change insignificant.")
                    break

            except Exception as e:
                print(f"[Warning] Iteration {i+1} failed: {e}")
                break

        return last_successful



    def choose_epsf_alpha(self, image, residual, correction, epsf_data, alphas=np.linspace(0, 1, 11)):
        best_alpha = 0
        best_rms = np.inf

        for alpha in alphas:
            test_epsf = epsf_data + alpha * correction
            test_epsf /= np.nansum(test_epsf)
            test_model = EPSFModel(test_epsf)

            _, test_residual = self.run_iterative_psf_photometry(image, test_model, self.last_positions)
            test_rms = np.nanstd(test_residual)

            if test_rms < best_rms:
                best_rms = test_rms
                best_alpha = alpha

        return best_alpha
        
    def extract_residual_stars(self, residual, positions, cutout_size):
        ndresid = NDData(data=residual)
        tbl = Table()
        tbl['x'] = positions[:, 0]
        tbl['y'] = positions[:, 1]

        residual_stars = extract_stars(ndresid, tbl, size=cutout_size)

        # Optional: reject bad ones here
        good_residuals = [
            s for s in residual_stars
            if np.isfinite(s.data).all()]
        return good_residuals
    
    def build_residual_correction(self, residual_stars):
        stack = np.array([s.data for s in residual_stars])
        correction = np.nanmedian(stack, axis=0)  # Or mean
        return correction
    
    def estimate_epsf_correction(self, residual, positions, epsf_shape, smoothing_sigma=1.0):
        """
        Estimate a correction to the current ePSF model using residuals.

        Parameters
        ----------
        residual : 2D ndarray
            The residual image from PSF subtraction.
        positions : list of tuples
            (x, y) positions of stars used in photometry.
        epsf_shape : tuple
            Shape of the ePSF model (e.g., (25, 25)).
        smoothing_sigma : float
            Standard deviation for optional Gaussian smoothing of the correction.

        Returns
        -------
        correction : 2D ndarray or None
            The residual correction to apply to the current ePSF, or None if not enough valid data.
        """

        residual_stars = self.extract_residual_stars(residual, positions, cutout_size=epsf_shape[0])

        if residual_stars is None or len(residual_stars) == 0:
            print("[Warning] No good residual stars found for EPSF correction.")
            return None

        correction = self.build_residual_correction(residual_stars)

        if correction is None or not np.any(np.isfinite(correction)):
            print("[Warning] Residual correction failed or contains no finite values.")
            return None
        
        return correction



class Simple_ePSF():
    def __init__(self, data, simple_psf = False):
    
        bkg = self.estimate_background(data, box_size=25)
        image = data = bkg
        positions = self.position_func(image)
        self.positions = positions
        
        if simple_psf:
        
            psf_kernel, fwhm = self.build_psf(image, positions)
            self.psf_kernel = psf_kernel
            self.fwhm = fwhm
    
    def estimate_background(self, image, box_size=25):
        bkg = Background2D(image, box_size=box_size, filter_size=(3, 3), bkg_estimator=MedianBackground())
        return bkg.background
    
    def position_func(self, data):
    
        nddata = NDData(data=data) 
        mean, median, std = sigma_clipped_stats(data, sigma=3.0) 

        fwhm = 5

        daofind = DAOStarFinder(fwhm=fwhm, threshold=10.*std)  
        sources = daofind(data) 

        sources.sort('flux', reverse=True)
        
        sharp = sources['sharpness']
        round1 = sources['roundness1']
        round2 = sources['roundness2']

        sharp_min, sharp_max = 0.2, 0.95   
        round1_min, round1_max = -0.5, 0.5 
        round2_min, round2_max = -0.5, 0.5  

        mask = (
            (sharp > sharp_min) & (sharp < sharp_max) &
            (round1 > round1_min) & (round1 < round1_max) &
            (round2 > round2_min) & (round2 < round2_max)
        )

        filtered_sources = sources[mask]

        positions = np.zeros((len(filtered_sources), 2))
        positions[:,0] = filtered_sources['xcentroid'].value
        positions[:,1] = filtered_sources['ycentroid'].value
        
        return positions
    
    def build_psf(self, image, positions):
        stars_tbl = Table(names=['x', 'y'], data=[positions[:, 1], positions[:, 0]])
        nddata = NDData(data=image)
        
        cutout_size = self.cutting_board(positions, fwhm_guess=2.5, padding_factor=3.0, safety_margin=True)
        
        stars = extract_stars(nddata, stars_tbl, size=cutout_size)
        
        fwhms = []
        star_cutouts = []

        for star in stars:
            data = star.data - np.nanmin(star.data)  # Normalize floor to 0
            norm_data = data / np.nansum(data)
            

            y, x = np.indices(data.shape)
            p_init = Gaussian2D(amplitude=1.0,
                                x_mean=data.shape[1] / 2,
                                y_mean=data.shape[0] / 2,
                                x_stddev=2.0,
                                y_stddev=2.0)

            fit_p = LevMarLSQFitter()(p_init, x, y, norm_data)

            fwhm = 2*np.sqrt(2*np.log(2)) * np.nanmean([fit_p.x_stddev.value, fit_p.y_stddev.value])
            fwhms.append(fwhm)

            model_data = fit_p(x, y)
            model_data /= np.nansum(model_data)
            star_cutouts.append(model_data)

        stacked = np.array(star_cutouts)
        psf_kernel = np.nanmean(stacked, axis=0)

        return psf_kernel, np.nanmedian(fwhms)
    
    def cutting_board(self, sources, fwhm_guess=3.0, padding_factor=3.0, safety_margin=True):
        cutout_size = int(np.round(padding_factor * fwhm_guess))
        if cutout_size % 2 == 0:
            cutout_size += 1

        tree = cKDTree(sources)
        distances, _ = tree.query(sources, k=2)  # k=2 because the closest is the point itself
        nearest_neighbor = distances[:, 1]  # Second closest (skip self)

        if safety_margin:
            min_neighbor_dist = np.nanpercentile(nearest_neighbor, 25)
            cutout_size = min(cutout_size, int(min_neighbor_dist / 2))
            if cutout_size % 2 == 0:
                cutout_size += 1

        if cutout_size < 5:
            return 5
        else:
            return cutout_size + 2