import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.wcs import WCS
from astropy.nddata import NDData
from astropy.table import Table
from astropy.convolution import convolve

from astroscrappy import detect_cosmics

from scipy.signal import fftconvolve, convolve2d
from scipy.ndimage import label
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

from photutils.detection import find_peaks
from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry, IterativePSFPhotometry
from photutils.background import LocalBackground, MMMBackground
from photutils.psf import SourceGrouper

from sklearn.cluster import DBSCAN
from skimage.transform import estimate_transform, warp
from skimage import filters, morphology, measure
from skimage import feature, transform, draw

from AstroColour.image_rotation import unrotate_image
from AstroColour.psf_analysis import PSF_Analysis, Simple_ePSF
from AstroColour.pdastro import pdastrostatsclass
from AstroColour.hidden_prints import hidden_prints

import os
from copy import deepcopy
from tqdm import tqdm
import cv2
import warnings

warnings.filterwarnings('ignore')

plt.rc('font', family='serif')

fig_width_pt = 244.0  # Get this from LaTeX using \the\columnwidth
text_width_pt = 508.0 # Get this from LaTeX using \the\textwidth

inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt*1.5 # width in inches
fig_width_full = text_width_pt*inches_per_pt*1.5  # 17
fig_height =fig_width*golden_mean # height in inches
fig_size = [fig_width,fig_height] #(9,5.5) #(9, 4.5)
fig_height_full = fig_width_full*golden_mean

class RGB():
    def __init__(self, images, 
                 colours = ['red', 'green', 'blue'], 
                 intensities = [1, 1, 1],
                 uppers = [98, 98, 98], lowers = [2, 2, 2],
                 save = False, save_name = '', save_folder = '',
                 figure_size = None, manual_override = None, dpi = 900, 
                 norm = 'linear', gamma = 1, epsf = True, 
                 epsf_plot = False, bkg_plot = False, temp_save = True, run = True):
        '''
        Create a RGB image from three images.

        Parameters
        ----------
        images : List
            List of Numpy Arrays of data images.
        colours : List
            List of Tuples or Strings of colour choice.
        intensities : List
            List of Floats between 0 and 1 for image intensities.
        uppers : List
            List of Floats between 0 and 100 for upper percentile.
        lowers : List
            List of Floats between 0 and 100 for lower percentile
        save : Boolean
            Whether to save the image.
        save_name : String
            Detail to add in the saved filename.
        save_folder : String
            Folder to save the image.
        figure_size : Float
            Dimension of the image.
        manual_override : Boolean
            Whether to manually override the limits.
        dpi : Integer
            DPI of the saved image.
        norm : String
            Normalisation of the image. An option of 'linear', 'sqrt', 'log', 'asinh' or 'sinh'.
        gamma : Float
            Gamma correction of the image. Power to raise the image to.
        epsf : Boolean
            Whether to use the EPSF method.
        epsf_plot : Boolean
            Whether to plot the EPSF kernel.
        run : Boolean
            Whether to process images or just use the framework
        '''
        length = len(images)
        
        if len(colours) != length:
            raise ValueError("Length of colours does not match the number of images.")
        if len(intensities) != length:
            raise ValueError("Length of intensities does not match the number of images.")
        if len(uppers) != length:
            raise ValueError("Length of uppers does not match the number of images.")
        if len(lowers) != length:
            raise ValueError("Length of lowers does not match the number of images.")

        self.save = save
        self.save_name = save_name
        self.save_folder = save_folder
        self.dpi = dpi
        self.epsf = epsf
        self.epsf_plot = epsf_plot
        self.bkg_plot = bkg_plot
        self.temp_save = temp_save

        self.figure_size = figure_size
        if self.figure_size == None:
            self.figure_size = fig_width_full

        if manual_override == None:
            manual_override = 100
        self.manual_override = manual_override
        
        if isinstance(norm, str):
            if ['linear', 'sqrt', 'log', 'asinh', 'sinh'].__contains__(norm):
                norms = [norm]*len(images)
            else:
                raise ValueError("Not a valid norm. Try 'linear', 'sqrt', 'log', 'asinh' or 'sinh'.")
            
        elif isinstance(norm, list) & (len(norm) == length):
            norms = []
            for i in range(len(norm)):
                if not ['linear', 'sqrt', 'log', 'asinh', 'sinh'].__contains__(norm[i]):
                    raise ValueError("Not a valid norm. Try 'linear', 'sqrt', 'log', 'asinh' or 'sinh'.")
                else:
                    norms.append(norm[i])
        
        self.gamma = gamma
        
        if run:
            images = self._preprocess_images(images)
            self.process_images(images, colours, intensities, uppers, lowers, norms)
        
    def _preprocess_images(self, images):
        
        new_images = []
        for image in images:
            
            bleed_image = self.bleeding_trails(image)
            
            # sat_image = self.satellite_trails(image)
            
            new_image, _ = unrotate_image(bleed_image)
            
            # plt.figure()
            # plt.imshow(sat_image - image, origin='lower', 
            #            vmin = np.nanpercentile(sat_image - image, 1), 
            #            vmax = np.nanpercentile(sat_image - image, 99), cmap= 'gray')
            # plt.show()         
            
            new_images.append(new_image)
        
        return new_images

    def process_images(self, images, colours, intensities, uppers, lowers, norms):
        """
        Process the images.
        
        Parameters
        ----------
        images : List
            List of Numpy Arrays of data images.
        colours : List
            List of Tuples or Strings of colour choice.
        intensities : List
            List of Floats between 0 and 1 for image intensities.
        uppers : List
            List of Floats between 0 and 100 for upper percentile.
        lowers : List
            List of Floats between 0 and 100 for lower percentile
        """
        
        calib_images = []
        files = []
        
        # try:
        if self.epsf:
            fwhms = []
            epsfs = []
            positions = []
            for image in images:
                
                psf_analysis = PSF_Analysis(image, epsf_plot=self.epsf_plot)
                
                fwhms.append(psf_analysis.fwhm_estimate)
                epsfs.append(psf_analysis.epsf_data)
                positions.append(psf_analysis.positions)
                
            idx = np.nanargmax(fwhms)
            # largest_fwhm_kernel = epsfs[idx]  # Identify the EPSF with the largest FWHM
            
            self.epsf = True
        else:
            self.epsf = False
            idx = 0
            positions = []
            for image in images:
                psf_analysis = Simple_ePSF(image)
                positions.append(psf_analysis.positions)
            
        # fwhms = []
        # epsfs = []
        # positions = []
        
        # for image in images:
            
        #     psf_analysis = Simple_ePSF(image, simple_psf = True)
        #     fwhms.append(psf_analysis.fwhm)
        #     epsfs.append(psf_analysis.psf_kernel)
        #     positions.append(psf_analysis.positions)
        # idx = np.nanargmax(fwhms)

        # print('ZZZ fwhms:', fwhms)
        
        
        with hidden_prints():
            for i, image in enumerate(images):
                if i != idx:
                    bkg_image = self.backgrounding(image, plot=self.bkg_plot)
                    crmask, cleaned_image = self.cosmic_ray_removal(bkg_image)
                    
                    aligned_image = self.positioning_aligning(positions[idx], positions[i], cleaned_image)
                    if self.epsf:
                        corrected_image = self.match_psf(aligned_image, epsfs[i])
                    else:
                        corrected_image = aligned_image.copy()
                    calib_images.append(corrected_image)
                else:
                    bkg_image = self.backgrounding(image, plot=self.bkg_plot)
                    crmask, cleaned_image = self.cosmic_ray_removal(bkg_image)
                    calib_images.append(cleaned_image)
                
                coloured_image = self.running_norm_colour(cleaned_image, 
                                                          lower = lowers[i], upper = uppers[i], 
                                                          norm = norms[i], colour_choice = colours[i], 
                                                          gamma = self.gamma, intensity = intensities[i])
                
                files.append(coloured_image)
                
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                im_composite = np.clip(np.nansum(files, axis=0), 0, 5)
        
            self.im_composite = im_composite
            self.calib_images = calib_images

            if self.temp_save:
                np.save('image_composite.npy', im_composite)
                np.save('calib_images.npy', calib_images)


    def running_norm_colour(self, image, lower = 2, upper = 98, 
                            norm = 'linear', colour_choice = 'red', 
                            gamma = 1, intensity = 1):
                
        p_norm = self.percent_norm(image, lower = lower, upper = upper, norm = norm, gamma = gamma)
        colour = self.colour_check(colour_choice)
                
        coloured_image = self.colourise(p_norm, colour, intensity)
        
        return coloured_image
        
    def cosmic_ray_removal(self,  image_data):
        crmask, cleaned = detect_cosmics(image_data, 
                                  sigclip=4.5,       # sigma threshold for detection
                                  sigfrac=0.3,        # neighboring pixels fraction
                                  objlim=5.0,         # contrast between CR and object
                                  gain=1.0,           # if not gain-corrected, set appropriately
                                  readnoise=5.0,      # your camera's read noise (in e-)
                                  satlevel=65535.0,   # saturation limit (to avoid false positives)
                                  niter=4,            # number of iterations
                                  cleantype='medmask' # how to interpolate over CRs
                                  )
        
        return crmask, cleaned
        
    def backgrounding(self, data, plot = None):
        
        pdac = pdastrostatsclass()
        
        cut_data = data.copy()
        
        thresh_hi = np.nanpercentile(cut_data, 70)
        thresh_low = np.nanpercentile(cut_data, 5)
        
        cut_data = cut_data[(cut_data < thresh_hi)]
        cut_data = cut_data[(cut_data > thresh_low)]
        
        
        cut_data_med = np.nanpercentile(cut_data, 40)
        
        df = pd.DataFrame()
        
        pdac.t = df
        pdac.t['dm']=  cut_data - cut_data_med
        
        pdac.calcaverage_sigmacutloop('dm', verbose=0, percentile_cut_firstiteration=50)
        
        bkg = pdac.statparams['mean'] + cut_data_med
        
        return data - bkg
    
    def _data_slicing(self, background):
        
        prep_data = background.ravel().copy()
        prep_data = prep_data[np.isfinite(prep_data)]

        prep_data = prep_data[(prep_data > np.nanpercentile(prep_data, 5)) & (prep_data < np.nanpercentile(prep_data, 95))]
        
        return prep_data
        
    def colourise(self, im, colour, intensity):
        '''
        Colourise the image. and scale the intensity.

        Parameters
        ----------
        im : 2d array
            FITS Image.
        colour : Tuple
            3 tuple scaled between 0 and 1.
        intensity : Float
            Multiply brightness by this amount

        Returns
        -------
        3d image
            Final 3d array in colour channel form.
        '''
        
        im_scaled = np.atleast_3d(im)

        colour = np.asarray(colour).reshape((1, 1, -1)) # Reshape the color (here, we assume channels last)
        return im_scaled * colour * intensity

    def colour_check(self, colour):
        '''
        Change the colour to a RGB tuple.

        Parameters
        ----------
        colour : Tuple or String
            Defining what colour choice for the channel

        Raises
        ------
        ValueError
            Needs to be a named colour in matplotlib or a tuple in the form (255,255,255).

        Returns
        -------
        colour : Tuple
            RGB colour tuple scaled between 0 and 1.
        '''

        if isinstance(colour, tuple):
            colour = tuple(ci/255 for ci in colour)
        elif isinstance(colour, str):
            colour = tuple(int(c*255) for c in mcolors.to_rgb(colour))
            colour = tuple(ci/255 for ci in colour)
        else:
            raise ValueError("Not a valid input. Try a named colour in matplotlib or a tuple in the form (255,255,255).")
        return colour

    def percent_norm(self, x, lower = 2, upper = 98, norm = 'linear', gamma = 1):
        '''
        Rescale the image to a percentage scale.

        Parameters
        ----------
        x : 2d array
            FITS image.
        lower : float
            lowest percent data to cut out.
        upper : float
            highest percent data to cut out.

        Returns
        -------
        arr_rescaled : 2d array
            Normalised percentage scaled 2d array.
        '''
        
        normed = simple_norm(x, stretch=norm, min_percent=lower, max_percent=upper)
        arr_rescaled = normed(x)
        
        arr_rescaled = np.power(arr_rescaled, gamma)
        
        return arr_rescaled

    def plot(self, image = None):
        '''
        Plot the final image.

        Parameters
        ----------
        im_composite : 3d array
            2d arrays that are stacked in a third axis.
        '''
        
        if image is not None:
            im_composite = image.copy()
        else:
            try:
                im_composite = np.load('image_composite.npy')
            except:
                im_composite = self.im_composite
            finally:
                raise ValueError('Either put in an array')

        plt.figure(figsize = (self.figure_size,self.figure_size))
        plt.imshow(im_composite)
        plt.axis('off')
        plt.xlim(self.manual_override , im_composite.shape[1] - self.manual_override )
        plt.ylim(self.manual_override , im_composite.shape[0] - self.manual_override )
        if self.save:
            plt.savefig(os.path.join(self.save_folder, self.save_name + '.pdf'), format = 'pdf', bbox_inches = 'tight')
            plt.savefig(os.path.join(self.save_folder, self.save_name + '.png'), dpi = self.dpi, bbox_inches = 'tight')
        plt.show()
        
        return im_composite

    def positioning_aligning(self, positions_ref, positions_target, target_image):
        tree = cKDTree(positions_ref)
        dist, idx = tree.query(positions_target, distance_upper_bound=3)  # max matching radius
        matched_ref = positions_ref[idx[dist != np.inf]]
        matched_target = positions_target[dist != np.inf]
        
        tform = estimate_transform('similarity', matched_target, matched_ref)
        
        aligned_image = warp(target_image, inverse_map=tform.inverse, order=3, preserve_range=True)
        
        return aligned_image
        
    def inpaint_masked_pixels(self, image, mask):
        image_norm = (image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image))
        image_8bit = np.uint8(image_norm * 255)
        mask_8bit = np.uint8(mask * 255)

        inpainted = cv2.inpaint(image_8bit, mask_8bit, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return inpainted.astype(float) / 255 * (np.nanmax(image) - np.nanmin(image)) + np.nanmin(image)
    
    def match_psf(self, image, psf):
        """
        Match the PSF of two images.
        
        Parameters
        ----------
        image : 2d array
            Image to be processed.
        psf : 2d array
            PSF kernel of the largest image.
            
        Returns
        -------
        corrected_image : 2d array
            Image with matched PSF.
        """

        corrected_image = convolve(image, psf)
        
        return corrected_image
    
    def bleeding_trails(self, data):

        norm = (data - np.nanmedian(data)) / np.nanstd(data)
        
        bright_mask = norm > 8   # Threshold for very bright structures

        selem = morphology.rectangle(10, 1)  # Morphological opening to remove point sources, keep vertical streaks
        vertical_mask = morphology.opening(bright_mask, selem) # Vertical structuring element (length 20 pixels, width 1)

        labels = measure.label(vertical_mask) # Label connected vertical structures
        props = measure.regionprops(labels)

        final_mask = np.zeros_like(vertical_mask, dtype=bool)
        for p in props:
            minr, minc, maxr, maxc = p.bbox
            height = maxr - minr
            width = maxc - minc
            if height > 20 and width <= 4:  # vertical streak condition
                final_mask[labels == p.label] = True

        #  Repair by column interpolation
        cleaned = data.copy()
        for c in np.where(final_mask.any(axis=0))[0]:
            col_mask = final_mask[:, c]
            left = cleaned[:, c-1] if c > 0 else cleaned[:, c+1]
            right = cleaned[:, c+1] if c < cleaned.shape[1]-1 else cleaned[:, c-1]
            interp_col = np.median(np.vstack([left, right]), axis=0)
            cleaned[col_mask, c] = interp_col[col_mask]
            
        return cleaned
    
    # def satellite_trails(self, data):
        
    #     norm = (data - np.nanmedian(data)) / np.nanstd(data)
        
    #     # Step 1: Edge detection (Canny)
    #     edges = feature.canny(norm, sigma=2.0)

    #     # Step 2: Hough transform for line detection
    #     tested_angles = np.linspace(-np.pi/2, np.pi/2, 360)
    #     hspace, angles, dists = transform.hough_line(edges, theta=tested_angles)
    #     accum, angles_peaks, dists_peaks = transform.hough_line_peaks(
    #         hspace, angles, dists, threshold=0.3 * np.max(hspace)
    #     )

    #     # Step 3: Create mask for detected lines
    #     line_mask = np.zeros_like(data, dtype=bool)
    #     for angle, dist in zip(angles_peaks, dists_peaks):
    #         # Generate long lines through the image
    #         for offset in np.linspace(-data.shape[1], data.shape[1], 2000):
    #             x = int(dist * np.cos(angle) + offset * np.sin(angle))
    #             y = int(dist * np.sin(angle) - offset * np.cos(angle))
    #             if 0 <= x < data.shape[0] and 0 <= y < data.shape[1]:
    #                 line_mask[x, y] = True

    #     # Step 4: Keep only connected line segments > 40 pixels
    #     labelled = measure.label(line_mask)  # FIXED: from skimage.measure
    #     final_mask = np.zeros_like(line_mask)
    #     for region in measure.regionprops(labelled):
    #         if region.area > 40:
    #             final_mask[labelled == region.label] = True

    #     # Step 5: Inpaint/replace lines by median of surroundings
    #     cleaned = data.copy()
    #     dilated_mask = morphology.binary_dilation(final_mask, morphology.disk(1))
    #     rows, cols = np.where(dilated_mask)
    #     for r, c in zip(rows, cols):
    #         rr_min, rr_max = max(r-3, 0), min(r+4, data.shape[0])
    #         cc_min, cc_max = max(c-3, 0), min(c+4, data.shape[1])
    #         patch = cleaned[rr_min:rr_max, cc_min:cc_max]
    #         patch_mask = final_mask[rr_min:rr_max, cc_min:cc_max]
    #         cleaned[r, c] = np.median(patch[~patch_mask])
            
    #     return cleaned
