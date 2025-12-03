import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

import ipywidgets as widgets
from IPython.display import display, clear_output

# from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.convolution import convolve
# from astropy.convolution import convolve_fft

from astroscrappy import detect_cosmics

from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import fourier_shift
from scipy.ndimage import gaussian_filter

from photutils.psf.matching import create_matching_kernel, TukeyWindow

from skimage.registration import phase_cross_correlation
from skimage import morphology, measure
from skimage.filters import sobel
from skimage.morphology import binary_dilation

# from AstroColour.image_rotation import unrotate_image
# from AstroColour.psf_analysis import PSF_Analysis, Simple_ePSF
# from AstroColour.spatial_psf_matcher import SpatialPSFMatcher
from AstroColour.pdastro import pdastrostatsclass
from AstroColour.hidden_prints import hidden_prints
from AstroColour.analysis_tools import Image_Analysis

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
                 save = False, save_name = '', save_folder = '',
                 figure_size = None, manual_override = None, dpi = 900,
                 epsf = True, epsf_plot = False, bkg_plot = False, 
                 temp_save = True, run = True):
        '''
        Create a RGB image from three images.

        Parameters
        ----------
        images : List
            List of Numpy Arrays of data images.
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
        epsf : Boolean
            Whether to use the EPSF method.
        epsf_plot : Boolean
            Whether to plot the EPSF kernel.
        run : Boolean
            Whether to process images or just use the framework
        '''

        self.save = save
        self.save_name = save_name
        self.save_folder = save_folder
        self.dpi = dpi
        self.epsf = epsf

        self.figure_size = figure_size
        if self.figure_size == None:
            self.figure_size = fig_width_full

        if manual_override == None:
            manual_override = 100
        self.manual_override = manual_override
        
        if run:
            ianalysis = Image_Analysis(images, epsf_plot = epsf_plot, bkg_plot = bkg_plot, temp_save = temp_save)
            self.calib_images = ianalysis.calib_images
            self.diff_images = ianalysis.diff_images
                
    def _master_plot(self, cleaned_images, lowers, uppers, norms, colours, intensities, gamma, method):
        
        length = len(cleaned_images)
        
        if len(colours) != length:
            raise ValueError("Length of colours does not match the number of images.")
        
        if isinstance(intensities, float): 
            intensities = [intensities] * len(cleaned_images)
        elif len(intensities) != length:
            raise ValueError("Length of intensities does not match the number of images.")
        
        if isinstance(uppers, float): 
            uppers = [uppers] * len(cleaned_images)
        elif len(uppers) != length:
            raise ValueError("Length of uppers does not match the number of images.")
        
        if isinstance(lowers, float): 
            lowers = [lowers] * len(cleaned_images)
        elif len(lowers) != length:
            raise ValueError("Length of lowers does not match the number of images.")
        
        if isinstance(gamma, float): 
            gamma = [gamma] * len(cleaned_images)
        elif len(gamma) != length:
            raise ValueError("Length of lowers does not match the number of images.")
        
        if isinstance(norms, str):
            if ['linear', 'sqrt', 'log', 'asinh', 'sinh'].__contains__(norms):
                norms = [norms]*length
            else:
                raise ValueError("Not a valid norm. Try 'linear', 'sqrt', 'log', 'asinh' or 'sinh'.")
            
        elif isinstance(norms, list) & (len(norms) == length):
            new_norms = []
            for i in range(len(norms)):
                if not ['linear', 'sqrt', 'log', 'asinh', 'sinh'].__contains__(norms[i]):
                    raise ValueError("Not a valid norm. Try 'linear', 'sqrt', 'log', 'asinh' or 'sinh'.")
                else:
                    new_norms.append(norms[i])
            norms = new_norms
        
        files = []
        for i in range(len(cleaned_images)):
            
            coloured_image = self.running_norm_colour(cleaned_images[i], 
                                                lower = lowers[i], upper = uppers[i], 
                                                norm = norms[i], colour_choice = colours[i], 
                                                gamma = gamma[i], intensity = intensities[i], method = method)
            
            files.append(coloured_image)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im_composite = np.clip(np.nansum(files, axis=0), 0, 5)
        im_composite = self.plot(im_composite)
        
        return im_composite
    
    def master_plot(self, cleaned_images, lowers, uppers, norms, colours, 
                    intensities, gamma, interactive = True, method = 'percent'):
        """
        Master RGB composite plotter.
        
        Parameters
        ----------
        cleaned_images : List
            List of cleaned 2D numpy arrays.
        lowers : Float or List of Float
            Lower percentile(s) (or counts) for normalization.
        uppers : Float or List of Float
            Upper percentile(s) (or counts) for normalization.
        norms : String or List of String
            Normalization method(s) ('linear', 'sqrt', 'log', 'asinh', 'sinh').
        colours : List
            List of colour choices (tuples or strings).
        intensities : Float or List of Float
            Intensity scaling factor(s).
        gamma : Float or List of Float
            Gamma correction factor(s).
        interactive : Boolean
            Whether to use interactive widgets for parameter adjustment.
        method : String
            Normalization method type ('percent', 'sigma', 'counts').
        
        Returns
        -------
        im_comp : 2D array
            The final RGB composite image.
        """
        
        if interactive:
            return self.master_plot_interactive(cleaned_images, lowers, uppers, norms, colours, intensities, gamma, method)
        else:
            im_comp =  self._master_plot(cleaned_images, lowers, uppers, norms, colours, intensities, gamma, method)
            return im_comp

    def master_plot_interactive(self, cleaned_images, lowers, uppers, norms, colours, intensities, gammas, method):
        """
        Interactive RGB composite plotter for Jupyter notebooks.
        Gives per-channel controls for lower, upper, norm, and intensity,
        plus a shared gamma slider.
        """

        n_channels = len(cleaned_images)
        
        lower_sliders = [widgets.FloatSlider(value=lowers[i], min=0, max=50, step=0.5, description=f'Lower {colours[i]}')
                         for i in range(n_channels)]
        upper_sliders = [widgets.FloatSlider(value=uppers[i], min=50, max=100, step=0.5, description=f'Upper {colours[i]}')
                         for i in range(n_channels)]
        intensity_sliders = [widgets.FloatSlider(value=intensities[i], min=0.1, max=3, step=0.05, description=f'Intensity {colours[i]}')
                             for i in range(n_channels)]
        norm_dropdowns = [widgets.Dropdown(options=['linear', 'sqrt', 'log', 'asinh', 'sinh'], value=norms[i],
                                           description=f'Norm {colours[i]}') for i in range(n_channels)]
        
        gamma_sliders = [widgets.FloatSlider(value=gammas[i], min=0.05, max=5, step=0.05, description=f'Gamma {colours[i]}')
                         for i in range(n_channels)]
        
        # gamma_slider = widgets.FloatSlider(value=gamma, min=0.1, max=5, step=0.1, description='Gamma (all)')

        go_button = widgets.Button(description="Update Plot", button_style='success')

        def update_plot(_):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                files = []
                for i in range(n_channels):
                    col_img = self.running_norm_colour(
                        cleaned_images[i],
                        lower=lower_sliders[i].value,
                        upper=upper_sliders[i].value,
                        norm=norm_dropdowns[i].value,
                        colour_choice=colours[i],
                        gamma=gamma_sliders[i].value,
                        intensity=intensity_sliders[i].value,
                        method=method
                    )
                    files.append(col_img)
                im_composite = np.clip(np.nansum(files, axis=0), 0, 5)

            clear_output(wait=True)
            display(ui)  # Redisplay widgets
            plt.figure(figsize=(6, 6))
            plt.imshow(im_composite)
            plt.axis('off')
            plt.show()

        go_button.on_click(update_plot)

        channel_controls = []
        for i in range(n_channels):
            channel_controls.append(
                widgets.VBox([
                    lower_sliders[i],
                    upper_sliders[i],
                    intensity_sliders[i],
                    norm_dropdowns[i], 
                    gamma_sliders[i]
                ])
            )

        ui = widgets.VBox(channel_controls + [go_button])
        display(ui)

        update_plot(None)

    def running_norm_colour(self, image, lower = 2, upper = 98, 
                            norm = 'linear', colour_choice = 'red', 
                            gamma = 1, intensity = 1, method = 'percent'):
                
        p_norm = self.percent_norm(image, lower = lower, upper = upper, norm = norm, gamma = gamma, method = method)
        colour = self.colour_check(colour_choice)
                
        coloured_image = self.colourise(p_norm, colour, intensity)
        
        return coloured_image
    
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

    def percent_norm(self, x, lower = 2, upper = 98, norm = 'linear', gamma = 1, method = 'percent'):
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
        
        if method.lower() == 'percent':
            normed = simple_norm(x, stretch=norm, min_percent=lower, max_percent=upper)
        elif method.lower() == 'sigma':
            mean, median, std = sigma_clipped_stats(x, sigma_upper = upper, sigma_lower = lower, maxiters = 5)
            vmin = median - (lower * std)
            vmax = median + (upper * std)
            normed = simple_norm(x, stretch=norm, vmin=vmin, vmax=vmax)
        elif method.lower() == 'counts':
            normed = simple_norm(x, stretch=norm, vmin=vmin, vmax=vmax)
        else:
            raise ValueError("Not a valid method. Try 'percent', 'sigma' or 'counts'.")
        
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
        plt.imshow(im_composite, origin = 'lower')
        plt.axis('off')
        plt.xlim(self.manual_override , im_composite.shape[1] - self.manual_override )
        plt.ylim(self.manual_override , im_composite.shape[0] - self.manual_override )
        if self.save:
            plt.savefig(os.path.join(self.save_folder, self.save_name + '.pdf'), format = 'pdf', bbox_inches = 'tight')
            plt.savefig(os.path.join(self.save_folder, self.save_name + '.png'), dpi = self.dpi, bbox_inches = 'tight')
        plt.show()
        
        return im_composite
    
    def inpaint_masked_pixels(self, image, mask):
        image_norm = (image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image))
        image_8bit = np.uint8(image_norm * 255)
        mask_8bit = np.uint8(mask * 255)

        inpainted = cv2.inpaint(image_8bit, mask_8bit, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return inpainted.astype(float) / 255 * (np.nanmax(image) - np.nanmin(image)) + np.nanmin(image)
    
    # def positioning_aligning(self, positions_ref, positions_target, target_image):
    #     tree = cKDTree(positions_ref)
    #     dist, idx = tree.query(positions_target, distance_upper_bound=3)  # max matching radius
    #     matched_ref = positions_ref[idx[dist != np.inf]]
    #     matched_target = positions_target[dist != np.inf]
        
    #     tform = estimate_transform('similarity', matched_target, matched_ref)
        
    #     aligned_image = warp(target_image, inverse_map=tform.inverse, order=3, preserve_range=True)
        
    #     return aligned_image