# AstroColour
**Colour the Astro Images in bands/colours.**

## Install Instructions
```
git clone https://github.com/ZacharyLane1204/AstroColour.git
```
In the Folder: e.g. 
```cd AstroColour/``` Run: ```pip install .```
If you are in a Windows environment you should use powershell.

## 

in your script import the module. Here is an example:
```
from AstroColour.AstroColour import RGB

rgb = RGB(data_cube,
          save = False, save_name = 'test', save_folder = '/Users/zgl12/', 
          epsf_plot=False, epsf = True,
          bkg_plot = False, temp_save = True, run = True)

calib_images = rgb.calib_images
rgb.master_plot(calib_images, 
                colours = ['red', 'green', 'blue'], 
                intensities = [0.55, 1, 0.55], 
                gamma = 1.2,
                norms = ['asinh', 'asinh', 'asinh'], 
                uppers = [99, 99, 99], 
                lowers = [5, 5, 5], 
                interactive=True)
```

```
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
```

Versions:
- numpy == 1.26.4
- matplotlib == matplotlib
- astropy == 7.1.0
- pandas == 2.2.3
- photutils == 1.13.0
- scikit-learn == 1.5.1
- opencv-python == 4.9.0.80
- astroscrappy == 1.2.0
- ipympl == 0.9.7
- ipython == 8.28.0
- ipywidgets == 7.8.4