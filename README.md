# AstroColour
**Colour the Astro Images in bands/colours.**

## Install Instructions
```
git clone https://github.com/ZacharyLane1204/AstroColour.git
```
In the Folder: e.g. 
```cd AstroColour/ Run: pip install .```
If you are in a Windows environment you should use powershell.

## 

in your script import the module. Here is an example:
```
from AstroColour.AstroColour import RGB

rgb = RGB(
    data_cube,
    colours=['red', 'green', 'blue'],
    intensities=[0.55, 1, 0.6],
    uppers=[99, 99, 99],
    lowers=[5, 5, 5],
    save=False,
    save_name='test',
    save_folder='/Users/zgl12/',
    gamma=1.5,
    norm='asinh',
    min_separation=29,
    star_size=5,
    epsf_plot=False,
    epsf=True
)
colour = rgb.plot()
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
min_separation : Float
  Minimum separation between stars.
star_size : Float
  Size of the stars for the EPSF method.
epsf_plot : Boolean
  Whether to plot the EPSF kernel.
```
