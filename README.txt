
ORIGINAL README:

This is a Python tool that performs geometric triangulation of coronal features observed in a pair of images coming from any space mission (SoHO, STEREO, SDO).
The code mimics the IDL/SolarSoftWare routine scc_measure.pro.

The code consists of two python programs:

a) the routine triangulate.py 

b) the module stereoscopy.py

We provide a very brief description and a step by step guide to the program.


a) triangulate.py 

The graphical interface of the routine is presented in Fig. 'Figure_traingulation.png'.

triangulate.py is tested in Python 3.8 and requires standard packages (e.g., Numpy, MatPlotLib) and needs Astropy and SunPy. We also provide two FITS files contained in the folder 'data_example', with the images data already processed for the analysis of them. The routine works by having as input the FITS files. If one would like to use any difference image or filtered image, it is convenient to process the data in advance and save them as FITS files with Sunpy.
To run the program in the IPython shell, the following call must be used:


>run triangulate.py 'path/file\_01.fits' 'path/file\_02.fits' 'path/output.dat'}

The graphic window  contains two panels: the right one shows the image from file_01.fits, the left one will plot file_02.fits. It is possible to set the contrast for the images by setting the min and max values of the intensity in the text boxes located above the panels (for image difference in EUV it is advised to use min=-10, max=10). Below the two panels, there are four buttons:'Start','End', 'Save', 'Close'.
The user should follow these steps for a correct execution of the program.

1. Run the program 'triangulate.py' {with the call mentioned above}.
2. Once the window is open, set the intensity contrast in the text boxes above the panels. Focus on the region of interest by using one of the 'matplotlib.pyplot' functions shown in the graphical interface.
3. To start collecting tie points, click on the 'Start' button.
4. Move with the cursor of the mouse to one of the panels and click on a feature with the left button of the mouse.
5. The LoS will be drawn in the other panel. To select the tie point, click with the 'right' button. If the 'left' button is clicked, the tie-point will not be selected but the LoS will be drawn in the other panel.
6. 3D coordinates for the points will be calculated. To save it, click on the 'Save' button. The button 'Save' must be clicked any time after taking the tie-point.
7. To disable temporarily the cursor of the mouse, click on the button 'End'. To restart and enable the cursor of the mouse, click on 'Start'.
8. To terminate the process and to close the application, click on 'Close'. The program will close and the saved 3D points will be written on the 'output.dat' file.


b) stereoscopy.py

The module 'stereoscopy.py' contains different functions for plotting the 3D data points in the HEEQ coordinate system, in the local reference frame, computing principal components and plotting the results of PCA.

The module can be imported in the IPython shell as:

import stereoscopy as stereo

In the 'data_example' folder we provide a short Python program  {run \_example\_obs\_1.py} that executes some functions of the 'stereoscopy' module.

    
