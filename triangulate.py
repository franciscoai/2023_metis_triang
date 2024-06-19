"""
Created by Giuseppe Nisticò.
march 25, 2022
Performing geometric triangulation between two stereoscopic images.

To run the program, write in the IPython shell the following command

run triangulate.py 'data_example/195_diff.fits' 'data_example/aia_diff.fits' 'data_example/my_outputs.txt'

Yara change

"""
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button, TextBox #, RangeSlider
import astropy 
import sunpy
from sunpy.map import Map
import datetime
#from sunpy.coordinates import frames
#from mpl_toolkits.mplot3d import Axes3D
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates.ephemeris import get_horizons_coord
import os
import astroquery
HOME = os.environ.get('HOME') 

    
#############################
plt.ion()

x1 = np.nan
y1 = np.nan
x2 = np.nan
y2 = np.nan
global save_file, dataout, minint1, maxint1, minint2, maxint2, imga, imgb, map0, map1, check, overwrite

check=0
dataout = []
##########################################
### Functions to perform triangulation ###
##########################################
def make_los(xpix, ypix, map):
    LOS_LENGHT = 10 # Lenght of LOS , original was 2
	# converting pixels to hpc  
    hdr = map.fits_header
    p = map.pixel_to_world(xpix*u.pix, ypix*u.pix)
    d = hdr['DSUN_OBS']
    # defining the line-of-sight
    npoint = 5001 # number of points which define the LOS
    z = (LOS_LENGHT*np.arange(npoint)/(npoint-1) - 1.) * (2.*6.9657e8)
    x = np.ones(npoint)*(d-z)*np.tan(p.Tx.value/3600.*np.pi/180.)
    y = np.ones(npoint)*(d-z)*np.tan(p.Ty.value/3600.*np.pi/180.)
	
    return x, y, z # returning the heliocentric 3D coordinates relative to the spacecraft
	
def make_proj_los(x,y,z,d, map):
    thetax = np.arctan2(x, d-z)*180./np.pi*3600.
    thetay = np.arctan2(y, d-z)*180./np.pi*3600.
    coords = SkyCoord(thetax*u.arcsec, thetay*u.arcsec, frame=map.coordinate_frame)
    pix = map.world_to_pixel(coords)
    print('in make_proj_los Pix', pix.x)
    print('in make_proj_los Pix', pix.y)
    return pix.x,pix.y
	
def helio2heeq(p, lon, lat):
	
    m1 = [ [1, 0, 0],
     [0, np.cos(lat), -np.sin(lat)],
    [0, np.sin(lat), np.cos(lat)]]

    m2 = [ [-np.sin(lon), np.cos(lon), 0],
       [0, 0, 1],
    [np.cos(lon), np.sin(lon), 0]]

    m1 = np.array(m1)
    m2 = np.array(m2)

    pheeq = m2.transpose() @ m1.transpose() @ p

    return pheeq
	
def heeq2helio(pheeq, lon, lat):
	
    m1 = [ [1, 0, 0],
 	[0, np.cos(lat), -np.sin(lat)],
 	[0, np.sin(lat), np.cos(lat)]]
	
    m2 = [ [-np.sin(lon), np.cos(lon), 0],
 	[0, 0, 1],
 	[np.cos(lon), np.sin(lon), 0]]
	
    m1 = np.array(m1)
    m2 = np.array(m2)
    p = m1 @ m2 @ pheeq
	
    return p
	
def calculate_3dcoord( x, y, los, los_heeq):
	
    # calculation of the distance
    d2 = (los[0,:] - x)**2. + (los[1,:] - y)**2.
    # finding the minimum value and the corresponding index
    aa = np.where( d2 == d2.min())
    print('aa=',aa)
    ss = aa[0][0]
    # find the coordinate on the LO
    p0 = los[:,ss]
    p1 = los[:,ss+1]
    d02 = (p0[0] - x)**2. + (p0[1] - y)**2. 
    d12 = (p1[0] - x)**2. + (p1[1] - y)**2. 
    m = (p1[1] - p0[1])/(p1[0] - p0[0])    
    xlos = ( (d02 - d12)/(1 + m**2.) + p1[0]**2. - p0[0]**2.)/(2 * (p1[0] - p0[0]))
    ylos = m * (xlos - p0[0]) + p0[1]
    
    frac =  np.sqrt((xlos - p0[0])**2 + (ylos - p0[1])**2.)/np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2.)
    
    p0heeq = los_heeq[:,ss]
    p1heeq = los_heeq[:,ss+1]  
    xheeq =  frac*(p1heeq[0] - p0heeq[0]) + p0heeq[0]    
    yheeq =  frac*(p1heeq[1] - p0heeq[1]) + p0heeq[1]    
    zheeq =  frac*(p1heeq[2] - p0heeq[2]) + p0heeq[2]    
	
    print('Frac=', frac)
    return xlos, ylos, xheeq, yheeq, zheeq	
 

def load_images(map_b, map_a):
	
    # reading the fits files of the images
    #map_a = Map(file_a)
    #map_b = Map(file_b)

    fig = plt.figure(figsize=(12,6))
    axs = []
    axs.append(fig.add_subplot(1,2,1, projection=map_b))
    axs.append(fig.add_subplot(1,2,2, projection=map_a))

    #map_a.plot(axes=axs[1], vmin=map_a.data.min(), vmax=map_a.data.max())
    #map_b.plot(axes=axs[0], vmin=map_b.data.min(), vmax=map_b.data.max())
    imga = map_a.plot(axes=axs[1], vmin=minint2, vmax=maxint2)
    imgb = map_b.plot(axes=axs[0], vmin=minint1, vmax=maxint1)
    
    
    return fig, axs, imga, imgb
#############################
def click_point(event):
    # the axes instance
    xd = event.xdata
    yd = event.ydata
    panel = event.inaxes
    return xd, yd, panel
     
def on_move(event):
    global linedrawn, x0, y0, x1, y1, pheeq, xheeq, yheeq, zheeq, losx1, losy1, losx0, losy0, check
    
    print('Check line 150=',check)
    if event.button is MouseButton.LEFT:# and x1 == 0./0. and y1 == 0./0.:
         if check == int(1):
              linedrawn.pop(0).remove()
         
         check=int(1)
        
         x, y, panel = click_point(event)
         px = panel.set_xlim()
         py = panel.set_ylim()
         if (panel == axs[0]):
              x0 = x                                              
              y0 = y
              axs[0].plot([x0,x0], [y0,y0], 'r+')
              #####################
              hcx0, hcy0, hcz0 = make_los(x0, y0, map0) # creating the LOS
              pheeq = helio2heeq([hcx0,hcy0,hcz0], lon0, lat0) # in HEEQ coordinates
              hcx1, hcy1, hcz1 = heeq2helio(pheeq, lon1, lat1) # deproject to A
              losx1, losy1 = make_proj_los(hcx1,hcy1,hcz1,d1, map1) # create the LOS in B
              linedrawn = axs[1].plot(losx1, losy1, color='black')
              fig.canvas.draw()
              print('x=',x0, 'y=',y0, 'panel=',panel)
         elif (panel == axs[1]):
              x1 = x
              y1 = y
              axs[1].plot([x1,x1], [y1,y1], 'r+')
              #####################
              hcx1, hcy1, hcz1 = make_los(x1, y1, map1) # creating the LOS
              pheeq = helio2heeq([hcx1,hcy1,hcz1], lon1, lat1) # in HEEQ coordinates
              hcx0, hcy0, hcz0 = heeq2helio(pheeq, lon0, lat0) # deproject to B
              losx0, losy0 = make_proj_los(hcx0,hcy0,hcz0,d0, map0) # create the LOS in B
              linedrawn = axs[0].plot( losx0, losy0, color='black')
              fig.canvas.draw()
              print('x=',x1, 'y=',y1, 'panel=',panel)
    if event.button is MouseButton.RIGHT:# x1 != np.nan and y1 != np.nan:
         x, y, panel = click_point(event)
         px = panel.set_xlim()
         py = panel.set_ylim()
         
         check=int(0)
         
         if (panel == axs[0]):
              x0 = x
              y0 = y
              print('data coords %f %f' % (x0, y0))
              axs[0].plot([x0,x0], [y0,y0], 'r+')
              los0 = [losx0, losy0]
              los0 = np.array(los0)

              xlos, ylos, xheeq, yheeq, zheeq = calculate_3dcoord(x0,y0,los0, pheeq)
              x0 = xlos
              y0 = ylos
              print('data coords after triang %f %f' % (x0, y0))
         elif (panel == axs[1]):
              x1 = x
              y1 = y
              print('data coords %f %f' % (x1, y1))
              axs[1].plot([x1,x1], [y1,y1], 'r+')
              los1 = [losx1, losy1]
              los1 = np.array(los1)
              xlos, ylos, xheeq, yheeq, zheeq = calculate_3dcoord(x1,y1,los1, pheeq)
              x1 = xlos
              y1 = ylos
              print('data coords after triang %f %f' % (x1, y1))
         linedrawn.pop(0).remove()
         fig.canvas.draw()
 
###################################
########## Button functions #######                      
def select_points(label):
    print('Start ')
    global binding_id
    #binding_id = plt.connect('button_press_event', on_move)
    binding_id = fig.canvas.mpl_connect('button_press_event', on_move)	
    print('Bind', binding_id)
    
    
def stop_selecting_points(label):
    global check
    print('Stop selecting points')
    #plt.disconnect('all')
    check = int(0)
    fig.canvas.mpl_disconnect(binding_id)
    
def save_points(label):
    global check
    dataout.append([x0,y0,x1,y1, xheeq/6.96e8, yheeq/6.96e8, zheeq/6.96e8])
    print('Data saved in an array')
    check = int(0)
        
def close_plot(val):
    if ((not os.path.exists(save_file)) or overwrite=='True'):
        print('Saving triangulation points ' + save_file)
        np.savetxt(save_file, dataout, delimiter=' ', fmt='%10.5f')
    else:
        print('ERROR: The file was not saved because it exist and overwrite is False')
    plt.close('all')
    
    
"""    
def update(val):
    # The val passed to a callback by the RangeSlider will
    # be a tuple of (min, max)

    # Update the image's colormap
    imgb.norm.vmin = val[0]
    imgb.norm.vmax = val[1]
    # Redraw the figure to ensure it updates
    fig.canvas.draw_idle()

"""    
def submit1min(expression):
    minint1 = eval(expression)
    imgb.norm.vmin=minint1
     #map1.plot(axes=axs[0], vmin=minint1, vmax=maxint1) 
    fig.canvas.draw_idle()

def submit1max(expression):
    maxint1 = eval(expression)
    imgb.norm.vmax=maxint1          
    #map1.plot(axes=axs[0], vmin=minint1, vmax=maxint1)
    fig.canvas.draw_idle()
      
	    
def submit2min(expression):
    minint2= eval(expression)
    imga.norm.vmin=minint2 
    fig.canvas.draw_idle()
     
def submit2max(expression):
    maxint2 = eval(expression)
    imga.norm.vmax=maxint2 
    fig.canvas.draw_idle()

def proc_lasco_header(hdr):
    hdr1=hdr
    if hdr1['TELESCOP'] == 'SOHO': 
        coordL2 = get_horizons_coord(-21, datetime.datetime.strptime(hdr1['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"), 'id')
        coordL2carr = coordL2.transform_to(sunpy.coordinates.frames.HeliographicCarrington(observer='earth'))
        coordL2ston = coordL2.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
        hdr1['DSUN_OBS'] = coordL2.radius.m
        hdr1['CRLT_OBS'] = coordL2carr.lat.deg
        hdr1['CRLN_OBS'] = coordL2carr.lon.deg
        hdr1['HGLT_OBS'] = coordL2ston.lat.deg
        hdr1['HGLN_OBS'] = coordL2ston.lon.deg
    return hdr1
    

################# Main program ###############################
overwrite = sys.argv[1]
event_name = sys.argv[2]
file0 = sys.argv[3]#HOME+'/python/3dloop/code/20090213_055530_n4euB.fts'
file1 = sys.argv[4]# HOME+'/python/3dloop/code/20090213_055530_n4euA.fts'

if len(sys.argv) == 6:
    save_file = sys.argv[5]
else:
    save_file = "C:\\Users\\deleo\\Desktop\\metis_py\\mendoza_cmes\\triang_points\\"
    save_file = os.path.join(save_file,os.path.basename(file0)+"_"+os.path.basename(file1+'_'+event_name+'.txt'))
    
    
#creating the maps
Rsun_m=6.957e08  
#print(Rsun_m)
print('Loading '+ os.path.basename(file0) + ' in Map0')
print('Loading '+ os.path.basename(file1) + ' in Map1')


if 'solo_L2_metis' in file0:
    hdul = astropy.io.fits.open(file0)[0]
    data,header=hdul.data,hdul.header
    print(np.shape(data))
    header['RSUN'] = header['RSUN_ARC']
    map0 = sunpy.map.Map(data, header)
    angle=map0.fits_header['SC_ROLL']
    #map0=map0.rotate(angle=-angle*u.deg)
else:
    map0 = Map(file0)  
    
if 'solo_L2_metis' in file1:
    print('ERROR:to use metis file pass it as the first argument')

map1 = Map(file1)


hdr1 = map1.fits_header
hdr0 = map0.fits_header

if 'LASCO' in hdr1['INSTRUME']:
    hdr1=proc_lasco_header(hdr1)
    print(hdr1['DSUN_OBS'])
if 'LASCO' in hdr0['INSTRUME']:
    hdr0=proc_lasco_header(hdr0)
    print(hdr0['DSUN_OBS'])
    
d1 = hdr1['DSUN_OBS']
lon1 = hdr1['HGLN_OBS']*np.pi/180.
lat1 = hdr1['HGLT_OBS']*np.pi/180.
d0 = hdr0['DSUN_OBS']
lon0 = hdr0['HGLN_OBS']*np.pi/180.
lat0 = hdr0['HGLT_OBS']*np.pi/180.



    
#cmaps
# map0.plot_settings['cmap'] = 'Greys_r'
# map1.plot_settings['cmap'] = 'Greys_r'

# minint1=np.percentile(map0.data,3)# map0.data.min() #ydl: general value valid for all the images
# maxint1=np.percentile(map0.data,99)  # map0.data.max()
# minint2=np.percentile(map1.data,3) # np.mean(map1.data) - 1.5* np.std(map1.data) #map1.data.min()
# maxint2=np.percentile(map1.data,99) # np.mean(map1.data) + 1.5* np.std(map1.data) #map1.data.max()

minint1=np.nanmean(map0.data) - 2* np.nanstd(map0.data)
maxint1=np.nanmean(map0.data) + 2* np.nanstd(map0.data)
minint2=np.nanmean(map1.data) - 2* np.nanstd(map1.data)
maxint2=np.nanmean(map1.data) + 2* np.nanstd(map1.data)

print('Using ranges ',minint1,maxint1,minint2,maxint2)
fig, axs, imga, imgb = load_images(map0, map1)
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
axs[0].set_xlim()
axs[0].set_ylim()
axs[1].set_xlim()
axs[1].set_ylim()

#---BUTTON-----
# xposition, yposition, width, height
ax1_button = plt.axes([0.05, 0.01, 0.08, 0.05])
ax2_button = plt.axes([0.15, 0.01, 0.08, 0.05])
ax3_button = plt.axes([0.25, 0.01, 0.08, 0.05])
ax4_button = plt.axes([0.35, 0.01, 0.08, 0.05])
#ax5_button = plt.axes([0.45, 0.01, 0.08, 0.05])
"""
# slider
# Create the RangeSlider
slider_ax = plt.axes([0.10, 0.90, 0.30, 0.08])
slider = RangeSlider(slider_ax, "Threshold", map0.data.min(), map0.data.max())
"""

# properties of the button
take_button  = Button(ax1_button, 'Start', color='white', hovercolor='gray')
end_button   = Button(ax2_button, 'End ', color='white', hovercolor='gray')
save_button  = Button(ax3_button, 'Save', color='white', hovercolor='gray')
#foot_button
close_button = Button(ax4_button, 'Close', color='white', hovercolor='gray')

# properties of the text boxes
t1minbox = plt.axes([0.15,0.94,0.08,0.05])
text1min = TextBox(t1minbox, "Min")
text1min.on_submit(submit1min)
#text1min.set_val(map0.data.min())

t1maxbox = plt.axes([0.45,0.94,0.08,0.05])
text1max = TextBox(t1maxbox, "Max")
text1max.on_submit(submit1max)
#text1max.set_val(map0.data.max())

t2minbox = plt.axes([0.65,0.94,0.08,0.05])
text2min = TextBox(t2minbox, "Min")
text2min.on_submit(submit2min)
#text2min.set_val(map1.data.min())

t2maxbox = plt.axes([0.85,0.94,0.08,0.05])
text2max = TextBox(t2maxbox, "Max")
text2max.on_submit(submit2max)
#text2max.set_val(map1.data.max())
"""

#################################################


slider.on_changed(update)
""" 
close_button.on_clicked(close_plot) 
take_button.on_clicked(select_points)
end_button.on_clicked(stop_selecting_points)
save_button.on_clicked(save_points)


