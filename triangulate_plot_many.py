# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:00:09 2023

@author: deleo
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import stereoscopy as stereo

#plt.ion()
#------------------------------------------------
def prominence_plot(namef, opath):
    # reading the content of file namef
    points=stereo.read_triangulation_output(namef)
    print(points)
    
    #plotting the data points in the HEEQ reference frames
    if 'inn' in namef:
        color='red'
    else:
        color='blue'
    fig, ax = stereo.plot_heeq_projections(points, color)
    # calculation of the midpoint
    midpoint = np.array(0.5 * (points[:,0] + points[:,-1]))
    #print('Midpoint=', midpoint)
    
    #performin PCA
    evalue, evector = stereo.pca(points, midpoint)
    
    # plot PCA eigenvectors and defining the best-fitting curve
    stereo.plot_pca_heeq(ax, points, midpoint, evalue, evector)
    ss, loop, loop_heeq = stereo.define_loop(midpoint, evalue, evector)
    stereo.plot_loop(loop_heeq, midpoint, ax)
    
    # plot the data points and the best-fitting curve in the local reference frame
    fig1, ax1 = stereo.plot_loop_local_frame(points, midpoint, loop_heeq, evalue, evector)
    
    fig.savefig(os.path.join(opath,'heeq_'+os.path.basename(namef)+'.png'), dpi=300,bbox_inches='tight')
    fig1.savefig(os.path.join(opath,'local_'+os.path.basename(namef)+'.png'), dpi=300,bbox_inches='tight')
    
    plt.close()
    plt.close()
    
    return points

def plot_3d(points, color, ax=None, show=None, fig=None, opath=None):
    if ax is None:
        #plotting in 3D
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d') 
        # Generate a sphere with center (0, 0, 0) and radius 1
        theta = np.linspace(0, 2*np.pi, 200)
        phi = np.linspace(0, np.pi, 200)
        X = np.outer(np.cos(theta), np.sin(phi))
        Y = np.outer(np.sin(theta), np.sin(phi))
        Z = np.outer(np.ones(np.size(theta)), np.cos(phi))
        ax.plot_surface(X, Y, Z)  
        # Set aspect ratio to be equal
        ax.set_aspect('auto') #ax.set_aspect('equal')
        # St labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #plot the points
        ax.scatter(points[0,:],points[1,:],points[2,:], color=color)
        ax.plot(points[0,:],points[1,:],points[2,:], color=color)
        # Set view angle: elevation and orizontal
        ax.view_init(elev=20,azim=50) 
        ax.set_title('3D Plot of the promincence evolution')
        ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))   
    else:
        #plot the points
        ax.scatter(points[0,:],points[1,:],points[2,:], color=color)
        ax.plot(points[0,:],points[1,:],points[2,:], color=color)        
    if show is not None:
       plt.show()
       fig.savefig(os.path.join(opath,'3D.png'), dpi=300,bbox_inches='tight')
    return ax, fig
       
####################################################
####MAIN
####################################################

# input path
path = "C:/Users/deleo/Desktop/metis_py/mendoza_cmes/triang_points"
files=['t_AIA_EUVI_304_133045_d1.txt','t_AIA_EUVI_304_140045_d1.txt'] # 't_met_cor2_d2_183830.txt','t_met_cor2_d2_185330.txt'
colors=['darkorange','red','blue','green','darkred']
opath = "C:/Users/deleo/Desktop/metis_py/mendoza_cmes/triang_plots"

#creates opath
os.makedirs(opath,exist_ok=True)
# plotting 
points =  []

for i in range(len(files)):
    points.append(prominence_plot(os.path.join(path,files[i]), opath))
    
#plotting in 3D all together
ax, fig = plot_3d(points[0],colors[0])
for i in range(len(files)-1):
    plot_3d(points[i+1],colors[i+1], ax=ax)
plot_3d(points[len(files)-1],colors[len(files)-1], ax=ax, show=True, fig=fig, opath=opath)










