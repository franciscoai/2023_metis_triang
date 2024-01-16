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
import datetime

#plt.ion()

markers = ['*','.','*']

def cart2polar(x, y, z):
    """
    Converts 3D Cartesian coordinates to polar coordinates.

    Args:
        x: Numpy array of x coordinates.
        y: Numpy array of y coordinates.
        z: Numpy array of z coordinates.

    Returns:
        r: Numpy array of radial distances.
        theta: Numpy array of azimuthal angles.
        phi: Numpy array of polar angles.
    """

    r = np.sqrt(x**2 + y**2 + z**2)  # Calculate radial distance
    theta = np.arctan2(y, x)  # Calculate azimuthal angle
    phi = np.arccos(z / r)  # Calculate polar angle

    return r, theta, phi
#------------------------------------------------
def prominence_plot(namef, opath, savefig=True):
    '''
    Function description by Copilo
        Plot the data points and the best-fitting curve in the HEEQ reference frame.
        Parameters      
        ----------
        namef : str
            Name of the file containing the triangulation output.
        opath : str
            Path where to save the figure.
        savefig : bool, optional    
            Whether to save the figure.
        Returns
        -------
        points : array-like, shape (3, n)   
            Data points.
    '''

    # reading the content of file namef
    points=stereo.read_triangulation_output(namef)

    if savefig is True:
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

def plot_3d(points, color, ax=None, show=None, savefig=None, fig=None, opath=None):
    '''
    Function description by Copilot
        Plot points in 3D, on a sphere of radius 1 centered in (0, 0, 0).
        Parameters
        ----------
        points : array-like, shape (3, n)
            Points to plot.
        color : str
            Color of the points.
        ax : Axes3D, optional 
            Axes on which to plot.
        show : bool, optional
            Whether to show the plot.
        savefig : bool, optional    
            Whether to save the figure.
        fig : Figure, optional  
            Figure on which to plot.
        opath : str, optional   
            Path where to save the figure.
        Returns     
        ------- 
        ax : Axes3D
            Axes on which the points are plotted.
        fig : Figure    
            Figure on which the points are plotted. 
    '''
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
        # St labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #plot the points
        ax.scatter(points[0,:],points[1,:],points[2,:], color=color)
        ax.plot(points[0,:],points[1,:],points[2,:], color=color)
        # Set view angle: elevation and orizontal
        #ax.view_init(elev=20,azim=50) 
        ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))   
    else:
        #plot the points
        ax.scatter(points[0,:],points[1,:],points[2,:], color=color)
        ax.plot(points[0,:],points[1,:],points[2,:], color=color)                
    if show is not None:
       plt.show()       
    if savefig is not None:
       # Set aspect ratio to be equal
       ax.set_aspect('equal') #ax.set_aspect('auto') #           
       ax.set_title('3D Plot of feature '+savefig+' evolution')       
       fig.savefig(os.path.join(opath,'3D_'+savefig+'.png'), dpi=300,bbox_inches='tight')
    return ax, fig

def plot_Cartesian_vs_time(points, date, opath, feature, colors, instruments, all_features):
    '''
    Function description by Copilot
        Plot the x, y, z coordinates of the point groups  vs date in three separated subplots.
        Parameters
        ----------
        points : list of arrays per date, len n, for each n there are 3, m points
        date : list of datetime objects, len n
        opath : str
            Path where to save the figure.
        feature : str
            Name of the feature.
        colors : list of colors, len n
        Returns
        -------
        None
    '''
    fig, ax = plt.subplots(3,1,figsize=(20, 15))
    fig.suptitle('Evolution of the '+feature+' feature')
    # asign marker for each type of feature in all_features
    unique_features = np.unique(all_features)   
    all_markers = [markers[unique_features.tolist().index(f)] for f in all_features]
    for i in range(len(points)):
        ax[0].scatter(np.repeat(date[i],len(points[i][0,:])),points[i][0,:], color=colors[i], label=instruments[i], marker=all_markers[i])
        ax[1].scatter(np.repeat(date[i],len(points[i][1,:])),points[i][1,:], color=colors[i], label=instruments[i], marker=all_markers[i])
        ax[2].scatter(np.repeat(date[i],len(points[i][2,:])),points[i][2,:], color=colors[i], label=instruments[i], marker=all_markers[i])
    ax[0].set_ylabel('X')
    ax[1].set_ylabel('Y')
    ax[2].set_ylabel('Z')
    ax[2].set_xlabel('Date')
    #legend of only unique labels
    handles, labels = ax[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # overwrite legend marker to use 'o' but keeps original color
    for k in by_label.keys():
        by_label[k]._marker = markers[1]
    ax[0].legend(by_label.values(), by_label.keys())

    fig.savefig(os.path.join(opath,'cart_coord_'+feature+'.png'), dpi=300,bbox_inches='tight')
    plt.close()
       
# same as above but first converts to polar coordinates
def plot_Ploar_vs_time(points, date, opath, feature, colors, instruments, all_features):
    '''
    Function description by Copilot
        Plot the r, theta, phi coordinates of the point groups  vs date in three separated subplots.
        Parameters
        ----------
        points : list of arrays per date, len n, for each n there are 3, m points
        date : list of datetime objects, len n
        opath : str
            Path where to save the figure.
        feature : str
            Name of the feature.
        colors : list of colors, len n
        Returns
        -------
        None
    '''
    fig, ax = plt.subplots(3,1,figsize=(15, 10))
    fig.suptitle('Evolution of the '+feature+' feature')
    # asign marker for each type of feature in all_features
    unique_features = np.unique(all_features)   
    all_markers = [markers[unique_features.tolist().index(f)] for f in all_features]    
    for i in range(len(points)):
        r, theta, phi = cart2polar(points[i][0,:],points[i][1,:],points[i][2,:])
        ax[0].scatter(np.repeat(date[i],len(r)),r, color=colors[i], label=instruments[i], marker=all_markers[i])
        ax[1].scatter(np.repeat(date[i],len(theta)),theta, color=colors[i], label=instruments[i], marker=all_markers[i])
        ax[2].scatter(np.repeat(date[i],len(phi)),phi, color=colors[i], label=instruments[i], marker=all_markers[i])
    ax[0].set_ylabel('r')
    ax[1].set_ylabel('theta')
    ax[2].set_ylabel('phi')
    ax[2].set_xlabel('Date')
    #legend of only unique labels
    handles, labels = ax[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # overwrite legend marker to use 'o' but keeps original color
    for k in by_label.keys():
        by_label[k]._marker = markers[1]
    ax[0].legend(by_label.values(), by_label.keys())
   
    fig.savefig(os.path.join(opath,'polar_coord_'+feature+'.png'), dpi=300,bbox_inches='tight')
    plt.close()

# function to compute the angular distance between two 3d cartesian points
def angular_distance(p1, p2):
    '''
    Function description by Copilot
        Compute the angular distance between two 3D Cartesian points.
        Parameters
        ----------
        p1 : array-like, shape (3,)
            First point.
        p2 : array-like, shape (3,)
            Second point.
        Returns
        -------
        dist : float
            Angular distance between the two points.
    '''
    # normalize the points
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)
    # compute the dot product
    dot = np.dot(p1, p2)
    # compute the distance in degrees
    dist = np.degrees(np.arccos(dot))
    return dist

# function to compute the maximum angular distance between the points in each date and plot it vs date
def plot_max_angular_distance(points, date, opath, feature, colors, instruments, all_features):
    '''
    Function description by Copilot
        Plot the maximum angular distance between the points of each date vs date.
        Parameters
        ----------
        points : list of arrays per date, len n, for each n there are 3, m points
        date : list of datetime objects, len n
        opath : str
            Path where to save the figure.
        feature : str
            Name of the feature.
        colors : list of colors, len n
        Returns
        -------
        None
    '''
    fig, ax = plt.subplots(1,1,figsize=(12, 8))
    fig.suptitle('Feature ' + feature + ' angular width evolution')
    max_dist = []
    for i in range(len(points)):
        # computes the angle between each point and the others
        dist = []
        for j in range(len(points[i][0,:])):
            for k in range(j+1,len(points[i][0,:])):
                dist.append(angular_distance(points[i][:,j],points[i][:,k]))
        # computes the maximum angle
        max_dist.append(np.max(dist))
    unique_features = np.unique(all_features)
    all_markers = [markers[unique_features.tolist().index(f)] for f in all_features]
    for i in range(len(points)):
        ax.scatter(date[i],max_dist[i], color=colors[i], label=instruments[i], marker=all_markers[i])
    ax.set_ylabel('Angular width [deg]')
    ax.set_xlabel('Date')
   
    fig.savefig(os.path.join(opath,'angular_width_'+feature+'.png'), dpi=300,bbox_inches='tight')
    plt.close()

####################################################
####MAIN
####################################################

# input path
root_path = '/gehme/projects/2023_metis_triang'
path = root_path+'/Triangulation_files_yara/triang_output_files'
instruments = ['AIA_EUVI_304', 'COR1_LASCO_C2','COR2_LASCO_C2','Metis_UV_COR2']
all_features = ['d1','d2','d3','d2_and_d3']
inst_colors=['blue','green','red','violet'] # colors for each instrument pair
opath = root_path + '/output_plots'

for feature in all_features:
    # read input files
    full_paths = []
    colors=[]
    for i in instruments:
        files=os.listdir(path+'/'+i)
        full_paths.append([path+'/'+i+'/'+f for f in files])
        colors.append([inst_colors[instruments.index(i)] for f in files])
    full_paths = [item for sublist in full_paths for item in sublist]
    colors = [item for sublist in colors for item in sublist]
    # keeps full paths containing the feature
    if '_and_' in feature:
        colors = [colors[i] for i in range(len(full_paths)) if feature.split('_and_')[0] in full_paths[i] or feature.split('_and_')[1] in full_paths[i]]
        full_paths = [full_paths[i] for i in range(len(full_paths)) if feature.split('_and_')[0] in full_paths[i] or feature.split('_and_')[1] in full_paths[i]]
    else:
        colors = [colors[i] for i in range(len(full_paths)) if feature in full_paths[i]]
        full_paths = [full_paths[i] for i in range(len(full_paths)) if feature in full_paths[i]]
    
    # instrument name per point 
    all_instruments = [os.path.basename(os.path.dirname(f)) for f in full_paths]

    #creates opath
    os.makedirs(opath,exist_ok=True)
    # extracts datetime objects from file names
    date=[]
    for fp in full_paths:
        f = os.path.basename(fp)
        # if starts with 'cor1' it extracts the datetime with YYYY-MM-DDTHH_MM_SS format from char in pos 15
        if f.startswith('cor1'):
            date.append(datetime.datetime.strptime(f[15:34],'%Y-%m-%dT%H_%M_%S'))
        # if starts with 'cor2' it extracts the datetime with YYYY-MM-DDTHH_MM_SS format from char in pos 13
        elif  f.startswith('cor2'):
            date.append(datetime.datetime.strptime(f[13:32],'%Y-%m-%dT%H_%M_%S'))
        # if starts with 'solo' it extracts the datetime with YYYYMMDDTHHMMSS format from char in pos 24
        elif  f.startswith('solo'):
            date.append(datetime.datetime.strptime(f[23:38],'%Y%m%dT%H%M%S'))
        # if startsd with AIA it extracts the datetime with YYYY-MM-DDTHH:MM:SS format from char in pos 25
        elif f.startswith('AIA'):
            date.append(datetime.datetime.strptime(f[25:44],'%Y-%m-%dT%H_%M_%S'))        
        # else it extracts the datetime with YYYYMMDD_HHMMSS format from char in pos 0
        else:
            date.append(datetime.datetime.strptime(f[0:15],'%Y%m%d_%H%M%S'))\
    #sort by date
    date, full_paths = zip(*sorted(zip(date, full_paths)))

    # plotting 
    points =  []
    current_features = []
    for i in range(len(full_paths)):
        points.append(prominence_plot(full_paths[i], opath, savefig=False))
        for f in all_features:
            if f in os.path.basename(full_paths[i]):
                current_features.append(f)
        
    #plot in 3D all together
    ax, fig = plot_3d(points[0],colors[0])
    for i in range(len(full_paths)-1):
        plot_3d(points[i+1],colors[i+1], ax=ax)
    plot_3d(points[len(full_paths)-1],colors[len(full_paths)-1], ax=ax, show=None, savefig=feature, fig=fig, opath=opath)

    # plot x,y,z coordinate vs date
    plot_Cartesian_vs_time(points, date, opath, feature, colors, all_instruments, current_features)
    # plot r,theta,phi coordinate vs date
    plot_Ploar_vs_time(points, date, opath, feature, colors, all_instruments, current_features)
    # plot max angular distance vs date
    plot_max_angular_distance(points, date, opath, feature, colors, all_instruments, current_features)










