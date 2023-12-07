"""
Created by Giuseppe Nisticò
March 25, 2022.

An example of program on how to plot the 3D coordinate data and 
performing PCA analysis. 
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import stereoscopy as stereo

plt.ion()

#------------------------------------------------
ft='t_304_inn_133045.txt'#'my_outputs.txt'
#ft = '/gehme/projects/2023_metis_yara/output/my_outputs.txt'

# reading the content of file ft
points = stereo.read_triangulation_output(ft)

#plotting the data points in the HEEQ reference frames
fig, ax = stereo.plot_heeq_projections(points, 'blue')

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

fig.savefig('heeq_'+ft[6:16]+'.png', dpi=300,bbox_inches='tight')
fig1.savefig('local_'+ft[6:16]+'.png', dpi=300,bbox_inches='tight')


#plotting in 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=10,azim=5) 
#plot the points
ax.scatter(points[0,:],points[1,:],points[2,:], color='blue')
ax.plot(points[0,:],points[1,:],points[2,:], color='blue')
#plots the loop
#ax.scatter(loop_heeq[0,:],loop_heeq[1,:],loop_heeq[2,:],color='red', s=0.2)
# Generate a sphere with center (0, 0, 0) and radius 1
theta = np.linspace(0, 2*np.pi, 200)
phi = np.linspace(0, np.pi, 200)
X = np.outer(np.cos(theta), np.sin(phi))
Y = np.outer(np.sin(theta), np.sin(phi))
Z = np.outer(np.ones(np.size(theta)), np.cos(phi))
# Plot the sphere
ax.plot_surface(X, Y, Z)
# Set aspect ratio to be equal
ax.set_aspect('auto') #ax.set_aspect('equal')
# St labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set view angle: elevation and orizontal

# Set title for the plot
ax.view_init(elev=20,azim=50) 
ax.set_title('3D Plot of Points and Sphere')
ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
#plt.savefig('3d_surface_'+ft[6:16]+'.png', dpi=300,bbox_inches='tight')
# Show the plot
plt.show()
#breakpoint()
