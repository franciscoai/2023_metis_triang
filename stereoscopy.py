import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random



from mpl_toolkits import mplot3d



# some settings
#plt.ion()
matplotlib.rcParams.update({'font.size': 16})
#------------------------------------------------
def read_triangulation_output(file):
    
    data = np.loadtxt(file)
    points = np.transpose(data[:, 4:7]) # transpose to get points array as [coordinate, number of points]
    
    return points
#------------------------------------------------    
def read_scc_measure_file(file):
    data = np.loadtxt(file) # very very good

    lon = data[:,0] * np.pi/180.
    lat = data[:,1] * np.pi/180.
    r = data [:,2]

    # calculation of the HEEQ coordinates

    x = r * np.cos(lon) * np.cos(lat)
    y = r * np.sin(lon) * np.cos(lat)
    z = r * np.sin(lat)
    
    points = np.zeros((3, len(x)))
    points[0,:] = x
    points[1,:] = y
    points[2,:] = z
    
    print('Functions executed.')
    return points
    
#------------------------------------------------   
def plot_heeq_projections(points, color_):
    border = 0.8
    fig, ax = plt.subplots(1,3,figsize=(12,4), constrained_layout=True)


    theta = (np.arange(0,181)-90.) * np.pi/180.
    phi = np.arange(0,361) * np.pi/180.

    x = 1. * np.outer(np.cos(theta), np.cos(phi))
    y = 1. * np.outer(np.cos(theta), np.sin(phi))
    z = 1. * np.outer(np.sin(theta), np.ones(np.size(phi)))


    range_x = 0.5*(max(points[0,:])+min(points[0,:]))
    range_y = 0.5*(max(points[1,:])+min(points[1,:]))
    range_z = 0.5*(max(points[2,:])+min(points[2,:]))

    ax[0].set_xlabel(r'$X_{HEEQ}$')
    ax[0].set_ylabel(r'$Y_{HEEQ}$')

    ax[1].set_xlabel(r'$X_{HEEQ}$')
    ax[1].set_ylabel(r'$Z_{HEEQ}$')

    ax[2].set_xlabel(r'$Y_{HEEQ}$')
    ax[2].set_ylabel(r'$Z_{HEEQ}$')

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[2].set_aspect('equal')

    ax[0].set_xlim(range_x-border, range_x+border)
    ax[0].set_ylim(range_y-border, range_y+border)

    ax[1].set_xlim(range_x-border, range_x+border)
    ax[1].set_ylim(range_z-border, range_z+border)

    ax[2].set_xlim(range_y-border, range_y+border)
    ax[2].set_ylim(range_z-border, range_z+border)



    for i in range(0,361,15):
    	ax[0].plot(x[:,i], y[:,i], 'k')
    	ax[1].plot(x[:,i], z[:,i], 'k')
    	ax[2].plot(y[:,i], z[:,i], 'k')
		
	
    for i in range(0,181,15):
    	ax[0].plot(x[i,:], y[i,:], 'k')
    	ax[1].plot(x[i,:], z[i,:], 'k')
    	ax[2].plot(y[i,:], z[i,:], 'k')
        
    ax[0].scatter(points[0,:],points[1,:], color=color_)
    ax[1].scatter(points[0,:],points[2,:], color=color_)
    ax[2].scatter(points[1,:],points[2,:], color=color_)    
        
    return fig, ax    
#------------------------------------------------
def overplot_points(points, ax, color):
    
    ax[0].scatter(points[0,:],points[1,:], color=color)
    ax[1].scatter(points[0,:],points[2,:], color=color)
    ax[2].scatter(points[1,:],points[2,:], color=color)
    
#def calculate_midpoint():    

def pca(points, midpoint):
    npoints = len(points[0,:])
    
    # points referred with respect to the midpoint
    xp = np.ravel(points[0,:])-midpoint[0]
    yp = np.ravel(points[1,:])-midpoint[1]
    zp = np.ravel(points[2,:])-midpoint[2]
    # construction of the array data
    data = [xp,yp,zp]
    data = np.array(data) # convert it into a numpy array
    #cov_matrix = np.cov(data) # calculate the covariance matrix
                              # alternatively
    cov_matrix = 1./npoints * data @ data.transpose()
    print('Covariance Matrix')
    print(cov_matrix)

    # calculating pca
    evalue_, evector_ = la.eigh(cov_matrix)    
    # sorting the eigenvalues in ascending order
    #evalue_sort = np.sort(evalue)
    index = np.argsort(evalue_)
    print('index=', index)
    evalue = evalue_[index]
    evector = evector_[:, index]
    # control that the eigenvectors define a right triad
    if (np.round(np.dot(np.cross(evector[:,1],evector[:,2]),evector[:,0])) == -1.0):
         print('Correction direction vectors applied!')
         evector[:,[0,2]] = -1. * evector[:,[0,2]]
         print('Correction applied!')
    print('n: eigenvalue=', evalue[0], ' eigenvector=',evector[:,0])
    print('a: eigenvalue=', evalue[1], ' eigenvector=',evector[:,1])
    print('b: eigenvalue=', evalue[2], ' eigenvector=',evector[:,2])
    print('n: eigenvalue=', np.sqrt(2*evalue[0]))
    print('a: eigenvalue=', np.sqrt(2.*evalue[1]))
    print('b: eigenvalue=', np.sqrt(2.*evalue[2]))
    return evalue, evector
    
def plot_pca_heeq(ax, points, midpoint, evalue, evector):
    
    npoints = float(len(points[:,0]))
    ax[0].scatter(midpoint[0], midpoint[1], color='yellow')
    ax[1].scatter(midpoint[0], midpoint[2], color='yellow')
    ax[2].scatter(midpoint[1], midpoint[2], color='yellow')

    # eigenvectors
    n_vec = evector[:,0]
    a_vec = evector[:,1]
    b_vec = evector[:,2]
    # plot the eigenvectors
    eax = np.sqrt(evalue[1]*2.)*a_vec[0]
    eay = np.sqrt(evalue[1]*2.)*a_vec[1]
    eaz = np.sqrt(evalue[1]*2.)*a_vec[2]  

    ebx = np.sqrt(evalue[2]*2.)*b_vec[0]
    eby = np.sqrt(evalue[2]*2.)*b_vec[1]
    ebz = np.sqrt(evalue[2]*2.)*b_vec[2]  

    enx = 0.05*n_vec[0]
    eny = 0.05*n_vec[1]
    enz = 0.05*n_vec[2]  
	# plotting the eigenvector a
    ax[0].arrow(midpoint[0],midpoint[1],eax,eay, width=0.002, length_includes_head=True, color='red')
    ax[1].arrow(midpoint[0],midpoint[2],eax,eaz, width=0.002, length_includes_head=True, color='red')
    ax[2].arrow(midpoint[1],midpoint[2],eay,eaz, width=0.002, length_includes_head=True, color='red')
    # plotting the eigenvector b
    ax[0].arrow(midpoint[0],midpoint[1],ebx,eby, width=0.002, length_includes_head=True, color='blue')
    ax[1].arrow(midpoint[0],midpoint[2],ebx,ebz, width=0.002, length_includes_head=True, color='blue')
    ax[2].arrow(midpoint[1],midpoint[2],eby,ebz, width=0.002, length_includes_head=True, color='blue')
    # plotting the eigenvector n
    ax[0].arrow(midpoint[0],midpoint[1],enx,eny, width=0.002, length_includes_head=True, color='green')
    ax[1].arrow(midpoint[0],midpoint[2],enx,enz, width=0.002, length_includes_head=True, color='green')
    ax[2].arrow(midpoint[1],midpoint[2],eny,enz, width=0.002, length_includes_head=True, color='green')

def make_elliptical_loop(evalue):
    a = np.sqrt(2. * evalue[1])
    b = np.sqrt(2. * evalue[2])
    
    angle = np.arange(1000)/999.*2.*np.pi    
    x = a * np.cos(angle)
    y = b * np.sin(angle)
    z = x * 0.0
    
    return x, y, z
    

def define_loop(midpoint, evalue, evector):
    x, y, z = make_elliptical_loop(evalue)
    loop = np.array([x,y,z])
        
    # convert the loop coordinates into the HEEQ reference frame
    m = np.array([evector[:,1],evector[:,2], evector[:,0]])
    print('Change basis matrix m=',m)
    loop_heeq_ = m.transpose() @ loop
    #print('shape=',loop_heeq.shape)
    loop_heeq_[0,:] = loop_heeq_[0,:] + midpoint[0]
    loop_heeq_[1,:] = loop_heeq_[1,:] + midpoint[1]            
    loop_heeq_[2,:] = loop_heeq_[2,:] + midpoint[2]
    print('Size loop_heeq', loop_heeq_.shape)
    
    # return those points >= 1 R_sun
    dloop = np.sqrt(np.sum(loop_heeq_**2.,axis=0))
    print('dloop format=', dloop.shape)
    ss = np.where(dloop >= 1.0)
    
    #check on the continuity of ss array
    
    print('ss=', ss[0])
    n = len(ss[0])
    index = np.where((ss[0][1:n]-ss[0][0:n-1]) > 1 )
    print(len(index[0]))
    print(index)
    if (len(index[0]) != int(0)):
    #     print('index=',index)
         kk = index[0][0]
         order =[]
         for i in range(kk+1,n):
              order.append(i)
         for i in range(0, kk+1):
              order.append(i)
         ss_ = [ss[0][i] for i in order]
         loop_heeq = loop_heeq_[:,ss_]
    else:
    #print('ss_=', ss_)
        ss_ = ss
        loop_heeq = loop_heeq_[:,ss_[0]]
    
    return ss_, loop, loop_heeq
    
def plot_loop(loop, midpoint, ax):
    ax[0].plot(loop[0,:],loop[1,:], color='red', linestyle='dashed')
    ax[1].plot(loop[0,:],loop[2,:], color='red', linestyle='dashed')
    ax[2].plot(loop[1,:],loop[2,:], color='red', linestyle='dashed')
    
def plot_loop_local_frame(points, midpoint, loop_heeq, evalue, evector):
    
    # convert the datapoints in the local reference frame of the loop 
    #plt.rcParams.update({'font.size': 18})
    #fig, ax = plt.subplots(1,3,figsize=(12,4), constrained_layout=True)
    
    x_ = points[0,:] - midpoint[0]
    y_ = points[1,:] - midpoint[1]
    z_ = points[2,:] - midpoint[2]
    
    m = np.array([evector[:,1],evector[:,2], evector[:,0]])
    data = m @ [x_, y_, z_]
    
    x = data[0,:]
    y = data[1,:]
    z = data[2,:]
    #############################################
    # conversion of the fitting line from HEEQ to local
    loop_heeq_ = loop_heeq *0.0
    loop_heeq_[0,:] = loop_heeq[0,:] - midpoint[0]
    loop_heeq_[1,:] = loop_heeq[1,:] - midpoint[1]
    loop_heeq_[2,:] = loop_heeq[2,:] - midpoint[2]    
    loop = m @ loop_heeq_
    
    a = np.sqrt(2.* evalue[1])
    b = np.sqrt(2.* evalue[2])
    n = np.sqrt(2.* evalue[0])
    #############################################
    #plotting
    border=0.05
    fig = plt.figure(figsize=(12,4),constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    ax=[]
    ax.append(fig.add_subplot(spec[:, 0]))
    ax.append(fig.add_subplot(spec[0, 1]))
    ax.append(fig.add_subplot(spec[1, 1]))

    ax[0].scatter(x, y, color='red')
    ax[0].arrow(0,0, a,0, width=0.002, length_includes_head=True, color='red')
    ax[0].arrow(0,0,0,b, width=0.002, length_includes_head=True, color='blue')
    ax[0].scatter([0,0],[0,0], edgecolors='green', facecolors='none', s=150)
    ax[0].scatter([0,0],[0,0], edgecolors='green', facecolors='green', s=40)
    ax[0].plot(loop[0,:], loop[1,:], color='red')
    if (x.max() > y.max()):
        ax[0].set_xlim(-x.max() - border, x.max()+border)
        ax[0].set_ylim = ax[0].set_xlim
    else:
        ax[0].set_xlim(-y.max() - border, y.max()+border)
        ax[0].set_xlim = ax[0].set_ylim
            
    ax[0].set_xlabel(r'$X_{Local}$ [$R_\odot$]')
    ax[0].set_ylabel(r'$Y_{Local}$ [$R_\odot$]')
    ax[0].set_aspect('equal')

    ax[1].scatter(x, z, color='red')
    #ax[1].plot(xp,zp, 'bo')
    ax[1].arrow(0,0,a,0, width=0.002, length_includes_head=True, color='blue')
    ax[1].arrow(0,0,0,0.02, width=0.002, length_includes_head=True, color='green')
    ax[1].scatter([0,0],[0,0], edgecolors='red', facecolors='none', s=150)
    ax[1].scatter([0,0],[0,0], facecolors='red', s=90,marker='x')
    ax[1].plot(loop[0,:], loop[2,:], color='red')
    ax[1].set_ylim(-0.03,0.03)
    ax[1].set_xlabel(r'$X_{Local}$ [$R_\odot$]')
    ax[1].set_ylabel(r'$Z_{Local}$ [$R_\odot$]')
    ax[1].set_aspect('equal')

    ax[2].scatter(y, z, color='red')
    #ax[2].plot(yp, zp, 'bo')
    ax[2].arrow(0,0,b,0, width=0.002, length_includes_head=True, color='red')
    ax[2].arrow(0,0,0,0.02, width=0.002, length_includes_head=True, color='green')
    ax[2].scatter([0,0],[0,0], edgecolors='blue', facecolors='none', s=150)
    ax[2].scatter([0,0],[0,0], facecolors='blue', s=40)
    ax[2].plot(loop[1,:], loop[2,:], color='red')
    ax[2].set_ylim(-0.03,0.03)
    ax[2].set_xlabel(r'$Y_{Local}$ [$R_\odot$]')
    ax[2].set_ylabel(r'$Z_{Local}$ [$R_\odot$]')
    ax[2].set_aspect('equal')    
  
    return fig, ax

def helio2heeq(p, lon, lat):
	
#    m1 = [ [1, 0, 0],
#     [0, np.cos(lat), -np.sin(lat)],
#    [0, np.sin(lat), np.cos(lat)]]
    m1 = [[np.cos(lat), 0, np.sin(lat)],
    [0, 1., 0],
    [-np.sin(lat), 0, np.cos(lat)]]
    
#    m2 = [ [-np.sin(lon), np.cos(lon), 0],
#       [0, 0, 1],
#    [np.cos(lon), np.sin(lon), 0]]

    m2 = [[np.cos(lon), np.sin(lon), 0],
    [-np.sin(lon), np.cos(lon), 0],
        [0,0,1.]]
        
    m1 = np.array(m1)
    m2 = np.array(m2)

    pheeq = m2.transpose() @ m1.transpose() @ p

    return pheeq
    
def hpc2_heeqmidpoint(thetax, thetay, dsun, lon, lat):
    # transform arcsecs to radians
    thetax_arc = thetax/3600.*np.pi/180.
    thetay_arc = thetay/3600.*np.pi/180.
    # dsun normalised to the solar radii
    d = dsun/6.96e8
    b = np.tan(thetax_arc)**2 + np.tan(thetay_arc)**2
    
    x1 = (b*d + np.sqrt(b*(1-d**2)+1.))/(1+b)
    x2 = (b*d - np.sqrt(b*(1-d**2)+1.))/(1+b)
    x_ = np.array([x1,x2])
    print('x1=',x1, 'x2=',x2)
    x_ = x_.max()
    y_ = (d-x_)*np.tan(thetax_arc)
    z_ = (d-x_)*np.tan(thetay_arc)
    
    lonarc = lon * np.pi/180.
    latarc = lat * np.pi/180.
    
    x, y, z = helio2heeq([x_,y_,z_],lonarc, latarc)  
    
    return x, y, z
