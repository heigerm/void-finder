# imports
import os
import numpy as np
from numpy import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, distance
from astropy.io import ascii
import skimage.filters as filters

def test(x):
    return print('Loaded correctly')

def shoelace(x,y):
    """
    Find the area of a polygon specified by its vertices(ex. the area of the Voronoi cell) using the shoelace method.
    
    inputs
    --------------------
    x: list or 1D array of floats or ints
        x-coordinates of the polygon
    y: list or 1D array of floats or ints
        y-coordinates of the polygon
    
    outputs
    --------------------
    area: float
        area of the polygon
    """

    return 0.5 * np.abs(np.sum(x[:-1]*y[1:]) + x[-1]*y[0] - np.sum(x[1:]*y[:-1])-x[0]*y[-1])

def k_neigh(interpx,interpy, vv, densities, k, discard):
    """
    Interpolate the density at a point based on the mean density of its k closest neighbors.

    input
    -------------------
    interpx, interpy:
        x, y coordinate of the point at which to interpolate
    k: int
        number of neighbors to consider. k = 1 for nearest neighbor

    output
    -------------------
    density: float
        interpolated density at grid point (interpx,interpy)
    
    """

    tp = np.array([interpx,interpy]).reshape(1,2) # test point (x,y) to interpolate
    
    # calculate the euclidian distance from the interpolated point to each data point
    euc = distance.cdist(tp,vv.points[0:-discard])[0]

    # return the interpolated density: the average density of the 5 closest points
    return np.mean(densities[np.argsort(euc)[:k]])

def v_neigh(interpx, interpy, vv, densities):
    
    """
    Calculates the density at a point by averaging the density of its Voronoi neighbors.
    
    input
    -----------------------
    interpx, interpy: 
        x, y coordinates of the point at which to interpolate

    vv: Voronoi diagram
        the Voronoi diagram of the data

    densities:
        the densities of the data

    output
    -----------------------
    i_dens: float
        interpolated density at the input point

    """
    
    inpoint = np.array([interpx,interpy]).reshape(1,2) # reshape the grid point
    into = np.concatenate((vv.points,inpoint)) # add the grid point to the data points
    vi = Voronoi(into) # calculate the updated Voronoi diagram
    
    # make a note of....
    ind_vi = len(vi.points) - 1 # index of the grid point in the in the Voronoi data structure
    inpoint_region = vi.point_region[ind_vi] # grid point's "region"--the index that references its vertices in the Voronoi data structure
    inpoint_vertxs = vi.regions[inpoint_region] # index of the grid point's vertices (the vertex region)
    
    # compare to every other Voronoi cell, slowly but surely...
    for jndex,jalue in enumerate(vi.regions):
        # the values '-1' refer to vertices that are off the map, not actual vertices, so don't consider them--it gives false matches amongst cells at the edges of the diagram
        jalue = np.delete(np.asarray(jalue),np.where(jalue==-1))
        uniq = []
        for ialue in inpoint_vertxs: # compare the grid point's vertices to the other Voronoi cell's
            match = np.where((jalue==ialue))[0] # are any of them the same?
            if match.size: # if so, append them to a list
                uniq.append(match[0])
    
        # if 2 or more vertices are the same, the two cells might share a wall
        if (len(uniq)>=2)&(len(uniq)<len(inpoint_vertxs)): # but make sure they're not the SAME as grid point's cell
            vind = np.where(vv.points==vi.points[np.where(vi.point_region == jndex)][0])[0][0] # indices of neighboring points in the original diagram
            i_dens = np.mean(densities[vind]) # interpolated density at the grid point
            
    return i_dens

def interpolater(input_data, method, grid):
    """
    Interpolates the density at a grid point.

    input
    ------------------------
    input_data: array
        array of coordinates of actual data with shape (2,N)
    method: 'nearest', 'distance', 'voronoi', or 'all
        the type of interpolation to perform

    output
    ------------------------
    densities: array
        array of densities in shape of grid
        if 'all', then 1 list of 3 arrays will be returned

    densities_unblurred: array
        array of the unblurred densities
        if 'all', then 1 list of 3 arrays will be returned
        
    ix, iy: 
        the grid coordinates, for plotting purposes
    """
    
    # rearrange the data a bit to make it easier to use and reference
    x, y = input_data[:,0], input_data [:,1]

    # create a grid over which to interpolate
    gridsize = grid
    ix,iy = np.meshgrid(np.linspace(min(x),max(x),gridsize), np.linspace(min(y),max(y),gridsize))
    x_mesh, y_mesh = np.vstack([ix.ravel(),iy.ravel()]) # make it easier to plot the grid

    # create a border of dummy points to bound the Voronoi diagram
    dummx = np.concatenate((np.linspace(min(x-0.2), max(x+0.2), 50),
                                np.full(50,min(x-0.2)), 
                                np.full(50,max(x+0.2)),
                                np.linspace(min(x-0.2), max(x+0.2), 50)))

    dummy = np.concatenate((np.full(50, min(y-0.2)), 
                            np.linspace(min(y-0.2), max(y+0.2), 50),
                            np.linspace(min(y-0.2), max(y+0.2), 50),
                            np.full(50, max(y+0.2))))
    
    # the points to use to make the Voronoi diagram 
    points = np.concatenate((input_data, np.stack((dummx,dummy),axis=-1)))

    # calculate the Voronoi diagram
    v = Voronoi(points) 

    # calculate the areas of the Voronoi cells
    areas = np.ones((len(v.points) - len(dummx))) # initialize an array to fill
    for ind,value in enumerate(v.point_region[:-len(dummx)]): # for all of the actual data points...
        if -1 not in v.regions[value]: # double check that it's not an edge
            # calculate the area using the shoelace method defined above
            areas[ind] = shoelace(v.vertices[v.regions[value]][:,0],v.vertices[v.regions[value]][:,1])

    # calculate the densities at the data points (the density of the cell)
    densities = 1 / areas
    densities = np.concatenate((densities,np.zeros(len(dummx)))) # set rho = 0 for the dummy points

    if method == 'all':
        # initialize empty arrays that will be the density maps with 3 different interpolation methods
        rho_nn = np.zeros((gridsize**2)).reshape(gridsize,gridsize)
        rho_dw = np.zeros((gridsize**2)).reshape(gridsize,gridsize)
        rho_vn = np.zeros((gridsize**2)).reshape(gridsize,gridsize)

        for index in range(len(ix)): # for every x grid coordinate
            for jndex in range(len(iy)): # for every associated y grid coordinate
                rho_nn[index,jndex] = np.log(k_neigh(ix[index,jndex],iy[index,jndex], v, densities, 1, len(dummx))) # nearest neighbor
                rho_dw[index,jndex] = np.log(k_neigh(ix[index,jndex],iy[index,jndex], v, densities, 5, len(dummx))) # distance weighted
                rho_vn[index,jndex] = v_neigh(ix[index,jndex],iy[index,jndex], v, densities) # Voronoi neighbors
        
        rho_vn[np.where(rho_vn==0)] = np.mean(rho_vn) # the edges get a little screwed up with this method...
        rho_vn = np.log(rho_vn)        
        
        # smooth out the density maps
        blur_nn, blur_dw, blur_vn = filters.gaussian(rho_nn, sigma=(1, 1), truncate=3.5, multichannel=True), filters.gaussian(rho_dw, sigma=(1, 1), truncate=3.5, multichannel=True), filters.gaussian(rho_vn, sigma=(1, 1), truncate=3.5, multichannel=True) 

        return [blur_nn, blur_dw, blur_vn], [rho_nn, rho_dw, rho_vn], ix, iy

    else:
        # initialize an empty array that will be the density map
        rho = np.zeros((gridsize**2)).reshape(gridsize,gridsize)

        if np.logical_or(method == 'nearest', method == 'distance_weighted'): # if using nearest neighbor or distance weighted interpolation
            if 'nearest':
                k_in = 1
            else:
                k_in = 5

            for index in range(len(ix)): # for every x grid coordinate
                for jndex in range(len(iy)): # for every associated y grid coordinate
                    rho[index,jndex] = k_neigh(ix[index,jndex],iy[index,jndex], v, densities, k_in, len(dummx)) # calculate the density

        else: # if using Voronoi neighbor interpolation 
            for index in range(len(ix)): # for every x grid coordinate
                for jndex in range(len(iy)): # for every associated y grid coordinate
                    rho[index,jndex] = v_neigh(ix[index,jndex],iy[index,jndex], v, densities) # calculate the density

        # smooth the density map
        blur_rho = filters.gaussian(rho, sigma=(1, 1), truncate=3.5, multichannel=True)

        return blur_rho, rho, ix, iy