# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 23:31:27 2017

@author: Adam Ciszkiewicz
"""
from stl import mesh
from mayavi import mlab
import numpy as np
from numpy.linalg import norm
    
def mesh_reader(filename, dilate_mesh=False, dilate_scale=1.15):
    """
    modified remesher -> returns matrix of normals for dilation
    """
    
    mesh1 = mesh.Mesh.from_file(filename)
    siz = mesh1.v0.shape[0]
    
    tria = np.zeros((siz, 3)).astype('int')
#    normals = np.zeros((siz, 3)).astype('int')
    
    for i in range(siz):
        tria[i,:] = np.array([i, i+siz, i+2*siz]).astype('int')
#        normals[i,:] = mesh1.normals[i]/norm(mesh1.normals[i])
        mesh1.normals[i] = mesh1.normals[i]/norm(mesh1.normals[i])

    normals_stacked = np.vstack((mesh1.normals,mesh1.normals,mesh1.normals))
    verts = np.concatenate((mesh1.v0, mesh1.v1, mesh1.v2), axis=0)
    
    if dilate_mesh:
        s = dilate_scale
        centr_pre = verts.mean(axis=0)
        vert2 = verts*s
        centr_post = vert2.mean(axis=0)
        p = centr_post-centr_pre
        vert2[:,0] -= p[0]
        vert2[:,1] -= p[1]
        vert2[:,2] -= p[2]
        
        verts = vert2
    
    return verts, tria, normals_stacked

def vert_rescaler(verts, dilate_scale):
    s = dilate_scale
    centr_pre = verts.mean(axis=0)
    vert2 = verts*s
    centr_post = vert2.mean(axis=0)
    p = centr_post-centr_pre
    vert2[:,0] -= p[0]
    vert2[:,1] -= p[1]
    vert2[:,2] -= p[2]
    
    return vert2
        
def plot_mesh(vert, tria, col = (0.3, 0.3, 1.0), opac=1.0):
    mlab.triangular_mesh(vert[:,0], vert[:,1], vert[:,2], tria, color=col, opacity = opac)

def transform_verts_general(vert, transl, rot):
    vert_transf = np.zeros_like(vert)

    vert_transf[:,0] = vert[:,0]*rot[0,0] + vert[:,1]*rot[0,1] + vert[:,2]*rot[0,2] + transl[0]
    vert_transf[:,1] = vert[:,0]*rot[1,0] + vert[:,1]*rot[1,1] + vert[:,2]*rot[1,2] + transl[1]
    vert_transf[:,2] = vert[:,0]*rot[2,0] + vert[:,1]*rot[2,1] + vert[:,2]*rot[2,2] + transl[2]
    
    return vert_transf

def compute_rot_matrix(rot):
    # rot = [alpha, beta, gamma] => [rad]
    # based on PhD by Franci, left leg, eq. 2.8
    
    c = np.cos(rot)
    s = np.sin(rot)

    r = np.zeros((3, 3))    
    
    r[0, 0] = c[0] * c[2] + s[0] * s[1] * s[2]
    r[1, 0] = s[2] * c[0] - c[2] * s[1] * s[0]
    r[2, 0] = c[1] * s[0]
    
    r[0, 1] = -s[2] * c[1]
    r[1, 1] = c[2] * c[1]
    r[2, 1] = s[1]
    
    r[0, 2] = -c[2] * s[0] + s[2] * s[1] * c[0]
    r[1, 2] = -s[2] * s[0] - c[2] * s[1] * c[0]
    r[2, 2] = c[1] * c[0]

    return r