# -*- coding: utf-8 -*-
"""
Created on Sun May 12 19:11:16 2019

@author: Adam Ciszkiewicz
"""
import numpy as np

from UtilityLib import mesh_reader
from os import listdir
from os.path import isfile, join

from scipy import optimize
from scipy.spatial.distance import cdist

class Parameter():
    def __init__(self):
        
        # gora tibii
        TT = np.array([91.98, -113.02, 334.37])
        
        MC = np.array([35.66, -77.08, 359.67])
        LC = np.array([106.48, -74.99, 370.04])
        
        IC = (MC + LC) / 2.0
        
        # dol tibii
        MM = np.array([52.50, -86.18, -5.54])
        LM = np.array([97.75, -52.86, -24.24])
        
        IM = (MM + LM) / 2.0
        
        # assigning local reference frames
        k_t = (MM - LM) / np.linalg.norm(MM - LM)
        
        v1 = MM - IC 
        v2 = LM - IC 
        v3 = np.cross(v1, v2)
        i_t = v3 / np.linalg.norm(v3)
        
        j_t = np.cross(k_t, i_t)
        
        Ttg = np.zeros((4, 4))
        Ttg[3, 3] = 1.0
        Ttg[0:3, 0] = i_t
        Ttg[0:3, 1] = j_t
        Ttg[0:3, 2] = k_t
        Ttg[0:3, 3] = IM
        Tgt = np.linalg.inv(Ttg)
        
        ##
        j_c = (IC - IM) / np.linalg.norm(IC - IM)
        
        v1 = LC - IM 
        v2 = MC - IM 
        v3 = np.cross(v1, v2)
        i_c = v3 / np.linalg.norm(v3)
        
        k_c = np.cross(i_c, j_c)
        
        Tcg = np.zeros((4, 4))
        Tcg[3, 3] = 1.0
        Tcg[0:3, 0] = i_c
        Tcg[0:3, 1] = j_c
        Tcg[0:3, 2] = k_c
        Tcg[0:3, 3] = IM
        Tgc = np.linalg.inv(Tcg)
        
        # make matrices available outside the class
        self.Tgt = Tgt        
        self.Tgc = Tgc        
        
        # transformation from c to t and initial transformation parameters
        Tct = np.dot(Tgt, Tcg)
        
        self.beta = np.arctan2(Tct[2, 1], np.sqrt(Tct[2,0] **2 + Tct[2,2] **2))

        self.alpha = np.arctan2(Tct[2, 0] / np.cos(self.beta), 
                                Tct[2, 2] / np.cos(self.beta))
        
        self.gamma = np.arctan2(-Tct[0, 1] / np.cos(self.beta), 
                                Tct[1, 1] / np.cos(self.beta))        

        self.p = Tct[0:3, 3]
        
        self.Tct = Tct
        
        # define ligaments in a global reference frame
        ATT_c = np.array([60.27, -97.27, -14.80])
        ATT_t = np.array([51.84, -89.86, -2.83])
        
        PTT_c = np.array([50.98, -65.09, -10.22])
        PTT_t = np.array([46.14, -73.26, 2.40])
        
        CT_c = np.array([54.88, -79.48, -25.90])
        CT_t = np.array([45.55, -83.16, 3.16])
        
        CF_c = np.array([93.38, -63.04, -40.07])
        CF_t = np.array([99.72, -51.45, -21.46])
        
        PTF_c = np.array([63.96, -51.08, -17.12])
        PTF_t = np.array([97.36, -48.45, -15.57])
        
        ATF_c = np.array([86.59, -92.03, -10.12])
        ATF_t = np.array([99.15, -72.86, -5.40])
        
        lig_c = [ATT_c, PTT_c, CT_c, CF_c, PTF_c, ATF_c]
        lig_t = [ATT_t, PTT_t, CT_t, CF_t, PTF_t, ATF_t]
        
        # transform ligaments to local reference frames
        self.a = np.zeros((len(lig_t), 3))
        self.b_l = np.zeros_like(self.a)
        
        temp = np.zeros(4)
        temp[3] = 1.0
        
        for i in range(len(lig_t)):
            temp[0:3] = lig_c[i]
            self.b_l[i] = np.dot(Tgc, temp)[0:3]

            temp[0:3] = lig_t[i]
            self.a[i] = np.dot(Tgt, temp)[0:3]