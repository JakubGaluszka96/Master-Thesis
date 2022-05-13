#Import potrzebnych bibliotek
from mayavi import mlab
from tvtk.api import tvtk
from UtilityLib import compute_rot_matrix, transform_verts_general, mesh_reader
from traits.api import HasTraits, Range, Instance, on_trait_change
from traitsui.api import View, Item, Group, HGroup
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, MlabSceneModel
import numpy as np                           #operacje na macierzach i wektorach (mnozenie, dodawanie, itd.)
import math
from math import sin, cos, sqrt              #funkcja sin, cos i sqrt
import vtk
output=vtk.vtkFileOutputWindow()
output.SetFileName("log.txt")
vtk.vtkOutputWindow().SetInstance(output)
"Polozenie przyczepow wiezadel na kosci udowej"
a = np.array([[-0.01511, -0.00879,-0.0012],      #wiezadlo 1 [X,Y,Z]
              [0.007236, -0.00753, -0.00668],      #...
              [0.04308, 0.00231, -0.00980],      #...
              [-0.04344, 0.01126, -0.00638]])     #wiezadlo n [X,Y,Z]
                                    #...
                                    #...
                                    ##Mozna uzyc dowolnej ilosci wiezadel

"Polozenie przyczepow wiezadel na kosci piszczelowej w jej ukladzie wspolrzednych"
b_t = np.array([[0.0001, -0.02599, -0.00427],      #wiezadlo 1 [X,Y,Z]
                [0.00001, 0.00002, 0.00003],      #...
                [0.03961,-0.04578, -0.01650],      #...
                [-0.02067,0.01874, -0.01222]])     #wiezadlo n [X,Y,Z]
                                    #...
                                    #...
l=np.array([0.02876,0.02359,0.06037,0.05084])                   #dlugosci wiezadel [l1, l2,...]
k=np.array([30000000,35000000,15000000,15000000])           #wspolczynniki proprocjonalnosci wiezadel [k1, k2,...]
Fmax=np.array([868,622,433,534])            #maksymalne sily przenoszone przez wiezadla [Fmax1, Fmax2,...]
epMax=np.array([1.03,1.05,1.03,1.05])         #maksymalne wydłużenie wiezadeł [epMax1, epMax2,...]
"Definiowanie zakresu katow podelgajacych sprawdzeniu."
rozkat=1                           #rozdzielczosc katowa musi byc taka sama dla wszystkich wartosci katowych. Jest to niezbedne do przeprowadzenia operacji mnozenia macierzy.
rozpol=25                           #rozdzielczosc przemieszczeniowa musi byc taka sama dla wszystkich XYZ. Jest to niezbedne do przeprowadzenia operacji mnozenia macierzy.
X=[-0.035,0.035]
Y=[-0.04,0.035]
Z=[-0.055,0.02]
"Do sily na styku kosci"
"Geometryczne"
PunktyHertza=np.array([[0.00001,0.00002,0.00003],  #punkty leżące na płaszczyznie okreslajacej konic blizszy kosci piszczelowej
                       [0.02892,-0.00201,0.00021],  #UWAGA! zaleca sie nie wpisywac zer, bo czasami  nie dziala
                       [-0.00163,-0.02036,-0.00090]])  
scp1=[0.02441,0.00018,-0.00129]                       #wspolrzedne okreslajace srodek sfery definujacej ksztalt i polozenie kłykcia przyrodkowego kosci udowej
scp2=[-0.02998,0.00447,-0.00247]                       #wspolrzedne okreslajace srodek sfery definujacej ksztalt i polozenie kłykcia strzałkowego kosci udowej
r1=0.02398                                     #promien sfery korelającej kłykieć przysrodkowy
r2=0.02369                                     #promien sfery korelającej kłykieć strzalkowy
A=1                                     #do obliczenia rownania plaszczyzny (nie edytowac)

"Mechaniczne"
E_1 = 5000000                           #modul Younga kosci udowej [Pa]
E_2 = 5000000                           #modul Younga chrzastki kosci piszcelowej [Pa]
v_1 = 0.46                              #wspolczynnik Poissona chrzastki kosci udowej
v_2 = 0.46                              #wspolczynnik Poissona chrzastki kosci piszczelowej
E=(E_1*E_2)/(((1-v_1**2)*E_2)+((1-v_2**2)*E_1))        #zastepczy modul Younga do obliczenia sily Hertza
#sigma=1000000                         #makxymalne naprężenie w chrząstce [Pa]
silaMAXhertz = (4/3)*E*(r1**(1/2))*(0.00241**(3/2))###(((sigma*3.14159265)**3)*(r1**2))/(6*E**2)                #maksymalna sila Hertza w punkcie styku [N]
sigma=(1/3.14156)*((6*silaMAXhertz*E**2)/(r1**2))**(1/3)

"Wynikiem wprowadzenia zmiennych katowych jest macierz obrotu wg katow Eulera"
def obracanie(theta, gamma, psi):
    Rot=np.array([[np.cos(theta)*np.cos(gamma), np.sin(gamma)*np.sin(psi)*np.cos(theta)-np.sin(theta)*np.cos(psi), np.cos(theta)*np.sin(gamma)*np.cos(psi)+np.sin(theta)*np.sin(psi)],
                  [np.sin(theta)*np.cos(gamma), np.sin(theta)*np.sin(gamma)*np.sin(psi)+np.cos(theta)*np.cos(psi), np.sin(theta)*np.sin(gamma)*np.cos(psi)-np.cos(theta)*np.sin(psi)],
                  [-np.sin(gamma), np.cos(gamma)*np.sin(psi), np.cos(gamma)*np.cos(psi)]])
    return Rot
    
"Wynikiem wprowadzenia zmiennych katowych i wspolrzednych kartezjanskich sa odleglosci miedzy przyczepami wiezadel"     
def distance(x, y, z, theta, gamma, psi):
    Rot=obracanie(theta,gamma,psi)
    b = np.dot(b_t, Rot) + [x, y, z]
    dl_wiez = np.linalg.norm((b.T - a.T),axis=1)
    return dl_wiez
    
"""Roznica dlugosci swobodnej wiezadel i odleglosci miedzy punktami mnozona jest przez wspolczynnik 
   proporcjonalnosci. Wynikiem jest sila przenoszona przez wiezadlo"""
def wyznacz_sile(x,y,z,theta,gamma,psi):
    dl_wiez=distance(x,y,z,theta,gamma,psi)
    F=k*np.power((((dl_wiez-l)).clip(0)),2)
    return F
"Koniec algorytmu wyznaczającego siły w więzadłach"
"---------------------------------------------------------------------------------------------"

"Algorytm obliczajacy sile Hertza"
"1. Transformacja punktow do globalnego ukladu wspolrzednych"
def trans_point_Hertz(x, y, z, theta, gamma, psi): 
    Rot=obracanie(theta, gamma, psi)
    pointH1 = np.dot(PunktyHertza[0], Rot) + [x, y, z]
    pointH2 = np.dot(PunktyHertza[1], Rot) + [x, y, z]
    pointH3 = np.dot(PunktyHertza[2], Rot) + [x, y, z]    
    return pointH1, pointH2, pointH3

"2. Obliczenie parametrow równania płaszczyzny"

def solve_matrix(pointH1, pointH2, pointH3):
    Aa = np.array([np.array([pointH1[1], pointH2[1], pointH3[1]]),
                  np.array([pointH1[2], pointH2[2], pointH3[2]]),
                  np.array([np.ones(np.size(pointH1[1])), np.ones(np.size(pointH2[1])), np.ones(np.size(pointH3[1]))])]).T
    Bb = np.array([pointH1[0], pointH2[0], pointH3[0]]).T
    o=np.linalg.solve(Aa, Bb)
    return o
  
"3. Wyznaczenie odleglosci miedzy plaszczyzna a srodkiem sfery"

def odleglosc(o,scp):
    c=(A*scp[0]+o.T[0]*scp[1]+o.T[1]*scp[2]+o.T[2])/np.sqrt(A**2+o.T[0]**2+o.T[1]**2)
    return c

"4. Wyzancznie sily na styku kosci wedlug zalozen Hertza"

def silaHertza(c,r):
    Fhertz = (4/3)*E*(r**(1/2))*((r-c).clip(0)**(3/2))
    return Fhertz

"5. Sprawdzenie punktów, czy spelniaja przyjęte założenia"

def czy_bezpieczne_hertza(Fhertz1):
    a = (silaMAXhertz-Fhertz1)>0
    return a

"Złożenie wczystkich funcji w jedną"
def Hertz(x, y, z, theta, gamma, psi, scp1, r1):
    pointH1, pointH2, pointH3=trans_point_Hertz(x, y, z, theta, gamma, psi)
    o=solve_matrix(pointH1, pointH2, pointH3)
    c1=odleglosc(o,scp1)    
    Fhertz1=silaHertza(c1, r1)
    a=czy_bezpieczne_hertza(Fhertz1)
    return a
    
"Koniec procedury obliczajacej sile Hertza"

"---------------------------------------------------------------------------------------------"
    
"""Sila wystepujaca w wiezadlach porownywana jest z maksymalna, ktora moze przeniesc wiezadlo.
   Jezeli zadne z wiezadel nie przekroczylo wartosci maksymalnej sily funkcja zwraca wartosc True""" 
def czy_bezpieczne(x,y,z,theta,gamma,psi):
    a = np.logical_not(np.any((wyznacz_sile(x,y,z,theta,gamma,psi)-Fmax).clip(0),axis=1))
    return a

"Wynikiem jest polozenie srodkow kul bedacych przestrzeniami roboczymi dla wiezadel."
def oblicz_srodki(theta, gamma, psi):
    Rot=obracanie(theta,gamma,psi)                             #macierz obrotu
    s=(a.reshape(l.size,3,1) - np.dot(b_t, Rot))                                   #obliczenie polozenia srodka kul bedacych przestrzeniami roboczymi dla wiezadel    
    return s     

"Wynikiem jest zawezone pole poszukiwan przestrzeni roboczej."
def zakres_XYZ(theta,gamma,psi):
    Rot=obracanie(theta,gamma,psi)                             #macierz obrotu                                  #obliczenie polozenia srodka kul bedacych przestrzeniami roboczymi dla wiezadel    
    s=(a.reshape(l.size,3,1)-np.dot(b_t, Rot))
    s=(s.T-l*epMax).T
    sm=(s.T+l*epMax).T
    mini=np.min(np.min(sm,axis=2),axis=0)
    maxi=np.max(np.max(s,axis=2),axis=0)
    return mini, maxi

"Procedura sprawdzająca które punkty są dozwolone według przyjętyh założeń"
    
def punkty_dozwolone():
    theta=np.linspace(math.radians(-20), math.radians(20), num=5, endpoint=True)   #(Poczatek[deg]*3.14159265/180.0, koniec, num=rozdzielczosc, enpoint=True(przedzial domkniety)) - obrot wokol osi X
    gamma=np.linspace(math.radians(-30), math.radians(30), num=5, endpoint=True)   #(Poczatek[deg]*3.14159265/180.0, koniec, num=rozdzielczosc, enpoint=True(przedzial domkniety)) - obrot wokol osi Y
    psi=np.linspace(math.radians(-10), math.radians(180), num=20, endpoint=True)     #(Poczatek[deg]*3.14159265/180.0, koniec, num=rozdzielczosc, enpoint=True(przedzial domkniety)) - obrot wokol osi Z
    #mini,maxi=zakres_XYZ(theta, gamma, psi)
    x=np.linspace(X[0], X[1],num=rozpol,endpoint=True)
    y=np.linspace(Y[0], Y[1],num=rozpol,endpoint=True)
    z=np.linspace(Z[0], Z[1],num=rozpol,endpoint=True)   
    theta, gamma, psi, x, y, z=np.meshgrid(theta, gamma, psi, x, y, z)
    theta=np.reshape(theta,(theta.size))
    gamma=np.reshape(gamma,(gamma.size))
    psi=np.reshape(psi,(psi.size))
    x=np.reshape(x,(x.size))
    y=np.reshape(y,(y.size))
    z=np.reshape(z,(z.size))
    g=czy_bezpieczne(x,y,z,theta,gamma,psi)
    x=np.extract(g, x)
    y=np.extract(g, y)
    z=np.extract(g, z)
    theta=np.extract(g, theta)
    gamma=np.extract(g, gamma)
    psi=np.extract(g, psi)
    g=Hertz(x, y, z, theta, gamma, psi, scp1, r1)
    x=np.extract(g, x)
    y=np.extract(g, y)
    z=np.extract(g, z)
    theta=np.extract(g, theta)
    gamma=np.extract(g, gamma)
    psi=np.extract(g, psi)
    g=Hertz(x, y, z, theta, gamma, psi, scp2, r2)
    x=np.extract(g, x)
    y=np.extract(g, y)
    z=np.extract(g, z)
    theta=np.extract(g, theta)
    gamma=np.extract(g, gamma)
    psi=np.extract(g, psi)
    return x, y, z, theta, gamma, psi



def narysujWykres3D():

    x, y, z, theta, gamma, psi=punkty_dozwolone()       #wyznaczenie punktów dozwolonych
    t_set=np.array([x,y,z]).T                           #procedura usuwająca powtarzające się punkty
    t_set=np.unique(t_set, axis=0).T                    
    x, y, z=t_set[0], t_set[1], t_set[2]
    print('Objetosc przestrzeni roboczej jest rowna=', x.size*(X[1]-X[0])*(Y[1]-Y[0])*(Z[1]-Z[0])/rozpol**3*1000**3, "mm3")
    vert, tria, normals = mesh_reader('femur.stl')
    vert_c, tria_c, normals_c = mesh_reader('tibia.stl')     
    class Visualization(HasTraits):
        atheta   = Range(-90.0, 90.0, 0.0)
        agamma   = Range(-90.0, 90.0, 0.0)
        apsi     = Range(-90.0, 90.0, 0.0)
        px      = Range(-35.0, 35.0, 0.0)
        py      = Range(-35.0, 35.0, 0.0)
        pz      = Range(-35.0, 35.0, 0.0)

        scene   = Instance(MlabSceneModel, ())

        def __init__(self):
            HasTraits.__init__(self)
            corxyz=np.array([-2.882, 8.568, -25.126]) #zmienic w obu miejscach
             
            self.scene.mlab.triangular_mesh(vert[:,0], 
                                            vert[:,1], 
                                            vert[:,2], 
                                            tria, color=(199/255.0, 204/255.0, 224/255.0),
                                            opacity = 0.3)
            self.scene.scene.background = (1.0, 1.0, 1.0) #kolor tla
            p = np.array([self.px, self.py, self.pz])
            R = obracanie(self.atheta, self.agamma, self.apsi)
            vert_c_cur = transform_verts_general(vert_c,
                                                 np.array([self.px, self.py, self.pz]), 
                                                 R)
            

            col=(0.8, 0.4, 0.0)

                
            self.plot = self.scene.mlab.triangular_mesh(vert_c_cur[:,0],
                                                        vert_c_cur[:,1], 
                                                        vert_c_cur[:,2], 
                                                        tria_c, color=col,
                                                        opacity=0.3)                

            sc = 100
            lw = 2
            self.i = mlab.quiver3d(self.px+corxyz[0], self.py+corxyz[1], self.pz+corxyz[2], 
                                   R[0, 0], R[1, 0], R[2, 0], 
                                   line_width=lw, scale_factor=sc, color = (1, 0, 0))
            self.j = mlab.quiver3d(self.px+corxyz[0], self.py+corxyz[1], self.pz+corxyz[2], 
                                   R[0, 1], R[1, 1], R[2, 1], 
                                   line_width=lw, scale_factor=sc, color = (0, 1, 0))
            self.k = mlab.quiver3d(self.px+corxyz[0], self.py+corxyz[1], self.pz+corxyz[2], 
                                   R[0, 2], R[1, 2], R[2, 2], 
                                   line_width=lw, scale_factor=sc, color = (0, 0, 1))        

            # glob ukl wsp
            lw = 2
            sc = 85
            self.ig = mlab.quiver3d(self.px, self.py, self.pz, 
                                    1., 0., 0., 
                                    line_width=lw, scale_factor=sc, color = (1, 0, 0))
            self.jg = mlab.quiver3d(self.px, self.py, self.pz, 
                                    0., 1., 0., 
                                    line_width=lw, scale_factor=sc, color = (0, 1, 0))
            self.kg = mlab.quiver3d(self.px, self.py, self.pz, 
                                    0., 0., 1., 
                                    line_width=lw, scale_factor=sc, color = (0, 0, 1)) 
            
            self.sa=mlab.points3d(scp1[0]*1000,scp1[1]*1000, scp1[2]*1000, color=(0,1,1), opacity=0.5, scale_factor=2*r1*1000)
            self.sb=mlab.points3d(scp2[0]*1000,scp2[1]*1000, scp2[2]*1000, color=(0,1,1), opacity=0.5, scale_factor=2*r2*1000)
            
            """
            self.pep = mlab.quiver3d(0.0, 0.0, 0.0, 
                                   corxyz[0], corxyz[1], corxyz[2], 
                                   line_width=4,scale_factor=1, color = (1, 1, 0)) 
            self.pip = mlab.quiver3d(0.0, 0.0, 0.0, 
                                   a.T[0][2]*1000, a.T[1][2]*1000, a.T[2][2]*1000, 
                                   line_width=4,scale_factor=1, color = (0.5, 0, 0.5))
            self.pup = mlab.quiver3d(a.T[0][2]*1000, a.T[1][2]*1000, a.T[2][2]*1000, 
                                   b_t.T[0][2]*1000+p[0]-a.T[0][2]*1000+corxyz[0], b_t.T[1][2]*1000+p[1]-a.T[1][2]*1000+corxyz[1], b_t.T[2][2]*1000+p[2]-a.T[2][2]*1000+corxyz[2], 
                                   line_width=4,scale_factor=1, color = (1, 0.7, 0.75))
            self.pyp = mlab.quiver3d(corxyz[0], corxyz[1], corxyz[2], 
                                   b_t.T[0][2]*1000+p[0], b_t.T[1][2]*1000+p[1], b_t.T[2][2]*1000+p[2], 
                                   line_width=4,scale_factor=1, color = (0.7, 0.8, 0.9))
            
            self.ga=mlab.points3d(a.T[0][2]*1000,a.T[1][2]*1000,a.T[2][2]*1000, color=(1,0,0), opacity=1, scale_factor=4)            
            self.gp=mlab.points3d(b_t.T[0][2]*1000+corxyz[0],b_t.T[1][2]*1000+corxyz[1],b_t.T[2][2]*1000+corxyz[2], color=(0,0,1), opacity=1, scale_factor=4)
            """
            xy=x*1000
            yy=y*1000
            zy=z*1000
            p=p+corxyz
            
            self.pg=mlab.points3d(xy,yy,zy, mode='cube', color=(0,0.75,0), opacity=0.9, scale_factor=2.5) #wyswietlenie punktów za pomocą biblioteki mayavi
            
            self.ag=mlab.points3d(a.T[0]*1000,a.T[1]*1000,a.T[2]*1000, color=(1,0,0), opacity=1, scale_factor=5)            
            self.pg=mlab.points3d(b_t.T[0]*1000+p[0],b_t.T[1]*1000+p[1],b_t.T[2]*1000+p[2], color=(0,0,1), opacity=1, scale_factor=5)
            
            self.aw=mlab.plot3d([a.T[0][0]*1000,b_t.T[0][0]*1000+p[0]],[a.T[1][0]*1000,b_t.T[1][0]*1000+p[1]],[a.T[2][0]*1000,b_t.T[2][0]*1000+p[2]], tube_radius=1, tube_sides = 20, color = (1,1,0)) #wiezadlo 1
            self.bw=mlab.plot3d([a.T[0][1]*1000,b_t.T[0][1]*1000+p[0]],[a.T[1][1]*1000,b_t.T[1][1]*1000+p[1]],[a.T[2][1]*1000,b_t.T[2][1]*1000+p[2]], tube_radius=1, tube_sides = 20, color = (1,1,0)) #wiezadlo 2
            self.cw=mlab.plot3d([a.T[0][2]*1000,b_t.T[0][2]*1000+p[0]],[a.T[1][2]*1000,b_t.T[1][2]*1000+p[1]],[a.T[2][2]*1000,b_t.T[2][2]*1000+p[2]], tube_radius=1, tube_sides = 20, color = (1,1,0)) #wiezadlo 3
            self.dw=mlab.plot3d([a.T[0][3]*1000,b_t.T[0][3]*1000+p[0]],[a.T[1][3]*1000,b_t.T[1][3]*1000+p[1]],[a.T[2][3]*1000,b_t.T[2][3]*1000+p[2]], tube_radius=1, tube_sides = 20, color = (1,1,0)) #wiezadlo 4
            
            
        @on_trait_change('atheta,agamma,apsi,px,py,pz')
        def update_plot(self):
            Rot1=obracanie(np.array([np.radians(self.atheta)]), np.array([np.radians(self.agamma)]), np.array([np.radians(self.apsi)]))
            R = compute_rot_matrix(np.array([np.radians(self.atheta), 
                                         np.radians(self.agamma), 
                                         np.radians(self.apsi)]) * 15.0)
            p = np.array([self.px, self.py, self.pz])
            corxyz=np.array([-2.882, 8.568, -25.126]) #zmienic tez na 1 miejscu
            vert_c_cur = transform_verts_general(vert_c,
                                                np.array([self.px, self.py, self.pz]), 
                                                R)
            
            b_te=transform_verts_general(b_t*1000+corxyz,
                                         np.array([self.px, self.py, self.pz]), 
                                         R)
            
            
            col=(0.8, 0.4, 0.0)

                      
            
            self.plot.mlab_source.set(x=vert_c_cur[:,0], y=vert_c_cur[:,1], z=vert_c_cur[:,2], color=col)
            
            Rot1=obracanie(np.array([np.radians(self.atheta)]), np.array([np.radians(self.agamma)]), np.array([np.radians(self.apsi)]))            
            corxyz=np.array([[-2.882, 8.568, -25.126]])
            corxyz=transform_verts_general(corxyz, np.array([self.px, self.py, self.pz]), R)
            
            self.i.mlab_source.set(x=corxyz[0][0], y=corxyz[0][1], z=corxyz[0][2], 
                               u=R[0, 0], v=R[1, 0], w=R[2, 0])
            self.j.mlab_source.set(x=corxyz[0][0], y=corxyz[0][1], z=corxyz[0][2], 
                               u=R[0, 1], v=R[1, 1], w=R[2, 1])
            self.k.mlab_source.set(x=corxyz[0][0], y=corxyz[0][1], z=corxyz[0][2], 
                               u=R[0, 2], v=R[1, 2], w=R[2, 2])
            
            self.pg.mlab_source.set(x=b_te.T[0], y=b_te.T[1], z=b_te.T[2])
            
            self.aw.mlab_source.set(x=[a.T[0][0]*1000,b_te.T[0][0]], y=[a.T[1][0]*1000,b_te.T[1][0]], z=[a.T[2][0]*1000,b_te.T[2][0]])
            self.bw.mlab_source.set(x=[a.T[0][1]*1000,b_te.T[0][1]], y=[a.T[1][1]*1000,b_te.T[1][1]], z=[a.T[2][1]*1000,b_te.T[2][1]])
            self.cw.mlab_source.set(x=[a.T[0][2]*1000,b_te.T[0][2]], y=[a.T[1][2]*1000,b_te.T[1][2]], z=[a.T[2][2]*1000,b_te.T[2][2]])
            self.dw.mlab_source.set(x=[a.T[0][3]*1000,b_te.T[0][3]], y=[a.T[1][3]*1000,b_te.T[1][3]], z=[a.T[2][3]*1000,b_te.T[2][3]])
        view = View(Item('scene', height=800, show_label=False,
                    editor=SceneEditor()),
                    HGroup('_', 'atheta', 'agamma', 'apsi', '_', 'px', 'py', 'pz'), 
                    resizable=True)        
    Visualization().configure_traits()
    
narysujWykres3D()