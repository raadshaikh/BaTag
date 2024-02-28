import numpy as np
from stl import mesh
from matplotlib import pyplot as plt
import pickle
from scipy.interpolate import interp1d

#utilities

Epsilon=2**-18 #idk smth small

def filter_decays():
    global decayPos
    global decayDir
    global t1
    global success_indices
    decayPos=decayPos[success_indices]
    decayDir=decayDir[success_indices]
    t1=t1[success_indices]
def filter_decays_po():
    global poPos
    global poDir
    global success_indices
    global yp
    global zp
    poPos=poPos[success_indices]
    poDir=poDir[success_indices]
    yp=yp[success_indices]
    zp=zp[success_indices]

def save(filename, *args):
    # Get global dictionary
    glob = globals()
    d = {}
    for v in args:
        # Copy over desired values
        d[v] = glob[v]
    with open(filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)

def load(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            glob[k] = v



dt=1 #timestep in seconds
timesteps=int(0.8*24*60*60/dt)

#geometry
xmin, xmax = -21, 21 #bounding box for volume under consideration
ymin, ymax = -21, 21
zmin, zmax = 20, 64
xrange=xmax-xmin
yrange=ymax-ymin
zrange=zmax-zmin
volume=xrange*yrange*zrange
cylinderR = 20 #inner radius of teflon cylinder
detectR = 6 #side length of square SiPM detector


'''start the radon hitting'''
# Load the STL files...
stlname='detector_chamber4_cut_noTefl.stl'
my_mesh = mesh.Mesh.from_file(stlname)
my_mesh=np.reshape(my_mesh, (len(my_mesh),3,3)) #before this, the whole triangle was in a (9,) array. I am splitting the vertices into their own arrays

#geometry
if stlname=='detector_chamber4_cut.stl' or 'detector_chamber4_cut_noTefl.stl':
    cylinder_zmin, cylinder_zmax = 2, 22
    detectZ = 12
elif stlname=='detector_chamber4_cut_shift.stl':
    cylinder_zmin, cylinder_zmax = 44, 64
    detectZ = 54
elif stlname=='detector_chamber4_cut_shift_long.stl':
    cylinder_zmin, cylinder_zmax = 2, 64
    detectZ = 54
elif stlname=='detector_chamber4_cut_long_cent.stl' or 'detector_chamber4_cut_long_cent_noTefl.stl':
    cylinder_zmin, cylinder_zmax = 0, 55
    detectZ = 24

N=int(4) #24000 Rn/cc, over 1 day, means 4000 decays/cc. temporarily reducing this for speed
print('starting with {} radon decays'.format(N))

#generate random positions and direction rays for each photon.
#N*M rows, 3 columns. each row identifies a photon, and there is a column for each coordinate
#M is number of photons from each decay. so total number of photons is N*M
#i don't want to bother renaming everything, so i'll keep the names 'decaypos' and 'decaydir'.
#but now, 'decaypos' has M copies of each of the N decay positions
#although 'decaydir' is still just N*M different random direction cosines since each photon has a different direction regardless of decay
M=1
#sipmeff=0.5 IMPORTANT manually reduce the efficiency by this much later
print('each producing {} photons, so total {} photons'.format(M, N*M))

option=2
print('option=',option)
#2 kinds of simulations: start with uniformly populated uv photons or start with the photons produced by the one radon decay

if option==1:
    decayPos=np.random.uniform([xmin, ymin, zmin], [xmax, ymax, zmax], (N,3))
    decayPos=np.repeat(decayPos, M, axis=0)
elif option==2:
    #this is the one artificial decay from the pips going straight up case
    big=16
    load('scintNRG_output_{}'.format(big))
    N=int(np.sum(np.rint(energies/np.min(energies))))
    print('actually starting with ', N, 'photons')
    positions=lengths+dx/2
    positions=10*positions #scintNRG dealt in cm, here i work with mm
    weights=np.int_(np.rint(energies/np.min(energies)))
    decayPosZ=np.repeat(positions, weights)
    decayPos=np.zeros((len(decayPosZ),3))
    decayPos[:,2]=decayPosZ


decayDir=np.random.normal(0,1,(len(decayPos),3))
decayDir=decayDir/np.reshape(np.sqrt(np.einsum('ij...,ij->i...',decayDir,decayDir)), (len(decayDir),1))
#this einsum thing is to get the elementwise dot product between two lists of vectors (here to get a list of norms of a list of vectors), to normalise the directions

#directly hitting uv photons on sipm but reusing old code so cutting out the middle and stitching it together like this haha
poPos=decayPos
poDir=decayDir

#find solid angle of feasability. sipm is a 6x6mm square, normal to x-axis, centred at (21, 0, 54)
yp=poPos[:,1]+(poDir[:,1]/poDir[:,0])*(cylinderR+1-poPos[:,0]) #y_Plane, referring to the y-coord of the point where the photon intersects with the Plane of the sipm
zp=poPos[:,2]+(poDir[:,2]/poDir[:,0])*(cylinderR+1-poPos[:,0])
#the last condition here imposes that 't' is positive
success_indices=np.logical_and.reduce([yp>-detectR/2, yp<detectR/2, zp>detectZ-(detectR/2), zp<detectZ+(detectR/2), (cylinderR+1-poPos[:,0])/poDir[:,0]>0])
filter_decays_po()

#i cut the sipms out of the model, so we'll pass the feasible photons that don't intersect with anything
lenpoPos=len(poPos)
success_indices=np.zeros(lenpoPos, dtype='bool')
sipm_intersections=[]
for i in range(lenpoPos):
    if int(10000*i/lenpoPos)%1==0: print("blue photons: {:.2f}%".format(100*i/lenpoPos), end='\r')
    O=np.tile(poPos[i], (len(my_mesh),1))
    D=np.tile(poDir[i], (len(my_mesh),1))
    v1=my_mesh[:,0]
    v2=my_mesh[:,1]
    v3=my_mesh[:,2]
    #first check what triangles are parallel to the ray and exclude them from further calculations
    normals_dcs_dot=np.einsum('ij...,ij->i...',np.cross(v2-v1, v3-v1),D)
    mask=np.logical_or(normals_dcs_dot<-Epsilon, normals_dcs_dot>Epsilon)
    
    #Moeller-Trumbore
    MTmatrix=np.array([-D[mask].T, (v2-v1)[mask].T,(v3-v1)[mask].T]).T
    MTvector=O[mask]-v1[mask]
    #this gives us intersection of ray with the plane of the triangle:
    tuv=np.linalg.solve(MTmatrix, MTvector)
    #now to see if the intersection is actually within the triangle, and exclude if not
    mask2=np.logical_and.reduce([tuv[:,0]>Epsilon, tuv[:,1]>Epsilon, tuv[:,2]>Epsilon, tuv[:,1]<1+Epsilon, tuv[:,1]+tuv[:,2]<1+Epsilon])
    if mask2.sum()==0: #this photon hits the detector and nothing else, so keep it
        success_indices[i]=True
        t=O[0]-np.array([cylinderR+1,yp[i],zp[i]])
        t=np.sqrt(np.dot(t,t))
        #print(O[0], D[0])
        sipm_intersections.append(list(O[0]+t*D[0])) #O+tD

sipm_intersections=np.array(sipm_intersections)
filter_decays_po()

print('\nblue photons done')


'''saving stuff'''
decayDensity=N/volume
num_successes=len(poPos)
efficiency=num_successes/(N*M)
#save variables you need
save('scintMC3_{}_output{}_{}x{}'.format(stlname[18:-4],option,N,M), 'N', 'M', 'volume', 'num_successes', 'poPos', 'sipm_intersections')
#load('pipsMC2_output')

print('{} photons detected.'.format(num_successes))
print('efficiency: {}/{} = {}'.format(num_successes,N*M,efficiency))
