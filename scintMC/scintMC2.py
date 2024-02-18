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
    poPos=poPos[success_indices]
    poDir=poDir[success_indices]

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


'''#interpolate the given data to find out the activity of each species at every point in time
#these are all single-decay reactions, so just integrate activity of the poloniums to find out how many of their decays there are

times_given=np.array([0,86400,172800,259200,345600,432000,864000])

# A for Activity
Ra226A=np.array([5000.00000025,4999.994062,4999.988124,4999.982185,4999.976247,4999.970309,4999.940618]) #t=0 value extrapolated
Rn222A=np.array([0,833.6894579	,1528.370218	,2107.220312	,2589.553113	,2991.46172	,4193.130921])
Po218A=np.array([0,	831.3278231	,1526.402361	,2105.580574	,2588.186785	,2990.323214	,4192.673588])
Pb214A=np.array([0,	810.8106227	,1509.306184	,2091.335006	,2576.316521	,2980.4322	,4188.700407])
Bi214A=np.array([0	,795.5202288	,1496.565299	,2080.718531	,2567.470235	,2973.060946	,4185.739404])
Po214A=np.array([0	,795.5202267	,1496.565297	,2080.71853	,2567.470234	,2973.060945	,4185.739404])
Pb210A=np.array([0	,0.033151067	,0.131646674	,0.284703018	,0.483216622	,0.719601905	,2.281773287])

#Aif for Activity interpolation function
Ra226Aif=interp1d(times_given, Ra226A, kind='cubic')
Rn222Aif=interp1d(times_given, Rn222A, kind='cubic')
Po218Aif=interp1d(times_given, Po218A, kind='cubic')
Pb214Aif=interp1d(times_given, Pb214A, kind='cubic')
Bi214Aif=interp1d(times_given, Bi214A, kind='cubic')
Po214Aif=interp1d(times_given, Po214A, kind='cubic')
Pb210Aif=interp1d(times_given, Pb210A, kind='cubic')

times_smooth=np.arange(0,timesteps,dt)

#Ai for Activity interpolated
Ra226Ai=Ra226Aif(times_smooth)
Rn222Ai=Rn222Aif(times_smooth)
Po218Ai=Po218Aif(times_smooth)
Pb214Ai=Pb214Aif(times_smooth)
Bi214Ai=Bi214Aif(times_smooth)
Po214Ai=Po214Aif(times_smooth)
Pb210Ai=Pb210Aif(times_smooth)

#AiI for Activity interpolated Integrated

Ra226AiI=Ra226Ai.sum()*dt
Rn222AiI=Rn222Ai.sum()*dt
Po218AiI=Po218Ai.sum()*dt
Pb214AiI=Po218Ai.sum()*dt
Bi214AiI=Bi214Ai.sum()*dt
Po214AiI=Po214Ai.sum()*dt
Pb210AiI=Pb210Ai.sum()*dt

#these are just the old variable names I was using.
#dividing by total volume to get density, then multiplying by my bounding box to get how many decays to consider
total_volume= 60e6 #in mm^3
N=int(volume*Rn222AiI/total_volume)
Po218s=int(Po218AiI*volume/total_volume)
Po214s=int(Po214AiI*volume/total_volume)

print('decay calculations done\n')'''

'''start the radon hitting'''
# Load the STL files...
stlname='detector_chamber4_cut_long_cent.stl'
my_mesh = mesh.Mesh.from_file(stlname)
my_mesh=np.reshape(my_mesh, (len(my_mesh),3,3)) #before this, the whole triangle was in a (9,) array. I am splitting the vertices into their own arrays

#geometry
if stlname=='detector_chamber4_cut.stl':
    cylinder_zmin, cylinder_zmax = 2, 22
    detectZ = 12
elif stlname=='detector_chamber4_cut_shift.stl':
    cylinder_zmin, cylinder_zmax = 44, 64
    detectZ = 54
elif stlname=='detector_chamber4_cut_shift_long.stl':
    cylinder_zmin, cylinder_zmax = 2, 64
    detectZ = 54
elif stlname=='detector_chamber4_cut_long_cent.stl':
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

zs=decayPos[:,2]
'''rhosquareds=np.power(decayPos[:,0], 2)+np.power(decayPos[:,1], 2)
fielddecaymask=np.logical_and.reduce([zs<41.25, rhosquareds<67.24])
fielddecays=fielddecaymask.sum()'''

decayDir=decayDir/np.reshape(np.sqrt(np.einsum('ij...,ij->i...',decayDir,decayDir)), (len(decayDir),1))
#this einsum thing is to get the elementwise dot product between two lists of vectors (here to get a list of norms of a list of vectors), to normalise the directions

#determine if the generated direction ray falls within the solid angle of feasibility
#i.e. would it hit the teflon cylinder if nothing was in the way
#work this out on pen-paper
a=np.power(decayDir[:,0], 2)+np.power(decayDir[:,1], 2)
b=2*(decayPos[:,0]*decayDir[:,0] + decayPos[:,1]*decayDir[:,1])
c=np.power(decayPos[:,0], 2)+np.power(decayPos[:,1], 2)-cylinderR**2
t1=(-b+np.power(np.power(b,2)-4*a*c, 0.5))/(2*a)
cylinder_intersect_z=decayPos[:,2]+t1*decayDir[:,2]
success_indices=np.logical_and(cylinder_intersect_z>cylinder_zmin, cylinder_intersect_z<cylinder_zmax)
filter_decays()

#use ray-casting algorithm
#the algorithm takes a point (where photon started), a direction, and a triangle (surface of solid object).
#it returns the intersection of the ray with the triangle (and whether it even intersects)
#for each decay, iterate over all triangles in the assembly to get a list of intersection points
#choose the nearest intersection point
#does it correspond to the teflon cylinder's interior?
#if yes, the photon succeeded, proceed to change direction and find out whether it succeeds the next stage (hitting the sipm
#do this for all decays parallely? use numpy magic to end up with a mask of success_indices that then filters the decays

#finally find length of decayPos, i.e. how many photons succeeded. this comes after the sipm detection!

lendecayPos=len(decayPos)
success_indices=np.zeros(lendecayPos, dtype='bool')
teflon_intersections=[]
for i in range(lendecayPos):
    if int(10000*i/lendecayPos)%1==0: print("uv photons: {:.2f}%".format(100*i/lendecayPos), end='\r')
    O=np.tile(decayPos[i], (len(my_mesh),1))
    D=np.tile(decayDir[i], (len(my_mesh),1))
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
    #print(O[mask][mask2])
    #print(tuv[mask2][:,0])
    #print(D[mask][mask2])
    try:
        if np.min(tuv[mask2][:,0]) < t1[i]+Epsilon or np.min(tuv[mask2][:,0]) > t1[i]-Epsilon: #is the closest intersection on the cylinder inner surface?
            success_indices[i]=True
            teflon_intersections.append(list(O[0]+t1[i]*D[0])) #O+tD
    except ValueError:
        pass
teflon_intersections=np.array(teflon_intersections)
filter_decays()

print('\nteflon cylinder check done \n\nstarting with {} blue photons'.format(len(decayPos)))

teflon_hits=len(decayPos)

'''done with teflon cylinder. move to sipm.'''

#don't want to rename everything, so 'poPos' and 'poDir' represent blue photons from the tpb trying to go to the sipm.

#each incident photon produces some number of wavelength-shifted photons, determined by photoluminescence quantum yield (plqy)
#generate random directions for each blue photon
#N rows, 3 columns. each row identifies an ion, and there is a column for each coordinate
#plqy=0.6 IMPORTANT manually reduce the efficiency at the end by this much
poPos=teflon_intersections.copy()
poDir=np.random.normal(0,1,(len(poPos),3))
poDir=poDir/np.reshape(np.sqrt(np.einsum('ij...,ij->i...',poDir,poDir)), (len(poDir),1))

#find solid angle of feasability. sipm is a 6x6mm square, normal to x-axis, centred at (21, 0 54)
yp=(poDir[:,1]/poDir[:,0])*(cylinderR+1-poPos[:,0]) #y_Plane, referring to the y-coord of the point where the photon intersects with the Plane of the sipm
zp=(poDir[:,2]/poDir[:,0])*(cylinderR+1-poPos[:,0])
success_indices=np.logical_and.reduce([yp>-detectR/2, yp<detectR/2, zp>detectZ-(detectR/2), zp<detectZ+(detectR/2)])
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
        sipm_intersections.append(list(O[0]+t1[i]*D[0])) #O+tD #WRONG t1 is WRONG HERE!!!!!

sipm_intersections=np.array(sipm_intersections)
filter_decays_po()

print('\nblue photons done')


'''saving stuff'''
decayDensity=N/volume
num_successes=len(poPos)
efficiency=num_successes/(N*M)
#save variables you need
save('scintMC_{}_output{}_{}x{}'.format(stlname[18:-4],option,N,M), 'N', 'M', 'volume', 'num_successes', 'teflon_hits', 'decayPos', 'teflon_intersections', 'poPos', 'sipm_intersections')
#load('pipsMC2_output')

print('{} photons detected.'.format(num_successes))
print('efficiency: {}/{} = {}'.format(num_successes,N*M,efficiency))
