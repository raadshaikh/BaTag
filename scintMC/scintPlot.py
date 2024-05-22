import numpy as np
import pickle
import matplotlib.pyplot as plt

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

N=194713 #how many photons were produced by scintNRG.py, it depends on 'big' and pressure. this is for big=16. 
#i should have had scintMC3's output also note down the pressure, but i guess it doesn't matter as long as i keep everything else the same and keep track of stuff
pressure = 2.0 #manually keeping track
M=1
option=2 #straight alpha (i was calling it 'artificial' but it's actually being produced by americium source)
stlname='detector_chamber4_cut_noTefl.stl'
load('scintMC3_{}_output{}_{}x{}'.format(stlname[18:-4],option,N,M))
xpo=poPos[:,0]
ypo=poPos[:,1]
zpo=poPos[:,2]
xsi=sipm_intersections[:,0]
ysi=sipm_intersections[:,1]
zsi=sipm_intersections[:,2]
'''plt.subplot(311)
plt.hist(xsi)
plt.subplot(312)
plt.hist(ysi)
plt.subplot(313)
plt.hist(zsi)'''

'''
#not needed since incident angle i am assuming to be irrelevant

incident_rays=sipm_intersections-poPos
incident_rays=incident_rays/np.reshape(np.sqrt(np.einsum('ij...,ij->i...',incident_rays,incident_rays)), (len(incident_rays),1)) #normalise
cos_theta_i=incident_rays[:,0]
theta_i=np.arccos(cos_theta_i)
theta_i=theta_i*180/np.pi #radians to degrees
angle_hist=plt.hist(theta_i, bins=50)
bins, heights = angle_hist[1], angle_hist[0]
bin_centres=bins[:-1]+((bins[1]-bins[0])*0.5)'''

#careful here !

'''
#this is wrong!
weights=np.cos(bin_centres*np.pi/180)**2 #tpb reemission
weights2=np.ones(len(bin_centres))-(0.2/55)*bin_centres #sipm pde
weights3=weights*weights2
#print(bins, bin_centres, heights)
angle_accounted = np.dot(heights, weights3)'''
angles=np.linspace(90,180,1000)
weights=np.cos(angles*np.pi/180)**2 #tpb reemission
weights=weights/np.sum(weights) #normalising the integral to 1 so this is a probability distribution
angles=180-angles
weights2=np.ones(len(angles))-(0.2/55)*angles #sipm pde
weights3=np.dot(weights,weights2)
angle_accounted=np.array([790, 790, 807])*weights3 #assuming angle of incidence of the uv photons on tpb is irrelevant. the numbers here are manually entered after running scintMC3.py and seeing the output
fraction=82333/N #82333 is how many photons are actually produced in argon by a 5.59 MeV alpha
##wait, these simulations are actually for the americium source which is 5.486 MeV! but pretty close so eh
final_counts=angle_accounted*fraction*0.5*0.6 #0.5 sipm qe and 0.6 tpb plqy
print(pressure, 'bars argon:\n', np.mean(final_counts), '+/-', np.std(final_counts))
#plt.show()
exit()

#should have saved this from scintMC2 but i'm not running that again right now
big=16
pressure = 0.7
load('scintNRG_output_{}_{}'.format(big, pressure))
N=int(np.sum(np.rint(energies/np.min(energies))))
positions=lengths+dx/2
positions=10*positions #scintNRG dealt in cm, here i work with mm
weights=np.int_(np.rint(energies/np.min(energies)))
decayPosZ=np.repeat(positions, weights)

x=teflon_intersections[:,0]
y=teflon_intersections[:,1]
z=teflon_intersections[:,2]

r=np.sqrt(x**2+y**2)
theta=np.arctan2(y,x)
theta=theta*180/np.pi

plt.suptitle(stlname)
plt.subplot(411)
plt.xlim((0,67))
plt.hist(decayPosZ, label='z all photons (mm)')
plt.legend()
plt.subplot(412)
plt.xlim((0,67))
plt.hist(decayPos[:,2], label='z successful photons')
plt.legend()
plt.subplot(413)
plt.xlim((0,67))
plt.hist(z, label='z teflon hits')
plt.legend()
plt.subplot(414)
plt.hist(theta, label='theta teflon hits (deg)')
plt.legend()

'''plt.scatter(theta, z, alpha=0.7, s=1)
plt.xlabel('$theta$ (deg)')
plt.ylabel('z (mm)')
plt.show()'''
plt.savefig('scintPlot_{}_output{}_{}x{}.png'.format(stlname[18:-4],option,N,M), bbox_inches='tight')
plt.show()