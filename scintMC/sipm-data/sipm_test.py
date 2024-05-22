import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import scipy.stats as ss

def fname(c,r,v,j,n): #construct filename from parameters to fetch the waveform data files
    #head = 'c'+str(c)+'r'+str(r)+'v'+str(v)+'_'+str(j)
    #the above line was for the led driver. the below lines are more adhoc and may change depending on what im working on right now
    head = 'sipm_argon_C'
    return head+'/'+head+'_'+str(n).zfill(3)+'.csv'

c=6 #LED driver channel
r=1 #LED pulse rate in MHz
v=12.5 #LED control voltage
Bpps=[] #array containing peak to peak voltages of channel B waveforms
Cpps=[]
Bareas=[] #array containing area-under-the-curve of channel B waveforms (unused, i thought of doing this but it turned out to be more trouble than it was worth)
Careas=[]
for j in range(1,1+1): #last time I took ten runs of each condition and iterated over/combined them so i could get 10,000 waveforms in all (picoscope only supports 1000 in one file). got too lazy for that this time, sorry
    for i in range(1,500+1):
        #print(i)
        data=np.loadtxt(fname(c,r,v,j,i), delimiter=',', skiprows=3)
        #Time,Channel A,Channel B,Channel C
        #(ns),(mV),(mV),(mV)
        t=data[:,0]
        dt=t[1]-t[0]
        #A=data[:,1]
        #B=data[:,2]
        #C=data[:,3]
        #the above lines were for the led driver tests
        B=data[:,1]
        C=data[:,2]
        Bpp = np.ptp(B)
        Cpp = np.ptp(C)
        Bpps.append(Bpp)
        Cpps.append(Cpp)
        # Barea=np.trapz(B,dx=dt) #i was gonna try analysing curve integrals but there were some complications so never mind
        # Carea=np.trapz(C,dx=dt)
        # Bareas.append(Barea)
        # Careas.append(Carea)
Bpps=1*np.array(Bpps) #at one point i needed to multiply by 1000 to convert mV and V. don't need to anymore, but i also don't feel like deleting this line in case i need to do it again
Cpps=1*np.array(Cpps)

#plt.suptitle('Channel '+str(c)+', Voltage '+str(v)+' V')
#print('Channel '+str(c)+', Voltage '+str(v)+' V')
#the above lines were for the led driver tests

arr=Bpps
#arr=arr[arr<8]
x=np.linspace(np.mean(arr)-3*np.std(arr),np.mean(arr)+3*np.std(arr), 50)
print(np.mean(arr), ' +/- ', np.std(arr))
N=int((np.mean(arr)/np.std(arr))**2) #Dr. Gornea's statistical photoelectron count estimator (i still don't completely understand this to be honest)
print(N)
print(np.mean(arr)/N) #mV per photoelectron
plt.subplot(1,2,1)
#labelB='Pulse ptp$={}\pm{}$ mV\n$N={}$\nGain={} mV/pe'.format(round(np.mean(arr),4), round(np.std(arr),4), N, round(np.mean(arr)/N,4))
labelB='Pulse ptp$={}\pm{}$ mV'.format(round(np.mean(arr),4), round(np.std(arr),4))
plt.hist(arr, bins=15, density=False, label=labelB)
#number of bins to be adjusted by hand for each dataset!
# arrFit = ss.norm.pdf(x,np.mean(arr), np.std(arr))
# plt.plot(x,arrFit)
plt.title('sipm 1, 29.6V')
plt.legend()

arr=Cpps
#arr=arr[arr<4]
x=np.linspace(np.mean(arr)-3*np.std(arr),np.mean(arr)+3*np.std(arr), 50)
print(np.mean(arr), ' +/- ', np.std(arr))
N=int((np.mean(arr)/np.std(arr))**2)
print(N)
print(np.mean(arr)/N)
plt.subplot(1,2,2)
#labelC='Pulse ptp$={}\pm{}$ mV\n$N={}$\nGain={} mV/pe'.format(round(np.mean(arr),4), round(np.std(arr),4), N, round(np.mean(arr)/N,4))
labelC='Pulse ptp$={}\pm{}$ mV'.format(round(np.mean(arr),4), round(np.std(arr),4))
plt.hist(arr, bins=15, density=False, label=labelC)
# arrFit = ss.norm.pdf(x,np.mean(arr), np.std(arr))
# plt.plot(x,arrFit)
plt.title('sipm 2, 29.4V')
plt.legend()

## to check correlation between both sipms
# plt.title('scatter plot of sipm1 and sipm2 amplitudes')
# plt.scatter(Bpps,Cpps)

plt.show()