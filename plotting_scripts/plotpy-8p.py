import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import glob
import os

# TO MAKE A MOVIE
# DO SOMETHING LIKE THIS
# mencoder "mf://*.png" -mf w=800:h=600:fps=3:type=png -ovc copy -oac copy -o movie8.avi



filesList = glob.glob("*-p0.dat")
timeList = []
for filename in filesList:
    time = filename.replace("-p0.dat",'')
    timeList.append(str(time))


zcoord = np.floor(31.0/2.0)
zcoord = 14
logscale = True
   
clmin = 0.0
clmax = 0.0005

logclmin = -4
logclmax = 0

to_dir = "xy-slice_z" + str(zcoord)

if not os.path.isdir(to_dir): os.mkdir(to_dir)

for time in timeList:
    data = np.genfromtxt(str(time) + "-p2.dat")    
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    UE = data[:,3]
    UB = data[:,4]
    UT = data[:,5]
    AA = data[:,6]
    
    select = np.where(z == zcoord)
    xvals = x[select]
    yvals = y[select]
    UEvals = UE[select]
    UBvals = UB[select]
    UTvals = UT[select]
    
    N = np.sqrt(len(xvals))
    xvals = xvals.reshape((N,N))
    yvals = yvals.reshape((N,N))
    
    UEvals = UEvals.reshape((N,N))
    UBvals = UBvals.reshape((N,N))
    UTvals = UTvals.reshape((N,N))
    
    if(logscale==True):
        UEvals = np.log10(UEvals)
        UBvals = np.log10(UBvals)
        UTvals = np.log10(UTvals)
        clmin = logclmin
        clmax = logclmax

    color = 'gist_heat'
    
    fig = plt.figure(figsize = [7+7,6+6])
    fig.subplots_adjust(wspace=0.01,hspace=0.01)
    ax = fig.add_subplot(221)
    

    
    ax.set_axis_bgcolor("#000000")
    plt3 = ax.pcolormesh(xvals,yvals,UTvals, cmap=color, vmin=clmin,vmax=clmax)
    ax.set_ylabel('P2 Total Energy at t = ' + time)
    ax.set_xticks([])
    plt.xlim(np.min(xvals),np.max(xvals))
    plt.ylim(np.min(yvals),np.max(yvals))    
   # plt.colorbar(plt3)
 #################   
    data = np.genfromtxt(str(time) + "-p3.dat")    
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    UE = data[:,3]
    UB = data[:,4]
    UT = data[:,5]
    
    select = np.where(z == zcoord)
    xvals = x[select]
    yvals = y[select]
    UEvals = UE[select]
    UBvals = UB[select]
    UTvals = UT[select]
    
    N = np.sqrt(len(xvals))
    xvals = xvals.reshape((N,N))
    yvals = yvals.reshape((N,N))
    
    UEvals = UEvals.reshape((N,N))
    UBvals = UBvals.reshape((N,N))
    UTvals = UTvals.reshape((N,N))
    
    if(logscale==True):
        UEvals = np.log10(UEvals)
        UBvals = np.log10(UBvals)
        UTvals = np.log10(UTvals)
        clmin = logclmin
        clmax = logclmax

    color = 'gist_heat'
    
    ax = fig.add_subplot(222)
    ax.set_axis_bgcolor("#000000")
    plt3 = ax.pcolormesh(xvals,yvals,UTvals, cmap=color, vmin=clmin,vmax=clmax)
    ax.yaxis.set_label_position("right")
    ax.set_ylabel('P3 Total Energy at t = ' + time)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlim(np.min(xvals),np.max(xvals))
    plt.ylim(np.min(yvals),np.max(yvals))     
    #plt.colorbar(plt3)
    ##############################
    data = np.genfromtxt(str(time) + "-p0.dat")    
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    UE = data[:,3]
    UB = data[:,4]
    UT = data[:,5]
    
    select = np.where(z == zcoord)
    xvals = x[select]
    yvals = y[select]
    UEvals = UE[select]
    UBvals = UB[select]
    UTvals = UT[select]
    
    N = np.sqrt(len(xvals))
    xvals = xvals.reshape((N,N))
    yvals = yvals.reshape((N,N))
    
    UEvals = UEvals.reshape((N,N))
    UBvals = UBvals.reshape((N,N))
    UTvals = UTvals.reshape((N,N))
    
    if(logscale==True):
        UEvals = np.log10(UEvals)
        UBvals = np.log10(UBvals)
        UTvals = np.log10(UTvals)
        clmin = logclmin
        clmax = logclmax

    color = 'gist_heat'
    ax = fig.add_subplot(223)
    ax.set_axis_bgcolor("#000000")
    plt3 = ax.pcolormesh(xvals,yvals,UTvals, cmap=color, vmin=clmin,vmax=clmax)
    ax.set_ylabel('P0 Total Energy at t = ' + time)
    plt.xlim(np.min(xvals),np.max(xvals))
    plt.ylim(np.min(yvals),np.max(yvals))    
#    colorbar_ax = ax.add_axes(
    #plt.colorbar(plt3,orientation='horizontal',pad=0.001)
    #####################
    data = np.genfromtxt(str(time) + "-p1.dat")    
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    UE = data[:,3]
    UB = data[:,4]
    UT = data[:,5]
    
    select = np.where(z == zcoord)
    xvals = x[select]
    yvals = y[select]
    UEvals = UE[select]
    UBvals = UB[select]
    UTvals = UT[select]
    
    N = np.sqrt(len(xvals))
    xvals = xvals.reshape((N,N))
    yvals = yvals.reshape((N,N))
    
    UEvals = UEvals.reshape((N,N))
    UBvals = UBvals.reshape((N,N))
    UTvals = UTvals.reshape((N,N))
    
    if(logscale==True):
        UEvals = np.log10(UEvals)
        UBvals = np.log10(UBvals)
        UTvals = np.log10(UTvals)
        clmin = logclmin
        clmax = logclmax

    color = 'gist_heat'
    
    ax = fig.add_subplot(224)
    ax.set_axis_bgcolor("#000000")
    plt3 = ax.pcolormesh(xvals,yvals,UTvals, cmap=color, vmin=clmin,vmax=clmax)
    ax.yaxis.set_label_position("right")
    ax.set_ylabel('P1 Total Energy at t = ' + time)
    ax.set_yticks([])
    plt.xlim(np.min(xvals),np.max(xvals))
    plt.ylim(np.min(yvals),np.max(yvals))    
   # plt.colorbar(plt3,orientation='horizontal')
        
    
    fig.savefig(to_dir + "/" + time + "_energies_z"+str(zcoord)+".png")
    plt.close(fig)
    
    
