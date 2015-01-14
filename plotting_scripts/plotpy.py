import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import glob
import os

# TO MAKE A MOVIE
# DO SOMETHING LIKE THIS
# mencoder "mf://*.png" -mf w=800:h=600:fps=2:type=png -ovc copy -oac copy -o movie.avi



filesList = glob.glob("*-p0.dat")
zcoord = np.floor(31.0/2.0)
zcoord = 14
logscale = True
   
clmin = 0.0
clmax = 0.0005

logclmin = -4
logclmax = 0

to_dir = "t6_xy-slice_z" + str(zcoord)

if not os.path.isdir(to_dir): os.mkdir(to_dir)

for filename in filesList:
    time = filename.replace('-p0.dat','')

    data = np.genfromtxt(filename)
    
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    
   # Ex = data[:,3]
    #Ey = data[:,4]
    #Ez = data[:,5]
    
    UE = data[:,3]
    UB = data[:,4]
    UT = data[:,5]

    AA = data[:,6]
    AAan = data[:,7]
    
    select = np.where(z == zcoord)
    
    xvals = x[select]
    yvals = y[select]
    UEvals = UE[select]
    UBvals = UB[select]
    UTvals = UT[select]
    AAvals = AA[select]
    AAanvals = AAan[select]
    
    N = np.sqrt(len(xvals))
    xvals = xvals.reshape((N,N))
    yvals = yvals.reshape((N,N))
    
    UEvals = UEvals.reshape((N,N))
    UBvals = UBvals.reshape((N,N))
    UTvals = UTvals.reshape((N,N))
    AAvals = AAvals.reshape((N,N))
    AAanvals = AAanvals.reshape((N,N))
    
    if(logscale==True):
        UEvals = np.log10(UEvals)
        UBvals = np.log10(UBvals)
        UTvals = np.log10(UTvals)
        AAvals = np.log10(AAvals)
        AAanvals = np.log10(AAanvals)
        clmin = logclmin
        clmax = logclmax
    
    #clmin = np.min(UTvals)
   # clmax = np.max(UTvals)
    color = 'gist_heat'
    
    fig = plt.figure(figsize = [24,6])
    fig.subplots_adjust(wspace=0.01,hspace=0.01)
    
    
    
    ax = fig.add_subplot(141)
    ax.set_axis_bgcolor("#000000")
    subset = UE[select]
    plt1 = ax.pcolormesh(xvals,yvals,UEvals, cmap=color, vmin=clmin,vmax=clmax)
    ax.set_title('Electric Field Energy at t = ' + time)
    plt.xlim(0,np.max(xvals))
    plt.ylim(0,np.max(yvals))
  #  plt.colorbar(plt1)
    
    ax = fig.add_subplot(142)
    ax.set_axis_bgcolor("#000000")
    subset = UB[select]
    plt2 = ax.pcolormesh(xvals,yvals,UBvals, cmap=color, vmin=clmin,vmax=clmax)
    ax.set_title('Magnetic Field Energy at t = ' + time)
    ax.set_yticks([])
    plt.xlim(0,np.max(xvals))
    plt.ylim(0,np.max(yvals))
  #  plt.colorbar(plt2)
    
    ax = fig.add_subplot(143)
    ax.set_axis_bgcolor("#000000")
    subset = UB[select]
    plt3 = ax.pcolormesh(xvals,yvals,UTvals, cmap=color, vmin=clmin,vmax=clmax)
    ax.set_title('Total Energy at t = ' + time)
    ax.set_yticks([])
    plt.xlim(0,np.max(xvals))
    plt.ylim(0,np.max(yvals))
    
    ax = fig.add_subplot(144)
    ax.set_axis_bgcolor("#000000")
    subset = UB[select]
    plt3 = ax.pcolormesh(xvals,yvals,AAvals, cmap=color, vmin=clmin,vmax=clmax)
    ax.set_title('A*A at t = ' + time)
    ax.set_yticks([])
    plt.xlim(0,np.max(xvals))
    plt.ylim(0,np.max(yvals))

    
 #   plt.colorbar(plt3)
    fig.savefig(to_dir + "/" + time + "_energies_z"+str(zcoord)+".png")
    plt.close(fig)
    
    fig = plt.figure(figsize = [6,6])
    ax = fig.add_subplot(111)
    ax.set_axis_bgcolor("#000000")
    plt3 = ax.pcolormesh(xvals,yvals,AAanvals, cmap=color)
    ax.set_title('A*A analytical at t = ' + time)
    ax.set_yticks([])
    plt.xlim(0,np.max(xvals))
    plt.ylim(0,np.max(yvals))
    plt.colorbar(plt3)
    fig.savefig(to_dir+"/aa_anal/"+time+"_AAanal_z"+str(zcoord)+".png")
    plt.close(fig)
    
    
    
