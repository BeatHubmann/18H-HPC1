import mpl_toolkits.mplot3d
from numpy import *
from pylab import *

u = loadtxt("density0.dat")
N = int(sqrt(u.shape[0]))
print("N=%d" % N)
u.shape = (N,N)

fig = figure()
x,y = mgrid[-1:1:N*1j, -1:1:N*1j]

pcolormesh(x,y,u)
colorbar()
show()
fig.savefig('density0000.png')


fig = figure()
u = loadtxt("density1000.dat")
u.shape = (N,N)
pcolormesh(x,y,u)
show()
fig.savefig('density1000.png')

fig = figure()
u = loadtxt("density2000.dat")
u.shape = (N,N)
pcolormesh(x,y,u)
show()
fig.savefig('density2000.png')

fig = figure()
u = loadtxt("density3000.dat")
u.shape = (N,N)
pcolormesh(x,y,u)
show()
fig.savefig('density3000.png')

fig = figure()
u = loadtxt("density4000.dat")
u.shape = (N,N)
pcolormesh(x,y,u)
show()
fig.savefig('density4000.png')

fig = figure()
u = loadtxt("density5000.dat")
u.shape = (N,N)
pcolormesh(x,y,u)
show()
fig.savefig('density5000.png')

fig = figure()
u = loadtxt("density6000.dat")
u.shape = (N,N)
pcolormesh(x,y,u)
show()
fig.savefig('density6000.png')

fig = figure()
u = loadtxt("density7000.dat")
u.shape = (N,N)
pcolormesh(x,y,u)
show()
fig.savefig('density7000.png')

fig = figure()
u = loadtxt("density8000.dat")
u.shape = (N,N)
pcolormesh(x,y,u)
show()
fig.savefig('density8000.png')

fig = figure()
u = loadtxt("density9000.dat")
u.shape = (N,N)
pcolormesh(x,y,u)
show()
fig.savefig('density9000.png')