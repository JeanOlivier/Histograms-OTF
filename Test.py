
from  numpy import array, meshgrid, arange, zeros
from numpy.random import normal, randint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import histograms_otf as hist
from ctypes import c_uint64, c_uint32, c_uint16

from IPython import get_ipython
ipython = get_ipython()


L = (10**8)
x = array(normal(0,1,L), dtype = 'float32');
y = array(normal(0,1,L), dtype = 'float32');
#t = arange(L)

n = 8;
k = 2**n;
H = zeros(k, dtype=c_uint64)


#ipython.magic ("timeit HH = H")

ipython.magic ("timeit hist.hist1dNbits(x, n,ihist=H, max=6)");
HH = zeros((k,k), dtype=c_uint64)
ipython.magic ("timeit hist.hist2dNbits(x, y, n, force_n=False, option='', ihist=HH, max=6)");
HH = zeros((k,k), dtype=c_uint64)
ipython.magic ("timeit hist.hist2dNbits(x, y, n, force_n=False, option='serial_uint8', ihist=HH, max=6)");
# HH = zeros((k,k), dtype=c_uint64)
# ipython.magic ("timeit hist.hist2dNbits(x, y, n, force_n=False, option='atomic', ihist=HH, max=6)");
# HH = zeros((k,k), dtype=c_uint64)
# ipython.magic ("timeit hist.hist2dNbits(x, y, n, force_n=False, option='par_red', ihist=HH, max=6)");
# HH = zeros((k,k), dtype=c_uint64)
# ipython.magic ("timeit hist.hist2dNbits(x, y, n, force_n=False, option='atomic_uint8', ihist=HH, max=6)");
# HH = zeros((k,k), dtype=c_uint64)
# ipython.magic ("timeit hist.hist2dNbits(x, y, n, force_n=False, option='par_rred', ihist=HH, max=6)");

#H_x = hist.hist1dNbits(x, n,ihist=H, max=6);
#plt.plot(H_x);

#X, Y = meshgrid(arange(1<<n),arange(1<<n));

# HH = zeros((k,k), dtype=c_uint64)
# A = hist.hist2dNbits(x, y, n, force_n=False, option='', ihist=HH, max=6);

# HH = zeros((k,k), dtype=c_uint64)
# B = hist.hist2dNbits(x, y, n, force_n=False, option='serial_uint8', ihist=HH, max=6);
# disp(np.sum(A)-np.sum(B))

# HH = zeros((k,k), dtype=c_uint64)
# B = hist.hist2dNbits(x, y, n, force_n=False, option='atomic', ihist=HH, max=6);
# disp(np.sum(A)-np.sum(B))

# HH = zeros((k,k), dtype=c_uint64)
# B = hist.hist2dNbits(x, y, n, force_n=False, option='par_red', ihist=HH, max=6);
# disp(np.sum(A)-np.sum(B))

# HH = zeros((k,k), dtype=c_uint64)
# B = hist.hist2dNbits(x, y, n, force_n=False, option='atomic_uint8', ihist=HH, max=6);
# disp(np.sum(A)-np.sum(B))

# HH = zeros((k,k), dtype=c_uint64)
# B = hist.hist2dNbits(x, y, n, force_n=False, option='par_rred', ihist=HH, max=6);
# disp(np.sum(A)-np.sum(B))








# fig = plt.figure();
# ax = Axes3D(fig);
# ax.plot_surface(X,Y,A,cmap='viridis');

# fig = plt.figure();
# ax = Axes3D(fig);
# ax.plot_surface(X,Y,B,cmap='viridis');


del x , H # 1d
#del H_x #1d graph 
del y, HH  #2d
#del X,Y, H_xy #2d graph