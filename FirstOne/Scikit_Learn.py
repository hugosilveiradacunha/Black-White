import numpy as np

# Generating a random array
X = np.random.random((3, 5))  # a 3 x 5 array

print(X)

# Accessing elements

# get a single element
print(X[0, 0])

# get a row
print(X[1])

# get a column
print(X[:, 1])

# Transposing an array
print(X.T)

# Turning a row vector into a column vector
y = np.linspace(0, 12, 5)
print(y)


# make into a column vector
print(y[:, np.newaxis])



# getting the shape or reshaping an array
print(X.shape)
print(X.reshape(5, 3))


# indexing by an array of integers (fancy indexing)
indices = np.array([3, 1, 0])
print(indices)
X[:, indices]



# Scipy Sparse Matrices



from scipy import sparse

# Create a random array with a lot of zeros
X = np.random.random((10, 5))
print(X)


# set the majority of elements to zero
X[X < 0.7] = 0
print(X)


# turn X into a csr (Compressed-Sparse-Row) matrix
X_csr = sparse.csr_matrix(X)
print(X_csr)


# convert the sparse matrix to a dense array
print(X_csr.toarray())




# Create an empty LIL matrix and add some items
X_lil = sparse.lil_matrix((5, 5))

for i, j in np.random.randint(0, 5, (15, 2)):
    X_lil[i, j] = i + j

print(X_lil)

print(X_lil.toarray())

print(X_lil.tocsr())



# Matplotlib

#%matplotlib inline

import matplotlib.pyplot as plt

# plotting a line
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))

# scatter-plot points
x = np.random.normal(size=500)
y = np.random.normal(size=500)
plt.scatter(x, y)

# showing images
x = np.linspace(1, 12, 100)
y = x[:, np.newaxis]

im = y * np.sin(x) * np.cos(y)
print(im.shape)

# imshow - note that origin is at the top-left by default!
plt.imshow(im)

# Contour plot - note that origin here is at the bottom-left by default!
plt.contour(im)

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D
ax = plt.axes(projection='3d')
xgrid, ygrid = np.meshgrid(x, y.ravel())
ax.plot_surface(xgrid, ygrid, im, cmap=plt.cm.jet, cstride=2, rstride=2, linewidth=0)

