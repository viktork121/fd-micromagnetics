import numpy as np
from scipy import ndimage, constants

class ExchangeField(object):
    def __init__(self, mesh, material):
        self._mesh = mesh
        self._A = material["A"]
        self._Ms = material["Ms"]

        # initialize laplace kernel
        self._kernel = np.zeros((3,3,3))
        # TODO setup 3D Laplace kernel
        self._kernel[:, 1, 1] += 1/self._mesh.dx[0]**2*np.array([1., -2., 1.])
        self._kernel[1, :, 1] += 1/self._mesh.dx[1]**2*np.array([1., -2., 1.])
        self._kernel[1, 1, :] += 1/self._mesh.dx[2]**2*np.array([1., -2., 1.])

        # initialize scratch space
        self._h = np.zeros(mesh.n + (3,))

    def h(self, t, m):
        # TODO
        # Implement exchange field:
        # 2 * A / (mu_0 * Ms) * Laplace(m)
        # TIP: use ndimage.convolve with self._kernel
        for i in range(2):
            self._h[:,:,:,i] = 2*self._A/(constants.mu_0*self._Ms)*ndimage.convolve(m[:,:,:,i], self._kernel)  
        return self._h

    def E(self, t, m):
        return -1/2*constants.mu_0*self._Ms*self._mesh.cell_volume * np.sum(m*self._h)
