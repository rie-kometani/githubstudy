import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

from pycuda.compiler import SourceModule

MAT_SIZE_X = 1000
MAT_SIZE_Y = 1000

BLOCKSIZE = 32

mod = SourceModule("""
__global__ void add_matrix_gpu(const float* __restrict__ dMat_A, const float* __restrict__ dMat_B, float *dMat_G, const int mat_size_x, const int mat_size_y) {
        int mat_x = threadIdx.x + blockIdx.x * blockDim.x;
        int mat_y = threadIdx.y + blockIdx.y * blockDim.y;
        if (mat_x >= mat_size_x) {
            return;
        } 
        if (mat_y >= mat_size_y) {
            return;
        }

        const int index = mat_y * mat_size_x + mat_x;

        dMat_G[index] = dMat_A[index] + dMat_B[index];
    
}
""")
add_matrix_gpu = mod.get_function("add_matrix_gpu")

block = (BLOCKSIZE, BLOCKSIZE, 1)
grid = ((MAT_SIZE_X + block[0] - 1) // block[0], (MAT_SIZE_Y + block[1] - 1) // block[1])

h_a = numpy.random.randn(MAT_SIZE_X, MAT_SIZE_Y).astype(numpy.float32)
h_b = numpy.random.randn(MAT_SIZE_X, MAT_SIZE_Y).astype(numpy.float32)
h_d = numpy.empty_like(h_a)

add_matrix_gpu(cuda.In(h_a), cuda.In(h_b), cuda.Out(h_d), numpy.int32(MAT_SIZE_X), numpy.int32(MAT_SIZE_Y), block = block, grid = grid)
       
for y in range(MAT_SIZE_Y):
    for x in range(MAT_SIZE_X):
        i = y * MAT_SIZE_X + x
        if i < 10:
            print("A[%d]=%8.2f, B[%d]=%8.2f, D[%d]=%8.2f" % (i, h_a[x][y], i, h_b[x][y], i, h_d[x][y]))
        else:
            break 
