import numpy as np

matA = np.loadtxt("matA.txt")
matB = np.loadtxt("matB.txt")
matC = np.loadtxt("matC.txt")

np_res = np.matmul(matA, matB)
diff_mat = np.abs(np_res - matC)

print(diff_mat)

print("Total diff: ", diff_mat.sum(), " of ", matC.shape)
print("Aveaging to ", diff_mat.sum() / diff_mat.size)