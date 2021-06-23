import os
import random
for num in [9, 16, 25]:
    for n in range(1, 500):
        i = random.randint(1, 100)
        j = random.randint(1, 100)
        k = random.randint(1, 100)
        print("Running on: " + str(i) + " * " + str(j) + " * " + str(k) + ", procnum = " + str(num) + ", roound: " + str(n))
        os.system("./matgen " + str(i) +  " " + str(j) + " matA.bin")
        os.system("./matgen " + str(j) +  " " + str(k) + " matB.bin")
        os.system("./serialMul matA.bin matB.bin serialRes.bin")
        val = os.system("mpirun -np " + str(num) + " ./cannon")
        if val != 0:
            print("ERROR!!!!!")
            exit(-1)
        # val = os.system("./comp matC.bin serialRes.bin")
        val = os.system("./comp matC.bin serialRes.bin | grep succeed")
        if val != 0:
            val = os.system("./comp matC.bin serialRes.bin")
            print("ERROR!!!!!")
            exit(-1)
        print("--------------------------------------------------------")
