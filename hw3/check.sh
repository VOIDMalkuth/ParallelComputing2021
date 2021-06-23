./print_mod matA.bin > matA.txt
sed -i '1d' matA.txt
./print_mod matB.bin > matB.txt
sed -i '1d' matB.txt
./print_mod matC.bin > matC.txt
sed -i '1d' matC.txt

python3 check_correctness.py