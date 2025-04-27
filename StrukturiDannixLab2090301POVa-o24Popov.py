import numpy as np
import time
from scipy.linalg.blas import cgemm  # BLAS implementation

N_LARGE = 4096
N_SMALL = 200

np.random.seed(42)
A_large = np.random.rand(N_LARGE, N_LARGE).astype(np.complex64) + 1j * np.random.rand(N_LARGE, N_LARGE).astype(np.complex64)
B_large = np.random.rand(N_LARGE, N_LARGE).astype(np.complex64) + 1j * np.random.rand(N_LARGE, N_LARGE).astype(np.complex64)

A_small = np.random.rand(N_SMALL, N_SMALL).astype(np.complex64) + 1j * np.random.rand(N_SMALL, N_SMALL).astype(np.complex64)
B_small = np.random.rand(N_SMALL, N_SMALL).astype(np.complex64) + 1j * np.random.rand(N_SMALL, N_SMALL).astype(np.complex64)

complexity_small = 2 * N_SMALL**3
complexity_large = 2 * N_LARGE**3

def measure_performance(func, *args):
    start_time = time.time()
    result = func(*args)
    elapsed_time = time.time() - start_time
    mflops = complexity_small / (elapsed_time * 1e6) if args[0].shape[0] == N_SMALL else complexity_large / (elapsed_time * 1e6)
    return result, elapsed_time, mflops

def matrix_multiply_formula(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.complex64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

print("Работу выполнил: Попов Олег Михайлович 09.03.01ПОВа-o24")
print("=" * 50)
print("1-Й ВАРИАНТ: УМНОЖЕНИЕ ПО ФОРМУЛЕ ИЗ ЛИНЕЙНОЙ АЛГЕБРЫ")
print(f"Размер матрицы: {N_SMALL}x{N_SMALL}")
C_formula, time_formula, mflops_formula = measure_performance(matrix_multiply_formula, A_small, B_small)
print(f"Время выполнения: {time_formula:.2f} секунд")
print(f"Производительность: {mflops_formula:.2f} MFLOPS")
print("=" * 50)

def matrix_multiply_blas(A, B):
    return cgemm(alpha=1.0, a=A, b=B)

print("\n" + "=" * 50)
print("2-Й ВАРИАНТ: ИСПОЛЬЗОВАНИЕ CBALS_CGEMM ИЗ БИБЛИОТЕКИ BLAS")
print(f"Размер матрицы: {N_LARGE}x{N_LARGE}")
C_blas, time_blas, mflops_blas = measure_performance(matrix_multiply_blas, A_large, B_large)
print(f"Время выполнения: {time_blas:.2f} секунд")
print(f"Производительность: {mflops_blas:.2f} MFLOPS")
print("=" * 50)

def matrix_multiply_optimized(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.complex64)
    block_size = 256
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                # Блочное умножение
                C[i:i+block_size, j:j+block_size] += np.dot(
                    A[i:i+block_size, k:k+block_size],
                    B[k:k+block_size, j:j+block_size]
                )
    return C
print("\n" + "=" * 50)
print("3-Й ВАРИАНТ: ОПТИМИЗИРОВАННЫЙ АЛГОРИТМ")
print(f"Размер матрицы: {N_LARGE}x{N_LARGE}")
C_optimized, time_optimized, mflops_optimized = measure_performance(matrix_multiply_optimized, A_large, B_large)
print(f"Время выполнения: {time_optimized:.2f} секунд")
print(f"Производительность: {mflops_optimized:.2f} MFLOPS")
print("=" * 50)

print("\n" + "=" * 50)
print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ:")
print(f"1-й вариант (размер {N_SMALL}x{N_SMALL}): {mflops_formula:.2f} MFLOPS")
print(f"2-й вариант (размер {N_LARGE}x{N_LARGE}): {mflops_blas:.2f} MFLOPS")
print(f"3-й вариант (размер {N_LARGE}x{N_LARGE}): {mflops_optimized:.2f} MFLOPS")

performance_ratio = mflops_optimized / mflops_blas
print(f"Отношение производительности (3-й / 2-й): {performance_ratio:.2f}")
print("=" * 50)
