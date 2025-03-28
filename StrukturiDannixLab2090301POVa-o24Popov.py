import numpy as np
import time
import sys
import torch
from numba import njit, prange, set_num_threads


def generate_matrix(n):
    print(f"  - Генерация {n}x{n}...", end=' ')
    mat = (np.random.rand(n, n).astype(np.float32) + 
           1j*np.random.rand(n, n).astype(np.float32)).astype(np.complex64)
    print(f"OK ({mat.nbytes//1024**2} МБ)")
    return np.ascontiguousarray(mat)

def classic_matmul(A, B, n):
    C = np.zeros((n, n), dtype=np.complex64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

def optimized_matmul(A, B):
    try:
        # Попытка использовать GPU через PyTorch
        if torch.cuda.is_available():
            with torch.no_grad():
                A_t = torch.tensor(A, device='cuda', dtype=torch.complex64)
                B_t = torch.tensor(B, device='cuda', dtype=torch.complex64)
                C_t = torch.matmul(A_t, B_t)
                return C_t.cpu().numpy()
        else:
            raise Exception("CUDA не доступен")
    except:
        try:
            return np.dot(A, B)
        except:
            # Аварийный вариант через Numba
            @njit(parallel=True, fastmath=True)
            def fallback_matmul(A, B):
                n = A.shape[0]
                C = np.empty((n, n), dtype=np.complex64)
                for i in prange(n):
                    C[i] = np.dot(A[i], B)
                return C
            return fallback_matmul(A, B)

def main():
    print("ЛАБОРАТОРНАЯ РАБОТА: ПЕРЕМНОЖЕНИЕ МАТРИЦ")
    print("========================================\n")

    # Вариант 1: Классический (400x400)
    print("1. Классический метод (400x400)")
    n1 = 400
    A1 = generate_matrix(n1)
    B1 = generate_matrix(n1)
    
    start = time.perf_counter()
    classic_matmul(A1, B1, n1)
    time1 = time.perf_counter() - start
    mflops1 = (2 * n1**3) / (time1 * 1e6)
    
    print(f"\n  Результаты:")
    print(f"  Время: {time1:.6f} с")
    print(f"  MFLOPS: {mflops1:.2f}")
    print(f"  Память: {A1.nbytes * 3 / 1e6:.2f} МБ")
    print("----------------------------------------\n")

    # Вариант 2: BLAS (4096x4096)
    print("2. BLAS (4096x4096)")
    n2 = 4096
    A2 = generate_matrix(n2)
    B2 = generate_matrix(n2)
    
    start = time.perf_counter()
    np.dot(A2, B2)
    time2 = time.perf_counter() - start
    mflops2 = (2 * n2**3) / (time2 * 1e6)
    
    print(f"\n  Результаты:")
    print(f"  Время: {time2:.6f} с")
    print(f"  MFLOPS: {mflops2:.2f}")
    print(f"  Память: {A2.nbytes * 3 / 1e6:.2f} МБ")
    print("----------------------------------------\n")

    # Вариант 3: Гибридное адаптивное перемножение матриц
    print("3. Свой метод (Гибридное адаптивное перемножение матриц)")
    n3 = 4096
    A3 = generate_matrix(n3)
    B3 = generate_matrix(n3)
    
    start = time.perf_counter()
    C3 = optimized_matmul(A3, B3)
    time3 = time.perf_counter() - start
    mflops3 = (2 * n3**3) / (time3 * 1e6)
    
    print(f"\n  Результаты:")
    print(f"  Время: {time3:.6f} с")
    print(f"  MFLOPS: {mflops3:.2f}")
    print(f"  Память: {A3.nbytes * 3 / 1e6:.2f} МБ")
    print(f"  Производительность: {mflops3/mflops2*100:.1f}% от BLAS")
    print("\n========================================")

    print("\nАВТОР:")
    print("Попов Олег Михайлович")
    print("090301-ПОВа-О24")

if __name__ == "__main__":
    main()
