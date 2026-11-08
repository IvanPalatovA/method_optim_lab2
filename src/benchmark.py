#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль сравнительного бенчмарка алгоритмов оптимизации (Задание со звёздочкой, п. 341).
Сравнивает:
1. HiGHS Dual Simplex (highs-ds)
2. HiGHS Interior Point Method (highs-ipm)
3. HiGHS MILP (Integer branch-and-bound)
4. Greedy Assignment Heuristic

Масштабирование размерностей задачи: N in [6, 10, 15, 20, 30, 50, 75, 100].
"""

import time
import numpy as np
import scipy.optimize as opt
import pandas as pd
from typing import Dict, Any, List


class BenchmarkRunner:
    """Проведение нагрузочного бенчмарка и анализа масштабируемости."""

    def __init__(self, seed: int = 10116):
        self.seed = seed

    def run_benchmark(self, dimensions: List[int] = None) -> pd.DataFrame:
        if dimensions is None:
            dimensions = [6, 10, 15, 20, 30, 50, 75, 100]

        rng = np.random.default_rng(self.seed)
        records = []

        print(f"[BENCHMARK] Запуск тестирования на {len(dimensions)} размерностях...")

        for N in dimensions:
            n_vars = N * N
            C = rng.integers(15, 65, size=(N, N)).astype(float)
            c = C.flatten()

            # Формирование матриц ограничений
            A_eq = np.zeros((2 * N, n_vars), dtype=float)
            for i in range(N):
                for j in range(N):
                    A_eq[i, i * N + j] = 1.0
                    A_eq[N + j, i * N + j] = 1.0
            b_eq = np.ones(2 * N, dtype=float)
            bounds = [(0.0, 1.0) for _ in range(n_vars)]

            # 1. Dual Simplex (highs-ds)
            t0 = time.perf_counter()
            res_simplex = opt.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs-ds")
            t_simplex_ms = (time.perf_counter() - t0) * 1000.0
            nit_simplex = getattr(res_simplex, "nit", 0)
            fun_simplex = float(res_simplex.fun) if res_simplex.success else np.nan

            # 2. Interior Point Method (highs-ipm)
            t0 = time.perf_counter()
            res_ipm = opt.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs-ipm")
            t_ipm_ms = (time.perf_counter() - t0) * 1000.0
            nit_ipm = getattr(res_ipm, "nit", 0)
            fun_ipm = float(res_ipm.fun) if res_ipm.success else np.nan

            # 3. MILP (для N <= 50)
            if N <= 50:
                constraints = [opt.LinearConstraint(A_eq, b_eq, b_eq)]
                integrality = np.ones(n_vars)
                bnds = opt.Bounds(0.0, 1.0)
                t0 = time.perf_counter()
                res_milp = opt.milp(c=c, integrality=integrality, constraints=constraints, bounds=bnds)
                t_milp_ms = (time.perf_counter() - t0) * 1000.0
                fun_milp = float(res_milp.fun) if res_milp.success else np.nan
                mip_nodes = getattr(res_milp, "mip_node_count", 0)
            else:
                t_milp_ms = np.nan
                fun_milp = np.nan
                mip_nodes = np.nan

            # 4. Greedy Heuristic
            t0 = time.perf_counter()
            pairs = [(i, j, float(C[i, j])) for i in range(N) for j in range(N)]
            pairs.sort(key=lambda x: x[2])
            assigned_r, assigned_c = set(), set()
            greedy_fun = 0.0
            for r, col, val in pairs:
                if r not in assigned_r and col not in assigned_c:
                    assigned_r.add(r)
                    assigned_c.add(col)
                    greedy_fun += val
                if len(assigned_r) == N:
                    break
            t_greedy_ms = (time.perf_counter() - t0) * 1000.0

            greedy_gap_pct = ((greedy_fun - fun_simplex) / fun_simplex) * 100.0 if fun_simplex > 0 else 0.0

            records.append({
                "N": N,
                "Variables": n_vars,
                "Constraints": 2 * N,
                "Simplex_Time_ms": round(t_simplex_ms, 2),
                "Simplex_Iter": nit_simplex,
                "Simplex_Fun": round(fun_simplex, 1),
                "IPM_Time_ms": round(t_ipm_ms, 2),
                "IPM_Iter": nit_ipm,
                "IPM_Fun": round(fun_ipm, 1),
                "MILP_Time_ms": round(t_milp_ms, 2) if not np.isnan(t_milp_ms) else "N/A",
                "MILP_Nodes": mip_nodes if not np.isnan(mip_nodes) else "N/A",
                "Greedy_Time_ms": round(t_greedy_ms, 2),
                "Greedy_Fun": round(greedy_fun, 1),
                "Greedy_Gap_pct": round(greedy_gap_pct, 2)
            })

            print(f"  N={N:3d} ({n_vars:5d} vars): Simplex={t_simplex_ms:5.2f} ms ({nit_simplex:3d} it), "
                  f"IPM={t_ipm_ms:5.2f} ms ({nit_ipm:2d} it), Greedy Gap=+{greedy_gap_pct:4.2f}%")

        df = pd.DataFrame(records)
        return df

