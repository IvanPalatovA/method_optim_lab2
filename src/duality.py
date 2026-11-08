#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль двойственной теории, узловых потенциалов и условий ККХ (п. 243, 349).
Формулирует и решает явную двойственную задачу:
  max sum(u_i) + sum(v_j)
  при u_i + v_j <= c_{ij}  для всех (i, j)
Проверяет теорему сильной двойственности и условия дополняющей нежёсткости (KKT).
"""

import numpy as np
import scipy.optimize as opt
from typing import Dict, Any, List


class DualityAnalyzer:
    """Анализ двойственности, потенциалов и приведенных стоимостей в задаче о назначениях."""

    def __init__(self, C_matrix: np.ndarray,
                 brigade_codes: List[str],
                 object_codes: List[str]):
        self.C = np.array(C_matrix, dtype=float)
        self.brigade_codes = brigade_codes
        self.object_codes = object_codes
        self.n, self.m = self.C.shape

    def solve_explicit_dual(self) -> Dict[str, Any]:
        """
        Явная формулировка двойственной задачи линейного программирования:
        Переменные: u_1..u_n (потенциалы бригад), v_1..v_m (потенциалы объектов).
        Цель: max sum(u_i) + sum(v_j)  <=>  min -sum(u_i) - sum(v_j)
        Ограничения: u_i + v_j <= c_{ij}  (m*n неравенств)
        """
        n, m = self.n, self.m
        n_vars = n + m

        # Целевая функция на минимум
        c_dual = -np.ones(n_vars, dtype=float)

        # Матрица A_ub * y <= b_ub
        A_ub = np.zeros((n * m, n_vars), dtype=float)
        b_ub = np.zeros(n * m, dtype=float)

        for i in range(n):
            for j in range(m):
                row = i * m + j
                A_ub[row, i] = 1.0       # u_i
                A_ub[row, n + j] = 1.0   # v_j
                b_ub[row] = self.C[i, j]

        # Переменные свободны по знаку
        bounds = [(None, None) for _ in range(n_vars)]

        res = opt.linprog(c_dual, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs-ds")

        u = res.x[:n]
        v = res.x[n:]
        dual_val = -float(res.fun)

        return {
            "status": res.status,
            "success": res.success,
            "message": res.message,
            "dual_objective": dual_val,
            "u_potentials": u,
            "v_potentials": v,
            "nit": getattr(res, "nit", 0),
            "res": res
        }

    def verify_complementary_slackness(self, x_primal: np.ndarray,
                                        u: np.ndarray,
                                        v: np.ndarray,
                                        primal_val: float,
                                        dual_val: float,
                                        tol: float = 1e-7) -> Dict[str, Any]:
        """
        Проверка сильной двойственности и условий дополняющей нежёсткости:
        1. Сильная двойственность: |Z_primal - W_dual| <= tol
        2. Дополняющая нежёсткость: x_{ij} * (c_{ij} - u_i - v_j) = 0 для всех (i, j)
        3. Допустимость двойственной: c_{ij} - u_i - v_j >= -tol
        """
        n, m = self.n, self.m
        x_mat = x_primal.reshape((n, m))

        # Вычисление матрицы приведенных стоимостей c_bar_{ij} = c_{ij} - u_i - v_j
        reduced_costs = np.zeros((n, m), dtype=float)
        kkt_violations = []

        max_kkt_violation = 0.0
        max_dual_infeasibility = 0.0

        for i in range(n):
            for j in range(m):
                c_bar = self.C[i, j] - (u[i] + v[j])
                reduced_costs[i, j] = c_bar
                if c_bar < -tol:
                    max_dual_infeasibility = max(max_dual_infeasibility, -c_bar)

                prod = x_mat[i, j] * c_bar
                if abs(prod) > tol:
                    max_kkt_violation = max(max_kkt_violation, abs(prod))

        duality_gap = abs(primal_val - dual_val)
        strong_duality_holds = duality_gap <= tol
        kkt_holds = max_kkt_violation <= tol and max_dual_infeasibility <= tol

        return {
            "primal_val": primal_val,
            "dual_val": dual_val,
            "duality_gap": duality_gap,
            "strong_duality_holds": strong_duality_holds,
            "kkt_holds": kkt_holds,
            "reduced_costs_matrix": reduced_costs,
            "max_kkt_violation": max_kkt_violation,
            "max_dual_infeasibility": max_dual_infeasibility
        }

