#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль независимой верификации решения на чистом NumPy (требование п. 1.2 и п. 240).
Проверяет:
1. Баланс потока во всех узлах сети (бригады и объекты).
2. Соблюдение пропускных способностей и границ на всех дугах (0 <= x_ij <= 1).
3. Строгую целочисленность решения (интегральность вершин политопа ТУМ).
4. Независимый пересчет значения целевой функции.
"""

import numpy as np
from typing import Dict, Any, List, Optional


class SolutionVerifier:
    """Класс строгой независимой проверки решения без обращения к объекту солвера."""

    @staticmethod
    def verify(
        x: np.ndarray,
        c: np.ndarray,
        A_eq: Optional[np.ndarray],
        b_eq: Optional[np.ndarray],
        bounds: List[tuple],
        reported_fun: float,
        A_ub: Optional[np.ndarray] = None,
        b_ub: Optional[np.ndarray] = None,
        constraint_names: Optional[List[str]] = None,
        tolerance: float = 1e-7
    ) -> Dict[str, Any]:
        x = np.asarray(x, dtype=float)
        c = np.asarray(c, dtype=float)

        # 1. Независимый пересчет целевой функции
        recalc_fun = float(np.dot(c, x))
        fun_discrepancy = abs(recalc_fun - reported_fun)
        fun_passed = fun_discrepancy <= tolerance

        # 2. Проверка ограничений равенства A_eq * x = b_eq (балансы узлов)
        max_eq_residual = 0.0
        eq_residuals = []
        if A_eq is not None and b_eq is not None:
            residuals = np.abs(np.dot(A_eq, x) - b_eq)
            max_eq_residual = float(np.max(residuals))
            eq_residuals = residuals.tolist()
        eq_passed = max_eq_residual <= tolerance

        # 3. Проверка ограничений неравенства A_ub * x <= b_ub (если есть)
        max_ub_violation = 0.0
        if A_ub is not None and b_ub is not None:
            violations = np.maximum(0.0, np.dot(A_ub, x) - b_ub)
            max_ub_violation = float(np.max(violations))
        ub_passed = max_ub_violation <= tolerance

        # 4. Проверка границ переменных lb <= x <= ub
        lb_violation = 0.0
        ub_bound_violation = 0.0
        for i, (lower, upper) in enumerate(bounds):
            if lower is not None and x[i] < lower - tolerance:
                lb_violation = max(lb_violation, lower - x[i])
            if upper is not None and x[i] > upper + tolerance:
                ub_bound_violation = max(ub_bound_violation, x[i] - upper)
        bounds_passed = (lb_violation <= tolerance) and (ub_bound_violation <= tolerance)

        # 5. Проверка целочисленности (Unimodularity / Integrality check)
        # Так как матрица инцидентности вполне унимодулярна, решение обязано быть {0, 1}
        dist_to_int = np.abs(x - np.round(x))
        max_dist_to_int = float(np.max(dist_to_int))
        integrality_passed = max_dist_to_int <= tolerance

        # 6. Суммарный статус проверки
        all_passed = fun_passed and eq_passed and ub_passed and bounds_passed and integrality_passed

        # Формирование отчета
        report_lines = [
            "=" * 76,
            "  РЕЗУЛЬТАТЫ НЕЗАВИСИМОЙ ПРОВЕРКИ РЕШЕНИЯ (NUMPY VERIFICATION)",
            "=" * 76,
            f"Машинный допуск (tolerance)      : {tolerance:.1e}",
            f"Статус верификации               : {'[PASS] УСПЕШНО' if all_passed else '[FAIL] ОШИБКА'}",
            "-" * 76,
            f"1. Целевая функция солвера       : {reported_fun:,.4f} ч",
            f"   Пересчитанная цель (c^T * x)   : {recalc_fun:,.4f} ч",
            f"   Абсолютная невязка цели        : {fun_discrepancy:.2e} -> {'OK' if fun_passed else 'VIOLATION'}",
            "-" * 76,
            f"2. Баланс узлов (A_eq * x = b_eq): макс. невязка = {max_eq_residual:.2e} -> {'OK' if eq_passed else 'VIOLATION'}",
            f"3. Ограничения неравенства       : макс. нарушение = {max_ub_violation:.2e} -> {'OK' if ub_passed else 'VIOLATION'}",
            f"4. Пропускные способности дуг    : нижняя невязка = {lb_violation:.2e}, верхняя = {ub_bound_violation:.2e} -> {'OK' if bounds_passed else 'VIOLATION'}",
            f"5. Целочисленность (x in {0,1})  : макс. отклонение от целого = {max_dist_to_int:.2e} -> {'OK' if integrality_passed else 'VIOLATION'}",
            "=" * 76
        ]

        if constraint_names and len(eq_residuals) == len(constraint_names):
            report_lines.append("\nДетализация баланса по вершинам сети:")
            for name, res in zip(constraint_names, eq_residuals):
                report_lines.append(f"  - Вершина {name:20s}: невязка = {res:.2e}")

        report_text = "\n".join(report_lines)

        return {
            "passed": all_passed,
            "recalculated_fun": recalc_fun,
            "fun_discrepancy": fun_discrepancy,
            "max_eq_residual": max_eq_residual,
            "max_ub_violation": max_ub_violation,
            "max_bound_violation": max(lb_violation, ub_bound_violation),
            "max_dist_to_int": max_dist_to_int,
            "report_text": report_text
        }

