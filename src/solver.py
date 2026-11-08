#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль интерфейса к промышленному солверу HiGHS (highspy / scipy.optimize.linprog/milp).
Обеспечивает решение задач LP (Dual Simplex, IPM) и MILP, измерение времени,
извлечение двойственных переменных (теневых цен) и формирование подробных логов.
"""

import time
import numpy as np
import scipy.optimize as opt
from typing import Dict, Any, Optional


class HighsSolver:
    """Обертка над промышленным решателем HiGHS."""

    def __init__(self, method: str = "highs-ds", presolve: bool = True):
        """
        :param method: 'highs', 'highs-ds' (Dual Simplex), или 'highs-ipm' (Interior Point Method)
        :param presolve: включение пресолвера HiGHS
        """
        self.method = method
        self.presolve = presolve

    def solve_lp(self, lp_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Решение задачи линейного программирования через HiGHS."""
        c = lp_dict["c"]
        A_eq = lp_dict.get("A_eq", None)
        b_eq = lp_dict.get("b_eq", None)
        A_ub = lp_dict.get("A_ub", None)
        b_ub = lp_dict.get("b_ub", None)
        bounds = lp_dict.get("bounds", None)

        options = {
            "presolve": self.presolve,
            "disp": False
        }

        start_time = time.perf_counter()
        res = opt.linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=self.method,
            options=options
        )
        solve_time_sec = time.perf_counter() - start_time

        # Извлечение двойственных переменных (потенциалов) и приведенных стоимостей
        eq_marginals = np.array([])
        ineq_marginals = np.array([])
        lower_marginals = np.array([])
        upper_marginals = np.array([])

        if hasattr(res, "eqlin") and res.eqlin is not None and hasattr(res.eqlin, "marginals"):
            eq_marginals = np.array(res.eqlin.marginals)
        if hasattr(res, "ineqlin") and res.ineqlin is not None and hasattr(res.ineqlin, "marginals"):
            ineq_marginals = np.array(res.ineqlin.marginals)
        if hasattr(res, "lower") and res.lower is not None and hasattr(res.lower, "marginals"):
            lower_marginals = np.array(res.lower.marginals)
        if hasattr(res, "upper") and res.upper is not None and hasattr(res.upper, "marginals"):
            upper_marginals = np.array(res.upper.marginals)

        # Формирование детального лога солвера
        log_lines = [
            "=" * 76,
            f"  HiGHS LP SOLVER LOG | Method: {self.method} | Presolve: {self.presolve}",
            "=" * 76,
            f"Status code     : {res.status}",
            f"Solver message  : {res.message}",
            f"Success flag    : {res.success}",
            f"Iterations (nit): {getattr(res, 'nit', 0)}",
            f"Solve time      : {solve_time_sec * 1000:.3f} ms ({solve_time_sec:.6f} s)",
            f"Objective (fun) : {res.fun if res.success else 'N/A'}"
        ]

        if res.success:
            log_lines.append("-" * 76)
            log_lines.append(f"Primal solution dimension: {len(res.x)} variables")
            log_lines.append(f"Non-zero variables count : {np.sum(res.x > 1e-5)}")
            if len(eq_marginals) > 0:
                log_lines.append(f"Equality dual marginals  : {len(eq_marginals)} values")
            if len(ineq_marginals) > 0:
                log_lines.append(f"Inequality dual marginals: {len(ineq_marginals)} values")
            if len(lower_marginals) > 0:
                log_lines.append(f"Lower bound reduced costs: min={lower_marginals.min():.4f}, max={lower_marginals.max():.4f}")
        log_lines.append("=" * 76)
        log_text = "\n".join(log_lines)

        return {
            "status": res.status,
            "success": res.success,
            "message": res.message,
            "fun": float(res.fun) if res.success else None,
            "x": res.x if res.success else None,
            "nit": getattr(res, "nit", 0),
            "solve_time_sec": solve_time_sec,
            "eq_marginals": eq_marginals,
            "ineq_marginals": ineq_marginals,
            "lower_marginals": lower_marginals,
            "upper_marginals": upper_marginals,
            "log": log_text,
            "raw_res": res
        }

    def solve_milp(self, lp_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Решение задачи в смешанно-целочисленной постановке (MILP) через HiGHS."""
        c = lp_dict["c"]
        A_eq = lp_dict.get("A_eq", None)
        b_eq = lp_dict.get("b_eq", None)
        bounds_list = lp_dict.get("bounds", None)

        n_vars = len(c)
        integrality = np.ones(n_vars)  # Все переменные бинарные {0, 1}

        # Формируем линейные ограничения для scipy.optimize.milp
        constraints = []
        if A_eq is not None and b_eq is not None:
            constraints.append(opt.LinearConstraint(A_eq, b_eq, b_eq))

        A_ub = lp_dict.get("A_ub", None)
        b_ub = lp_dict.get("b_ub", None)
        if A_ub is not None and b_ub is not None:
            constraints.append(opt.LinearConstraint(A_ub, -np.inf, b_ub))

        lb = [b[0] for b in bounds_list] if bounds_list else np.zeros(n_vars)
        ub = [b[1] for b in bounds_list] if bounds_list else np.ones(n_vars)
        bounds = opt.Bounds(lb, ub)

        start_time = time.perf_counter()
        res = opt.milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)
        solve_time_sec = time.perf_counter() - start_time

        log_lines = [
            "=" * 76,
            "  HiGHS MILP SOLVER LOG | Binary Integrality Constraints",
            "=" * 76,
            f"Status code     : {res.status}",
            f"Solver message  : {res.message}",
            f"Success flag    : {res.success}",
            f"Solve time      : {solve_time_sec * 1000:.3f} ms ({solve_time_sec:.6f} s)",
            f"Objective (fun) : {res.fun if res.success else 'N/A'}",
            f"MIP Node count  : {getattr(res, 'mip_node_count', 0)}",
            f"MIP Dual bound  : {getattr(res, 'mip_dual_bound', 0.0)}",
            f"MIP Gap         : {getattr(res, 'mip_gap', 0.0)}"
        ]
        log_lines.append("=" * 76)

        return {
            "status": res.status,
            "success": res.success,
            "message": res.message,
            "fun": float(res.fun) if res.success else None,
            "x": res.x if res.success else None,
            "solve_time_sec": solve_time_sec,
            "mip_node_count": getattr(res, "mip_node_count", 0),
            "mip_gap": getattr(res, "mip_gap", 0.0),
            "log": "\n".join(log_lines),
            "raw_res": res
        }

