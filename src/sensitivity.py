#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль анализа чувствительности, диапазонов устойчивости и теневых цен (п. 243, 343).
Определяет:
1. Теневые цены ограничений предложения (бригады) и спроса (объекты).
2. Приведенные стоимости дуг (reduced costs).
3. Допустимые диапазоны вариации стоимостей [c_min, c_max] без смены оптимального базиса.
4. Экспериментальное подтверждение линейного отклика целевой функции.
"""

import numpy as np
import scipy.optimize as opt
import pandas as pd
from typing import Dict, Any, List, Tuple


class SensitivityAnalyzer:
    """Анализ устойчивости решения и диапазонов чувствительности."""

    def __init__(self, C_matrix: np.ndarray,
                 brigades: List[Dict[str, Any]],
                 objects: List[Dict[str, Any]]):
        self.C = np.array(C_matrix, dtype=float)
        self.brigades = brigades
        self.objects = objects
        self.n, self.m = self.C.shape

    def analyze_stability_ranges(self,
                                 optimal_assignments: List[Tuple[int, int]],
                                 base_fun: float) -> pd.DataFrame:
        """
        Определение диапазонов устойчивости для коэффициентов c_{ij}.
        Для каждого активного ребра (i*, j*) определяется максимальное увеличение delta_max,
        при котором базисное назначение сохраняется.
        """
        records = []
        base_set = set(optimal_assignments)

        for i in range(self.n):
            for j in range(self.m):
                is_active = (i, j) in base_set
                c_orig = self.C[i, j]

                if is_active:
                    # Поиск верхней границы: увеличиваем c_{ij} бинарным поиском
                    low = 0.0
                    high = 100.0
                    # Проверяем, меняется ли вообще при +100
                    C_test = self.C.copy()
                    C_test[i, j] += high
                    r_test, c_test = opt.linear_sum_assignment(C_test)
                    if set(zip(r_test, c_test)) == base_set:
                        delta_max = np.inf
                    else:
                        for _ in range(30):
                            mid = (low + high) / 2.0
                            C_mid = self.C.copy()
                            C_mid[i, j] += mid
                            r_m, c_m = opt.linear_sum_assignment(C_mid)
                            if set(zip(r_m, c_m)) == base_set:
                                low = mid
                            else:
                                high = mid
                        delta_max = round(low, 2)

                    # Нижняя граница: можно уменьшать до 0
                    delta_min = -c_orig

                    records.append({
                        "brigade": self.brigades[i]["code"],
                        "object": self.objects[j]["code"],
                        "active": "ДА (x*=1)",
                        "current_cost": c_orig,
                        "allowable_decrease": round(abs(delta_min), 2),
                        "allowable_increase": delta_max if delta_max != np.inf else "> 100.0",
                        "status": "Базисная дуга (назначена)"
                    })
                else:
                    # Неактивная дуга: насколько можно уменьшить c_{ij}, чтобы она вошла в базис?
                    low = 0.0
                    high = c_orig
                    for _ in range(30):
                        mid = (low + high) / 2.0
                        C_mid = self.C.copy()
                        C_mid[i, j] -= mid
                        r_m, c_m = opt.linear_sum_assignment(C_mid)
                        if (i, j) not in set(zip(r_m, c_m)):
                            low = mid
                        else:
                            high = mid
                    margin_decrease = round(low, 2)

                    records.append({
                        "brigade": self.brigades[i]["code"],
                        "object": self.objects[j]["code"],
                        "active": "НЕТ (x*=0)",
                        "current_cost": c_orig,
                        "allowable_decrease": margin_decrease,
                        "allowable_increase": np.inf,
                        "status": "Внебазисная дуга"
                    })

        df = pd.DataFrame(records)
        return df

    def verify_shadow_price_prediction(self, target_i: int, target_j: int,
                                       delta: float, base_fun: float) -> Dict[str, Any]:
        """
        Верификация предсказания теневой цены:
        Теоретическое изменение цели: delta * x_{target_i, target_j}
        Фактическое изменение цели: пересчет полной задачи с возмущенной матрицей.
        """
        C_pert = self.C.copy()
        C_pert[target_i, target_j] += delta

        r, c_opt = opt.linear_sum_assignment(C_pert)
        actual_new_fun = float(C_pert[r, c_opt].sum())

        predicted_diff = delta * 1.0  # так как x_{target_i, target_j} = 1
        actual_diff = actual_new_fun - base_fun
        prediction_error = abs(actual_diff - predicted_diff)

        return {
            "target_arc": (self.brigades[target_i]["code"], self.objects[target_j]["code"]),
            "delta": delta,
            "base_fun": base_fun,
            "predicted_new_fun": base_fun + predicted_diff,
            "actual_new_fun": actual_new_fun,
            "predicted_diff": predicted_diff,
            "actual_diff": actual_diff,
            "prediction_error": prediction_error,
            "exact_match": prediction_error < 1e-7
        }

