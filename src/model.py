#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль построения оптимизационных ЛП и MILP моделей для задачи о назначениях.
Вариант 6: "Бригады на объекты, минимум суммарного времени" (6x6).

Сетевая структура:
- Вершины истока (Бригады B1..B6, supply = +1)
- Вершины стока (Объекты O1..O6, demand = +1)
- Дуги (i, j) с пропускной способностью u_ij = 1.0 и временем c_ij (ч).
"""

import json
import numpy as np
from typing import Dict, Any, Tuple, List, Optional


class AssignmentLPModel:
    """Класс построения сетевой ЛП-модели задачи о назначениях."""

    def __init__(self, data_path: str = "data/input_data.json"):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.metadata = self.data["metadata"]
        self.brigades = self.data["brigades"]
        self.objects = self.data["objects"]
        self.extra_objects = self.data.get("extra_objects", [])

        self.n_brigades = len(self.brigades)
        self.n_objects = len(self.objects)

        self.C_base = np.array(self.data["time_matrix_base"], dtype=float)
        self.C_extra = np.array(self.data.get("time_matrix_extra", []), dtype=float)

    def get_var_index(self, i: int, j: int, n_cols: Optional[int] = None) -> int:
        """Индекс одномерной переменной x_{ij} в векторе решения."""
        cols = n_cols if n_cols is not None else self.n_objects
        return i * cols + j

    def build_base_lp(self) -> Dict[str, Any]:
        """
        Построение базовой непрерывной ЛП-модели задачи о назначениях.
        Переменные: x_{ij} in [0, 1] — поток / доля назначения бригады i на объект j.
        Целевая функция: min sum_{i,j} c_{ij} * x_{ij}
        Баланс бригад: sum_j x_{ij} = 1  (i = 1..n)
        Баланс объектов: sum_i x_{ij} = 1  (j = 1..m)
        """
        n, m = self.n_brigades, self.n_objects
        n_vars = n * m

        # Вектор коэффициентов целевой функции (время выполнения)
        c = self.C_base.flatten()

        # Матрица ограничений равенства A_eq * x = b_eq
        # n ограничений по бригадам + m ограничений по объектам
        A_eq = np.zeros((n + m, n_vars), dtype=float)
        b_eq = np.ones(n + m, dtype=float)

        constraint_names = []
        # 1. Баланс предложения бригад: sum_j x_{ij} = 1
        for i in range(n):
            for j in range(m):
                A_eq[i, self.get_var_index(i, j, m)] = 1.0
            constraint_names.append(f"supply_{self.brigades[i]['code']}")

        # 2. Баланс спроса объектов: sum_i x_{ij} = 1
        for j in range(m):
            for i in range(n):
                A_eq[n + j, self.get_var_index(i, j, m)] = 1.0
            constraint_names.append(f"demand_{self.objects[j]['code']}")

        # Границы переменных: 0 <= x_ij <= 1
        bounds = [(0.0, 1.0) for _ in range(n_vars)]

        var_names = [f"x_{self.brigades[i]['code']}_{self.objects[j]['code']}"
                     for i in range(n) for j in range(m)]

        return {
            "scenario": "base",
            "c": c,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "bounds": bounds,
            "n_brigades": n,
            "n_objects": m,
            "n_vars": n_vars,
            "var_names": var_names,
            "constraint_names": constraint_names,
            "C_matrix": self.C_base.copy()
        }

    def build_arc_failure_lp(self, failed_i: int, failed_j: int) -> Dict[str, Any]:
        """
        Сценарий «отказ дуги»: закрытие маршрута (failed_i, failed_j).
        Устанавливает верхнюю границу пропускной способности дуги в 0.0 (x_{ij} = 0).
        """
        lp = self.build_base_lp()
        lp["scenario"] = f"arc_failure_{self.brigades[failed_i]['code']}_{self.objects[failed_j]['code']}"

        idx = self.get_var_index(failed_i, failed_j, self.n_objects)
        bounds = list(lp["bounds"])
        bounds[idx] = (0.0, 0.0)  # дуга полностью закрыта
        lp["bounds"] = bounds
        lp["failed_arc"] = (failed_i, failed_j)
        lp["failed_brigade"] = self.brigades[failed_i]["code"]
        lp["failed_object"] = self.objects[failed_j]["code"]

        return lp

    def build_demand_growth_lp(self, num_extra: int = 1) -> Dict[str, Any]:
        """
        Сценарий «рост спроса на 20%»: добавление объектов без увеличения штата бригад.
        При num_extra=1: 7 объектов и 6 бригад (рост спроса ~17-20%).
        Показывает математическую несовместность (Infeasible) баланса.
        """
        n = self.n_brigades
        m = self.n_objects + num_extra
        n_vars = n * m

        # Объединяем матрицу стоимостей
        C_combined = np.hstack([self.C_base, self.C_extra[:, :num_extra]])
        c = C_combined.flatten()

        all_objects = self.objects + self.extra_objects[:num_extra]

        A_eq = np.zeros((n + m, n_vars), dtype=float)
        b_eq = np.ones(n + m, dtype=float)

        constraint_names = []
        for i in range(n):
            for j in range(m):
                A_eq[i, self.get_var_index(i, j, m)] = 1.0
            constraint_names.append(f"supply_{self.brigades[i]['code']}")

        for j in range(m):
            for i in range(n):
                A_eq[n + j, self.get_var_index(i, j, m)] = 1.0
            constraint_names.append(f"demand_{all_objects[j]['code']}")

        bounds = [(0.0, 1.0) for _ in range(n_vars)]
        var_names = [f"x_{self.brigades[i]['code']}_{all_objects[j]['code']}"
                     for i in range(n) for j in range(m)]

        return {
            "scenario": f"demand_growth_{num_extra}",
            "c": c,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "bounds": bounds,
            "n_brigades": n,
            "n_objects": m,
            "n_vars": n_vars,
            "var_names": var_names,
            "constraint_names": constraint_names,
            "all_objects": all_objects,
            "C_matrix": C_combined
        }

    def build_elastic_recovery_lp(self, num_extra: int = 1, penalty_cost: float = 200.0) -> Dict[str, Any]:
        """
        Восстановление совместности при росте спроса через эластичный фильтр /
        привлечение внешнего субподрядчика (штраф penalty_cost за каждый дефицитный объект).
        
        Ограничения:
          sum_j x_{ij} <= 1  (бригады могут выполнить не более 1 объекта)
          sum_i x_{ij} + s_j = 1  (каждый объект либо выполняется бригадой, либо отдается на аутсорсинг со штрафом s_j)
        Цель:
          min sum c_{ij} x_{ij} + sum penalty * s_j
        """
        n = self.n_brigades
        m = self.n_objects + num_extra
        all_objects = self.objects + self.extra_objects[:num_extra]
        C_combined = np.hstack([self.C_base, self.C_extra[:, :num_extra]])

        # Переменные: n * m назначений x_{ij} + m переменных дефицита s_j
        n_vars = n * m + m

        c = np.zeros(n_vars, dtype=float)
        c[:n * m] = C_combined.flatten()
        c[n * m:] = penalty_cost  # штраф за аутсорсинг

        # Ограничения: n неравенств для бригад (<= 1) и m равенств для объектов (= 1)
        # scipy linprog поддерживает A_ub и A_eq
        A_ub = np.zeros((n, n_vars), dtype=float)
        b_ub = np.ones(n, dtype=float)
        for i in range(n):
            for j in range(m):
                A_ub[i, self.get_var_index(i, j, m)] = 1.0

        A_eq = np.zeros((m, n_vars), dtype=float)
        b_eq = np.ones(m, dtype=float)
        for j in range(m):
            for i in range(n):
                A_eq[j, self.get_var_index(i, j, m)] = 1.0
            A_eq[j, n * m + j] = 1.0  # s_j

        bounds = [(0.0, 1.0) for _ in range(n * m)] + [(0.0, 1.0) for _ in range(m)]

        var_names = [f"x_{self.brigades[i]['code']}_{all_objects[j]['code']}"
                     for i in range(n) for j in range(m)] + [f"deficit_{all_objects[j]['code']}" for j in range(m)]

        constraint_names = [f"supply_limit_{self.brigades[i]['code']}" for i in range(n)] + \
                           [f"demand_cover_{all_objects[j]['code']}" for j in range(m)]

        return {
            "scenario": f"elastic_recovery_{num_extra}",
            "c": c,
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "bounds": bounds,
            "n_brigades": n,
            "n_objects": m,
            "n_vars": n_vars,
            "var_names": var_names,
            "constraint_names": constraint_names,
            "all_objects": all_objects,
            "penalty_cost": penalty_cost,
            "C_matrix": C_combined
        }

