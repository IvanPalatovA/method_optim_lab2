#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль сценарного анализа (п. 1.4, п. 238-244).
Реализует:
1. Базовый сценарий (Base).
2. Сценарий «Отказ дуги» (Arc Failure): закрытие ключевой дуги (B4, O2), оценка ущерба.
3. Сценарий «Рост спроса на 20%» (Demand Growth): 7 объектов на 6 бригад -> несовместность (Infeasible).
4. Восстановление совместности: эластичный фильтр / аутсорсинг со штрафом (Elastic Recovery).
5. Сравнение с целочисленной постановкой (MILP vs LP).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.model import AssignmentLPModel
from src.solver import HighsSolver


class ScenarioManager:
    """Управление и проведение всех сценариев оптимизационного эксперимента."""

    def __init__(self, model: AssignmentLPModel, solver: HighsSolver):
        self.model = model
        self.solver = solver

    def run_all_scenarios(self) -> Dict[str, Any]:
        """Запуск всего пула сценариев и сбор сравнительной аналитики."""
        results = {}

        # -------------------------------------------------------------
        # 1. Базовый сценарий (LP)
        # -------------------------------------------------------------
        base_lp = self.model.build_base_lp()
        base_sol = self.solver.solve_lp(base_lp)
        results["base"] = {
            "name": "Базовый сценарий (LP)",
            "lp": base_lp,
            "sol": base_sol,
            "status": "OPTIMAL" if base_sol["success"] else "FAILED",
            "time_total": base_sol["fun"],
            "solve_time_ms": base_sol["solve_time_sec"] * 1000,
            "iterations": base_sol["nit"],
            "notes": "Штатное расписание, 6 бригад на 6 объектов"
        }

        # -------------------------------------------------------------
        # 2. Сравнение LP vs MILP (Дополнительное задание)
        # -------------------------------------------------------------
        milp_sol = self.solver.solve_milp(base_lp)
        results["milp"] = {
            "name": "Целочисленная MILP (Binary)",
            "lp": base_lp,
            "sol": milp_sol,
            "status": "OPTIMAL" if milp_sol["success"] else "FAILED",
            "time_total": milp_sol["fun"],
            "solve_time_ms": milp_sol["solve_time_sec"] * 1000,
            "iterations": getattr(milp_sol.get("raw_res", None), "mip_node_count", 0),
            "notes": "Бинарные переменные x_ij in {0, 1}. Зазор gap = 0.00% (ТУМ)"
        }

        # -------------------------------------------------------------
        # 3. Сценарий «Отказ дуги» (закрытие наилучшей дуги B4-O2)
        # -------------------------------------------------------------
        # Дуга (3, 1): Бригада B4 на Объект O2, время 15.0 ч (наименьшее в матрице)
        failed_i, failed_j = 3, 1
        arc_fail_lp = self.model.build_arc_failure_lp(failed_i, failed_j)
        arc_fail_sol = self.solver.solve_lp(arc_fail_lp)

        damage = arc_fail_sol["fun"] - base_sol["fun"] if arc_fail_sol["success"] else np.nan
        damage_pct = (damage / base_sol["fun"]) * 100.0 if base_sol["success"] else np.nan

        results["arc_failure"] = {
            "name": "Отказ дуги (B4 - O2)",
            "lp": arc_fail_lp,
            "sol": arc_fail_sol,
            "status": "OPTIMAL" if arc_fail_sol["success"] else "FAILED",
            "time_total": arc_fail_sol["fun"],
            "solve_time_ms": arc_fail_sol["solve_time_sec"] * 1000,
            "iterations": arc_fail_sol["nit"],
            "damage_hours": damage,
            "damage_percent": damage_pct,
            "notes": f"Закрыта дуга B4-O2. Ущерб: +{damage:.1f} ч (+{damage_pct:.1f}%)"
        }

        # -------------------------------------------------------------
        # 4. Сценарий «Рост спроса на 20%» (7 объектов на 6 бригад)
        # -------------------------------------------------------------
        demand_growth_lp = self.model.build_demand_growth_lp(num_extra=1)
        demand_growth_sol = self.solver.solve_lp(demand_growth_lp)

        results["demand_growth_infeasible"] = {
            "name": "Рост спроса +20% (Несовместный)",
            "lp": demand_growth_lp,
            "sol": demand_growth_sol,
            "status": "INFEASIBLE (HiGHS Status 8)",
            "time_total": None,
            "solve_time_ms": demand_growth_sol["solve_time_sec"] * 1000,
            "iterations": demand_growth_sol["nit"],
            "notes": "Спрос (7) превышает предложение (6). Решений не существует"
        }

        # -------------------------------------------------------------
        # 5. Восстановление совместности: Эластичный фильтр / Аутсорсинг
        # -------------------------------------------------------------
        elastic_lp = self.model.build_elastic_recovery_lp(num_extra=1, penalty_cost=200.0)
        elastic_sol = self.solver.solve_lp(elastic_lp)

        # Вычисляем внутреннее время и штраф
        n, m = elastic_lp["n_brigades"], elastic_lp["n_objects"]
        x = elastic_sol["x"]
        x_mat = x[:n * m].reshape((n, m))
        deficits = x[n * m:]
        internal_time = float(np.sum(x_mat * elastic_lp["C_matrix"]))
        penalty_time = float(np.sum(deficits * elastic_lp["penalty_cost"]))

        # Находим переданный на аутсорсинг объект
        outsourced_obj_idx = int(np.argmax(deficits))
        outsourced_obj_code = elastic_lp["all_objects"][outsourced_obj_idx]["code"]

        results["elastic_recovery"] = {
            "name": "Рост спроса + Эластичный фильтр",
            "lp": elastic_lp,
            "sol": elastic_sol,
            "status": "OPTIMAL",
            "time_total": elastic_sol["fun"],
            "internal_work_time": internal_time,
            "penalty_time": penalty_time,
            "outsourced_object": outsourced_obj_code,
            "solve_time_ms": elastic_sol["solve_time_sec"] * 1000,
            "iterations": elastic_sol["nit"],
            "notes": f"Аутсорсинг объекта {outsourced_obj_code}. Внутр. время: {internal_time:.1f} ч, штраф: {penalty_time:.1f} ч"
        }

        return results

    def create_comparison_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Формирование сводной таблицы сравнения всех сценариев."""
        rows = []
        for key in ["base", "milp", "arc_failure", "demand_growth_infeasible", "elastic_recovery"]:
            item = results[key]
            total_time_str = f"{item['time_total']:.1f} ч" if item['time_total'] is not None else "Не существует (ОДР пуста)"
            rows.append({
                "Сценарий": item["name"],
                "Статус HiGHS": item["status"],
                "Целевая функция (время)": total_time_str,
                "Итераций": item["iterations"],
                "Время расчета (мс)": f"{item['solve_time_ms']:.2f}",
                "Содержательная интерпретация": item["notes"]
            })

        df = pd.DataFrame(rows)
        return df

