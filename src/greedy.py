#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль жадной эвристики (Greedy Assignment Heuristic) и оценки зазора субоптимальности (п. 244).
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class GreedyAssignment:
    """Жадная эвристика решения задачи о назначениях."""

    @staticmethod
    def solve(C_matrix: np.ndarray,
              brigade_codes: List[str],
              object_codes: List[str]) -> Dict[str, Any]:
        """
        Жадный алгоритм:
        1. Сортирует все возможные пары (i, j) по возрастанию времени c_{ij}.
        2. Итеративно фиксирует назначение с наименьшим временем, если бригада i
           и объект j ещё не заняты.
        """
        n, m = C_matrix.shape
        pairs = []
        for i in range(n):
            for j in range(m):
                pairs.append((i, j, float(C_matrix[i, j])))

        pairs.sort(key=lambda item: item[2])

        assigned_brigades = set()
        assigned_objects = set()
        assignments = []
        steps_trace = []
        total_time = 0.0

        step_num = 1
        for i, j, time_val in pairs:
            if i not in assigned_brigades and j not in assigned_objects:
                assigned_brigades.add(i)
                assigned_objects.add(j)
                assignments.append((i, j, time_val))
                total_time += time_val
                steps_trace.append({
                    "step": step_num,
                    "brigade_idx": i,
                    "object_idx": j,
                    "brigade": brigade_codes[i],
                    "object": object_codes[j],
                    "time": time_val,
                    "cumulative_time": total_time
                })
                step_num += 1

            if len(assignments) == min(n, m):
                break

        return {
            "total_time": total_time,
            "assignments": assignments,
            "steps_trace": steps_trace,
            "n_assigned": len(assignments)
        }

    @staticmethod
    def compare_with_optimal(greedy_res: Dict[str, Any],
                              optimal_time: float) -> Dict[str, Any]:
        """Вычисление субоптимального зазора (gap) между жадной эвристикой и точным оптимумом."""
        greedy_time = greedy_res["total_time"]
        abs_diff = greedy_time - optimal_time
        rel_gap_pct = (abs_diff / optimal_time) * 100.0 if optimal_time > 0 else 0.0

        explanation = (
            f"Жадный алгоритм набрал {greedy_time:.1f} ч против точного оптимума {optimal_time:.1f} ч "
            f"(зазор субоптимальности: +{rel_gap_pct:.2f}%, перерасход: +{abs_diff:.1f} ч).\n"
            f"Причина субоптимальности: близорукий (миопический) выбор. Занимая локально самые дешёвые дуги "
            f"на первых шагах, жадная стратегия оставляет на последних шагах несовместимые бригады и объекты "
            f"с катастрофически высоким временем выполнения."
        )

        return {
            "greedy_time": greedy_time,
            "optimal_time": optimal_time,
            "absolute_diff": abs_diff,
            "gap_percent": rel_gap_pct,
            "explanation": explanation
        }

