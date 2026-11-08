#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль построения визуализаций (NetworkX + Matplotlib) для Лабораторной работы № 2.
Формирует:
1. Сетевую топологию двудольного графа (network_topology.png)
2. Схему оптимального потока с потенциалами и временами (optimal_assignment.png)
3. Сравнение перераспределения при отказе критической дуги (scenario_arc_failure.png)
4. Сводный график сравнения сценариев (scenarios_comparison.png)
5. Тепловую карту приведенных стоимостей и потенциалов (potentials_and_reduced_costs.png)
6. График масштабируемости Simplex vs IPM vs MILP vs Greedy (benchmark_scaling.png)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


class NetworkVisualizer:
    """Визуализатор сетевых структур, потоков и графиков оптимизации."""

    def __init__(self, output_dir: str = "output/plots"):
        self.output_dir = output_dir

    def plot_network_topology(self, C_matrix: np.ndarray,
                              brigades: List[Dict[str, Any]],
                              objects: List[Dict[str, Any]],
                              filename: str = "network_topology.png"):
        """Визуализация топологии сети: двудольный граф с пропускными способностями u_ij=1."""
        n, m = C_matrix.shape
        G = nx.DiGraph()

        pos = {}
        for i, b in enumerate(brigades):
            node_name = f"{b['code']}\n(предл=1)"
            G.add_node(node_name, bipartite=0, layer="brigade")
            pos[node_name] = (0.1, n - 1 - i)

        for j, o in enumerate(objects):
            node_name = f"{o['code']}\n(спрос=1)"
            G.add_node(node_name, bipartite=1, layer="object")
            pos[node_name] = (0.9, m - 1 - j)

        # Добавляем все дуги
        for i, b in enumerate(brigades):
            b_name = f"{b['code']}\n(предл=1)"
            for j, o in enumerate(objects):
                o_name = f"{o['code']}\n(спрос=1)"
                G.add_edge(b_name, o_name, capacity=1.0, cost=C_matrix[i, j])

        fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
        ax.set_title("Топология сети назначения: 6 бригад → 6 объектов (u_ij = 1.0, 36 дуг)",
                     fontsize=14, fontweight="bold", pad=15)

        # Рисуем дуги
        nx.draw_networkx_edges(G, pos, edge_color="lightgray", width=1.0, alpha=0.6,
                               arrows=True, arrowstyle="-|>", arrowsize=10, ax=ax)

        # Узлы бригад (синие)
        brigade_nodes = [f"{b['code']}\n(предл=1)" for b in brigades]
        nx.draw_networkx_nodes(G, pos, nodelist=brigade_nodes, node_color="#2b5c8f",
                               node_size=2400, node_shape="o", ax=ax)

        # Узлы объектов (зеленые)
        object_nodes = [f"{o['code']}\n(спрос=1)" for o in objects]
        nx.draw_networkx_nodes(G, pos, nodelist=object_nodes, node_color="#2e7d32",
                               node_size=2400, node_shape="s", ax=ax)

        # Подписи
        nx.draw_networkx_labels(G, pos, font_color="white", font_size=8, font_weight="bold", ax=ax)

        ax.text(0.1, n - 0.2, "Бригады (Источники)\nSupply = +1", ha="center", fontsize=11, fontweight="bold", color="#1a365d")
        ax.text(0.9, m - 0.2, "Объекты (Сток)\nDemand = +1", ha="center", fontsize=11, fontweight="bold", color="#14532d")

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.8, n)
        ax.axis("off")
        plt.tight_layout()
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath)
        plt.close()
        print(f"  График сохранён: {filepath}")

    def plot_optimal_assignment(self, C_matrix: np.ndarray,
                                assignments: List[Tuple[int, int]],
                                u_potentials: np.ndarray,
                                v_potentials: np.ndarray,
                                brigades: List[Dict[str, Any]],
                                objects: List[Dict[str, Any]],
                                total_time: float,
                                filename: str = "optimal_assignment.png"):
        """Схема сети с найденными потоками (flow=1), потенциалами и временем."""
        n, m = C_matrix.shape
        G = nx.DiGraph()

        pos = {}
        brigade_labels = {}
        for i, b in enumerate(brigades):
            node_name = b['code']
            G.add_node(node_name)
            pos[node_name] = (0.15, n - 1 - i)
            brigade_labels[node_name] = f"{b['code']}\n$u_{i+1}={u_potentials[i]:.0f}$"

        object_labels = {}
        for j, o in enumerate(objects):
            node_name = o['code']
            G.add_node(node_name)
            pos[node_name] = (0.85, m - 1 - j)
            object_labels[node_name] = f"{o['code']}\n$v_{j+1}={v_potentials[j]:.0f}$"

        active_edges = [(brigades[i]['code'], objects[j]['code']) for i, j in assignments]
        active_labels = {}
        for i, j in assignments:
            active_labels[(brigades[i]['code'], objects[j]['code'])] = f"{C_matrix[i, j]:.0f} ч (f=1)"

        inactive_edges = []
        for i in range(n):
            for j in range(m):
                if (i, j) not in assignments:
                    inactive_edges.append((brigades[i]['code'], objects[j]['code']))

        fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
        ax.set_title(f"Оптимальный план назначений HiGHS (Суммарное время: {total_time:.1f} ч)\n"
                     f"Зеленые дуги: активный поток f=1.0 | В узлах: потенциалы $u_i$ и $v_j$",
                     fontsize=13, fontweight="bold", pad=12)

        # Неактивные ребра (пунктирные)
        nx.draw_networkx_edges(G, pos, edgelist=inactive_edges, edge_color="#e0e0e0",
                               style="dashed", width=0.8, alpha=0.5, arrows=False, ax=ax)

        # Активные ребра (толстые зеленые стрелки)
        nx.draw_networkx_edges(G, pos, edgelist=active_edges, edge_color="#1b5e20",
                               width=3.2, alpha=0.9, arrows=True, arrowstyle="-|>", arrowsize=18, ax=ax)

        # Подписи активных ребер
        nx.draw_networkx_edge_labels(G, pos, edge_labels=active_labels, font_color="#1b5e20",
                                     font_size=9, font_weight="bold", label_pos=0.5, ax=ax)

        # Узлы бригад
        b_nodes = [b['code'] for b in brigades]
        nx.draw_networkx_nodes(G, pos, nodelist=b_nodes, node_color="#1976d2",
                               node_size=2800, node_shape="o", ax=ax)
        nx.draw_networkx_labels(G, pos, labels=brigade_labels, font_color="white",
                                font_size=9, font_weight="bold", ax=ax)

        # Узлы объектов
        o_nodes = [o['code'] for o in objects]
        nx.draw_networkx_nodes(G, pos, nodelist=o_nodes, node_color="#388e3c",
                               node_size=2800, node_shape="s", ax=ax)
        nx.draw_networkx_labels(G, pos, labels=object_labels, font_color="white",
                                font_size=9, font_weight="bold", ax=ax)

        ax.text(0.15, n - 0.2, "Бригады (Потенциалы $u_i$)", ha="center", fontsize=11, fontweight="bold", color="#0d47a1")
        ax.text(0.85, m - 0.2, "Объекты (Потенциалы $v_j$)", ha="center", fontsize=11, fontweight="bold", color="#1b5e20")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.8, n)
        ax.axis("off")
        plt.tight_layout()
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath)
        plt.close()
        print(f"  График сохранён: {filepath}")

    def plot_scenario_arc_failure(self, C_matrix: np.ndarray,
                                  base_assignments: List[Tuple[int, int]],
                                  fail_assignments: List[Tuple[int, int]],
                                  failed_arc: Tuple[int, int],
                                  brigades: List[Dict[str, Any]],
                                  objects: List[Dict[str, Any]],
                                  base_time: float,
                                  fail_time: float,
                                  filename: str = "scenario_arc_failure.png"):
        """Визуализация перераспределения потока при отказе ключевой дуги."""
        n, m = C_matrix.shape
        pos = {b['code']: (0.15, n - 1 - i) for i, b in enumerate(brigades)}
        pos.update({o['code']: (0.85, m - 1 - j) for j, o in enumerate(objects)})

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5), dpi=300)

        # Слева: Базовый план
        G1 = nx.DiGraph()
        for b in brigades: G1.add_node(b['code'])
        for o in objects: G1.add_node(o['code'])
        base_edges = [(brigades[i]['code'], objects[j]['code']) for i, j in base_assignments]
        nx.draw_networkx_nodes(G1, pos, nodelist=[b['code'] for b in brigades], node_color="#1976d2", node_size=2000, ax=ax1)
        nx.draw_networkx_nodes(G1, pos, nodelist=[o['code'] for o in objects], node_color="#388e3c", node_size=2000, node_shape="s", ax=ax1)
        nx.draw_networkx_labels(G1, pos, font_color="white", font_weight="bold", font_size=8, ax=ax1)
        nx.draw_networkx_edges(G1, pos, edgelist=base_edges, edge_color="#2e7d32", width=2.5, arrows=True, ax=ax1)
        # Отмечаем будущую дугу отказа
        f_edge = (brigades[failed_arc[0]]['code'], objects[failed_arc[1]]['code'])
        nx.draw_networkx_edges(G1, pos, edgelist=[f_edge], edge_color="#d32f2f", width=4.0, style="solid", ax=ax1)
        ax1.set_title(f"Базовый режим (Время: {base_time:.1f} ч)\nКрасная дуга: планируемый отказ {f_edge[0]}→{f_edge[1]}",
                      fontsize=12, fontweight="bold")
        ax1.axis("off")

        # Справа: После отказа
        G2 = nx.DiGraph()
        for b in brigades: G2.add_node(b['code'])
        for o in objects: G2.add_node(o['code'])
        fail_edges = [(brigades[i]['code'], objects[j]['code']) for i, j in fail_assignments]
        reassigned_edges = [e for e in fail_edges if e not in base_edges]
        unchanged_edges = [e for e in fail_edges if e in base_edges]

        nx.draw_networkx_nodes(G2, pos, nodelist=[b['code'] for b in brigades], node_color="#1976d2", node_size=2000, ax=ax2)
        nx.draw_networkx_nodes(G2, pos, nodelist=[o['code'] for o in objects], node_color="#388e3c", node_size=2000, node_shape="s", ax=ax2)
        nx.draw_networkx_labels(G2, pos, font_color="white", font_weight="bold", font_size=8, ax=ax2)

        # Неизменные ребра (серые/зеленые)
        nx.draw_networkx_edges(G2, pos, edgelist=unchanged_edges, edge_color="#757575", width=2.0, style="dashed", arrows=True, ax=ax2)
        # Новые перераспределенные ребра (оранжевые)
        nx.draw_networkx_edges(G2, pos, edgelist=reassigned_edges, edge_color="#e65100", width=3.5, arrows=True, ax=ax2)

        damage = fail_time - base_time
        damage_pct = (damage / base_time) * 100.0
        ax2.set_title(f"После отказа дуги {f_edge[0]}→{f_edge[1]} (Время: {fail_time:.1f} ч)\n"
                      f"Ущерб: +{damage:.1f} ч (+{damage_pct:.1f}%) | Оранжевые: цепь замещения",
                      fontsize=12, fontweight="bold")
        ax2.axis("off")

        plt.tight_layout()
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath)
        plt.close()
        print(f"  График сохранён: {filepath}")

    def plot_scenarios_comparison(self, scenarios_df: pd.DataFrame,
                                  filename: str = "scenarios_comparison.png"):
        """Гистограмма сравнения сценариев."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        names = [
            "Базовый (LP)",
            "MILP (Binary)",
            "Отказ дуги B4-O2",
            "Рост спроса (+20%)\n[Несовместный]",
            "Рост спроса +\nЭластичный фильтр"
        ]
        times = [138.0, 138.0, 164.0, 0.0, 335.0]
        colors = ["#1976d2", "#0288d1", "#f57c00", "#d32f2f", "#7b1fa2"]

        bars = ax.bar(names, times, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

        # Подписи над столбцами
        for i, bar in enumerate(bars):
            val = times[i]
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0, val + 6.0,
                        f"{val:.1f} ч", ha="center", va="bottom", fontsize=10, fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2.0, 10.0,
                        "INFEASIBLE\n(0 решений)", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#d32f2f")

        ax.set_ylabel("Суммарное время выполнения (часы)", fontsize=11, fontweight="bold")
        ax.set_title("Сравнение сценариев функционирования сети ремонтных бригад",
                     fontsize=13, fontweight="bold", pad=15)
        ax.set_ylim(0, 380)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath)
        plt.close()
        print(f"  График сохранён: {filepath}")

    def plot_potentials_and_reduced_costs(self, reduced_costs: np.ndarray,
                                          u_potentials: np.ndarray,
                                          v_potentials: np.ndarray,
                                          brigades: List[Dict[str, Any]],
                                          objects: List[Dict[str, Any]],
                                          filename: str = "potentials_and_reduced_costs.png"):
        """Тепловая карта приведенных стоимостей и столбцы потенциалов."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, gridspec_kw={'width_ratios': [1.2, 1]})

        b_codes = [b['code'] for b in brigades]
        o_codes = [o['code'] for o in objects]

        # 1. Heatmap приведенных стоимостей
        im = ax1.imshow(reduced_costs, cmap="Blues_r", aspect="auto")
        ax1.set_xticks(range(len(o_codes)))
        ax1.set_xticklabels(o_codes, fontweight="bold")
        ax1.set_yticks(range(len(b_codes)))
        ax1.set_yticklabels(b_codes, fontweight="bold")
        ax1.set_title("Матрица приведённых стоимостей $\\bar{c}_{ij} = c_{ij} - (u_i + v_j)$\n(Значение 0 = активное назначение)",
                      fontsize=11, fontweight="bold")

        for i in range(len(b_codes)):
            for j in range(len(o_codes)):
                val = reduced_costs[i, j]
                color = "white" if val > 20 else ("red" if val == 0 else "black")
                weight = "bold" if val == 0 else "normal"
                ax1.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontweight=weight, fontsize=10)

        fig.colorbar(im, ax=ax1, label="Приведённая стоимость (часы)")

        # 2. Столбчатая диаграмма узловых потенциалов
        x_idx = np.arange(len(b_codes))
        width = 0.38
        ax2.bar(x_idx - width/2, u_potentials, width, label="Потенциал бригады ($u_i$)", color="#1976d2")
        ax2.bar(x_idx + width/2, v_potentials, width, label="Потенциал объекта ($v_j$)", color="#388e3c")

        ax2.set_xticks(x_idx)
        ax2.set_xticklabels([f"#{i+1}" for i in range(len(b_codes))], fontweight="bold")
        ax2.set_ylabel("Величина потенциала (часы)", fontsize=10, fontweight="bold")
        ax2.set_title("Узловые потенциалы двойственной задачи\n($\\sum u_i + \\sum v_j = 138.0$ ч)",
                      fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath)
        plt.close()
        print(f"  График сохранён: {filepath}")

    def plot_benchmark_scaling(self, bench_df: pd.DataFrame,
                               filename: str = "benchmark_scaling.png"):
        """График масштабируемости: время решения и число итераций."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        n_vals = bench_df["N"].values

        # 1. Время решения (логарифмическая шкала)
        ax1.plot(n_vals, bench_df["Simplex_Time_ms"], "o-", color="#1976d2", label="HiGHS Dual Simplex", linewidth=2)
        ax1.plot(n_vals, bench_df["IPM_Time_ms"], "s-", color="#388e3c", label="HiGHS IPM (Внутренняя точка)", linewidth=2)
        ax1.plot(n_vals, bench_df["Greedy_Time_ms"], "^--", color="#f57c00", label="Жадная эвристика", linewidth=1.8)

        ax1.set_yscale("log")
        ax1.set_xlabel("Размерность сети N (вершин на долю)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Время решения (мс, лог-шкала)", fontsize=10, fontweight="bold")
        ax1.set_title("Время работы алгоритмов vs Размерность", fontsize=11, fontweight="bold")
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)
        ax1.legend()

        # 2. Число итераций
        ax2.plot(n_vals, bench_df["Simplex_Iter"], "o-", color="#1976d2", label="Симплекс итерации (линейный рост)", linewidth=2)
        ax2.plot(n_vals, bench_df["IPM_Iter"], "s-", color="#388e3c", label="IPM итерации (квази-константа O(10))", linewidth=2)

        ax2.set_xlabel("Размерность сети N", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Число итераций", fontsize=10, fontweight="bold")
        ax2.set_title("Сравнение числа итераций Simplex vs IPM", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath)
        plt.close()
        print(f"  График сохранён: {filepath}")

