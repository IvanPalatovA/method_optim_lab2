#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор исходных данных для Лабораторной работы № 2.
Вариант 6: "Задача о назначениях: бригады на объекты, минимум суммарного времени" (6 бригад, 6 объектов).

Индивидуальные параметры (по формуле из п. 1.5 регламента):
  Студент: Палатов Иван Александрович
  Учебная группа: Б24-517 (Q = 1)
  Порядковый номер по списку: N = 16
  Год выполнения: 2026 (Y = 26)
  Номер варианта: V = 1 + ((17*16 + 7*1 + 26) mod 20) = 1 + (305 mod 20) = 1 + 5 = 6
  Seed генерации данных: seed = 10000 + 100*Q + N = 10000 + 100*1 + 16 = 10116
"""

import json
import csv
import numpy as np


def generate_variant_data(seed: int = 10116,
                          output_json: str = "data/input_data.json",
                          output_csv: str = "data/time_matrix.csv"):
    rng = np.random.default_rng(seed)

    # 1. Бригады (источники предложения, supply = 1)
    brigades = [
        {
            "id": 1,
            "code": "B1",
            "name": "Бригада автоматики и КИПиА",
            "specialization": "Системы автоматического управления, прецизионные датчики, SCADA",
            "capacity": 1.0
        },
        {
            "id": 2,
            "code": "B2",
            "name": "Бригада высоковольтных электромонтажников",
            "specialization": "Трансформаторы 110 кВ, распределительные устройства, силовые кабели",
            "capacity": 1.0
        },
        {
            "id": 3,
            "code": "B3",
            "name": "Бригада механомонтажников и турбинистов",
            "specialization": "Турбоагрегаты, насосные комплексы, динамическая балансировка валов",
            "capacity": 1.0
        },
        {
            "id": 4,
            "code": "B4",
            "name": "Бригада дефектоскопии и неразрушающего контроля",
            "specialization": "Ультразвуковая и рентгеновская дефектоскопия, контроль сварных швов",
            "capacity": 1.0
        },
        {
            "id": 5,
            "code": "B5",
            "name": "Бригада трубопроводных и гидравлических систем",
            "specialization": "Контуры высокого давления, теплообменное оборудование, гидроиспытания",
            "capacity": 1.0
        },
        {
            "id": 6,
            "code": "B6",
            "name": "Бригада строительно-монтажных и такелажных работ",
            "specialization": "Фундаменты тяжелых агрегатов, виброизоляторы, металлоконструкции",
            "capacity": 1.0
        }
    ]

    # 2. Объекты базового сценария (потребители спроса, demand = 1)
    objects = [
        {
            "id": 1,
            "code": "O1",
            "name": "Подстанция ТП-110/10 кВ",
            "description": "Модернизация ячеек распределительного устройства и ошиновки",
            "demand": 1.0
        },
        {
            "id": 2,
            "code": "O2",
            "name": "Главный циркуляционный насос ЦН-1",
            "description": "Капитальный ремонт гидравлического тракта и замена рабочего колеса",
            "demand": 1.0
        },
        {
            "id": 3,
            "code": "O3",
            "name": "Система противоаварийной автоматики ПАА",
            "description": "Монтаж и пусконаладка микропроцессорных защит и датчиков",
            "demand": 1.0
        },
        {
            "id": 4,
            "code": "O4",
            "name": "Турбинный агрегат ТА-2",
            "description": "Центровка валопровода и ревизия опорных подшипников",
            "demand": 1.0
        },
        {
            "id": 5,
            "code": "O5",
            "name": "Теплообменный блок ТБ-4",
            "description": "Неразрушающий контроль сварных соединений и переобвязка трубок",
            "demand": 1.0
        },
        {
            "id": 6,
            "code": "O6",
            "name": "Резервная дизель-генераторная установка ДГУ",
            "description": "Установка виброопор, монтаж анкеров и сопряжение с рамой",
            "demand": 1.0
        }
    ]

    # Дополнительные объекты для сценария роста спроса (+20% и более)
    extra_objects = [
        {
            "id": 7,
            "code": "O7",
            "name": "Компрессорная станция КС-3",
            "description": "Внеплановое подключение воздуходувок технологического сжатия",
            "demand": 1.0
        },
        {
            "id": 8,
            "code": "O8",
            "name": "Насосная станция пожаротушения СПТ",
            "description": "Аварийная модернизация насосов высокого напора",
            "demand": 1.0
        }
    ]

    n_brigades = len(brigades)
    n_objects = len(objects)

    # 3. Генерация матрицы времен выполнения C (часы)
    # Диапазон от 15 до 65 часов в зависимости от специализации и сложности
    C_base = rng.integers(15, 65, size=(n_brigades, n_objects)).astype(float)
    C_extra = rng.integers(15, 65, size=(n_brigades, len(extra_objects))).astype(float)

    # Метаданные варианта
    metadata = {
        "student": "Палатов Иван Александрович",
        "group": "Б24-517",
        "group_index_Q": 1,
        "student_number_N": 16,
        "year_Y": 26,
        "variant": 6,
        "seed": seed,
        "topic": "Задача о назначениях: бригады на объекты, минимум суммарного времени",
        "n_brigades": n_brigades,
        "n_objects": n_objects,
        "time_unit": "часы",
        "arc_capacity_default": 1.0
    }

    full_data = {
        "metadata": metadata,
        "brigades": brigades,
        "objects": objects,
        "extra_objects": extra_objects,
        "time_matrix_base": C_base.tolist(),
        "time_matrix_extra": C_extra.tolist()
    }

    # Сохраняем JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)

    # Сохраняем матрицу времени в CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["Brigade_ID", "Brigade_Code"] + [obj["code"] for obj in objects]
        writer.writerow(header)
        for i, b in enumerate(brigades):
            row = [b["id"], b["code"]] + [f"{C_base[i, j]:.1f}" for j in range(n_objects)]
            writer.writerow(row)

    print(f"[DATA] Данные варианта 6 успешно сгенерированы (seed={seed}).")
    print(f"  Файлы сохранены в {output_json} и {output_csv}")
    print(f"  Размерность базовой матрицы: {n_brigades}x{n_objects} (36 дуг)")
    print(f"  Времена выполнения: min={C_base.min():.1f} ч, max={C_base.max():.1f} ч, mean={C_base.mean():.2f} ч")


if __name__ == "__main__":
    generate_variant_data()

