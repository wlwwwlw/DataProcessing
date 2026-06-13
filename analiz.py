# ============================================================
# Разведочный анализ данных для ВКР
# Тема: Анализ мотивации проживания на сельских территориях
#
# Скрипт строит 3 рисунка:
# 1) распределение территорий по численности населения;
# 2) распределение дорожных расстояний между территориями;
# 3) распределение территорий по количеству инфраструктурных объектов.
#
# Используемые входные файлы:
# - population_NN.xlsx
# - population_kirov.xlsx
# - distances_matrix_NN.xlsx
# - distances_matrix_kirov.xlsx
# - results_NN.xlsx
# - results_kirov.xlsx
# ============================================================

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 1. СЛУЖЕБНЫЕ ФУНКЦИИ
# ------------------------------------------------------------

def find_existing_file(*names: str) -> Path:
    """Ищет файл по нескольким возможным именам рядом со скриптом."""
    for name in names:
        path = BASE_DIR / name
        if path.exists():
            return path

    # На случай, если файл скачан с добавкой типа population_NN(5).xlsx
    for name in names:
        stem = Path(name).stem.lower()
        suffix = Path(name).suffix.lower()
        candidates = sorted(BASE_DIR.glob(f"{stem}*{suffix}"))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(f"Не найден файл. Искались варианты: {names}")


def normalize_text(value) -> str:
    """Нормализация текста для поиска колонок и сопоставления названий."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip().lower()
    text = text.replace("ё", "е")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[«»\"']", "", text)
    return text.strip()


def to_number(series: pd.Series) -> pd.Series:
    """Преобразует значения в числа, корректно обрабатывая пробелы и запятые."""
    s = series.astype(str)
    s = s.str.replace("\u00a0", " ", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def find_column(df: pd.DataFrame, variants: list[str], required: bool = True) -> str | None:
    """Ищет колонку по возможным вариантам названия."""
    norm_cols = {col: normalize_text(col) for col in df.columns}

    for variant in variants:
        variant_norm = normalize_text(variant)
        for col, col_norm in norm_cols.items():
            if variant_norm == col_norm or variant_norm in col_norm:
                return col

    if required:
        raise ValueError(f"Не найдена колонка из вариантов {variants}. Колонки файла: {list(df.columns)}")

    return None


def kde_count_curve(values, points: int = 350, bandwidth: float | None = None, bins: int | None = None):
    """
    Простая KDE-кривая без scipy/seaborn.
    Возвращает x и y, где y масштабирован примерно как количество объектов,
    а не как плотность вероятности.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.array([]), np.array([])

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))

    if abs(vmax - vmin) < 1e-12:
        return np.array([vmin]), np.array([len(arr)])

    pad = (vmax - vmin) * 0.08
    x = np.linspace(vmin - pad, vmax + pad, points)

    n = len(arr)
    std = float(np.std(arr, ddof=1)) if n > 1 else 1.0

    if bandwidth is None:
        # Правило Сильвермана, но с защитой от слишком малого значения.
        bandwidth = 1.06 * std * (n ** (-1 / 5)) if std > 0 else (vmax - vmin) / 10
        bandwidth = max(bandwidth, (vmax - vmin) / 25)

    z = (x[:, None] - arr[None, :]) / bandwidth
    density = np.exp(-0.5 * z ** 2).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))

    if bins is None:
        bins = max(8, min(18, int(np.sqrt(n)) + 5))

    bin_width = (vmax - vmin) / bins
    y = density * n * bin_width

    return x, y


def plot_two_kde(
    values_nn,
    values_kirov,
    xlabel: str,
    ylabel: str,
    title: str,
    output_name: str,
    bins: int | None = None,
) -> None:
    """Строит сравнительную кривую распределения для двух регионов."""
    plt.figure(figsize=(10.5, 5.7), dpi=160)

    x_nn, y_nn = kde_count_curve(values_nn, bins=bins)
    x_kirov, y_kirov = kde_count_curve(values_kirov, bins=bins)

    plt.plot(x_nn, y_nn, linewidth=2.8, label="Нижегородская область", color="#1f4e79")
    plt.plot(x_kirov, y_kirov, linewidth=2.8, linestyle="--", label="Кировская область", color="#b56589")

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13, pad=12)
    plt.grid(True, alpha=0.28)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_path = OUTPUT_DIR / output_name
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {out_path}")


# ------------------------------------------------------------
# 2. ЗАГРУЗКА ДАННЫХ
# ------------------------------------------------------------

def load_population(path: Path) -> pd.DataFrame:
    """
    Загружает таблицу населения.
    Файл может быть как с нормальной шапкой, так и без нее.
    """
    # Читаем без шапки, потому что в некоторых файлах первая строка может
    # ошибочно восприниматься как заголовок.
    raw = pd.read_excel(path, header=None)

    if raw.shape[1] < 2:
        raise ValueError(f"В файле {path.name} должно быть минимум 2 столбца: территория и население.")

    df = raw.iloc[:, :2].copy()
    df.columns = ["territory", "population"]

    df["territory"] = df["territory"].astype(str).str.strip()
    df["population"] = to_number(df["population"])

    df = df.dropna(subset=["population"])
    df = df[df["population"] > 0]
    df = df.drop_duplicates(subset=["territory"], keep="first")
    df = df.reset_index(drop=True)

    return df


def load_distances(path: Path) -> np.ndarray:
    """
    Загружает матрицу расстояний и возвращает все попарные расстояния
    из верхнего треугольника без диагонали.
    """
    raw = pd.read_excel(path)

    if raw.shape[1] < 3:
        raise ValueError(f"Файл {path.name} не похож на матрицу расстояний.")

    # Первый столбец — названия территорий, остальные — расстояния.
    matrix = raw.iloc[:, 1:].copy()

    for col in matrix.columns:
        matrix[col] = to_number(matrix[col])

    arr = matrix.to_numpy(dtype=float)

    # Обрезаем до квадратной части, если вдруг есть лишний столбец/строка.
    n = min(arr.shape[0], arr.shape[1])
    arr = arr[:n, :n]

    # Берем только верхний треугольник без диагонали, чтобы пары не дублировались.
    values = arr[np.triu_indices(n, k=1)]
    values = values[np.isfinite(values)]
    values = values[values > 0]

    return values


def load_object_counts(path: Path) -> pd.Series:
    """Считает количество инфраструктурных объектов по каждой территории."""
    df = pd.read_excel(path)

    territory_col = find_column(
        df,
        [
            "округ",
            "территория",
            "муниципальное образование",
            "населенный пункт",
            "населённый пункт",
            "город",
            "town",
            "city",
        ],
    )

    counts = df.groupby(territory_col).size().sort_values()
    return counts


# ------------------------------------------------------------
# 3. ОСНОВНОЙ АНАЛИЗ
# ------------------------------------------------------------

def main() -> None:
    print("=" * 90)
    print("Разведочный анализ данных по Нижегородской и Кировской областям")
    print("=" * 90)

    population_nn_path = find_existing_file("population_NN.xlsx", "population_NN(5).xlsx")
    population_kirov_path = find_existing_file("population_kirov.xlsx", "population_kirov(3).xlsx")

    distances_nn_path = find_existing_file("distances_matrix_NN.xlsx", "distances_matrix_NN(5).xlsx")
    distances_kirov_path = find_existing_file("distances_matrix_kirov.xlsx", "distances_matrix_kirov(3).xlsx")

    results_nn_path = find_existing_file("results_NN.xlsx", "results_NN(5).xlsx")
    results_kirov_path = find_existing_file("results_kirov.xlsx", "results_kirov(3).xlsx")

    population_nn = load_population(population_nn_path)
    population_kirov = load_population(population_kirov_path)

    distances_nn = load_distances(distances_nn_path)
    distances_kirov = load_distances(distances_kirov_path)

    object_counts_nn = load_object_counts(results_nn_path)
    object_counts_kirov = load_object_counts(results_kirov_path)

    print("\nИспользованные файлы:")
    print(f"- {population_nn_path.name}")
    print(f"- {population_kirov_path.name}")
    print(f"- {distances_nn_path.name}")
    print(f"- {distances_kirov_path.name}")
    print(f"- {results_nn_path.name}")
    print(f"- {results_kirov_path.name}")

    print("\nКраткая сводка:")
    print(f"Нижегородская область: территорий в population — {len(population_nn)}")
    print(f"Кировская область: территорий в population — {len(population_kirov)}")
    print(f"Нижегородская область: пар расстояний — {len(distances_nn)}")
    print(f"Кировская область: пар расстояний — {len(distances_kirov)}")
    print(f"Нижегородская область: территорий в results — {len(object_counts_nn)}")
    print(f"Кировская область: территорий в results — {len(object_counts_kirov)}")

    # Рисунок 1
    plot_two_kde(
        population_nn["population"].values,
        population_kirov["population"].values,
        xlabel="Численность населения, человек",
        ylabel="Количество населенных пунктов",
        title="Распределение населенных пунктов по численности населения",
        output_name="figure_1_population_distribution.png",
        bins=12,
    )

    # Рисунок 2
    plot_two_kde(
        distances_nn,
        distances_kirov,
        xlabel="Расстояние, км",
        ylabel="Количество пар населенных пунктов",
        title="Распределение дорожных расстояний между населенными пунктами",
        output_name="figure_2_distance_distribution.png",
        bins=16,
    )

    # Рисунок 3
    plot_two_kde(
        object_counts_nn.values,
        object_counts_kirov.values,
        xlabel="Количество инфраструктурных объектов",
        ylabel="Количество населенных пунктов",
        title="Распределение населенных пунктов по количеству инфраструктурных объектов",
        output_name="figure_3_infrastructure_distribution.png",
        bins=12,
    )

    # Дополнительно сохраняем сводную таблицу, чтобы можно было проверить числа.
    summary = pd.DataFrame(
        [
            {
                "region": "Нижегородская область",
                "population_territories": len(population_nn),
                "population_min": population_nn["population"].min(),
                "population_max": population_nn["population"].max(),
                "population_mean": population_nn["population"].mean(),
                "distance_pairs": len(distances_nn),
                "distance_min": np.min(distances_nn),
                "distance_max": np.max(distances_nn),
                "distance_mean": np.mean(distances_nn),
                "object_territories": len(object_counts_nn),
                "objects_min": object_counts_nn.min(),
                "objects_max": object_counts_nn.max(),
                "objects_mean": object_counts_nn.mean(),
            },
            {
                "region": "Кировская область",
                "population_territories": len(population_kirov),
                "population_min": population_kirov["population"].min(),
                "population_max": population_kirov["population"].max(),
                "population_mean": population_kirov["population"].mean(),
                "distance_pairs": len(distances_kirov),
                "distance_min": np.min(distances_kirov),
                "distance_max": np.max(distances_kirov),
                "distance_mean": np.mean(distances_kirov),
                "object_territories": len(object_counts_kirov),
                "objects_min": object_counts_kirov.min(),
                "objects_max": object_counts_kirov.max(),
                "objects_mean": object_counts_kirov.mean(),
            },
        ]
    )

    summary_path = OUTPUT_DIR / "eda_summary.xlsx"
    summary.to_excel(summary_path, index=False)
    print(f"Сохранено: {summary_path}")

    print("\nГотово. Рисунки сохранены в папку plots.")


if __name__ == "__main__":
    main()
