"""
Segmentación univariada: para una lista de variables y un valor objetivo,
encuentra en qué variables y con qué puntos de corte (numéricas) o categorías
(categóricas) la tasa promedio de una variable binaria objetivo supera ese valor.
"""

import numpy as np
import pandas as pd


def _wilson_lb(k, n, z=1.96):
    """Límite inferior del intervalo de Wilson para una proporción k/n."""
    p = k / n
    d = 1 + z**2 / n
    centro = p + z**2 / (2 * n)
    margen = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (centro - margen) / d


def segmentos_univariados(
    df,
    variables,
    target,
    valor_objetivo,
    min_soporte=0.01,      # <1: fracción de la población; >=1: nº de filas
    n_cortes=20,           # cuantiles candidatos por variable numérica
    max_cardinalidad=15,   # numéricas con <= este nº de valores únicos se tratan como categóricas
    usar_wilson=True,      # compara el límite inferior de Wilson en vez de la tasa puntual
    z=1.96,
):
    """
    Analiza cada variable de forma independiente (univariado) y devuelve los
    segmentos (regla de una sola variable) donde la tasa de `target` supera
    `valor_objetivo`, respetando un soporte mínimo.

    Parámetros
    ----------
    df : pd.DataFrame
    variables : list[str]
        Columnas a evaluar.
    target : str
        Columna binaria (0/1) objetivo.
    valor_objetivo : float
        Umbral de tasa que debe superarse.
    min_soporte : float
        Tamaño mínimo del segmento. Si < 1 se interpreta como fracción de la
        población total; si >= 1, como número de filas.
    n_cortes : int
        Número de cuantiles candidatos a probar por variable numérica.
    max_cardinalidad : int
        Variables numéricas con menos valores únicos que esto se tratan como
        categóricas (una regla por categoría).
    usar_wilson : bool
        Si True, compara el límite inferior de Wilson (más conservador y
        robusto a segmentos pequeños). Si False, compara la tasa puntual.
    z : float
        Z-score para el intervalo de Wilson (1.96 ~ 95% de confianza).

    Retorna
    -------
    pd.DataFrame con columnas:
        variable, regla, n, pct_poblacion, tasa, wilson_lb, lift
    Ordenado por pct_poblacion descendente.
    """
    n_tot = len(df)
    min_n = min_soporte * n_tot if min_soporte < 1 else min_soporte
    base = df[target].mean()
    filas = []

    def evaluar(var, regla, k, n):
        if n < min_n:
            return
        tasa = k / n
        lb = _wilson_lb(k, n, z)
        if (lb if usar_wilson else tasa) > valor_objetivo:
            filas.append(dict(
                variable=var, regla=regla, n=int(n),
                pct_poblacion=n / n_tot, tasa=tasa,
                wilson_lb=lb, lift=tasa / base,
            ))

    for v in variables:
        d = df[[v, target]].dropna()
        x, y = d[v], d[target].to_numpy()
        es_num = pd.api.types.is_numeric_dtype(x) and x.nunique() > max_cardinalidad

        if not es_num:  # categórica: cada categoría por separado
            g = d.groupby(v, observed=True)[target].agg(["sum", "count"])
            for cat, r in g.iterrows():
                evaluar(v, f"{v} == {cat!r}", r["sum"], r["count"])
            continue

        xv = x.to_numpy()
        cortes = np.unique(np.quantile(xv, np.linspace(0.05, 0.95, n_cortes)))
        for sentido in (">", "<="):
            mejor = None
            for c in cortes:
                m = xv > c if sentido == ">" else xv <= c
                n, k = m.sum(), y[m].sum()
                if n < min_n:
                    continue
                tasa = k / n
                lb = _wilson_lb(k, n, z)
                if (lb if usar_wilson else tasa) > valor_objetivo:
                    if mejor is None or n > mejor[0]:
                        mejor = (n, k, c)
            if mejor:
                n, k, c = mejor
                evaluar(v, f"{v} {sentido} {c:.4g}", k, n)

    cols = ["variable", "regla", "n", "pct_poblacion", "tasa", "wilson_lb", "lift"]
    return (
        pd.DataFrame(filas, columns=cols)
        .sort_values("pct_poblacion", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    # Ejemplo mínimo de uso
    rng = np.random.default_rng(0)
    n = 5000
    df_ej = pd.DataFrame({
        "edad": rng.normal(40, 12, n).clip(18, 80),
        "ingreso": rng.lognormal(10, 0.6, n),
        "tipo_producto": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
    })
    # y correlacionado con edad e ingreso, para que el ejemplo produzca resultados
    prob = 0.1 + 0.15 * (df_ej["edad"] > 55) + 0.1 * (df_ej["ingreso"] > df_ej["ingreso"].quantile(0.8))
    df_ej["y"] = rng.binomial(1, prob.clip(0, 1))

    resultado = segmentos_univariados(
        df_ej, ["edad", "ingreso", "tipo_producto"], target="y", valor_objetivo=0.15
    )
    print(resultado)
