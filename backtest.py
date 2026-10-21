#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest.py
===========
Backtest de las reglas del generador (etapa 2 de Weinstein + RS Mansfield > 0 +
ruptura de base con volumen) sin mirar al futuro.

Compara 5 variantes para ver cuanto aporta cada filtro:
  1. Ruptura sola
  2. Ruptura + volumen
  3. Etapa 2 + RS>0 (entrada cuando se activa)
  4. Etapa 2 + RS>0 + ruptura
  5. COMPLETO: etapa 2 + RS>0 + ruptura + volumen

Y dos formas de salir:
  - trailing: cierre semanal bajo la SMA30 (estilo Weinstein) o stop inicial
  - objetivo: altura de la base proyectada sobre el techo, o stop inicial (max. 120 barras)

Reglas de ejecucion (sin lookahead): la señal se calcula al cierre; se entra en la
apertura de la barra siguiente; stop inicial bajo el minimo de la base - 0.25 ATR
(o 2 ATR si queda a mas del 12%); si stop y objetivo caen en la misma barra, gana el stop.

AVISO DE SESGO DE SUPERVIVENCIA: el universo es el S&P 500 ACTUAL. Las empresas que
salieron del indice (quiebras, caidas fuertes, adquisiciones) no estan, y las que
entraron lo hicieron tras subir. Los resultados salen MEJORES de lo que habria dado
operar entonces. Usalos para comparar variantes entre si, no como rentabilidad esperada.

Uso:
  python backtest.py                         # tickers.txt o S&P 500, 10 años, cache_bt/
  python backtest.py --tickers AAPL,MSFT --años 10
  python backtest.py --solo-cache            # sin red
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generador_graficos as g

L_BASE, PROF_MAX, VOL_MIN, COOLDOWN = 40, 0.16, 1.5, 20
VARIANTES = ["Ruptura sola", "Ruptura + volumen", "Etapa 2 + RS>0 (al activarse)",
             "Etapa 2 + RS>0 + ruptura", "COMPLETO: etapa 2 + RS>0 + ruptura + volumen"]
CON_RUPTURA = [v for v in VARIANTES if "uptura" in v]
HS = (20, 60)


def preparar_serie(raw, bench_w):
    """Señales y auxiliares, todo calculado solo con informacion disponible en cada fecha."""
    dd = g.preparar(raw)
    dw, _ = g.semanal(dd)
    et, m30, _ = g.etapas_serie(dw)
    rs = g.mansfield(dw["Close"], bench_w, 52)
    idx = dd.index
    # ultimo valor semanal ya cerrado en cada fecha diaria (etiqueta viernes <= fecha)
    etapa_d = et.reindex(idx, method="ffill")
    rs_d = rs.reindex(idx, method="ffill")
    h, l, c, v = dd["High"], dd["Low"], dd["Close"], dd["Volume"]
    prev_max = h.shift(1).rolling(L_BASE).max()
    prev_min = l.shift(1).rolling(L_BASE).min()
    prof = (prev_max - prev_min) / prev_max
    vr = v / v.shift(1).rolling(50).mean()
    ruptura = (c > prev_max) & (prof <= PROF_MAX)
    estado = (etapa_d == 2) & (rs_d > 0)
    S = {
        VARIANTES[0]: ruptura,
        VARIANTES[1]: ruptura & (vr >= VOL_MIN),
        VARIANTES[2]: estado & ~estado.shift(1, fill_value=False),
        VARIANTES[3]: estado & ruptura,
        VARIANTES[4]: estado & ruptura & (vr >= VOL_MIN),
    }
    S = {k: x.fillna(False).astype(bool) for k, x in S.items()}
    below = (dw["Close"] < m30) & m30.notna()
    salida = pd.Series(False, index=idx)
    for F, flag in below.items():
        if flag:
            sub = idx[(idx <= F) & (idx > F - pd.Timedelta(days=7))]
            if len(sub):
                salida.loc[sub[-1]] = True
    return {"dd": dd, "S": S, "atr": g.atr(h, l, c), "pmin": prev_min, "pmax": prev_max,
            "salida": salida.values, "valido": (etapa_d.notna() & rs_d.notna()).values}


def operar(P, i, modo, slip):
    dd = P["dd"]
    n = len(dd)
    j = i + 1
    if j >= n:
        return None
    O, H, Lw, C = (dd[k].values for k in ("Open", "High", "Low", "Close"))
    entrada = O[j] * (1 + slip)
    a, pm, pM = P["atr"].iloc[i], P["pmin"].iloc[i], P["pmax"].iloc[i]
    if not (np.isfinite(a) and np.isfinite(pm) and np.isfinite(pM)):
        return None
    stop = pm - 0.25 * a
    if (entrada - stop) / entrada > 0.12:
        stop = entrada - 2 * a
    if stop >= entrada:
        return None
    riesgo = entrada - stop
    tp = pM + (pM - pm)
    if modo == "objetivo" and tp <= entrada * 1.01:
        return None
    px, motivo, kout = None, None, None
    for k in range(j, n):
        if modo == "trailing" and P["salida"][k - 1]:
            px, motivo, kout = O[k], "SMA30 semanal", k
            break
        if O[k] <= stop:
            px, motivo, kout = O[k], "stop", k
            break
        if Lw[k] <= stop:
            px, motivo, kout = stop, "stop", k
            break
        if modo == "objetivo":
            if O[k] >= tp:
                px, motivo, kout = O[k], "objetivo", k
                break
            if H[k] >= tp:
                px, motivo, kout = tp, "objetivo", k
                break
            if k - j >= 120:
                px, motivo, kout = O[k], "tiempo", k
                break
    if px is None:
        px, motivo, kout = C[n - 1], "abierta", n - 1
    px_net = px * (1 - slip)
    return {"fecha_senal": dd.index[i], "fecha_entrada": dd.index[j], "entrada": entrada, "stop": stop,
            "fecha_salida": dd.index[kout], "salida": px_net, "motivo": motivo,
            "R": (px_net - entrada) / riesgo, "ret": px_net / entrada - 1, "barras": kout - j, "k_out": kout}


def trades_variante(P, nombre, modo, slip):
    idxs = np.flatnonzero(P["S"][nombre].values & P["valido"])
    out, libre = [], -1
    for i in idxs:
        if i <= libre:
            continue
        t = operar(P, i, modo, slip)
        if t is None:
            continue
        out.append(t)
        libre = t["k_out"]
    return out


def retornos_fwd(P, nombre):
    dd = P["dd"]
    O, C = dd["Open"].values, dd["Close"].values
    res = {h: [] for h in HS}
    last = -10 ** 9
    for i in np.flatnonzero(P["S"][nombre].values & P["valido"]):
        if i - last < COOLDOWN:
            continue
        last = i
        for h in HS:
            if i + 1 + h < len(dd):
                res[h].append(C[i + 1 + h] / O[i + 1] - 1)
    return res


def baseline_fwd(P):
    dd = P["dd"]
    O, C = dd["Open"].values, dd["Close"].values
    n = len(dd)
    out = {}
    for h in HS:
        r = C[1 + h:] / O[1:n - h] - 1
        out[h] = r[P["valido"][:n - 1 - h]]
    return out


def resumen(trs):
    if not trs:
        return {"Trades": 0}
    R = np.array([t["R"] for t in trs])
    ret = np.array([t["ret"] for t in trs])
    gan, per = R[R > 0].sum(), -R[R <= 0].sum()
    se = R.std(ddof=1) / np.sqrt(len(R)) if len(R) > 1 else np.nan
    mot = pd.Series([t["motivo"] for t in trs]).value_counts(normalize=True) * 100
    return {"Trades": len(R), "Ganadoras %": round((R > 0).mean() * 100, 1),
            "R medio": round(R.mean(), 2), "R mediano": round(float(np.median(R)), 2),
            "Factor beneficio": round(gan / per, 2) if per > 0 else np.nan,
            "Retorno medio %": round(ret.mean() * 100, 1),
            "Barras medias": int(round(np.mean([t["barras"] for t in trs]))),
            "t-stat R": round(R.mean() / se, 2) if se and se > 0 else np.nan,
            "% stop": round(mot.get("stop", 0), 0),
            "% salida por regla": round(mot.get("SMA30 semanal", 0) + mot.get("objetivo", 0), 0),
            "% abierta/tiempo": round(mot.get("abierta", 0) + mot.get("tiempo", 0), 0)}


def tabla_md(df):
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    filas = ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + " |"
             for row in df.itertuples(index=False)]
    return "\n".join([cab, sep] + filas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers")
    ap.add_argument("--archivo", default="tickers.txt")
    ap.add_argument("--cache", default="cache_bt")
    ap.add_argument("--salida", default="resultados_backtest")
    ap.add_argument("--años", type=int, default=10)
    ap.add_argument("--solo-cache", action="store_true")
    ap.add_argument("--coste", type=float, default=0.0005, help="Deslizamiento por lado (0.0005 = 0.05%)")
    args = ap.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    txt = ""
    if args.archivo and os.path.exists(args.archivo):
        txt = open(args.archivo, encoding="utf-8").read()
    tickers = g.parse_tickers((args.tickers or "") + "," + txt)
    if not tickers:
        p = os.path.join(args.cache, "sp500.csv")
        sp = pd.read_csv(p) if args.solo_cache and os.path.exists(p) else (None if args.solo_cache else g.cargar_sp500(args.cache))
        if sp is None:
            sys.exit("Sin tickers y sin lista del S&P 500.")
        tickers = list(dict.fromkeys(sp["Ticker"].tolist()))
        print(f"Universo: S&P 500 actual ({len(tickers)} tickers)")
    datos, avisos = g.actualizar_cache(tickers + [g.BENCH], args.cache, args.años, False, args.solo_cache)
    avisos += g.descartar_provisional(datos, args.cache)
    if g.BENCH not in datos:
        sys.exit("Falta ^GSPC en la cache: se necesita para el RS Mansfield.")
    bench_w = datos[g.BENCH]["Close"].resample("W-FRI").last().dropna()

    trades = {(v, m): [] for v in VARIANTES for m in ("trailing", "objetivo") if m == "trailing" or v in CON_RUPTURA}
    fwd = {v: {h: [] for h in HS} for v in VARIANTES}
    base = {h: [] for h in HS}
    usados, ini, fin = 0, None, None
    for k, t in enumerate(tickers, 1):
        if t not in datos or len(datos[t]) < 400:
            continue
        if k % 100 == 0:
            print(f"  {k}/{len(tickers)}")
        try:
            P = preparar_serie(datos[t], bench_w)
        except Exception as e:  # noqa
            avisos.append(f"{t}: error ({type(e).__name__}: {e})")
            continue
        usados += 1
        val = np.flatnonzero(P["valido"])
        if len(val):
            a, b = P["dd"].index[val[0]], P["dd"].index[-1]
            ini = a if ini is None else min(ini, a)
            fin = b if fin is None else max(fin, b)
        for (v, m) in trades:
            for tr in trades_variante(P, v, m, args.coste):
                tr["Ticker"], tr["Variante"], tr["Salida"] = t, v, m
                trades[(v, m)].append(tr)
        for v in VARIANTES:
            r = retornos_fwd(P, v)
            for h in HS:
                fwd[v][h] += r[h]
        bf = baseline_fwd(P)
        for h in HS:
            base[h].append(bf[h])
    if not usados:
        sys.exit("Sin datos suficientes (se necesitan 400+ barras por ticker).")
    base = {h: float(np.concatenate(base[h]).mean() * 100) for h in HS}

    filas = []
    for (v, m), trs in trades.items():
        f = {"Variante": v, "Salida": m, **resumen(trs)}
        f["Señales (20b entre ellas)"] = len(fwd[v][20])
        for h in HS:
            x = fwd[v][h]
            f[f"Retorno +{h}b %"] = round(float(np.mean(x)) * 100, 1) if x else np.nan
        filas.append(f)
    df = pd.DataFrame(filas)
    for h in HS:
        df[f"Base +{h}b %"] = round(base[h], 1)
    df.to_csv(os.path.join(args.salida, "backtest_resumen.csv"), index=False, encoding="utf-8-sig")
    todos = [t for trs in trades.values() for t in trs]
    pd.DataFrame(todos).drop(columns=["k_out"]).to_csv(os.path.join(args.salida, "backtest_trades.csv"), index=False)

    comp = trades[(VARIANTES[4], "trailing")]
    por_año = pd.DataFrame()
    if comp:
        d = pd.DataFrame(comp)
        d["Año"] = d["fecha_entrada"].dt.year
        por_año = d.groupby("Año").agg(Trades=("R", "size"), **{"R medio": ("R", "mean")},
                                       **{"Ganadoras %": ("R", lambda x: (x > 0).mean() * 100)}).round(2).reset_index()
        if len(d) >= 5:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].hist(d["R"].clip(-3, 10), bins=30, color="#2e7d32", alpha=0.8)
            ax[0].axvline(0, color="k", lw=0.8)
            ax[0].set_title("Distribucion de R (COMPLETO, trailing)")
            ds = d.sort_values("fecha_salida")
            ax[1].plot(ds["fecha_salida"], ds["R"].cumsum(), color="#1f4e79")
            ax[1].set_title("R acumulado por fecha de salida (trades solapados)")
            ax[1].grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(args.salida, "backtest_R.png"), dpi=110)
            plt.close(fig)

    md = ["# Backtest de las reglas", "",
          f"- Periodo con señales validas: {ini:%Y-%m-%d} a {fin:%Y-%m-%d}; {usados} tickers.",
          f"- Deslizamiento: {args.coste * 100:.2f}% por lado. Precios sin ajustar por dividendos.",
          "- **Sesgo de supervivencia:** universo = S&P 500 actual; faltan las empresas que salieron del indice. "
          "Los resultados estan inflados: sirven para comparar variantes, no como rentabilidad esperada.",
          "- Los trades no son independientes (las rupturas se concentran en fases alcistas del mercado), "
          "asi que el t-stat sobreestima la significancia.",
          "- `Base +Nb %` es el retorno medio de entrar cualquier dia valido (sin señal) a N barras.", "",
          "## Resumen", "", tabla_md(df), ""]
    if len(por_año):
        md += ["## COMPLETO (trailing) por año de entrada", "", tabla_md(por_año), ""]
    open(os.path.join(args.salida, "backtest.md"), "w", encoding="utf-8").write("\n".join(md))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(df.drop(columns=["Barras medias", "% stop", "% salida por regla", "% abierta/tiempo"], errors="ignore").to_string(index=False))
    print(f"\nPeriodo {ini:%Y-%m-%d} a {fin:%Y-%m-%d}, {usados} tickers. Resultados en {os.path.abspath(args.salida)}")
    print("AVISO: sesgo de supervivencia (S&P 500 actual); comparar variantes, no tomar como rentabilidad esperada.")


if __name__ == "__main__":
    main()
