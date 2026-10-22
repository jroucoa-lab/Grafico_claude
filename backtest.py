#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest.py
===========
Backtest de las reglas del generador (etapa 2 de Weinstein + RS Mansfield > 0 +
ruptura de base con volumen) sin mirar al futuro.

Compara 10 variantes para ver cuanto aporta cada filtro:
  1-5. Ruptura sola / + volumen / Etapa 2 + RS>0 / + ruptura / COMPLETO
  6.   Estado + mercado (S&P 500 en etapa 2)
  7.   Estado + etapa joven (<= 12 semanas en etapa 2)
  8.   Estado + no extendida (<= 15% sobre la SMA30 semanal)
  9.   Estado + RS creciente (RS actual > RS de hace 4 semanas)
  10.  Combinado de 6, 7, 8 y 9
  11.  CONTROL: entrada en dias aleatorios (misma salida y mismo stop)
  12.  Etapa 2 sola, sin RS (para ver cuanto aporta el RS)
("Estado" = etapa 2 + RS>0; entra cuando la condicion se activa)
El CONTROL responde a la pregunta clave: ¿aportan algo las entradas, o el resultado sale
solo de la salida (stop + trailing semanal) en un mercado alcista? Se muestra la diferencia
de R frente al control con intervalo de confianza agrupado por mes.

Salida principal: trailing (cierre semanal bajo la SMA30, estilo Weinstein) o stop inicial.
Con --con-objetivo tambien prueba la salida por altura de base proyectada.

Ademas simula una CARTERA con tu tamaño real (riesgo por operacion, tope por posicion,
sin apalancamiento) y la compara con comprar y mantener el S&P 500. Cuando varias señales
coinciden el mismo dia y no hay caja para todas, se prioriza por RS Mansfield (y se mide
el ruido repitiendo con --semillas desempates aleatorios).

PRUEBA SIN LIMITE DE CAJA (backtest_sin_limite.csv): cartera equiponderada de TODAS las acciones
que cumplen la condicion cada dia (sin las 20 posiciones ni los desempates) frente a la de
todas las acciones. Es la forma mas limpia de ver si estar en etapa 2 con RS>0 aporta algo.
Incluye etapa 4 como contraste (deberia rendir peor que "todas") y una comparacion con IC95%.

PRUEBA DE MOMENTUM POR RS (backtest_rs_momentum.csv): cada mes se ordenan todas las acciones por
RS Mansfield y se mantienen a partes iguales los quintiles (Q1 = RS mas bajo ... Q5 = mas alto), el
decil superior y el quintil superior dentro de etapa 2, frente a todas. Sin limite de caja. Si el
RS ordena de verdad, el rendimiento debe subir de Q1 a Q5.

CUIDADO CON EL SOBREAJUSTE: probar muchas variantes sobre los mismos datos siempre
encuentra alguna que "funciona". Elige con la mitad anterior a --corte y comprueba
que se mantiene en la mitad posterior.

Reglas de ejecucion (sin lookahead): la señal se calcula al cierre; se entra en la
apertura de la barra siguiente; stop inicial bajo el minimo de la base - 0.25 ATR
(o 2 ATR si queda a mas del 12%); si stop y objetivo caen en la misma barra, gana el stop.

AVISO DE SESGO DE SUPERVIVENCIA: el universo es el S&P 500 ACTUAL. Las empresas que
salieron del indice (quiebras, caidas fuertes, adquisiciones) no estan, y las que
entraron lo hicieron tras subir. Los resultados salen MEJORES de lo que habria dado
operar entonces. Usalos para comparar variantes entre si, no como rentabilidad esperada.

Uso:
  python backtest.py --sp500                 # S&P 500 actual, 10 años, cache_bt/
  python backtest.py                         # tickers.txt (o S&P 500 si esta vacio)
  python backtest.py --tickers AAPL,MSFT --años 10
  python backtest.py --solo-cache            # sin red
"""
import argparse
import os
import sys
import zlib

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generador_graficos as g

L_BASE, PROF_MAX, VOL_MIN, COOLDOWN = 40, 0.16, 1.5, 20
VARIANTES = ["Ruptura sola", "Ruptura + volumen", "Etapa 2 + RS>0 (al activarse)",
             "Etapa 2 + RS>0 + ruptura", "COMPLETO: etapa 2 + RS>0 + ruptura + volumen",
             "Estado + mercado en etapa 2", "Estado + etapa joven (<=12 sem)",
             "Estado + no extendida (<=15%)", "Estado + RS creciente (4 sem)",
             "Combinado: mercado + RS creciente + joven + no extendida",
             "CONTROL: entrada aleatoria", "Etapa 2 sola (sin RS)"]
CONTROL = VARIANTES[10]
CON_RUPTURA = [v for v in VARIANTES if "uptura" in v]
HS = (20, 60)


def preparar_serie(raw, bench_w, et_mkt, seed=0, ticker=""):
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
    m30_d = m30.reindex(idx, method="ffill")
    dist = (c / m30_d - 1) * 100
    edad = et.groupby((et != et.shift()).cumsum()).cumcount() + 1
    joven = edad.reindex(idx, method="ffill") <= 12
    noext = dist <= 15
    rs_up = (rs > rs.shift(4)).astype(float).reindex(idx, method="ffill") == 1
    mkt = et_mkt.reindex(idx, method="ffill") == 2

    def act(x):
        x = x.fillna(False).astype(bool)
        return x & ~x.shift(1, fill_value=False)

    E = {"Estado (etapa 2 + RS>0)": estado,
         "Etapa 2 sola": (etapa_d == 2),
         "Estado + no extendida (<=15%)": estado & noext,
         "Estado + mercado en etapa 2": estado & mkt,
         "Etapa 4 (contraste)": (etapa_d == 4)}
    E = {k: x.fillna(False).astype(bool) for k, x in E.items()}
    S = {
        VARIANTES[0]: ruptura,
        VARIANTES[1]: ruptura & (vr >= VOL_MIN),
        VARIANTES[2]: act(estado),
        VARIANTES[3]: estado & ruptura,
        VARIANTES[4]: estado & ruptura & (vr >= VOL_MIN),
        VARIANTES[5]: act(estado & mkt),
        VARIANTES[6]: act(estado & joven),
        VARIANTES[7]: act(estado & noext),
        VARIANTES[8]: act(estado & rs_up),
        VARIANTES[9]: act(estado & mkt & rs_up & joven & noext),
        VARIANTES[10]: pd.Series(np.random.default_rng(seed + zlib.crc32(ticker.encode())).random(len(idx)) < 0.01,
                                 index=idx),
        VARIANTES[11]: act(etapa_d == 2),
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
            "salida": salida.values, "rs": rs_d.values, "valido": (etapa_d.notna() & rs_d.notna()).values,
            "E": E, "valido_s": (etapa_d.notna() & rs_d.notna())}


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
            "rs": P["rs"][i], "R": (px_net - entrada) / riesgo, "ret": px_net / entrada - 1, "barras": kout - j, "k_out": kout}


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


def simular_cartera(tr, px, bench_close, cuenta, riesgo, max_pos, seed=0, prioridad="aleatorio", ew=None):
    """Cartera con tu tamaño real: acciones = min(riesgo/(entrada-stop), max_pos/entrada),
    sin apalancamiento (si no hay caja, se omite la operacion). Valora a mercado cada dia.
    prioridad: con varias entradas el mismo dia, 'rs' = mayor RS Mansfield primero; 'aleatorio'."""
    if tr.empty:
        return None
    tr = tr.copy()
    if prioridad == "rs":
        tr["_o"] = -tr["rs"].fillna(-1e9)
    else:
        tr["_o"] = np.random.default_rng(seed).random(len(tr))
    tr = tr.sort_values(["fecha_entrada", "_o"])
    dias = px.index[(px.index >= tr["fecha_entrada"].min()) & (px.index <= tr["fecha_salida"].max())]
    por_dia = {d: g_.to_dict("records") for d, g_ in tr.groupby("fecha_entrada")}
    abiertas, cerradas, omitidas = [], [], 0
    cash = float(cuenta)
    eq, inv = [], []

    def cerrar(d):
        nonlocal cash
        for p in [p for p in abiertas if p["fecha_salida"] <= d]:
            cash += p["acc"] * p["salida"]
            cerradas.append((p["acc"] * (p["salida"] - p["entrada"]), p["acc"] * p["entrada"]))
            abiertas.remove(p)

    for d in dias:
        cerrar(d)
        for t in por_dia.get(d, []):
            r_acc = t["entrada"] - t["stop"]
            acc = int(min(riesgo // r_acc, max_pos // t["entrada"]))
            if acc < 1:
                continue
            if acc * t["entrada"] > cash:
                omitidas += 1
                continue
            cash -= acc * t["entrada"]
            abiertas.append({**t, "acc": acc})
        cerrar(d)
        val = sum(p["acc"] * px.at[d, p["Ticker"]] for p in abiertas)
        eq.append(cash + val)
        inv.append(val / (cash + val) if cash + val > 0 else np.nan)
    eq = pd.Series(eq, index=dias)
    años = max((dias[-1] - dias[0]).days / 365.25, 1e-9)
    dd = eq / eq.cummax() - 1
    mejor, ini = 0, None
    for d, b_ in (dd < -1e-9).items():
        if b_:
            ini = ini or d
            mejor = max(mejor, (d - ini).days)
        else:
            ini = None
    pnl = np.array([c[0] for c in cerradas])
    gan, per = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    bc = bench_close.reindex(dias).ffill()
    bdd = bc / bc.cummax() - 1
    extra = {}
    if ew is not None:
        ec = ew.reindex(dias).ffill()
        extra = {"Equipond. CAGR %": round(((ec.iloc[-1] / ec.iloc[0]) ** (1 / años) - 1) * 100, 1),
                 "Equipond. DD max %": round((ec / ec.cummax() - 1).min() * 100, 1)}
    return {"eq": eq, "dd": dd,
            "res": {"Operaciones tomadas": len(cerradas), "Omitidas (sin caja)": omitidas,
                    "Ganadoras %": round((pnl > 0).mean() * 100, 1) if len(pnl) else np.nan,
                    "Factor beneficio $": round(gan / per, 2) if per > 0 else np.nan,
                    "CAGR %": round(((eq.iloc[-1] / cuenta) ** (1 / años) - 1) * 100, 1),
                    "DD max %": round(dd.min() * 100, 1),
                    "Racha bajo maximos (años)": round(mejor / 365.25, 1),
                    "Capital invertido medio %": round(float(np.nanmean(inv)) * 100, 0),
                    "Cuenta final $": round(eq.iloc[-1]),
                    "S&P 500 CAGR %": round(((bc.iloc[-1] / bc.iloc[0]) ** (1 / años) - 1) * 100, 1),
                    "S&P 500 DD max %": round(bdd.min() * 100, 1), **extra}}


def equiponderado(px):
    """Indice equiponderado de los mismos tickers (comprar y mantener, rebalanceo mensual).
    Comparte el sesgo de supervivencia de la estrategia: es la referencia justa para medirlo."""
    r = px.pct_change()
    r = r.where(px.notna() & px.shift().notna())
    out = pd.Series(np.nan, index=px.index)
    val = 1.0
    for _, g_ in r.groupby(px.index.to_period("M")):
        cols = g_.columns[g_.notna().any()]
        if len(cols) == 0:
            out.loc[g_.index] = val
            continue
        path = (1 + g_[cols].fillna(0)).cumprod().mean(axis=1)
        out.loc[g_.index] = val * path.values
        val = out.loc[g_.index[-1]]
    return out


def cartera_sin_limite(ret, cond, coste):
    """Cartera equiponderada de las acciones con cond=True al cierre de t-1 (se mantienen durante t).
    Rebalanceo diario; sin miembros = caja. Descuenta coste por lado sobre la rotacion."""
    w = cond.shift(1, fill_value=False) & ret.notna()
    n = w.sum(axis=1)
    r = ret.where(w).sum(axis=1) / n.replace(0, np.nan)
    r = r.fillna(0.0)
    peso = w.div(n.replace(0, np.nan), axis=0).fillna(0.0)
    turn = (peso - peso.shift(1).fillna(0.0)).abs().sum(axis=1)
    r_net = r - turn * coste
    return r_net, n, turn


def cartera_rs_mensual(ret, rs, lo, hi, coste, filtro=None):
    """Cada mes: acciones con percentil de RS en (lo, hi] (calculado al cierre del mes anterior,
    solo entre las validas y, si hay filtro, entre las que lo cumplen), a partes iguales y sin tocar
    hasta fin de mes. Coste por lado sobre la rotacion real (con deriva de pesos)."""
    idx = ret.index
    r_out = pd.Series(0.0, index=idx)
    n_out = pd.Series(0, index=idx)
    t_out = pd.Series(0.0, index=idx)
    w_prev = pd.Series(dtype=float)
    prev_day = None
    for _, g_ in ret.groupby(idx.to_period("M")):
        miembros = []
        if prev_day is not None:
            rs_p = rs.loc[prev_day]
            ok = rs_p.notna()
            if filtro is not None:
                ok &= filtro.loc[prev_day]
            if ok.sum() >= 10:
                pct = rs_p[ok].rank(pct=True)
                miembros = [m for m in pct.index[(pct > lo) & (pct <= hi)] if g_[m].notna().any()]
        if not miembros:
            if len(w_prev):
                t_out.loc[g_.index[0]] = w_prev.sum()
            w_prev = pd.Series(dtype=float)
        else:
            w_new = pd.Series(1.0 / len(miembros), index=miembros)
            allw = w_new.index.union(w_prev.index)
            turn = (w_new.reindex(allw).fillna(0) - w_prev.reindex(allw).fillna(0)).abs().sum()
            cum = (1 + g_[miembros].fillna(0)).cumprod()
            path = cum.mean(axis=1)
            rr = path.pct_change()
            rr.iloc[0] = path.iloc[0] - 1
            rr.iloc[0] = (1 + rr.iloc[0]) * (1 - coste * turn) - 1
            r_out.loc[g_.index] = rr.values
            n_out.loc[g_.index] = len(miembros)
            t_out.loc[g_.index[0]] = turn
            fin = cum.iloc[-1]
            w_prev = fin / fin.sum()
        prev_day = g_.index[-1]
    return r_out, n_out, t_out


def exceso_ic(r_a, r_b, n=3000, seed=0):
    """Exceso diario medio de A sobre B anualizado, con IC95% remuestreando meses."""
    d = (r_a - r_b).dropna()
    mes = d.index.to_period("M")
    g_ = d.groupby(mes).agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(g_), (n, len(g_)))
    m = g_["sum"].values[idx].sum(1) / g_["count"].values[idx].sum(1) * 252 * 100
    lo, hi = np.percentile(m, [2.5, 97.5])
    return d.mean() * 252 * 100, lo, hi


def cagr_ventana(eq, a, b):
    """CAGR (%) y caida maxima (%) de la curva de cuenta entre a y b, reiniciando el capital en a."""
    x = eq[(eq.index >= a) & (eq.index < b)]
    if len(x) < 60:
        return np.nan, np.nan
    años = (x.index[-1] - x.index[0]).days / 365.25
    return ((x.iloc[-1] / x.iloc[0]) ** (1 / años) - 1) * 100, (x / x.cummax() - 1).min() * 100


def dif_vs_control(tdf, v, n=3000, seed=0):
    """Diferencia de R medio frente al control, con IC95% remuestreando meses de entrada."""
    t = tdf[tdf["Salida"] == "trailing"].copy()
    t["mes"] = t["fecha_entrada"].dt.to_period("M")
    a, c = t[t["Variante"] == v], t[t["Variante"] == CONTROL]
    meses = sorted(set(a["mes"]) | set(c["mes"]))
    ga = a.groupby("mes")["R"].agg(["sum", "count"]).reindex(meses).fillna(0)
    gc = c.groupby("mes")["R"].agg(["sum", "count"]).reindex(meses).fillna(0)
    idx = np.random.default_rng(seed).integers(0, len(meses), (n, len(meses)))
    ma = ga["sum"].values[idx].sum(1) / np.maximum(ga["count"].values[idx].sum(1), 1)
    mc = gc["sum"].values[idx].sum(1) / np.maximum(gc["count"].values[idx].sum(1), 1)
    d = ma - mc
    lo, hi = np.percentile(d, [2.5, 97.5])
    pa, pc = a["R"].mean(), c["R"].mean()
    return {"Variante": v, "R medio": round(pa, 2), "R control": round(pc, 2), "Diferencia": round(pa - pc, 2),
            "IC95 diferencia": f"[{lo:.2f}, {hi:.2f}]", "Supera al control": "SI" if lo > 0 else ("NO (peor)" if hi < 0 else "no se distingue")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers")
    ap.add_argument("--archivo", default="tickers.txt")
    ap.add_argument("--sp500", action="store_true", help="Ignora tickers.txt y usa el S&P 500 actual")
    ap.add_argument("--cache", default="cache_bt")
    ap.add_argument("--salida", default="resultados_backtest")
    ap.add_argument("--años", type=int, default=10)
    ap.add_argument("--solo-cache", action="store_true")
    ap.add_argument("--coste", type=float, default=0.0005, help="Deslizamiento por lado (0.0005 = 0.05%)")
    ap.add_argument("--corte", default="2022-01-01", help="Fecha que separa las dos mitades")
    ap.add_argument("--sin-tramo-desde", default="2024-01-01",
                    help="Tercera ventana: CAGR solo hasta esta fecha (quita el tramo final)")
    ap.add_argument("--con-objetivo", action="store_true", help="Prueba tambien la salida por altura de base")
    ap.add_argument("--cuenta", type=float, default=100000)
    ap.add_argument("--riesgo", type=float, default=800, help="Riesgo maximo por operacion en dinero")
    ap.add_argument("--max-pos", type=float, default=5000)
    ap.add_argument("--semilla", type=int, default=0, help="Semilla base (control aleatorio y desempates)")
    ap.add_argument("--semillas", type=int, default=5, help="Repeticiones con desempate aleatorio para medir el ruido")
    args = ap.parse_args()
    os.makedirs(args.salida, exist_ok=True)
    corte = pd.Timestamp(args.corte)

    tickers = []
    if not args.sp500:
        txt = ""
        if args.archivo and os.path.exists(args.archivo):
            txt = open(args.archivo, encoding="utf-8").read()
        tickers = g.parse_tickers((args.tickers or "") + "," + txt)
    if not tickers:
        p = os.path.join(args.cache, "sp500.csv")
        sp = (pd.read_csv(p) if os.path.exists(p) else None) if args.solo_cache else g.cargar_sp500(args.cache)
        if sp is None:
            sys.exit("Sin tickers y sin lista del S&P 500.")
        tickers = list(dict.fromkeys(sp["Ticker"].tolist()))
        print(f"Universo: S&P 500 actual ({len(tickers)} tickers)")
    else:
        print(f"Universo: {len(tickers)} tickers de la lista (no es el S&P 500 completo)")
    datos, avisos = g.actualizar_cache(tickers + [g.BENCH], args.cache, args.años, False, args.solo_cache)
    avisos += g.descartar_provisional(datos, args.cache)
    if g.BENCH not in datos:
        sys.exit("Falta ^GSPC en la cache: se necesita para el RS Mansfield y el filtro de mercado.")
    bench_close = datos[g.BENCH]["Close"]
    bench_w = bench_close.resample("W-FRI").last().dropna()
    et_mkt, _, _ = g.etapas_serie(pd.DataFrame({"Close": bench_w}))

    modos = {v: ["trailing"] + (["objetivo"] if args.con_objetivo and v in CON_RUPTURA else []) for v in VARIANTES}
    trades = {(v, m): [] for v in VARIANTES for m in modos[v]}
    fwd = {v: {h: [] for h in HS} for v in VARIANTES}
    base = {h: [] for h in HS}
    closes = {}
    cond_cols = {}
    valid_cols = {}
    rs_cols = {}
    usados, ini, fin = 0, None, None
    for k, t in enumerate(tickers, 1):
        if t not in datos or len(datos[t]) < 400:
            continue
        if k % 100 == 0:
            print(f"  {k}/{len(tickers)}")
        try:
            P = preparar_serie(datos[t], bench_w, et_mkt, args.semilla, t)
        except Exception as e:  # noqa
            avisos.append(f"{t}: error ({type(e).__name__}: {e})")
            continue
        usados += 1
        closes[t] = P["dd"]["Close"]
        valid_cols[t] = P["valido_s"]
        rs_cols[t] = pd.Series(P["rs"], index=P["dd"].index)
        for nom_c, ser_c in P["E"].items():
            cond_cols.setdefault(nom_c, {})[t] = ser_c & P["valido_s"]
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
        if trs:
            d = pd.DataFrame(trs)
            a_, b_ = d[d["fecha_entrada"] < corte]["R"], d[d["fecha_entrada"] >= corte]["R"]
            f["R antes corte"], f["n antes"] = round(a_.mean(), 2) if len(a_) else np.nan, len(a_)
            f["R desde corte"], f["n desde"] = round(b_.mean(), 2) if len(b_) else np.nan, len(b_)
        f["Señales (20b entre ellas)"] = len(fwd[v][20])
        for h in HS:
            x = fwd[v][h]
            f[f"Retorno +{h}b %"] = round(float(np.mean(x)) * 100, 1) if x else np.nan
        filas.append(f)
    df = pd.DataFrame(filas)
    rc = df[(df["Variante"] == CONTROL) & (df["Salida"] == "trailing")]["R medio"]
    if len(rc):
        df.insert(df.columns.get_loc("R medio") + 1, "R vs control", (df["R medio"] - rc.iloc[0]).round(2))
    for h in HS:
        df[f"Base +{h}b %"] = round(base[h], 1)
    df.to_csv(os.path.join(args.salida, "backtest_resumen.csv"), index=False, encoding="utf-8-sig")
    todos = [t for trs in trades.values() for t in trs]
    tdf = pd.DataFrame(todos)
    tdf.drop(columns=["k_out"]).to_csv(os.path.join(args.salida, "backtest_trades.csv"), index=False)

    # --- Diferencia frente al control
    dfd = pd.DataFrame([dif_vs_control(tdf, v) for v in VARIANTES if v != CONTROL]) if len(tdf) else pd.DataFrame()
    if len(dfd):
        dfd.to_csv(os.path.join(args.salida, "backtest_vs_control.csv"), index=False, encoding="utf-8-sig")

    # --- Cartera (prioridad por RS) y ruido con desempates aleatorios
    px_all = pd.DataFrame(closes).reindex(bench_close.index).ffill()
    ew = equiponderado(px_all)

    # --- Cartera sin limite de caja: condicion vs todas las acciones
    ret_all = px_all.pct_change()
    idx_all = px_all.index
    todas_m = pd.DataFrame(valid_cols).reindex(idx_all).fillna(False).astype(bool)
    r_todas, n_todas, turn_todas = cartera_sin_limite(ret_all, todas_m, args.coste)
    fecha_ini = idx_all[np.flatnonzero(todas_m.any(axis=1).values)[0]] if todas_m.values.any() else idx_all[0]
    r_todas = r_todas[r_todas.index >= fecha_ini]
    años_t = (r_todas.index[-1] - r_todas.index[0]).days / 365.25
    ventanas_sl = [(f"antes {corte:%Y-%m}", pd.Timestamp.min, corte), (f"desde {corte:%Y-%m}", corte, pd.Timestamp.max),
                   (f"hasta {pd.Timestamp(args.sin_tramo_desde):%Y-%m}", pd.Timestamp.min, pd.Timestamp(args.sin_tramo_desde))]

    def fila_sl(nombre, r_net, n, turn, es_base=False, base=None):
        r_net, n, turn = (x[x.index >= fecha_ini] for x in (r_net, n, turn))
        base_r = r_todas if base is None else base[base.index >= fecha_ini]
        eq = (1 + r_net).cumprod()
        f = {"Cartera": nombre,
             "CAGR %": round(((eq.iloc[-1]) ** (1 / años_t) - 1) * 100, 1),
             "DD max %": round((eq / eq.cummax() - 1).min() * 100, 1)}
        for nom, a, b in ventanas_sl:
            f[f"CAGR {nom}"] = round(cagr_ventana(eq, a, b)[0], 1)
        f["% del tiempo invertida"] = round((n > 0).mean() * 100, 0)
        f["Acciones medias"] = round(float(n[n > 0].mean()), 0) if (n > 0).any() else np.nan
        f["Rotacion anual (x cartera)"] = round(float(turn.sum() / años_t), 1)
        if not es_base:
            ex, lo, hi = exceso_ic(r_net, base_r)
            act = n > 0
            f["Exceso anual vs todas %"] = round(ex, 1)
            f["IC95 exceso"] = f"[{lo:.1f}, {hi:.1f}]"
            f["Exceso solo dias invertida %"] = round(float((r_net - base_r)[act].mean() * 252 * 100), 1)
            f["Veredicto"] = "SUPERA" if lo > 0 else ("PEOR" if hi < 0 else "no se distingue")
        return f, eq

    filas_sl, eqs_sl = [], {}
    f0, eq0 = fila_sl("TODAS las acciones (equiponderada diaria)", r_todas, n_todas, turn_todas, True)
    filas_sl.append(f0)
    eqs_sl["Todas"] = eq0
    for nom_c, cols_c in cond_cols.items():
        cm = pd.DataFrame(cols_c).reindex(idx_all).fillna(False).astype(bool)
        r_c, n_c, t_c = cartera_sin_limite(ret_all, cm, args.coste)
        f, eqc = fila_sl(nom_c, r_c, n_c, t_c)
        filas_sl.append(f)
        eqs_sl[nom_c] = eqc
    dfs = pd.DataFrame(filas_sl)
    dfs.to_csv(os.path.join(args.salida, "backtest_sin_limite.csv"), index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for k, v_ in eqs_sl.items():
        if k in ("Todas", "Estado (etapa 2 + RS>0)", "Etapa 2 sola", "Etapa 4 (contraste)", "Estado + mercado en etapa 2"):
            ax.plot(v_ / v_.iloc[0] * args.cuenta, lw=1.3, label=k)
    bs = bench_close.reindex(eqs_sl["Todas"].index).ffill()
    ax.plot(bs / bs.iloc[0] * args.cuenta, color="grey", ls="--", lw=1.2, label="S&P 500")
    ax.set_yscale("log")
    ax.set_title("Sin limite de caja: equiponderada de las acciones que cumplen la condicion (escala log)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.salida, "backtest_sin_limite.png"), dpi=110)
    plt.close(fig)

    # --- Momentum por RS (cuantiles mensuales, sin limite de caja)
    rs_m = pd.DataFrame(rs_cols).reindex(idx_all)
    et2_m = pd.DataFrame(cond_cols["Etapa 2 sola"]).reindex(idx_all).fillna(False).astype(bool)
    r_b, n_b, t_b = cartera_rs_mensual(ret_all, rs_m, 0.0, 1.0, args.coste)
    fb, eq_b = fila_sl("TODAS (mensual)", r_b, n_b, t_b, True)
    filas_rs, eqs_rs = [fb], {"Todas": eq_b}
    for nom_q, lo_q, hi_q, filt in [("Q1 (RS mas bajo)", 0, .2, None), ("Q2", .2, .4, None), ("Q3", .4, .6, None),
                                    ("Q4", .6, .8, None), ("Q5 (RS mas alto)", .8, 1.0, None),
                                    ("Decil superior (D10)", .9, 1.0, None),
                                    ("Etapa 2: quintil superior de RS", .8, 1.0, et2_m)]:
        r_q, n_q, t_q = cartera_rs_mensual(ret_all, rs_m, lo_q, hi_q, args.coste, filt)
        f_q, eq_q = fila_sl(nom_q, r_q, n_q, t_q, base=r_b)
        filas_rs.append(f_q)
        eqs_rs[nom_q] = eq_q
    dfr = pd.DataFrame(filas_rs)
    dfr.to_csv(os.path.join(args.salida, "backtest_rs_momentum.csv"), index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for k_, v_ in eqs_rs.items():
        if k_ in ("Todas", "Q1 (RS mas bajo)", "Q3", "Q5 (RS mas alto)", "Decil superior (D10)"):
            ax.plot(v_ / v_.iloc[0] * args.cuenta, lw=1.3, label=k_)
    ax.plot(bs / bs.iloc[0] * args.cuenta, color="grey", ls="--", lw=1.2, label="S&P 500")
    ax.set_yscale("log")
    ax.set_title("Momentum por RS: cuantiles mensuales equiponderados (escala log)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.salida, "backtest_rs_momentum.png"), dpi=110)
    plt.close(fig)
    carteras, filas_c = {}, []
    for v in VARIANTES:
        sub = tdf[(tdf["Variante"] == v) & (tdf["Salida"] == "trailing")] if len(tdf) else tdf
        c_ = simular_cartera(sub, px_all, bench_close, args.cuenta, args.riesgo, args.max_pos, prioridad="rs", ew=ew)
        if not c_:
            continue
        carteras[v] = c_
        sims = [simular_cartera(sub, px_all, bench_close, args.cuenta, args.riesgo, args.max_pos, seed=sd)
                for sd in range(args.semillas)]
        c_["eqs_alea"] = [x["eq"] for x in sims]
        cag = [x["res"]["CAGR %"] for x in sims]
        filas_c.append({"Variante": v, **c_["res"], "CAGR aleat. media %": round(float(np.mean(cag)), 1),
                        "CAGR aleat. min": round(min(cag), 1), "CAGR aleat. max": round(max(cag), 1)})
    dfc = pd.DataFrame(filas_c)
    excl = pd.Timestamp(args.sin_tramo_desde)
    ventanas = [(f"antes {corte:%Y-%m}", pd.Timestamp.min, corte), (f"desde {corte:%Y-%m}", corte, pd.Timestamp.max),
                (f"hasta {excl:%Y-%m}", pd.Timestamp.min, excl)]
    filas_m = []
    for v, c_ in carteras.items():
        f = {"Variante": v}
        for nom, a, b in ventanas:
            f[f"CAGR {nom} (RS)"] = round(cagr_ventana(c_["eq"], a, b)[0], 1)
        for nom, a, b in ventanas:
            f[f"CAGR {nom} (aleat.)"] = round(float(np.nanmean([cagr_ventana(e, a, b)[0] for e in c_["eqs_alea"]])), 1)
        filas_m.append(f)
    if carteras:
        sp_eq = bench_close
        f = {"Variante": "S&P 500 (comprar y mantener)"}
        for pri in ("(RS)", "(aleat.)"):
            for nom, a, b in ventanas:
                f[f"CAGR {nom} {pri}"] = round(cagr_ventana(sp_eq, max(a, list(carteras.values())[0]["eq"].index[0]), b)[0], 1)
        filas_m.append(f)
        f = {"Variante": "Equiponderado de los mismos tickers (mensual)"}
        for pri in ("(RS)", "(aleat.)"):
            for nom, a, b in ventanas:
                f[f"CAGR {nom} {pri}"] = round(cagr_ventana(ew, max(a, list(carteras.values())[0]["eq"].index[0]), b)[0], 1)
        filas_m.append(f)
    dfm = pd.DataFrame(filas_m)
    if len(dfm):
        sp, eqw = dfm.iloc[-2], dfm.iloc[-1]
        cols_rs = [c for c in dfm.columns if "(RS)" in c]
        dfm["Gana al S&P en las 2 mitades (RS)"] = [
            "SI" if all(r[c] > sp[c] for c in cols_rs[:2]) else "no" for _, r in dfm.iterrows()]
        dfm["Gana al equipond. en las 2 mitades (RS)"] = [
            "SI" if all(r[c] > eqw[c] for c in cols_rs[:2]) else "no" for _, r in dfm.iterrows()]
        for k in (-2, -1):
            dfm.iloc[k, dfm.columns.get_loc("Gana al S&P en las 2 mitades (RS)")] = ""
            dfm.iloc[k, dfm.columns.get_loc("Gana al equipond. en las 2 mitades (RS)")] = ""
        dfm.to_csv(os.path.join(args.salida, "backtest_mitades.csv"), index=False, encoding="utf-8-sig")
    if len(dfc):
        dfc.to_csv(os.path.join(args.salida, "backtest_cartera.csv"), index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
        sel = [v for v in (CONTROL, VARIANTES[2], VARIANTES[7], VARIANTES[4]) if v in carteras]
        for v in sel:
            ax[0].plot(carteras[v]["eq"], lw=1.3, label=v[:42])
            ax[1].plot(carteras[v]["dd"] * 100, lw=1.0)
        e0 = carteras[sel[0]]["eq"]
        bc = bench_close.reindex(e0.index).ffill()
        ax[0].plot(bc / bc.iloc[0] * args.cuenta, color="grey", lw=1.3, ls="--", label="S&P 500 (comprar y mantener)")
        ec_ = ew.reindex(e0.index).ffill()
        ax[0].plot(ec_ / ec_.iloc[0] * args.cuenta, color="black", lw=1.3, ls=":", label="Equiponderado mismos tickers")
        ax[0].set_title(f"Cartera simulada (prioridad RS): {args.riesgo:.0f} de riesgo, max {args.max_pos:.0f} por posicion")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        ax[1].set_ylabel("Caida desde maximo %")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.salida, "backtest_cartera.png"), dpi=110)
        plt.close(fig)

    # --- Grafico de R (estado) y por año
    ref = VARIANTES[2]
    d_ref = tdf[(tdf["Variante"] == ref) & (tdf["Salida"] == "trailing")] if len(tdf) else tdf
    por_año = pd.DataFrame()
    if len(d_ref):
        d = d_ref.copy()
        d["Año"] = d["fecha_entrada"].dt.year
        por_año = d.groupby("Año").agg(Trades=("R", "size"), **{"R medio": ("R", "mean")},
                                       **{"Ganadoras %": ("R", lambda x: (x > 0).mean() * 100)}).round(2).reset_index()
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(d["R"].clip(-3, 10), bins=30, color="#2e7d32", alpha=0.8)
        ax[0].axvline(0, color="k", lw=0.8)
        ax[0].set_title("Distribucion de R (Estado, trailing)")
        ds = d.sort_values("fecha_salida")
        ax[1].plot(ds["fecha_salida"], ds["R"].cumsum(), color="#1f4e79")
        ax[1].set_title("R acumulado (trades solapados, sin tamaño)")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.salida, "backtest_R.png"), dpi=110)
        plt.close(fig)

    md = ["# Backtest de las reglas", "",
          f"- Periodo con señales validas: {ini:%Y-%m-%d} a {fin:%Y-%m-%d}; {usados} tickers"
          + ("" if len(tickers) > 400 else " (NO es el S&P 500 completo)") + ".",
          f"- Deslizamiento: {args.coste * 100:.2f}% por lado. Precios sin ajustar por dividendos.",
          "- **Sesgo de supervivencia:** universo = componentes ACTUALES; faltan las empresas que salieron del indice. "
          "Los resultados estan inflados: sirven para comparar variantes, no como rentabilidad esperada.",
          "- **Sobreajuste:** se prueban 10 variantes sobre los mismos datos. Elige con la mitad anterior al corte "
          f"({corte:%Y-%m-%d}) y exige que se mantenga en la posterior.",
          "- Los trades no son independientes (las rupturas se concentran en fases alcistas): el t-stat sobreestima.",
          "- `Base +Nb %` = retorno medio de entrar cualquier dia valido (sin señal) a N barras.", "",
          "- **CONTROL:** entra en dias aleatorios (~1% de las barras validas) con el mismo stop y la misma salida. "
          "Si una variante no lo supera, sus entradas no aportan: el resultado viene de la salida y del mercado.", "",
          "## Operaciones (R = ganancia / riesgo inicial)", "", tabla_md(df), ""]
    if len(dfd):
        md += ["## Diferencia de R frente al control (IC95% agrupado por mes)", "", tabla_md(dfd), ""]
    md += ["## Prueba sin limite de caja (equiponderada diaria, coste incluido)", "",
           "Cada dia se mantiene a partes iguales las acciones que cumplian la condicion al cierre anterior; sin ninguna, caja. "
           "`Exceso` = diferencia con la cartera de TODAS las acciones, con IC95% remuestreando meses. "
           "Etapa 4 es un contraste: si la clasificacion sirve, debe rendir peor que todas.", "", tabla_md(dfs), ""]
    md += ["## Momentum por RS (cuantiles mensuales, sin limite de caja)", "",
           "Q1 = acciones con RS mas bajo, Q5 = RS mas alto (entre todas las validas, al cierre del mes anterior). "
           "Si el RS ordena de verdad, el CAGR debe subir de Q1 a Q5. Ojo: el universo son componentes actuales "
           "(algunas entraron al indice despues de subir), lo que favorece al momentum.", "", tabla_md(dfr), ""]
    if len(dfm):
        md += ["## CAGR por mitades (cada ventana reinicia el capital)", "",
               f"`antes/desde {corte:%Y-%m}` = dos mitades; `hasta {excl:%Y-%m}` = sin el tramo final. "
               "(RS) = prioridad por RS cuando falta caja; (aleat.) = media de los desempates aleatorios.", "",
               tabla_md(dfm), ""]
    if len(dfc):
        md += ["## Cartera con tu tamaño real (prioridad por RS cuando falta caja)", "",
               "`Equipond.` = comprar y mantener los mismos tickers a partes iguales (rebalanceo mensual): "
               "comparte el sesgo de supervivencia, asi que lo que la estrategia gane sobre el es lo que no se explica por ese sesgo.", "",
               f"`CAGR aleat.` = {args.semillas} repeticiones con desempate aleatorio: mide cuanto del resultado es suerte.", "",
               tabla_md(dfc), ""]
    if len(por_año):
        md += ["## Estado (etapa 2 + RS>0, trailing) por año de entrada", "", tabla_md(por_año), ""]
    open(os.path.join(args.salida, "backtest.md"), "w", encoding="utf-8").write("\n".join(md))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["Variante", "Salida", "Trades", "R medio", "R vs control", "Factor beneficio", "R antes corte", "R desde corte"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    if len(dfd):
        print()
        print(dfd.to_string(index=False))
    print()
    print(dfs[["Cartera", "CAGR %", "DD max %", "% del tiempo invertida", "Acciones medias", "Exceso anual vs todas %",
               "IC95 exceso", "Veredicto"] if "Veredicto" in dfs.columns else ["Cartera", "CAGR %"]].to_string(index=False))
    print()
    print(dfr[["Cartera", "CAGR %", "DD max %", "CAGR antes " + f"{corte:%Y-%m}", "CAGR desde " + f"{corte:%Y-%m}",
               "Acciones medias", "Rotacion anual (x cartera)", "Exceso anual vs todas %", "IC95 exceso",
               "Veredicto"] if "Veredicto" in dfr.columns else ["Cartera", "CAGR %"]].to_string(index=False))
    if len(dfm):
        print()
        print(dfm.to_string(index=False))
    if len(dfc):
        print()
        print(dfc[["Variante", "Operaciones tomadas", "Omitidas (sin caja)", "CAGR %", "CAGR aleat. media %",
                   "CAGR aleat. min", "CAGR aleat. max", "DD max %", "Racha bajo maximos (años)",
                   "S&P 500 CAGR %", "Equipond. CAGR %", "Equipond. DD max %"]].to_string(index=False))
    print(f"\nPeriodo {ini:%Y-%m-%d} a {fin:%Y-%m-%d}, {usados} tickers. Resultados en {os.path.abspath(args.salida)}")
    print("AVISO: sesgo de supervivencia y riesgo de sobreajuste: comparar variantes, no tomar como rentabilidad esperada.")


if __name__ == "__main__":
    main()
