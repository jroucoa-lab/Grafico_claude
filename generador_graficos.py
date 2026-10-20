#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generador_graficos.py
=====================
Descarga OHLCV (Yahoo/yfinance) y genera, para una lista de tickers:

  1. datos_ohlcv.csv          -> Ticker, Fecha, Open, High, Low, Close, AdjClose, Volume
  2. graficos_tecnicos.pdf    -> 2 paginas por accion (Diario + Semanal), velas japonesas,
                                 medias, volumen, RSI, MACD, RS Mansfield (y estocastico opcional)
  3. resumen_patrones.xlsx    -> hojas Resumen, Velas, Figuras (patrones de velas y figuras
                                 tecnicas detectados por codigo) + plan de referencia con
                                 tamano de posicion (tope por posicion y riesgo maximo)

Uso:
  python generador_graficos.py --tickers PFE,NUE,PR,CNC
  python generador_graficos.py --archivo tickers.txt --salida resultados
  python generador_graficos.py --csv-entrada datos_ohlcv.csv      # sin descargar (offline)

Dependencias:  pip install pandas numpy matplotlib yfinance openpyxl

AVISO: los patrones y figuras son detecciones heuristicas para REVISAR visualmente.
No es asesoramiento financiero.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BENCH = "^GSPC"
SECTOR_ETF = {
    "Technology": "XLK", "Information Technology": "XLK",
    "Financial Services": "XLF", "Financials": "XLF",
    "Healthcare": "XLV", "Health Care": "XLV",
    "Energy": "XLE", "Industrials": "XLI",
    "Consumer Cyclical": "XLY", "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP", "Consumer Staples": "XLP",
    "Basic Materials": "XLB", "Materials": "XLB",
    "Utilities": "XLU", "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


# ----------------------------------------------------------------------------
# Indicadores
# ----------------------------------------------------------------------------
def sma(s, n):
    return s.rolling(n).mean()


def wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def rsi(close, n=14):
    d = close.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad.replace(0, np.nan))


def macd(close, f=12, s=26, sg=9):
    m = close.ewm(span=f, adjust=False).mean() - close.ewm(span=s, adjust=False).mean()
    sig = m.ewm(span=sg, adjust=False).mean()
    return m, sig, m - sig


def estocastico(h, l, c, n=14, k=3, d=3):
    ll, hh = l.rolling(n).min(), h.rolling(n).max()
    raw = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    kk = raw.rolling(k).mean()
    return kk, kk.rolling(d).mean()


def atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def mansfield(close, bench, n):
    b = bench.reindex(close.index).ffill()
    rs = close / b
    return (rs / rs.rolling(n).mean() - 1) * 100


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
COLS = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]


def descargar(ticker, años):
    import yfinance as yf
    df = yf.download(ticker, period=f"{años}y", interval="1d", auto_adjust=False,
                     progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Adj Close": "AdjClose"})
    if "AdjClose" not in df:
        df["AdjClose"] = df["Close"]
    df = df[COLS].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def sector_de(ticker):
    try:
        import yfinance as yf
        return SECTOR_ETF.get(yf.Ticker(ticker).info.get("sector", ""), None)
    except Exception:
        return None


def ajustar(df):
    """OHLC ajustado por splits/dividendos (factor AdjClose/Close)."""
    f = (df["AdjClose"] / df["Close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return pd.DataFrame({"Open": df["Open"] * f, "High": df["High"] * f,
                         "Low": df["Low"] * f, "Close": df["AdjClose"],
                         "Volume": df["Volume"]}).dropna()


def semanal(d):
    w = d.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min",
                                  "Close": "last", "Volume": "sum"})
    return w.dropna(subset=["Close"])


# ----------------------------------------------------------------------------
# Patrones de velas
# ----------------------------------------------------------------------------
def patrones_velas(d):
    o, h, l, c = d["Open"], d["High"], d["Low"], d["Close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    up = h - np.maximum(o, c)
    lo = np.minimum(o, c) - l
    avgb = body.rolling(14).mean()
    bull, bear = c > o, c < o
    tdown = c.shift(1) < c.shift(6)
    tup = c.shift(1) > c.shift(6)
    s = lambda x, k: x.shift(k)

    P = {}
    P["Doji"] = ((body <= 0.1 * rng), "neutral")
    P["Martillo"] = ((lo >= 2 * body) & (up <= 0.15 * rng) & tdown, "alcista")
    P["Martillo invertido"] = ((up >= 2 * body) & (lo <= 0.15 * rng) & tdown & bull, "alcista")
    P["Hombre colgado"] = ((lo >= 2 * body) & (up <= 0.15 * rng) & tup, "bajista")
    P["Estrella fugaz"] = ((up >= 2 * body) & (lo <= 0.15 * rng) & tup, "bajista")
    P["Envolvente alcista"] = (s(bear, 1) & bull & (o <= s(c, 1)) & (c >= s(o, 1))
                               & (body > s(body, 1)) & tdown, "alcista")
    P["Envolvente bajista"] = (s(bull, 1) & bear & (o >= s(c, 1)) & (c <= s(o, 1))
                               & (body > s(body, 1)) & tup, "bajista")
    P["Perforante"] = (s(bear, 1) & bull & (o < s(c, 1)) & (c > (s(o, 1) + s(c, 1)) / 2)
                       & (c < s(o, 1)) & tdown, "alcista")
    P["Nube oscura"] = (s(bull, 1) & bear & (o > s(c, 1)) & (c < (s(o, 1) + s(c, 1)) / 2)
                        & (c > s(o, 1)) & tup, "bajista")
    P["Estrella de la manana"] = (s(bear, 2) & (s(body, 2) > s(avgb, 2))
                                  & (s(body, 1) <= 0.3 * s(body, 2)) & bull
                                  & (c > (s(o, 2) + s(c, 2)) / 2), "alcista")
    P["Estrella de la tarde"] = (s(bull, 2) & (s(body, 2) > s(avgb, 2))
                                 & (s(body, 1) <= 0.3 * s(body, 2)) & bear
                                 & (c < (s(o, 2) + s(c, 2)) / 2), "bajista")
    P["Tres soldados blancos"] = (bull & s(bull, 1) & s(bull, 2) & (c > s(c, 1))
                                  & (s(c, 1) > s(c, 2)) & (o > s(o, 1)) & (o < s(c, 1))
                                  & (s(o, 1) > s(o, 2)) & (s(o, 1) < s(c, 2))
                                  & (body > 0.5 * avgb), "alcista")
    P["Tres cuervos negros"] = (bear & s(bear, 1) & s(bear, 2) & (c < s(c, 1))
                                & (s(c, 1) < s(c, 2)) & (o < s(o, 1)) & (o > s(c, 1))
                                & (s(o, 1) < s(o, 2)) & (s(o, 1) > s(c, 2))
                                & (body > 0.5 * avgb), "bajista")
    P["Marubozu alcista"] = (bull & (body >= 0.9 * rng) & (body > 1.5 * avgb), "alcista")
    P["Marubozu bajista"] = (bear & (body >= 0.9 * rng) & (body > 1.5 * avgb), "bajista")
    P["Inside bar"] = ((h <= s(h, 1)) & (l >= s(l, 1)), "neutral")
    return {k: (v[0].fillna(False), v[1]) for k, v in P.items()}


def velas_recientes(t, tf, d, ultimas):
    rows = []
    P = patrones_velas(d)
    for i in range(max(0, len(d) - ultimas), len(d)):
        for nombre, (ser, sentido) in P.items():
            if bool(ser.iloc[i]):
                rows.append({"Ticker": t, "TF": tf, "Fecha": d.index[i].date(),
                             "Patron": nombre, "Sentido": sentido, "pos": i})
    return rows


# ----------------------------------------------------------------------------
# Pivotes, soportes/resistencias y figuras
# ----------------------------------------------------------------------------
def pivotes(d, n):
    h, l = d["High"].values, d["Low"].values
    ph, pl = [], []
    for i in range(n, len(d) - n):
        if h[i] == h[i - n:i + n + 1].max():
            ph.append(i)
        if l[i] == l[i - n:i + n + 1].min():
            pl.append(i)
    return ph, pl


def niveles_sr(d, ph, pl, ult, tol=0.015):
    ini = len(d) - ult
    precios = sorted([d["High"].iloc[i] for i in ph if i >= ini] +
                     [d["Low"].iloc[i] for i in pl if i >= ini])
    cl = []
    for p in precios:
        if cl and p <= cl[-1][-1] * (1 + tol):
            cl[-1].append(p)
        else:
            cl.append([p])
    return [(float(np.mean(c)), len(c)) for c in cl if len(c) >= 2]


def cercanos(niveles, precio):
    sop = [n for n, _ in niveles if n < precio]
    res = [n for n, _ in niveles if n > precio]
    return (max(sop) if sop else np.nan), (min(res) if res else np.nan)


def figuras(d, tf):
    out = []
    daily = tf == "D"
    n_piv = 5 if daily else 3
    L = 60 if daily else 26
    prof_max = 0.15 if daily else 0.25
    sep_min = 10 if daily else 6
    n52 = 252 if daily else 52
    nvol = 50 if daily else 10
    c = d["Close"].iloc[-1]
    ph, pl = pivotes(d, n_piv)

    # Base / rango lateral y ruptura
    if len(d) > L + 1:
        w = d.iloc[-L:]
        prof = (w["High"].max() - w["Low"].min()) / w["High"].max()
        prev_max = d["High"].iloc[-L:-1].max()
        vr = d["Volume"].iloc[-1] / d["Volume"].iloc[-nvol - 1:-1].mean()
        if c > prev_max and prof <= prof_max + 0.05:
            out.append(("Ruptura de base/rango",
                        f"cierre {c:.2f} > max {L} barras {prev_max:.2f}; vol x{vr:.2f}"))
        elif prof <= prof_max:
            out.append(("Base / rango lateral",
                        f"profundidad {prof*100:.1f}% en {L} barras; techo {w['High'].max():.2f}, "
                        f"suelo {w['Low'].min():.2f}"))

    # Cerca de maximo 52 semanas
    if len(d) >= n52:
        hi = d["High"].iloc[-n52:].max()
        dist = (c / hi - 1) * 100
        if dist >= -3:
            out.append(("Cerca de maximo 52s", f"maximo {hi:.2f}, distancia {dist:.1f}%"))

    # Doble suelo / doble techo
    if len(pl) >= 2:
        i1, i2 = pl[-2], pl[-1]
        l1, l2 = d["Low"].iloc[i1], d["Low"].iloc[i2]
        if abs(l1 - l2) / l1 <= 0.03 and i2 - i1 >= sep_min and len(d) - i2 <= 60:
            cuello = d["High"].iloc[i1:i2 + 1].max()
            if cuello >= max(l1, l2) * 1.08:
                est = "confirmado" if c > cuello else "pendiente de cierre > cuello"
                out.append(("Doble suelo", f"minimos {l1:.2f}/{l2:.2f}, cuello {cuello:.2f} ({est})"))
    if len(ph) >= 2:
        i1, i2 = ph[-2], ph[-1]
        h1, h2 = d["High"].iloc[i1], d["High"].iloc[i2]
        if abs(h1 - h2) / h1 <= 0.03 and i2 - i1 >= sep_min and len(d) - i2 <= 60:
            cuello = d["Low"].iloc[i1:i2 + 1].min()
            if cuello <= min(h1, h2) * 0.92:
                est = "confirmado" if c < cuello else "pendiente de cierre < cuello"
                out.append(("Doble techo", f"maximos {h1:.2f}/{h2:.2f}, cuello {cuello:.2f} ({est})"))

    # Triangulos (3 ultimos pivotes de cada tipo)
    if len(ph) >= 3 and len(pl) >= 3 and ph[-3] >= len(d) - (150 if daily else 80):
        H = [d["High"].iloc[i] for i in ph[-3:]]
        Lw = [d["Low"].iloc[i] for i in pl[-3:]]
        flat_h = (max(H) - min(H)) / np.mean(H) <= 0.02
        flat_l = (max(Lw) - min(Lw)) / np.mean(Lw) <= 0.02
        if flat_h and Lw[0] < Lw[1] < Lw[2]:
            out.append(("Triangulo ascendente", f"resistencia ~{np.mean(H):.2f}, minimos crecientes"))
        elif flat_l and H[0] > H[1] > H[2]:
            out.append(("Triangulo descendente", f"soporte ~{np.mean(Lw):.2f}, maximos decrecientes"))
        elif H[0] > H[1] > H[2] and Lw[0] < Lw[1] < Lw[2]:
            out.append(("Triangulo simetrico / cuna", "maximos decrecientes y minimos crecientes"))

    # Bandera alcista (solo diario)
    if daily and len(d) > 45:
        seg = d.iloc[-40:]
        hi_i = int(seg["High"].values.argmax())
        if hi_i >= 5:
            pre, cons = seg.iloc[:hi_i + 1], seg.iloc[hi_i:]
            lo_i = int(pre["Low"].values.argmin())
            base = pre["Low"].min()
            hi = seg["High"].max()
            gain = hi / base - 1
            retr = (hi - cons["Low"].min()) / (hi - base) if hi > base else 1
            rango = (cons["High"].max() - cons["Low"].min()) / hi
            if (gain >= 0.15 and hi_i - lo_i <= 20 and 4 <= len(cons) - 1 <= 20
                    and retr <= 0.5 and rango <= 0.12):
                cont = cons["Volume"].mean() < pre["Volume"].iloc[-10:].mean()
                out.append(("Bandera alcista (posible)",
                            f"mastil +{gain*100:.0f}%, retroceso {retr*100:.0f}%, "
                            f"volumen {'decreciente' if cont else 'no decreciente'}"))

    # Taza (solo semanal)
    if not daily and len(d) >= 40:
        N = min(60, len(d))
        w = d.iloc[-N:]
        left_i = int(w["High"].values[:N // 3].argmax())
        left = w["High"].iloc[left_i]
        bot_i = int(w["Low"].values[left_i:].argmin()) + left_i
        bottom = w["Low"].iloc[bot_i]
        prof = (left - bottom) / left
        right = w["High"].iloc[bot_i:].max()
        if (0.12 <= prof <= 0.40 and bot_i - left_i >= 8 and N - 1 - bot_i >= 6
                and right >= 0.9 * left and c >= 0.88 * left):
            out.append(("Taza (posible)", f"borde izq {left:.2f}, fondo {bottom:.2f} "
                                          f"({prof*100:.0f}%), pivote ~{left:.2f}"))
    return out


# ----------------------------------------------------------------------------
# Etapa de Weinstein simplificada (semanal)
# ----------------------------------------------------------------------------
def etapa(dw):
    s30 = sma(dw["Close"], 30)
    if s30.dropna().shape[0] < 5:
        return "n/d", np.nan
    c, m = dw["Close"].iloc[-1], s30.iloc[-1]
    pend = s30.iloc[-1] - s30.iloc[-5]
    if c > m and pend > 0:
        e = 2
    elif c > m:
        e = 3
    elif pend < 0:
        e = 4
    else:
        e = 1
    return e, (c / m - 1) * 100


# ----------------------------------------------------------------------------
# Graficos
# ----------------------------------------------------------------------------
def indicadores(d, bench, tf):
    c, h, l = d["Close"], d["High"], d["Low"]
    ind = {}
    if tf == "D":
        ind["wma10"], ind["wma50"], ind["wma150"] = wma(c, 10), wma(c, 50), wma(c, 150)
        ind["sma200"] = sma(c, 200)
        ind["vol_ma"] = sma(d["Volume"], 50)
        n_rs = 250
    else:
        ind["wma10"], ind["wma30"], ind["sma30"] = wma(c, 10), wma(c, 30), sma(c, 30)
        ind["vol_ma"] = sma(d["Volume"], 10)
        n_rs = 52
    ind["rsi"] = rsi(c)
    ind["macd"], ind["sig"], ind["hist"] = macd(c)
    ind["k"], ind["dd"] = estocastico(h, l, c)
    ind["atr"] = atr(h, l, c)
    ind["rs"] = mansfield(c, bench, n_rs) if bench is not None else None
    return ind


ESTILO_MAS = {
    "wma10": ("WMA10", "#1f77b4", 1.1, "-"), "wma50": ("WMA50", "#ff7f0e", 1.1, "-"),
    "wma150": ("WMA150", "#d62728", 1.4, "-"), "sma200": ("SMA200", "#7b3fa0", 1.2, "--"),
    "wma30": ("WMA30", "#ff7f0e", 1.1, "-"), "sma30": ("SMA30 (Weinstein)", "#2ca02c", 1.4, "--"),
}


def dibujar(pdf, t, d, tf, ind, velas, niveles, et, cfg):
    n = min(cfg["dias"] if tf == "D" else cfg["semanas"], len(d))
    w = d.iloc[-n:]
    idx = w.index
    x = np.arange(n)
    o, h, l, c = (w[k].values for k in ("Open", "High", "Low", "Close"))
    paneles = ["precio", "vol", "rsi", "macd"]
    if ind.get("rs") is not None:
        paneles.append("rs")
    if cfg["estoc"]:
        paneles.append("estoc")
    ratios = {"precio": 7, "vol": 1.6, "rsi": 1.9, "macd": 1.9, "rs": 1.7, "estoc": 1.7}
    fig, axs = plt.subplots(len(paneles), 1, figsize=(16.5, 11.7), sharex=True,
                            gridspec_kw={"height_ratios": [ratios[p] for p in paneles],
                                         "hspace": 0.06})
    axs = dict(zip(paneles, np.atleast_1d(axs)))
    ax = axs["precio"]

    # Velas
    verde, rojo = "#1a9850", "#d73027"
    col = np.where(c >= o, verde, rojo)
    ax.vlines(x, l, h, colors=col, linewidth=0.9, zorder=2)
    ax.bar(x, np.where(c >= o, c - o, o - c) + 1e-9, bottom=np.minimum(o, c),
           width=0.7, color=col, edgecolor=col, zorder=3)

    # Medias
    for k, (nombre, color, lw, ls) in ESTILO_MAS.items():
        if k in ind:
            s = ind[k].iloc[-n:].values
            if np.isfinite(s).any():
                ax.plot(x, s, color=color, lw=lw, ls=ls, label=nombre, zorder=4)

    # Maximo/minimo 52s y soportes/resistencias
    n52 = 252 if tf == "D" else 52
    hi52, lo52 = d["High"].iloc[-n52:].max(), d["Low"].iloc[-n52:].min()
    ax.axhline(hi52, color="grey", ls="--", lw=0.8)
    ax.axhline(lo52, color="grey", ls="--", lw=0.8)
    ax.text(n - 0.5, hi52, f" Max52 {hi52:.2f}", va="bottom", ha="right", fontsize=8, color="grey")
    ax.text(n - 0.5, lo52, f" Min52 {lo52:.2f}", va="top", ha="right", fontsize=8, color="grey")
    cierre = c[-1]
    cerc = sorted(niveles, key=lambda z: abs(z[0] - cierre))[:4]
    for nv, tq in cerc:
        ax.axhline(nv, color="#555555", ls=":", lw=0.9)
        ax.text(0, nv, f" {nv:.2f} ({tq})", fontsize=7.5, va="bottom", color="#555555")

    # Marcas de velas recientes
    rng_y = np.nanmax(h) - np.nanmin(l)
    for v in velas:
        p = v["pos"] - (len(d) - n)
        if 0 <= p < n and p >= n - cfg["marcar"]:
            if v["Sentido"] == "alcista":
                ax.annotate(v["Patron"], (p, l[p]), xytext=(p, l[p] - 0.04 * rng_y), fontsize=7,
                            color=verde, ha="center", va="top", rotation=90,
                            arrowprops=dict(arrowstyle="-", color=verde, lw=0.5))
            elif v["Sentido"] == "bajista":
                ax.annotate(v["Patron"], (p, h[p]), xytext=(p, h[p] + 0.04 * rng_y), fontsize=7,
                            color=rojo, ha="center", va="bottom", rotation=90,
                            arrowprops=dict(arrowstyle="-", color=rojo, lw=0.5))
            else:
                ax.annotate(v["Patron"], (p, h[p]), xytext=(p, h[p] + 0.04 * rng_y), fontsize=7,
                            color="#444444", ha="center", va="bottom", rotation=90)
    ymin, ymax = np.nanmin(l), np.nanmax(h)
    ax.set_ylim(ymin - 0.10 * (ymax - ymin), ymax + 0.12 * (ymax - ymin))
    ax.set_xlim(-1, n + 3)
    ax.annotate(f"{cierre:.2f}", (n - 1, cierre), xytext=(n + 0.5, cierre), fontsize=9,
                fontweight="bold", color="white", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="black"))
    ax.legend(loc="upper left", fontsize=8, ncol=5, frameon=False)
    ax.grid(alpha=0.25)
    tfn = "Diario" if tf == "D" else "Semanal"
    ax.set_title(f"{t} · {tfn} · ultima vela {idx[-1]:%d/%m/%Y} · cierre {cierre:.2f} · "
                 f"Etapa semanal {et}", fontsize=13, fontweight="bold", loc="left")

    # Volumen
    a = axs["vol"]
    a.bar(x, w["Volume"].values, width=0.7, color=col, alpha=0.6)
    a.plot(x, ind["vol_ma"].iloc[-n:].values, color="black", lw=1.0,
           label="Media vol " + ("50d" if tf == "D" else "10s"))
    a.set_ylabel("Volumen", fontsize=8)
    a.legend(loc="upper left", fontsize=7, frameon=False)
    a.grid(alpha=0.25)

    # RSI
    a = axs["rsi"]
    a.plot(x, ind["rsi"].iloc[-n:].values, color="#6a3d9a", lw=1.2)
    for y, ls in ((70, "--"), (50, ":"), (30, "--")):
        a.axhline(y, color="grey", ls=ls, lw=0.8)
    a.set_ylim(10, 90)
    a.set_ylabel("RSI 14", fontsize=8)
    a.grid(alpha=0.25)

    # MACD
    a = axs["macd"]
    hist = ind["hist"].iloc[-n:].values
    a.bar(x, hist, width=0.7, color=np.where(hist >= 0, verde, rojo), alpha=0.6)
    a.plot(x, ind["macd"].iloc[-n:].values, color="#1f77b4", lw=1.1)
    a.plot(x, ind["sig"].iloc[-n:].values, color="#ff7f0e", lw=1.1)
    a.axhline(0, color="grey", lw=0.8)
    a.set_ylabel("MACD 12/26/9", fontsize=8)
    a.grid(alpha=0.25)

    # RS Mansfield
    if "rs" in axs:
        a = axs["rs"]
        r = ind["rs"].iloc[-n:].values
        a.plot(x, r, color="#00695c", lw=1.3)
        a.fill_between(x, 0, r, where=np.nan_to_num(r) >= 0, color=verde, alpha=0.15)
        a.fill_between(x, 0, r, where=np.nan_to_num(r) < 0, color=rojo, alpha=0.15)
        a.axhline(0, color="grey", lw=0.8)
        a.set_ylabel("RS Mansfield", fontsize=8)
        a.grid(alpha=0.25)

    # Estocastico
    if "estoc" in axs:
        a = axs["estoc"]
        a.plot(x, ind["k"].iloc[-n:].values, color="#1f77b4", lw=1.1)
        a.plot(x, ind["dd"].iloc[-n:].values, color="#ff7f0e", lw=1.1)
        for y in (20, 80):
            a.axhline(y, color="grey", ls="--", lw=0.8)
        a.set_ylim(0, 100)
        a.set_ylabel("Estoc 14,3,3", fontsize=8)
        a.grid(alpha=0.25)

    # Eje X: meses
    ult = axs[paneles[-1]]
    pos, lab = [], []
    for i in range(n):
        if i == 0 or idx[i].month != idx[i - 1].month:
            if tf == "D" or idx[i].month in (1, 4, 7, 10):
                pos.append(i)
                lab.append(idx[i].strftime("%b %y"))
    if len(pos) > 1 and pos[1] - pos[0] < 8:
        pos, lab = pos[1:], lab[1:]
    ult.set_xticks(pos)
    ult.set_xticklabels(lab, fontsize=8)
    fig.text(0.995, 0.005, "Generado por generador_graficos.py", ha="right", fontsize=6, color="grey")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.04)
    pdf.savefig(fig)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Proceso principal
# ----------------------------------------------------------------------------
def leer_tickers(args):
    txt = args.tickers or ""
    if args.archivo:
        with open(args.archivo, encoding="utf-8") as f:
            txt += "," + f.read()
    ts = [t.strip().upper() for t in txt.replace("\n", ",").replace(";", ",").split(",")]
    vistos, out = set(), []
    for t in ts:
        if t and t not in vistos:
            vistos.add(t)
            out.append(t)
    return out


def cargar_csv(ruta):
    df = pd.read_csv(ruta, parse_dates=["Fecha"])
    if "AdjClose" not in df:
        df["AdjClose"] = df["Close"]
    out = {}
    for t, g in df.groupby("Ticker"):
        out[t] = g.set_index("Fecha")[COLS].sort_index()
    return out


def acciones_ref(cierre, atr_d, cfg):
    riesgo_acc = 2 * atr_d
    if not riesgo_acc or math.isnan(riesgo_acc) or riesgo_acc <= 0:
        return np.nan, np.nan, 0, 0.0
    stop = cierre - riesgo_acc
    n = int(min(cfg["riesgo_max"] // riesgo_acc, cfg["max_pos"] // cierre))
    return stop, riesgo_acc, n, n * riesgo_acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", help="Tickers separados por comas (formato Yahoo)")
    ap.add_argument("--archivo", help="Fichero de texto con tickers (comas o lineas)")
    ap.add_argument("--csv-entrada", help="CSV OHLCV existente (modo offline)")
    ap.add_argument("--salida", default="salida_graficos", help="Carpeta de salida")
    ap.add_argument("--años", type=int, default=6, help="Años de historico diario a descargar")
    ap.add_argument("--dias", type=int, default=170, help="Velas diarias en el grafico (~8 meses)")
    ap.add_argument("--semanas", type=int, default=156, help="Velas semanales en el grafico (~3 años)")
    ap.add_argument("--marcar", type=int, default=15, help="Marcar patrones de velas de las ultimas N barras")
    ap.add_argument("--estocastico", action="store_true", help="Añadir panel de estocastico")
    ap.add_argument("--cuenta", type=float, default=100000, help="Tamaño de cuenta")
    ap.add_argument("--riesgo", type=float, default=0.008, help="Riesgo maximo por operacion (fraccion)")
    ap.add_argument("--max-pos", type=float, default=5000, help="Capital maximo por posicion")
    args = ap.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    cfg = {"dias": args.dias, "semanas": args.semanas, "marcar": args.marcar,
           "estoc": args.estocastico, "riesgo_max": args.cuenta * args.riesgo,
           "max_pos": args.max_pos}

    # --- Datos
    if args.csv_entrada:
        datos = cargar_csv(args.csv_entrada)
        tickers = leer_tickers(args) or [t for t in datos if t != BENCH]
    else:
        tickers = leer_tickers(args)
        if not tickers:
            sys.exit("Indica --tickers, --archivo o --csv-entrada")
        datos = {}
        for t in tickers + [BENCH]:
            print(f"Descargando {t}...")
            try:
                df = descargar(t, args.años)
            except Exception as e:  # noqa
                print(f"  ! error {t}: {e}")
                df = None
            if df is not None:
                datos[t] = df

    bench_d = None
    if BENCH in datos:
        bench_d = datos[BENCH]["AdjClose"]
    bench_w = bench_d.resample("W-FRI").last().dropna() if bench_d is not None else None
    if bench_d is None:
        print("Aviso: sin ^GSPC; no se calculara RS Mansfield.")

    # CSV combinado
    partes = []
    for t in tickers:
        if t in datos:
            g = datos[t].copy()
            g.insert(0, "Ticker", t)
            g.index.name = "Fecha"
            partes.append(g.reset_index())
    if not partes:
        sys.exit("Sin datos.")
    pd.concat(partes).to_csv(os.path.join(args.salida, "datos_ohlcv.csv"), index=False)

    # --- Analisis y graficos
    resumen, velas_rows, fig_rows = [], [], []
    pdf_path = os.path.join(args.salida, "graficos_tecnicos.pdf")
    with PdfPages(pdf_path) as pdf:
        for t in tickers:
            if t not in datos or len(datos[t]) < 220:
                print(f"{t}: datos insuficientes (<220 barras), se omite")
                continue
            print(f"Analizando {t}...")
            dd = ajustar(datos[t])
            dw = semanal(dd)
            ind_d = indicadores(dd, bench_d, "D")
            ind_w = indicadores(dw, bench_w, "W")
            et, dist30 = etapa(dw)

            res = {}
            for tf, d, ind, ult in (("D", dd, ind_d, 10), ("W", dw, ind_w, 4)):
                ph, pl = pivotes(d, 5 if tf == "D" else 3)
                niv = niveles_sr(d, ph, pl, 250 if tf == "D" else 150)
                vel = velas_recientes(t, tf, d, ult)
                fgs = figuras(d, tf)
                res[tf] = (niv, vel, fgs)
                dibujar(pdf, t, d, tf, ind, vel, niv, et, cfg)
                velas_rows += [{k: v for k, v in r.items() if k != "pos"} for r in vel]
                fig_rows += [{"Ticker": t, "TF": tf, "Figura": f, "Detalle": det} for f, det in fgs]

            cierre = dd["Close"].iloc[-1]
            sop_d, res_d = cercanos(res["D"][0], cierre)
            sop_w, res_w = cercanos(res["W"][0], cierre)
            atr_d = ind_d["atr"].iloc[-1]
            stop, r_acc, nacc, riesgo = acciones_ref(cierre, atr_d, cfg)
            hi52 = dd["High"].iloc[-252:].max()
            resumen.append({
                "Ticker": t, "Fecha": dd.index[-1].date(), "Cierre": round(cierre, 2),
                "Etapa sem": et, "Dist SMA30s %": round(dist30, 1) if dist30 == dist30 else np.nan,
                "Dist max52 %": round((cierre / hi52 - 1) * 100, 1),
                "RSI dia": round(ind_d["rsi"].iloc[-1], 1), "RSI sem": round(ind_w["rsi"].iloc[-1], 1),
                "RS Mansfield sem": (round(ind_w["rs"].iloc[-1], 2) if ind_w["rs"] is not None else np.nan),
                "Vol ratio dia": round(dd["Volume"].iloc[-1] / dd["Volume"].iloc[-51:-1].mean(), 2),
                "ATR% dia": round(atr_d / cierre * 100, 2),
                "Soporte cercano (D)": round(sop_d, 2) if sop_d == sop_d else np.nan,
                "Resistencia cercana (D)": round(res_d, 2) if res_d == res_d else np.nan,
                "Soporte cercano (S)": round(sop_w, 2) if sop_w == sop_w else np.nan,
                "Resistencia cercana (S)": round(res_w, 2) if res_w == res_w else np.nan,
                "Velas D (10 barras)": "; ".join(sorted({r["Patron"] for r in res["D"][1]})),
                "Velas S (4 barras)": "; ".join(sorted({r["Patron"] for r in res["W"][1]})),
                "Figuras D": "; ".join(f for f, _ in res["D"][2]),
                "Figuras S": "; ".join(f for f, _ in res["W"][2]),
                "Ref stop (2xATR)": round(stop, 2) if stop == stop else np.nan,
                "Ref acciones": nacc, "Ref riesgo $": round(riesgo, 0),
                "Ref capital $": round(nacc * cierre, 0),
            })

    # --- Excel (o CSV si falta openpyxl)
    df_res = pd.DataFrame(resumen)
    df_vel = pd.DataFrame(velas_rows, columns=["Ticker", "TF", "Fecha", "Patron", "Sentido"])
    df_fig = pd.DataFrame(fig_rows, columns=["Ticker", "TF", "Figura", "Detalle"])
    try:
        ruta = os.path.join(args.salida, "resumen_patrones.xlsx")
        with pd.ExcelWriter(ruta, engine="openpyxl") as xw:
            df_res.to_excel(xw, sheet_name="Resumen", index=False)
            df_vel.to_excel(xw, sheet_name="Velas", index=False)
            df_fig.to_excel(xw, sheet_name="Figuras", index=False)
        from openpyxl import load_workbook
        wb = load_workbook(ruta)
        for ws in wb.worksheets:
            ws.freeze_panes = "B2"
            for col in ws.iter_cols():
                ancho = max(len(str(c.value or "")) for c in col) + 2
                ws.column_dimensions[col[0].column_letter].width = min(45, max(10, ancho))
        wb.save(ruta)
    except ImportError:
        df_res.to_csv(os.path.join(args.salida, "resumen.csv"), index=False)
        df_vel.to_csv(os.path.join(args.salida, "velas.csv"), index=False)
        df_fig.to_csv(os.path.join(args.salida, "figuras.csv"), index=False)
    print(f"\nListo. Resultados en: {os.path.abspath(args.salida)}")


if __name__ == "__main__":
    main()
