#!/usr/bin/env python3
"""
Developer: Akash Khatri
Created On: 2026-10-15
Last Updated: 2026-10-15
Maintainer: Akash Khatri
Contact: [kashkhatri@yahoo.com]
Organization: Personal
Version: 1.0.0
License: NA

market_scanner.py - one script, any stock/index.

SCAN MODE (default):   python market_scanner.py chain.csv
  Reads an option-chain CSV and finds 4-leg (2 calls + 2 puts, incl. box spreads) and
  2-leg vertical spreads where every leg is entered at a real market price (so Opstra
  shows (0) on every leg) and the expiry payoff is strictly green (> 0 everywhere).

SOLVE MODE:            python market_scanner.py solve
  Asks for your 4 legs + market data and prints the entry price one leg must have for a
  green payoff (for reference: this is NOT a market price).

CSV format - header names are matched loosely (case/spaces/underscores ignored), so you
can paste Opstra's columns as they are:
  required : StrikePrice, CallLTP, PutLTP
  optional : CallOI, PutOI, CallIV, PutIV           (OI / stale-price filters)
             CallBid, CallAsk, PutBid, PutAsk       (for --prices bidask)
  Market data can sit at the top of the CSV as a comment line, e.g.
             # spot=300.6 lot=1975 iv=27.19 dte=24
  Anything missing is asked for when you run the script.

Options (scan mode):
  --spot --lot --iv --dte   market data (override the CSV comment line)
  --prices ltp|bidask       ltp: all brackets (0). bidask: buy at Ask, sell at Bid (realistic)
  --min-oi N                ignore strikes with open interest below N (default 0)
  --parity-tol P            drop strikes whose put-call parity is off by > P points (default 5, 0=off)
  --floor-rs R              minimum guaranteed profit in rupees (chart base height, default 0)
  --top N                   results to show (default 5)
Requires: pip install numpy
"""
import argparse
import csv
import math
import re
import sys

import numpy as np

# ----------------------------------------------------------------------------------
# Payoff maths (per share, at expiry)
# ----------------------------------------------------------------------------------


def pnl(legs, S):
    t = 0.0
    for l in legs:
        iv = max(S - l["k"], 0) if l["typ"] == "CE" else max(l["k"] - S, 0)
        t += l["sign"] * (iv - l["p"])
    return t


def tail_slope(legs):  # payoff slope above the highest strike
    return sum(l["sign"] for l in legs if l["typ"] == "CE")


def analyse(legs):
    """(min payoff, max payoff) per share; +-inf when unbounded."""
    pts = [0.0] + sorted({l["k"] for l in legs})
    vals = [pnl(legs, x) for x in pts]
    lo, hi, ts = min(vals), max(vals), tail_slope(legs)
    if ts < 0:
        lo = -math.inf
    if ts > 0:
        hi = math.inf
    return lo, hi


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def pop(legs, spot, iv, dte):
    """Probability of profit, lognormal model."""
    sd = iv / 100 * math.sqrt(dte / 365)
    cdf = lambda x: (
        0.0
        if x <= 0
        else (1.0 if x == math.inf else ncdf((math.log(x / spot) + 0.5 * sd * sd) / sd))
    )
    pts = [0.0] + sorted({l["k"] for l in legs})
    segs = [
        (pts[i], pts[i + 1], pnl(legs, pts[i]), pnl(legs, pts[i + 1]))
        for i in range(len(pts) - 1)
    ]
    last, ts = pts[-1], tail_slope(legs)
    segs.append(
        (
            last,
            math.inf,
            pnl(legs, last),
            math.inf if ts > 0 else (-math.inf if ts < 0 else pnl(legs, last)),
        )
    )
    prob = 0.0
    for a, b, ya, yb in segs:
        if b == math.inf:
            if ya > 0 and yb > 0:
                prob += 1 - cdf(a)
            elif ya <= 0 and yb == math.inf:
                prob += 1 - cdf(a + (-ya) / ts)
            elif ya > 0 and yb == -math.inf:
                prob += cdf(a + ya / -ts) - cdf(a)
            continue
        if ya > 0 and yb > 0:
            prob += cdf(b) - cdf(a)
        elif ya > 0 >= yb or yb > 0 >= ya:
            x = a + (b - a) * (0 - ya) / (yb - ya)
            prob += (cdf(x) - cdf(a)) if ya > 0 else (cdf(b) - cdf(x))
    return min(prob, 1.0)


def metrics(legs, lot, spot, iv, dte):
    lo, hi = analyse(legs)
    credit = -sum(l["sign"] * l["p"] for l in legs)
    return {
        "pop": pop(legs, spot, iv, dte) * 100,
        "maxp": hi * lot,
        "maxl": 0 if lo > 0 else -lo * lot,
        "floor": lo * lot,
        "credit": credit * lot,
    }


def name(l):
    return f'{"BUY " if l["sign"] > 0 else "SELL"} {l["k"]:g} {l["typ"]}'


def fmt(m):
    mp = "Unlimited" if m["maxp"] == math.inf else f'₹{m["maxp"]:,.2f}'
    ml = "Unlimited" if m["maxl"] == math.inf else f'₹{m["maxl"]:,.2f}'
    return (
        f'Prob. of Profit: {m["pop"]:.2f}%\nMax Profit: {mp}\nMax Loss: {ml}\n'
        f'Net Credit: ₹{m["credit"]:,.2f}\nMin Profit (chart base): ₹{m["floor"]:,.2f}'
    )


def label(legs):
    c = sorted([l for l in legs if l["typ"] == "CE"], key=lambda l: l["k"])
    p = sorted([l for l in legs if l["typ"] == "PE"], key=lambda l: l["k"])
    if (
        len(c) == 2
        and len(p) == 2
        and c[0]["k"] == p[0]["k"]
        and c[1]["k"] == p[1]["k"]
    ):
        if (
            c[0]["sign"] == -p[0]["sign"]
            and c[1]["sign"] == -p[1]["sign"]
            and c[0]["sign"] == -c[1]["sign"]
        ):
            return "Box spread"
    return "4-leg" if len(legs) == 4 else "Credit/debit spread"


# ----------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------


def ask(prompt, cast=str, default=None, valid=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            val = cast(raw.upper() if valid else raw)
            if valid and val not in valid:
                raise ValueError
            if cast in (float, int) and val < 0:
                raise ValueError
            return val
        except ValueError:
            print("  Invalid input, try again.")


# ----------------------------------------------------------------------------------
# CSV loading
# ----------------------------------------------------------------------------------

ALIASES = {
    "strike": "strike",
    "strikeprice": "strike",
    "callltp": "cltp",
    "putltp": "pltp",
    "calloi": "coi",
    "putoi": "poi",
    "calliv": "civ",
    "putiv": "piv",
    "callbid": "cbid",
    "callask": "cask",
    "putbid": "pbid",
    "putask": "pask",
}


def load_csv(path, prices, min_oi, parity_tol):
    meta, lines = {}, []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                for k, v in re.findall(r"(\w+)\s*[=:]\s*([\d.]+)", line):
                    meta[k.lower()] = float(v)
            elif line.strip():
                lines.append(line)
    if not lines:
        sys.exit("CSV is empty.")
    rdr = csv.DictReader(lines)
    found = [re.sub("[^a-z]", "", h.lower()) for h in (rdr.fieldnames or [])]
    need = (
        ["strike", "cltp", "pltp"]
        if prices == "ltp"
        else ["strike", "cbid", "cask", "pbid", "pask"]
    )
    have = {ALIASES[h] for h in found if h in ALIASES}
    if not set(need) <= have:
        sys.exit(
            f"CSV is missing columns for --prices {prices}. Needed: {need}\n"
            f"Recognised in your file: {sorted(have)}\nYour headers: {rdr.fieldnames}"
        )
    rows, skipped = [], 0
    for r in rdr:
        d = {}
        for h, v in r.items():
            k = re.sub("[^a-z]", "", (h or "").lower())
            if k in ALIASES:
                d[ALIASES[k]] = v
        try:
            x = {
                k: float(str(v).replace(",", ""))
                for k, v in d.items()
                if str(v).strip() not in ("", "-")
            }
            if prices == "ltp":
                x["cb"] = x["cs"] = x["cltp"]
                x["pb"] = x["ps"] = x["pltp"]
            else:
                x["cb"], x["cs"], x["pb"], x["ps"] = (
                    x["cask"],
                    x["cbid"],
                    x["pask"],
                    x["pbid"],
                )
            if min(x["cb"], x["cs"], x["pb"], x["ps"]) <= 0 or "strike" not in x:
                raise ValueError
            if "cltp" not in x:
                x["cltp"], x["pltp"] = x["cb"], x["pb"]
        except (ValueError, KeyError):
            skipped += 1
            continue
        stale = x.get("civ", 1) == 0 or x.get("piv", 1) == 0
        lowoi = x.get("coi", 1e18) < min_oi or x.get("poi", 1e18) < min_oi
        if stale or lowoi:
            skipped += 1
            continue
        rows.append(x)
    if (
        parity_tol > 0 and rows
    ):  # drop strikes whose call/put prices disagree with the rest
        fw = [r["strike"] + r["cltp"] - r["pltp"] for r in rows]
        med = sorted(fw)[len(fw) // 2]
        keep = [r for r, f in zip(rows, fw) if abs(f - med) <= parity_tol]
        skipped += len(rows) - len(keep)
        rows = keep
    rows.sort(key=lambda r: r["strike"])
    return rows, skipped, meta


# ----------------------------------------------------------------------------------
# Scanner
# ----------------------------------------------------------------------------------


def leg_set(rows, typ):
    legs = []
    for r in rows:
        for sign in (1, -1):
            if typ == "CE":
                p = r["cb"] if sign > 0 else r["cs"]
            else:
                p = r["pb"] if sign > 0 else r["ps"]
            legs.append({"sign": sign, "k": r["strike"], "typ": typ, "p": p})
    return legs


def vec(l, grid):
    iv = (
        np.maximum(grid - l["k"], 0)
        if l["typ"] == "CE"
        else np.maximum(l["k"] - grid, 0)
    )
    return l["sign"] * (iv - l["p"])


def pairs(legs, skip_both_sell):
    out = []
    for a in range(len(legs)):
        for b in range(a + 1, len(legs)):
            if legs[a]["k"] == legs[b]["k"]:
                continue
            if skip_both_sell and legs[a]["sign"] < 0 and legs[b]["sign"] < 0:
                continue
            out.append((a, b))
    return out


def scan(rows, floor_ps, keep=3000):
    grid = np.array([0.0] + [r["strike"] for r in rows])
    cl, pl = leg_set(rows, "CE"), leg_set(rows, "PE")
    cp, pp = pairs(cl, True), pairs(pl, False)
    cv, pv = [vec(l, grid) for l in cl], [vec(l, grid) for l in pl]
    C = np.array([cv[a] + cv[b] for a, b in cp])
    P = np.array([pv[a] + pv[b] for a, b in pp])
    hits = []
    for i in range(len(C)):
        m = (C[i][None, :] + P).min(axis=1)
        for j in np.nonzero(m > floor_ps + 1e-9)[0]:
            hits.append(
                (m[j], [cl[cp[i][0]], cl[cp[i][1]], pl[pp[j][0]], pl[pp[j][1]]])
            )
        if len(hits) > keep * 4:
            hits.sort(key=lambda h: -h[0])
            hits = hits[:keep]
    for i, (a, b) in enumerate(cp):
        if C[i].min() > floor_ps + 1e-9:
            hits.append((C[i].min(), [cl[a], cl[b]]))
    for j, (a, b) in enumerate(pp):
        if P[j].min() > floor_ps + 1e-9:
            hits.append((P[j].min(), [pl[a], pl[b]]))
    hits.sort(key=lambda h: -h[0])
    return hits[:keep], len(C) * len(P)


def scan_main(argv):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--spot", type=float)
    ap.add_argument("--lot", type=int)
    ap.add_argument("--iv", type=float)
    ap.add_argument("--dte", type=float)
    ap.add_argument("--prices", choices=["ltp", "bidask"], default="ltp")
    ap.add_argument("--min-oi", type=float, default=0.0)
    ap.add_argument("--parity-tol", type=float, default=5.0)
    ap.add_argument("--floor-rs", type=float, default=0.0)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)
    path = a.csv or ask("Path to option chain CSV", str)

    rows, skipped, meta = load_csv(path, a.prices, a.min_oi, a.parity_tol)
    spot = a.spot or meta.get("spot") or ask("Spot price", float)
    lot = (
        a.lot or int(meta["lot"]) if (a.lot or "lot" in meta) else ask("Lot size", int)
    )
    iv = a.iv or meta.get("iv") or ask("IV (%)", float)
    dte = a.dte or meta.get("dte") or ask("DTE (days to expiry)", float)

    print(
        f"\nStrikes used: {len(rows)}  (skipped {skipped}: missing/zero price, IV=0, low OI, or parity mismatch)"
    )
    if len(rows) < 2:
        sys.exit("Not enough usable strikes.")
    hits, n = scan(rows, a.floor_rs / lot)
    print(
        f"Combinations tested: {n:,} four-leg + all vertical spreads  [prices: {a.prices}]"
    )
    if not hits:
        print("\nNo all-green combination exists at these prices.")
        return
    res = [(metrics(legs, lot, spot, iv, dte), legs) for _, legs in hits[:300]]
    res.sort(key=lambda r: (-r[0]["pop"], -r[0]["floor"]))
    print(
        f"All-green combinations found: {len(hits):,}+  (top {min(a.top, len(res))} by POP, then chart base)\n"
    )
    for i, (m, legs) in enumerate(res[: a.top], 1):
        print(f"#{i}  [{label(legs)}]")
        for l in legs:
            print(
                f"  {name(l)} @ {l['p']:.2f}" + ("  (0)" if a.prices == "ltp" else "")
            )
        print(fmt(m))
        print()


# ----------------------------------------------------------------------------------
# Solve mode: your own 4 legs -> price one leg must have for a green payoff
# ----------------------------------------------------------------------------------


def ask_leg(n):
    print(f"\n--- Leg {n} ---")
    print(
        "Type the whole leg on one line (e.g. B 320 CE 2.4), or press Enter to answer step by step."
    )
    while True:
        raw = input("Leg: ").strip()
        if raw == "":
            break
        try:
            side, k, typ, p = raw.split()
            if (
                side.upper() not in ("B", "S")
                or typ.upper() not in ("CE", "PE")
                or float(k) < 0
                or float(p) < 0
            ):
                raise ValueError
            return {
                "sign": 1 if side.upper() == "B" else -1,
                "k": float(k),
                "typ": typ.upper(),
                "p": float(p),
            }
        except ValueError:
            print("  Invalid. Format: B/S strike CE/PE premium, e.g. S 325 CE 9.3")
    side = ask("Buy or Sell (B/S)", valid=("B", "S"))
    return {
        "sign": 1 if side == "B" else -1,
        "k": ask("Strike", float),
        "typ": ask("Option type (CE/PE)", valid=("CE", "PE")),
        "p": ask("Entry price (premium)", float),
    }


def solve_leg(legs, idx, floor, tick):
    lo, _ = analyse(legs)
    if lo == -math.inf:
        return None
    shift = floor - lo if lo < floor else 0.0
    new = [dict(x) for x in legs]
    newp = legs[idx]["p"] - legs[idx]["sign"] * shift
    newp = (
        math.ceil(newp / tick - 1e-9)
        if legs[idx]["sign"] < 0
        else math.floor(newp / tick + 1e-9)
    ) * tick
    if newp < 0:
        return None
    new[idx]["p"] = round(newp, 2)
    return new if analyse(new)[0] > 0 else None


def solve_main():
    print("Enter the 4 legs (as in the Edit Position screens):")
    legs = [ask_leg(i) for i in range(1, 5)]
    print("\n--- Market data ---")
    spot, lot = ask("Spot price", float), ask("Lot size", int)
    iv, dte = ask("IV (%)", float), ask("DTE (days to expiry)", float)
    print("\n--- Solver settings (press Enter for default) ---")
    sells = [i + 1 for i, l in enumerate(legs) if l["sign"] < 0]
    adj = 0
    while adj not in (1, 2, 3, 4):
        adj = ask("Leg number to modify (1-4)", int, default=sells[0] if sells else 1)
    floor = (
        ask("Minimum guaranteed profit in ₹ (chart base height)", float, default=2000.0)
        / lot
    )
    print("\nCURRENT\n" + fmt(metrics(legs, lot, spot, iv, dte)))
    new = solve_leg(legs, adj - 1, floor, 0.05)
    print("\nREQUIRED (all-green) - 4 LEGS")
    if new is None:
        print(f"Not solvable by changing leg {adj} only. Try another leg.")
        return
    print("Entry:")
    for o, n in zip(legs, new):
        print(
            f"  {name(n)} @ {n['p']:.2f}"
            + (f"  (was {o['p']:.2f})" if o["p"] != n["p"] else "")
        )
    print(fmt(metrics(new, lot, spot, iv, dte)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "solve":
        solve_main()
    else:
        scan_main(sys.argv[1:])
