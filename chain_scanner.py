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

Option-chain scanner: finds 4-leg (2 calls + 2 puts, incl. box spreads) and 2-leg
vertical spreads where EVERY leg is entered at its real market price (LTP), so all
leg brackets in Opstra show (0), and the expiry payoff is strictly green (> floor).

CSV columns (any order/case; extra columns ignored):
  strike, call_ltp, put_ltp   (required)
  call_oi, put_oi             (optional, used by --min-oi)
  call_iv, put_iv             (optional; rows with IV = 0 are treated as stale and skipped)

Usage:
  python chain_scanner.py                      # asks for everything
  python chain_scanner.py --csv chain.csv --spot 22550 --lot 75 --iv 14 --dte 24 \
        --min-oi 100000 --floor-rs 0 --top 5
"""
import argparse
import csv
import re
import sys

import numpy as np

from green_payoff import ask, fmt, metrics, name

ALIASES = {
    "strike": "strike",
    "strikeprice": "strike",
    "callltp": "cltp",
    "putltp": "pltp",
    "calloi": "coi",
    "putoi": "poi",
    "calliv": "civ",
    "putiv": "piv",
}


def load(path, min_oi, parity_tol=0.0):
    rows, skipped = [], 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = {
                ALIASES[k]: v
                for h, v in r.items()
                if (k := re.sub("[^a-z]", "", h.lower())) in ALIASES
            }
            try:
                x = {
                    k: float(str(v).replace(",", ""))
                    for k, v in d.items()
                    if str(v).strip() != ""
                }
                if x["cltp"] <= 0 or x["pltp"] <= 0:
                    raise ValueError
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
    ):  # drop rows whose call/put LTPs disagree with the rest (stale)
        fw = [r["strike"] + r["cltp"] - r["pltp"] for r in rows]
        med = sorted(fw)[len(fw) // 2]
        keep = [r for r, f in zip(rows, fw) if abs(f - med) <= parity_tol]
        skipped += len(rows) - len(keep)
        rows = keep
    rows.sort(key=lambda r: r["strike"])
    return rows, skipped


def leg_set(rows, typ):
    legs = []
    for r in rows:
        for sign in (1, -1):
            legs.append(
                {
                    "sign": sign,
                    "k": r["strike"],
                    "typ": typ,
                    "p": r["cltp"] if typ == "CE" else r["pltp"],
                }
            )
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
                continue  # two short calls = unlimited upside loss
            out.append((a, b))
    return out


def label(legs):
    c = sorted([l for l in legs if l["typ"] == "CE"], key=lambda l: l["k"])
    p = sorted([l for l in legs if l["typ"] == "PE"], key=lambda l: l["k"])
    if (
        len(c) == 2
        and len(p) == 2
        and c[0]["k"] == p[0]["k"]
        and c[1]["k"] == p[1]["k"]
    ):
        # box = synthetic long at one strike + synthetic short at the other (either direction)
        if (
            c[0]["sign"] == -p[0]["sign"]
            and c[1]["sign"] == -p[1]["sign"]
            and c[0]["sign"] == -c[1]["sign"]
        ):
            return "Box spread"
    return f"{len(legs)}-leg" if len(legs) == 4 else "Credit/debit spread"


def scan(rows, floor_ps, keep=3000):
    ks = [r["strike"] for r in rows]
    grid = np.array([0.0] + ks)
    cl, pl = leg_set(rows, "CE"), leg_set(rows, "PE")
    cp, pp = pairs(cl, True), pairs(pl, False)
    cv = [vec(l, grid) for l in cl]
    pv = [vec(l, grid) for l in pl]
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
    # 2-leg verticals
    for i, (a, b) in enumerate(cp):
        if C[i].min() > floor_ps + 1e-9:
            hits.append((C[i].min(), [cl[a], cl[b]]))
    for j, (a, b) in enumerate(pp):
        if P[j].min() > floor_ps + 1e-9:
            hits.append((P[j].min(), [pl[a], pl[b]]))
    hits.sort(key=lambda h: -h[0])
    return hits[:keep], len(C) * len(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--spot", type=float)
    ap.add_argument("--lot", type=int)
    ap.add_argument("--iv", type=float)
    ap.add_argument("--dte", type=float)
    ap.add_argument("--min-oi", type=float, default=None)
    ap.add_argument("--floor-rs", type=float, default=None)
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument(
        "--parity-tol",
        type=float,
        default=None,
        help="drop strikes whose put-call parity is off by more than this many points (0=off)",
    )
    a = ap.parse_args()
    if not a.csv:
        a.csv = ask("Path to option chain CSV", str)
        a.spot = ask("Spot price", float)
        a.lot = ask("Lot size", int)
        a.iv = ask("IV (%)", float)
        a.dte = ask("DTE (days to expiry)", float)
        a.min_oi = ask(
            "Minimum open interest per strike (0 = no filter)", float, default=0.0
        )
        a.floor_rs = ask(
            "Minimum guaranteed profit in ₹ (chart base)", float, default=0.0
        )
        a.parity_tol = ask(
            "Stale-price filter: max put-call parity gap in points (0 = off)",
            float,
            default=5.0,
        )
        a.top = ask("How many results to show", int, default=5)
    a.min_oi = a.min_oi or 0.0
    a.floor_rs = a.floor_rs or 0.0
    a.top = a.top or 5
    a.parity_tol = 0.0 if a.parity_tol is None else a.parity_tol
    if None in (a.spot, a.lot, a.iv, a.dte):
        sys.exit("--spot, --lot, --iv, --dte are required")

    rows, skipped = load(a.csv, a.min_oi, a.parity_tol)
    print(
        f"\nStrikes used: {len(rows)}  (skipped {skipped}: stale/zero price, IV=0 or low OI)"
    )
    if len(rows) < 2:
        sys.exit("Not enough usable strikes.")
    hits, n = scan(rows, a.floor_rs / a.lot)
    print(f"Combinations tested: {n:,} (4-leg) + all vertical spreads")
    if not hits:
        print("\nNo all-green combination exists at these market prices.")
        return
    res = []
    for fl, legs in hits[:300]:
        m = metrics(legs, a.lot, a.spot, a.iv, a.dte)
        res.append((m, legs))
    res.sort(key=lambda r: (-r[0]["pop"], -r[0]["floor"]))
    print(
        f"All-green combinations found: {len(hits):,}+  (showing top {min(a.top, len(res))} by POP, then chart base)\n"
    )
    for n_, (m, legs) in enumerate(res[: a.top], 1):
        print(f"#{n_}  [{label(legs)}]")
        for l in legs:
            print(f"  {name(l)} @ {l['p']:.2f}  (0)")
        print(fmt(m))
        print()


if __name__ == "__main__":
    main()
