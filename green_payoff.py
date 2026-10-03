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

All-green payoff solver.

Input : 4 legs (B/S, strike, CE/PE, premium) + spot, lot size, IV, DTE
Output: modified entry premium that makes the expiry payoff strictly > 0
        everywhere, plus Prob. of Profit / Max Profit / Max Loss / Net Credit.

Usage:
  python green_payoff.py        # asks for every input (4 legs + spot, lot, IV, DTE)
  python green_payoff.py --spot 300.6 --lot 1975 --iv 27.19 --dte 24 \
      "S 325 CE 1.75" "B 340 CE 0.6" "B 295 PE 5.25" "S 310 PE 12.55"   # no prompts

Options:
  --adjust N   1-based leg whose premium is changed in the 4-leg result
               (default: first SELL leg, as in your screenshot)
  --floor X    minimum guaranteed profit PER SHARE (chart base). Interactive mode asks in ₹.
  --tick X     price tick for rounding (default 0.05)
"""
import argparse
import math
from itertools import combinations

DEFAULT = ["S 325 CE 1.75", "B 340 CE 0.6", "B 295 PE 5.25", "S 310 PE 12.55"]


def parse_leg(s):
    side, k, typ, p = s.split()
    return {
        "sign": 1 if side.upper() == "B" else -1,
        "k": float(k),
        "typ": typ.upper(),
        "p": float(p),
    }


def pnl(legs, S):
    t = 0.0
    for l in legs:
        iv = max(S - l["k"], 0) if l["typ"] == "CE" else max(l["k"] - S, 0)
        t += l["sign"] * (iv - l["p"])
    return t


def tail_slope(legs):  # slope of payoff above the highest strike
    return sum(l["sign"] for l in legs if l["typ"] == "CE")


def analyse(legs):
    """Per-share: min pnl, max pnl, bounded?  (loss bounded on both sides)"""
    ks = sorted({l["k"] for l in legs})
    pts = [0.0] + ks
    vals = [pnl(legs, x) for x in pts]
    ts = tail_slope(legs)
    lo, hi = min(vals), max(vals)
    if ts < 0:
        lo = -math.inf
    if ts > 0:
        hi = math.inf
    return lo, hi


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def pop(legs, spot, iv, dte):
    sd = iv / 100 * math.sqrt(dte / 365)
    cdf = lambda x: (
        0.0
        if x <= 0
        else (1.0 if x == math.inf else ncdf((math.log(x / spot) + 0.5 * sd * sd) / sd))
    )
    ks = sorted({l["k"] for l in legs})
    pts = [0.0] + ks
    segs = [
        (pts[i], pts[i + 1], pnl(legs, pts[i]), pnl(legs, pts[i + 1]))
        for i in range(len(pts) - 1)
    ]
    last = pts[-1]
    ts = tail_slope(legs)
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
                prob += 1 - cdf(a + (-ya) / ts) if ts > 0 else 0
            elif ya > 0 and yb == -math.inf:
                prob += cdf(a + ya / -ts) - cdf(a)
            continue
        if ya > 0 and yb > 0:
            prob += cdf(b) - cdf(a)
        elif ya > 0 >= yb or yb > 0 >= ya:
            x = a + (b - a) * (0 - ya) / (yb - ya)
            prob += (cdf(x) - cdf(a)) if ya > 0 else (cdf(b) - cdf(x))
    return min(prob, 1.0)


def solve(legs, idx, floor, tick):
    """Change premium of legs[idx] so min payoff >= floor. Returns new legs or None."""
    lo, _ = analyse(legs)
    if lo == -math.inf:
        return None
    shift = floor - lo if lo < floor else 0.0
    l = legs[idx]
    new = [dict(x) for x in legs]
    # pnl shifts by -sign*dp  ->  dp = -sign*shift
    dp = -l["sign"] * shift
    newp = l["p"] + dp
    newp = (
        math.ceil(newp / tick - 1e-9)
        if l["sign"] < 0
        else math.floor(newp / tick + 1e-9)
    ) * tick
    if newp < 0:
        return None
    new[idx]["p"] = round(newp, 2)
    return new if analyse(new)[0] > 0 else None


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
        f'Prob. of Profit: {m["pop"]:.2f}%\nMax Profit: {mp}\n'
        f'Max Loss: {ml}\nNet Credit: ₹{m["credit"]:,.2f}\n'
        f'Min Profit (chart base): ₹{m["floor"]:,.2f}'
    )


def ask(prompt, cast=str, default=None, valid=None):
    """Prompt until a valid answer is given. Enter accepts the default (if any)."""
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
            leg = parse_leg(raw)
            if (
                raw.split()[0].upper() not in ("B", "S")
                or leg["typ"] not in ("CE", "PE")
                or leg["k"] < 0
                or leg["p"] < 0
            ):
                raise ValueError
            return leg
        except (ValueError, IndexError):
            print("  Invalid. Format: B/S strike CE/PE premium, e.g. S 325 CE 9.3")
    side = ask("Buy or Sell (B/S)", valid=("B", "S"))
    strike = ask("Strike", float)
    typ = ask("Option type (CE/PE)", valid=("CE", "PE"))
    prem = ask("Entry price (premium)", float)
    return {"sign": 1 if side == "B" else -1, "k": strike, "typ": typ, "p": prem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legs", nargs="*")
    ap.add_argument("--spot", type=float)
    ap.add_argument("--lot", type=int)
    ap.add_argument("--iv", type=float)
    ap.add_argument("--dte", type=float)
    ap.add_argument("--adjust", type=int, default=0)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--tick", type=float, default=0.05)
    a = ap.parse_args()

    if a.legs:  # non-interactive mode
        legs = [parse_leg(s) for s in a.legs]
        if None in (a.spot, a.lot, a.iv, a.dte):
            ap.error(
                "--spot, --lot, --iv and --dte are required with legs on the command line"
            )
    else:  # interactive mode
        print("Enter the 4 legs (as in the Edit Position screens):")
        legs = [ask_leg(i) for i in range(1, 5)]
        print("\n--- Market data ---")
        a.spot = ask("Spot price", float)
        a.lot = ask("Lot size", int)
        a.iv = ask("IV (%)", float)
        a.dte = ask("DTE (days to expiry)", float)
        print("\n--- Solver settings (press Enter for default) ---")
        sells = [i + 1 for i, l in enumerate(legs) if l["sign"] < 0]
        a.adjust = 0
        while a.adjust not in (1, 2, 3, 4):
            a.adjust = ask(
                "Leg number to modify (1-4)", int, default=sells[0] if sells else 1
            )
        rs = ask(
            "Minimum guaranteed profit in ₹ for the whole position (chart base height)",
            float,
            default=2000.0,
        )
        a.floor = rs / a.lot  # per-share equivalent
        print()

    # ---- current state
    print("CURRENT")
    print(fmt(metrics(legs, a.lot, a.spot, a.iv, a.dte)))

    # ---- full 4-leg solution
    idx = (
        a.adjust - 1
        if a.adjust
        else next(i for i, l in enumerate(legs) if l["sign"] < 0)
    )
    new = solve(legs, idx, a.floor, a.tick)
    print("\nREQUIRED (all-green) - 4 LEGS")
    if new is None:
        print(
            f"Not solvable by changing leg {idx + 1} only. Try --adjust with another leg."
        )
    else:
        print("Entry:")
        for o, n in zip(legs, new):
            ch = f'  (was {o["p"]:.2f})' if o["p"] != n["p"] else ""
            print(f"  {name(n)} @ {n['p']:.2f}{ch}")
        print(fmt(metrics(new, a.lot, a.spot, a.iv, a.dte)))

    # ---- other sub-structures, ranked by POP
    rows = []
    for r in (2,):
        for combo in combinations(range(len(legs)), r):
            sub = [legs[i] for i in combo]
            if sub[0]["typ"] != sub[1]["typ"] or sub[0]["sign"] == sub[1]["sign"]:
                continue  # only vertical spreads (same type, one buy one sell)
            if analyse(sub)[0] == -math.inf:
                continue
            j = next(
                (k for k, i in enumerate(combo) if i == idx),
                next((k for k, l in enumerate(sub) if l["sign"] < 0), 0),
            )
            s2 = solve(sub, j, a.floor, a.tick)
            if s2:
                rows.append((metrics(s2, a.lot, a.spot, a.iv, a.dte), s2, sub))
    rows.sort(key=lambda x: (-x[0]["pop"], -x[0]["maxp"]))
    if rows:
        print("\nOTHER 2-LEG STRUCTURES (ranked by POP)")
        for m, s2, sub in rows:
            legtxt = " | ".join(f'{name(n)} @ {n["p"]:.2f}' for n in s2)
            print(
                f'{legtxt}\n  POP {m["pop"]:.2f}% | MaxP ₹{m["maxp"]:,.2f} | '
                f'MaxL ₹{m["maxl"]:,.2f} | Net Credit ₹{m["credit"]:,.2f} | Base ₹{m["floor"]:,.2f}'
            )


if __name__ == "__main__":
    main()
