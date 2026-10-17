"""Every headline number in README.md, where it comes from, and whether a run reproduces it.

  python -I scripts/headline.py <produced dir> [--expected public/data] [--readme README.md] [--md out.md]

For each row: read the value at <file> -> <key path> in the produced dir, round it the way the README states it,
and check (1) it equals the README's figure, (2) that figure appears in README.md, and (3) the raw value equals the
committed one in --expected. Exits 1 if any row fails. Also used to build the table in REPRODUCE.md.
"""
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

# (README claim, file, key path, decimals, absolute value?, README figure)
ROWS = [
    ("Sensor wearers in the program", "ml_effects.json", "cohort.walk.people", 0, False, "500"),
    ("Clean meal curves", "ml_effects.json", "cohort.walk.rows", 0, False, "18,218"),
    ("Walk after dinner: rise lower by", "ml_effects.json", "cohort.walkAfterDinner.est", 1, True, "18.3"),
    ("  95% CI bound (near)", "ml_effects.json", "cohort.walkAfterDinner.hi", 1, True, "16.6"),
    ("  95% CI bound (far)", "ml_effects.json", "cohort.walkAfterDinner.lo", 1, True, "20.1"),
    ("Walk after other meals", "ml_effects.json", "cohort.walkAfterOtherMeals.est", 1, True, "14.6"),
    ("Walk, all meals (double ML)", "ml_effects.json", "cohort.walk.est", 1, True, "17.8"),
    ("Walk, all meals (regression cross-check)", "cohort.json", "statistics.effects.walk.est", 1, True, "17.6"),
    ("Walk, type 2 diabetes", "ml_effects.json", "byGroup.type 2 diabetes.walk.est", 1, True, "23.3"),
    ("Walk, normal blood sugar", "ml_effects.json", "byGroup.normal.walk.est", 1, True, "13.2"),
    ("Negative control (pre-meal sugar)", "ml_effects.json", "cohort.negativeControl.est", 2, False, "-0.35"),
    ("  CI low", "ml_effects.json", "cohort.negativeControl.lo", 2, False, "-1.38"),
    ("  CI high", "ml_effects.json", "cohort.negativeControl.hi", 2, False, "0.68"),
    ("E-value", "cohort.json", "robustness.eValue.estimate", 1, False, "3.9"),
    ("Walk 10-19 min", "ml_effects.json", "walkDose.1.est", 1, True, "17.5"),
    ("Walk 20-29 min", "ml_effects.json", "walkDose.2.est", 1, True, "19.5"),
    ("Walk 30+ min", "ml_effects.json", "walkDose.3.est", 1, True, "16.6"),
    ("Late dinner (8 pm or later)", "ml_effects.json", "cohort.lateDinner.est", 1, False, "9.2"),
    ("  CI low", "ml_effects.json", "cohort.lateDinner.lo", 1, False, "7.1"),
    ("  CI high", "ml_effects.json", "cohort.lateDinner.hi", 1, False, "11.3"),
    ("Dinner 6-7 pm vs before 6", "ml_effects.json", "dinnerHour.1.est", 1, False, "3.9"),
    ("Dinner 7-8 pm", "ml_effects.json", "dinnerHour.2.est", 1, False, "8.4"),
    ("Dinner 8-9 pm", "ml_effects.json", "dinnerHour.3.est", 1, False, "13.3"),
    ("Dinner 9-10 pm", "ml_effects.json", "dinnerHour.4.est", 1, False, "18.2"),
    ("Dinner 10 pm or later", "ml_effects.json", "dinnerHour.5.est", 1, False, "25.4"),
    ("Fast vs coaching, HbA1c (randomized)", "cohort.json", "trial.contrasts.hba1c_pct.coaching + overnight fast vs coaching.est", 2, False, "-0.03"),
    ("  CI low", "cohort.json", "trial.contrasts.hba1c_pct.coaching + overnight fast vs coaching.lo", 2, False, "-0.06"),
    ("  CI high", "cohort.json", "trial.contrasts.hba1c_pct.coaching + overnight fast vs coaching.hi", 2, False, "0.00"),
    ("Fast effect, type 2 diabetes", "cohort.json", "trial.fastVsCoachingBySubgroup.hba1c_pct.blood sugar: type 2 diabetes.est", 2, False, "-0.08"),
    ("Interaction test Q p", "cohort.json", "trial.heterogeneity.hba1c_pct.p", 2, False, "0.15"),
    ("Holm-adjusted p (type 2 diabetes)", "cohort.json", "trial.fastVsCoachingBySubgroup.hba1c_pct.blood sugar: type 2 diabetes.pHolm", 2, False, "0.13"),
    ("Fast arm: hours fasted", "cohort.json", "trial.adherence.coaching + overnight fast.overnightFastHours", 2, False, "13.05"),
    ("Coaching arm: hours fasted", "cohort.json", "trial.adherence.coaching.overnightFastHours", 2, False, "12.07"),
    ("Fast arm: nights at 13 h+ (%)", "cohort.json", "trial.adherence.coaching + overnight fast.nightsAtLeast13hPct", 0, False, "62"),
    ("Coaching arm: nights at 13 h+ (%)", "cohort.json", "trial.adherence.coaching.nightsAtLeast13hPct", 0, False, "36"),
    ("Coaching vs email tips, HbA1c", "cohort.json", "trial.contrasts.hba1c_pct.coaching vs tips by email.est", 2, False, "-0.17"),
    ("CGMacros real adults", "realworld.json", "cgmacros.people", 0, False, "45"),
    ("CGMacros late dinner (double ML)", "ml_effects.json", "real.lateDinner.est", 1, False, "2.6"),
    ("  CI low", "ml_effects.json", "real.lateDinner.lo", 1, False, "-5.1"),
    ("  CI high", "ml_effects.json", "real.lateDinner.hi", 1, False, "10.3"),
    ("CGMacros walk after a meal (double ML)", "ml_effects.json", "real.walk10.est", 1, False, "-3.2"),
    ("  CI low", "ml_effects.json", "real.walk10.lo", 1, False, "-6.9"),
    ("  CI high", "ml_effects.json", "real.walk10.hi", 1, False, "0.5"),
    ("Dinner forecast: typical miss (MAE)", "ml_effects.json", "forecast.mae", 1, False, "14.0"),
    ("Dinner forecast: R²", "ml_effects.json", "forecast.r2", 2, False, "0.62"),
    ("Planted -10 effect, double ML error", "mlcompare.json", "effectBenchmark.nonlinear.Double ML (gradient boosting, shipped).bias", 1, False, "0.3"),
    ("Planted -10 effect, regression error", "mlcompare.json", "effectBenchmark.nonlinear.Fixed effects (statistics).bias", 1, False, "3.5"),
    ("Regression interval coverage (never)", "mlcompare.json", "effectBenchmark.nonlinear.Fixed effects (statistics).coverage95", 0, False, "0"),
    ("New people R², gradient boosting", "mlcompare.json", "predictProgram.splits.new people (5-fold).Gradient boosting.r2.est", 2, False, "0.44"),
    ("New people R², linear", "mlcompare.json", "predictProgram.splits.new people (5-fold).Linear, per-person intercept (fixed effects).r2.est", 2, False, "0.35"),
    ("Real data R², random forest", "mlcompare.json", "predictCgmacros.splits.new people (5-fold).Random forest.r2.est", 2, False, "0.30"),
    ("Real data R², linear", "mlcompare.json", "predictCgmacros.splits.new people (5-fold).Linear, per-person intercept (fixed effects).r2.est", 2, False, "0.11"),
]


def get(d, path):
    for k in path.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def get_path(d, path):
    """Key paths contain dots inside names ('5-fold).x'); resolve greedily against the actual keys."""
    parts = path.split(".")
    i = 0
    while i < len(parts):
        if isinstance(d, list):
            d, i = d[int(parts[i])], i + 1
            continue
        for j in range(len(parts), i, -1):
            k = ".".join(parts[i:j])
            if k in d:
                d, i = d[k], j
                break
        else:
            raise KeyError(".".join(parts[i:]))
    return d


def fmt(v, dp, absval):
    x = Decimal(str(abs(v) if absval else v)).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)
    s = f"{x:,}" if dp == 0 else str(x)
    return "0.00" if s == "-0.00" else s


def main():
    a = sys.argv[1:]
    prod = a[0]
    exp = a[a.index("--expected") + 1] if "--expected" in a else None
    readme = open(a[a.index("--readme") + 1]).read() if "--readme" in a else None
    cache, fails, lines = {}, 0, []
    lines.append("| README claim | File → key | Expected (README) | Produced | Matches committed JSON | Result |")
    lines.append("|---|---|---|---|---|---|")
    for label, f, path, dp, absval, want in ROWS:
        for root in filter(None, (prod, exp)):
            if (root, f) not in cache:
                p = os.path.join(root, f)
                cache[(root, f)] = json.load(open(p)) if os.path.exists(p) else None
        try:
            v = get_path(cache[(prod, f)], path)
            got = fmt(v, dp, absval)
        except (KeyError, TypeError, IndexError, ValueError) as e:
            v, got = None, f"missing ({e})"
        same = "n/a"
        if exp:
            try:
                c = get_path(cache[(exp, f)], path)
                # 1e-9 relative: BLAS on another CPU (x86 vs arm64) moves the last digits (seen: 1.3e-12 on one regression value)
                close = isinstance(c, (int, float)) and isinstance(v, (int, float)) and abs(c - v) <= 1e-9 * max(1.0, abs(c))
                same = "yes" if c == v else "yes (within 1e-9)" if close else f"no ({c})"
            except (KeyError, TypeError, IndexError):
                same = "no (not in committed)"
        in_readme = readme is None or want.lstrip("-") in readme
        ok = got == want and in_readme and same in ("yes", "yes (within 1e-9)", "n/a")
        fails += not ok
        lines.append(f"| {label} | `{f}` → `{path}` | {want} | {got} | {same} | {'PASS' if ok else 'FAIL' + ('' if in_readme else ' (figure not in README)')} |")
    lines.append("")
    lines.append(f"**{len(ROWS) - fails}/{len(ROWS)} headline numbers reproduced.**")
    out = "\n".join(lines)
    print(out)
    if "--md" in a:
        open(a[a.index("--md") + 1], "w").write(out + "\n")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
