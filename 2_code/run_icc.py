"""ICC variance decomposition of trait ratings.

US: rating ~ 1 + (1|participant) + (1|target)
CN: rating ~ 1 + (1|participant) + (1|celeb_idx) + (1|module)

ICC_k = sigma2_k / sigma2_total for each grouping factor k.

Outputs
-------
3_pipeline/us/icc_results.csv
3_pipeline/cn/icc_results.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "2_code"))

from config import DATASETS, PIPELINE_DIR, local_common_traits
from data_loader import load_ratings
from lmm_utils import configure_r


def _fit_null(df, formula):
    from pymer4.models import Lmer
    import warnings
    warnings.filterwarnings("ignore")
    m = Lmer(formula, data=df)
    m.fit(summarize=False)
    return m.ranef_var  # DataFrame indexed by grouping factor


def _icc_row(rv, keys):
    """Return dict of var_ and icc_ columns from a ranef_var DataFrame."""
    vals = {k: float(rv.loc[k, "Var"]) if k in rv.index else 0.0 for k in keys}
    vals["Residual"] = float(rv.loc["Residual", "Var"])
    total = sum(vals.values())
    row = {f"var_{k.lower()}": v for k, v in vals.items()}
    row["var_total"] = total
    row.update({f"icc_{k.lower()}": (v / total if total > 0 else np.nan)
                for k, v in vals.items()})
    return row


def run_us():
    cfg = DATASETS["us"]
    ratings = load_ratings(cfg)

    vmodule_path = ROOT / '1_data' / 'us' / 'vmodule.npy'
    vmodule = np.load(vmodule_path)  # shape (415,), values 1-10

    rows = []
    for trait, arr in ratings.items():
        n_part, n_targ = arr.shape
        records = [
            {"participant": str(p), "target": str(t),
             "module": str(int(vmodule[p])), "rating": arr[p, t]}
            for p in range(n_part) for t in range(n_targ)
            if not np.isnan(arr[p, t])
        ]
        sub = pd.DataFrame(records)
        rv = _fit_null(sub, "rating ~ 1 + (1|participant) + (1|target) + (1|module)")
        row = {"dataset": "us", "trait": trait}
        row.update({"n_obs": len(sub),
                    "n_participants": sub["participant"].nunique(),
                    "n_targets": sub["target"].nunique()})
        row.update(_icc_row(rv, ["target", "participant", "module"]))
        rows.append(row)
        print(f"  US {trait:12s}  ICC_identity={row['icc_target']:.3f}  "
              f"ICC_participant={row['icc_participant']:.3f}  "
              f"ICC_module={row['icc_module']:.3f}")

    # rename target → identity for clarity in downstream code
    df = pd.DataFrame(rows).rename(columns={
        "var_target": "var_identity", "icc_target": "icc_identity"
    })
    return df


def run_cn():
    cfg = DATASETS["cn"]
    df_raw = pd.read_csv(cfg["ratings_file"])

    rows = []
    for trait in cfg["traits"]:
        sub = (df_raw[df_raw["session_trait"] == trait]
               [["participant", "celeb_idx", "module", "rating_clean"]]
               .dropna(subset=["rating_clean"]))
        sub = sub.astype({"participant": str, "celeb_idx": str, "module": str})
        rv = _fit_null(
            sub,
            "rating_clean ~ 1 + (1|participant) + (1|celeb_idx) + (1|module)"
        )
        row = {"dataset": "cn", "trait": trait}
        row.update({"n_obs": len(sub),
                    "n_participants": sub["participant"].nunique(),
                    "n_targets": sub["celeb_idx"].nunique()})
        row.update(_icc_row(rv, ["celeb_idx", "participant", "module"]))
        # rename celeb_idx → identity
        for suffix in ("var", "icc"):
            row[f"{suffix}_identity"] = row.pop(f"{suffix}_celeb_idx")
        rows.append(row)
        print(f"  CN {trait:12s}  ICC_identity={row['icc_identity']:.3f}  "
              f"ICC_participant={row['icc_participant']:.3f}  "
              f"ICC_module={row['icc_module']:.3f}")

    return pd.DataFrame(rows)


def run_us_trait_space():
    """Pooled null model across all traits (trait-space level)."""
    cfg = DATASETS["us"]
    ratings = load_ratings(cfg)
    traits = local_common_traits("us")

    vmodule = np.load(ROOT / '1_data' / 'us' / 'vmodule.npy')

    records = []
    for trait in traits:
        arr = ratings[trait]
        n_part, n_targ = arr.shape
        records += [
            {"participant": str(p), "target": str(t),
             "module": str(int(vmodule[p])), "rating": arr[p, t]}
            for p in range(n_part) for t in range(n_targ)
            if not np.isnan(arr[p, t])
        ]
    sub = pd.DataFrame(records)
    rv = _fit_null(sub, "rating ~ 1 + (1|participant) + (1|target) + (1|module)")
    row = {"dataset": "us", "trait": "trait_space"}
    row.update({"n_obs": len(sub),
                "n_participants": sub["participant"].nunique(),
                "n_targets": sub["target"].nunique(),
                "n_traits": len(traits)})
    row.update(_icc_row(rv, ["target", "participant", "module"]))
    print(f"  US trait_space  ICC_identity={row['icc_target']:.3f}  "
          f"ICC_participant={row['icc_participant']:.3f}  "
          f"ICC_module={row['icc_module']:.3f}")
    return pd.DataFrame([row]).rename(columns={
        "var_target": "var_identity", "icc_target": "icc_identity"
    })


def run_cn_trait_space():
    """Pooled null model across all traits (trait-space level)."""
    cfg = DATASETS["cn"]
    df_raw = pd.read_csv(cfg["ratings_file"])
    traits = local_common_traits("cn")

    sub = (df_raw[df_raw["session_trait"].isin(traits)]
           [["participant", "celeb_idx", "module", "rating_clean"]]
           .dropna(subset=["rating_clean"]))
    sub = sub.astype({"participant": str, "celeb_idx": str, "module": str})
    rv = _fit_null(
        sub,
        "rating_clean ~ 1 + (1|participant) + (1|celeb_idx) + (1|module)"
    )
    row = {"dataset": "cn", "trait": "trait_space"}
    row.update({"n_obs": len(sub),
                "n_participants": sub["participant"].nunique(),
                "n_targets": sub["celeb_idx"].nunique(),
                "n_traits": len(traits)})
    row.update(_icc_row(rv, ["celeb_idx", "participant", "module"]))
    for suffix in ("var", "icc"):
        row[f"{suffix}_identity"] = row.pop(f"{suffix}_celeb_idx")
    print(f"  CN trait_space  ICC_identity={row['icc_identity']:.3f}  "
          f"ICC_participant={row['icc_participant']:.3f}  "
          f"ICC_module={row['icc_module']:.3f}")
    return pd.DataFrame([row])


if __name__ == "__main__":
    configure_r()

    print("\n=== US ===")
    df_us = run_us()
    out_us = PIPELINE_DIR / "us" / "icc_results.csv"
    df_us.to_csv(out_us, index=False)
    print(f"Saved {out_us}")

    print("\n=== CN ===")
    df_cn = run_cn()
    out_cn = PIPELINE_DIR / "cn" / "icc_results.csv"
    df_cn.to_csv(out_cn, index=False)
    print(f"Saved {out_cn}")

    print("\n=== US trait space ===")
    df_us_ts = run_us_trait_space()
    out_us_ts = PIPELINE_DIR / "us" / "icc_trait_space_results.csv"
    df_us_ts.to_csv(out_us_ts, index=False)
    print(f"Saved {out_us_ts}")

    print("\n=== CN trait space ===")
    df_cn_ts = run_cn_trait_space()
    out_cn_ts = PIPELINE_DIR / "cn" / "icc_trait_space_results.csv"
    df_cn_ts.to_csv(out_cn_ts, index=False)
    print(f"Saved {out_cn_ts}")

    print("\nDone.")
