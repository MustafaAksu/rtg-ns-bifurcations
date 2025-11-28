# rtg_post_diag.py
# Run nonlinear diagnostics at K_demo or (K_star + deltaK) with PSD and Lyapunov.

from __future__ import annotations
import os, argparse, json
import numpy as np
from rtg_core import (
    Params, build_k3_spec, post_ns_demo, lyap_qr,
    ensure_dir, now_tag
)

def main():
    ap = argparse.ArgumentParser(description="Post-NS diagnostics for K3 model")
    ap.add_argument("--family", type=str, default="inertial")
    ap.add_argument("--scheme", type=str, default="explicit", choices=["explicit","ec"])
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--eps_asym", type=float, default=0.2)
    ap.add_argument("--deg_norm", action="store_true")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--K_demo", type=float, help="Use this K directly")
    group.add_argument("--K_star", type=float, help="Use K_star + deltaK")

    ap.add_argument("--deltaK", type=float, default=0.03)
    ap.add_argument("--T", type=int, default=60000)
    ap.add_argument("--burn", type=int, default=30000)
    ap.add_argument("--noise", type=float, default=0.0)

    ap.add_argument("--lyap_demo", action="store_true")
    ap.add_argument("--le_q", type=int, default=1)

    ap.add_argument("--outdir", type=str, default=None)

    args = ap.parse_args()
    p = Params(
        family=args.family, scheme=args.scheme,
        dt=args.dt, gamma=args.gamma, directed=args.directed,
        eps_asym=args.eps_asym, deg_norm=args.deg_norm
    )

    spec = build_k3_spec(p)
    outdir = args.outdir or os.path.join(f"figs_post_{p.scheme}_{now_tag()}")
    ensure_dir(outdir)

    K_run = args.K_demo if args.K_demo is not None else (args.K_star + args.deltaK)
    diag = post_ns_demo(K_run - args.deltaK, p, spec, outdir, deltaK=args.deltaK,
                        T=args.T, burn=args.burn, noise=args.noise)
    print(f"[post] run at K={diag['K_run']:.6f} | r_mean={diag['r_mean']:.6f} | r_std={diag['r_std']:.3e} | f_peak_per_step={diag['f_peak_per_step']:.6f}")

    with open(os.path.join(outdir, "ns_post_diag.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    if args.lyap_demo:
        les = lyap_qr(K_run, p, spec, T=args.T, burn=args.burn, q=max(1,args.le_q), noise=args.noise)
        with open(os.path.join(outdir, "lyap_summary.txt"), "w", encoding="utf-8") as f:
            f.write("Largest Lyapunov exponents (per step):\n")
            for i, le in enumerate(les):
                f.write(f"  LE_{i+1} ~= {le:.6e}\n")
        print("Lyapunov summary written.")

if __name__ == "__main__":
    main()
