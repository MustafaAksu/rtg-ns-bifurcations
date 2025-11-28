# rtg_ns_scan.py
# Sweep -> bracket NS -> refine -> (optional) post-NS + Lyapunov.

from __future__ import annotations
import os, argparse, json, math
import numpy as np
from rtg_core import (
    Params, build_k3_spec, scan_sweep, find_ns_bracket, refine_ns,
    drho_dK_numeric, post_ns_demo, largest_lyap, lyap_qr,
    ensure_dir, now_tag, plot_angle_vs_K, plot_grid_overview
)

def main():
    ap = argparse.ArgumentParser(description="RTG NS scan (K3 model)")
    ap.add_argument("--family", type=str, default="inertial")
    ap.add_argument("--scheme", type=str, default="explicit", choices=["explicit","ec"])
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--eps_asym", type=float, default=0.2)
    ap.add_argument("--deg_norm", action="store_true")
    ap.add_argument("--Delta", type=float, nargs="+", default=[0.01,0.0,-0.01])

    ap.add_argument("--K_min", type=float, default=1.0)
    ap.add_argument("--K_max", type=float, default=6.0)
    ap.add_argument("--K_pts", type=int, default=80)
    ap.add_argument("--ang_tol", type=float, default=0.10)
    ap.add_argument("--outdir", type=str, default=None)

    ap.add_argument("--post_ns", action="store_true")
    ap.add_argument("--deltaK", type=float, default=0.03)
    ap.add_argument("--T_diag", type=int, default=60000)
    ap.add_argument("--burn_diag", type=int, default=30000)
    ap.add_argument("--noise", type=float, default=0.0)

    ap.add_argument("--lyap_demo", action="store_true")
    ap.add_argument("--le_q", type=int, default=1)

    args = ap.parse_args()
    p = Params(
        family=args.family, scheme=args.scheme,
        dt=args.dt, gamma=args.gamma,
        directed=args.directed, eps_asym=args.eps_asym,
        deg_norm=args.deg_norm, Delta=np.array(args.Delta, dtype=float),
        K_min=args.K_min, K_max=args.K_max, K_pts=args.K_pts,
        ang_tol=args.ang_tol, outdir=args.outdir,
        post_ns=args.post_ns, deltaK=args.deltaK,
        T_diag=args.T_diag, burn_diag=args.burn_diag,
        noise=args.noise, lyap_demo=args.lyap_demo, le_q=args.le_q
    )

    spec = build_k3_spec(p)
    outdir = p.outdir or os.path.join(f"figs_ns_scan_{p.scheme}_{now_tag()}")
    ensure_dir(outdir)

    # Theory hint
    A = 1.0 - p.gamma*p.dt
    if p.scheme == "ec":
        print(f"[theory] Inertial (Euler-Cromer): complex-pair modulus tends to sqrt(1 - gamma*dt) = {math.sqrt(max(0.0,A)):.6f}; NS crossing often suppressed for gamma>0.")
    else:
        print("[theory] Inertial (Explicit): |lambda| depends on K via dt^2 * M; NS can occur when coupling induces outward radial motion.")

    print(f"Target family={p.family} | scheme={p.scheme} | dt={p.dt} | gamma={p.gamma} | directed={p.directed} eps={p.eps_asym} | deg_norm={p.deg_norm}")
    print(f"K range [{p.K_min}, {p.K_max}] with {p.K_pts} points; ang_tol={p.ang_tol:.3f} rad\n")

    # Sweep
    res = scan_sweep(p, spec)
    K, rho, ang, lbl = res["K"], res["rho"], res["ang"], res["label"]

    tag = f"{p.family}_{p.scheme}_dt{p.dt:.3f}" + ("_dir" if p.directed else "")
    plot_angle_vs_K(K, ang, outdir, tag)
    plot_grid_overview(K, rho, ang, lbl, outdir, tag)

    # Save CSV
    csv_path = os.path.join(outdir, "k_sweep.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("K,rho,angle,label\n")
        for i in range(len(K)):
            f.write(f"{K[i]:.9f},{rho[i]:.9f},{ang[i]:.9f},{lbl[i]}\n")
    print(f"[save] sweep -> {csv_path}")

    # Bracket → refine
    br = find_ns_bracket(K, rho, ang, lbl, ang_tol=p.ang_tol)
    if br is None:
        print("No NS bracket found on this grid. Either a flip (real pi) precedes NS, or the complex pair stays within |lambda|<1 across the window.")
        return

    K_lo, K_hi = br
    print(f"NS-like crossing bracketed in [{K_lo:.6f}, {K_hi:.6f}] ... refining")
    K_star, r_star, a_star, lab_star = refine_ns(K_lo, K_hi, p, spec, iters=32)
    print(f"Refine result: K*={K_star:.9f} | rho=1.000000 | angle={a_star:+.6f} rad  [{lab_star}]")

    dR = drho_dK_numeric(K_star, p, spec, h=1e-3)
    print(f"[check] d|lambda|/dK at K* ~= {dR:.3e}")

    with open(os.path.join(outdir, "ns_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "family": p.family, "scheme": p.scheme, "dt": p.dt, "gamma": p.gamma,
            "directed": p.directed, "eps_asym": p.eps_asym, "deg_norm": p.deg_norm,
            "K_lo": K_lo, "K_hi": K_hi, "K_star": K_star,
            "rho_star": float(r_star), "angle_star": float(a_star),
            "drho_dK_at_star": float(dR)
        }, f, indent=2)

    # Post-NS
    if p.post_ns:
        diag = post_ns_demo(K_star, p, spec, outdir, p.deltaK, p.T_diag, p.burn_diag, p.noise)
        print(f"[confirm] post-NS at K={diag['K_run']:.6f}: r_mean={diag['r_mean']:.4f}, r_std={diag['r_std']:.4e}, f_peak_per_step={diag['f_peak_per_step']:.5f}")
        with open(os.path.join(outdir, "ns_post_diag.json"), "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)

    if p.lyap_demo:
        les = lyap_qr(K_star + p.deltaK, p, spec, T=max(20000,p.T_diag//2), burn=max(5000,p.burn_diag//2),
                      q=max(1,p.le_q), noise=p.noise)
        print("Lyapunov (per step):", ", ".join([f"{x:.6e}" for x in les]))
        with open(os.path.join(outdir, "lyap_summary.txt"), "w", encoding="utf-8") as f:
            f.write("Largest Lyapunov exponents (per step):\n")
            for i, le in enumerate(les):
                f.write(f"  LE_{i+1} ~= {le:.6e}\n")

if __name__ == "__main__":
    main()
