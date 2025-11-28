param(
    # Fixed global settings you usually don’t touch
    [string]$Topology = "quat_sym",
    [string]$Scheme   = "explicit",
    [float] $Gamma    = 0.102,
    [switch]$DegNorm,
    [switch]$Directed
)

# --- EDIT THESE LISTS TO SWEEP ---------------------------------------
# Example: 3 values of dt, 3 values of eps_asym, 2 values of w_diag, 2 of sigma2
$dtList      = @(0.096, 0.098, 0.100)
$epsList     = @(0.026, 0.028, 0.030)      # eps_asym
$wDiagList   = @(2.55, 2.60)               # w_diag
$sigma2List  = @(-0.10, -0.08)            # sigma2
$triadPhi    = 2.0943951024               # usually fixed

# K-scan and NS / Lyapunov settings (also safe to edit)
$Kmin        = 0.00013
$Kmax        = 5.61
$Kpts        = 20000
$angTol      = 0.08
$near2       = 0.03

$doPostNS    = $true
$deltaK      = 0.0005
$T           = 200000
$burn        = 100000
$noise       = 1e-6

$doLyap      = $true
$le_q        = 6
# ----------------------------------------------------------------------

# Location of Python & script
$python = "python"
$script = "rtg_k4_quat_scout_v3.py"

# Create a root directory for this sweep
$rootOut = "k4_sweeps_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $rootOut | Out-Null

# Convenience flags
$degFlag = $null
if ($DegNorm) { $degFlag = "--deg_norm" }

$dirFlag = $null
if ($Directed) { $dirFlag = "--directed" }

foreach ($dt in $dtList) {
    foreach ($eps in $epsList) {
        foreach ($wDiag in $wDiagList) {
            foreach ($sig2 in $sigma2List) {

                # Tag & output folder for this combo
                $tag = "dt${dt}_eps${eps}_w${wDiag}_s2${sig2}"
                $outdir = Join-Path $rootOut $tag
                New-Item -ItemType Directory -Path $outdir | Out-Null

                Write-Host "=== Running sweep: $tag ==="

                # Build argument list
                $args = @(
                    "--topology", $Topology,
                    "--scheme",   $Scheme,
                    "--dt",       $dt,
                    "--gamma",    $Gamma
                )

                if ($DegNorm)  { $args += $degFlag }
                if ($Directed) { $args += $dirFlag }

                $args += @(
                    "--eps_asym",  $eps,
                    "--w_diag",    $wDiag,
                    "--triad_phi", $triadPhi,
                    "--sigma2",    $sig2,

                    "--K_min",  $Kmin,
                    "--K_max",  $Kmax,
                    "--K_pts",  $Kpts,
                    "--ang_tol", $angTol,
                    "--near2",   $near2,

                    "--deltaK", $deltaK,
                    "--T",      $T,
                    "--burn",   $burn,
                    "--noise",  $noise,
                    "--le_q",   $le_q,
                    "--outdir", $outdir
                )

                if ($doPostNS) { $args += "--post_ns" }
                if ($doLyap)   { $args += "--lyap_demo" }

                # Run and tee full stdout to a log file
                $logPath = Join-Path $outdir "run.log"
                & $python $script @args 2>&1 | Tee-Object -FilePath $logPath

                Write-Host "   -> finished, results in $outdir"
                Write-Host ""
            }
        }
    }
}
