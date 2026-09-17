# Bayesian dual-route models for the recognition / familiarity split (brms + cmdstan).
#
# Four variants, matching the frequentist analyses:
#
#   twofit   one model per recognition arm; the contrast is the posterior
#            difference of the two slope posteriors (computed downstream)
#   joint    single model, recognition as 0/1, WITH distributional terms so the
#            residual SD may differ by condition (sigma ~ recognized) — the
#            Bayesian analogue of the glmmTMB dispformula fit
#   fammean  moderator is the pair's MEAN FAMILIARITY (continuous, z-scored)
#            rather than binary recognition, fitted on the full sample
#   famhigh  same as fammean but restricted to pairs where BOTH targets were
#            rated above 5.  NOT ESTIMABLE FOR US: its familiarity item is
#            binary (1/7), so both>5 implies both==7 and the moderator has zero
#            variance.  CN only.
#
# Predictors are z-scored; the outcome stays in rating units, so coefficients
# are semi-standardised and comparable with the lme4/glmmTMB results.
#
# Usage: Rscript brms_recognition.R <pairs.csv> <variant> <out_prefix> [chains] [iter]

suppressPackageStartupMessages({
  library(brms)
  library(cmdstanr)
})
options(brms.backend = "cmdstanr", mc.cores = 4)

args    <- commandArgs(trailingOnly = TRUE)
inp     <- args[1]
variant <- args[2]
outpre  <- args[3]
chains  <- if (length(args) >= 4) as.integer(args[4]) else 4
iter    <- if (length(args) >= 5) as.integer(args[5]) else 1000
nthread <- if (length(args) >= 6) as.integer(args[6]) else 1

df <- read.csv(inp)
df$participant <- factor(df$participant)
df$target1 <- factor(df$target1)
df$target2 <- factor(df$target2)

rand <- "(1 | participant) + (1 | target1) + (1 | target2)"

# Weakly informative priors on the semi-standardised scale.
pri <- c(prior(normal(0, 1), class = "b"),
         prior(student_t(3, 0, 2.5), class = "sd"),
         prior(student_t(3, 0, 2.5), class = "sigma"))

fit_one <- function(d, form, prior, ...) {
  # Within-chain parallelism (reduce_sum) when nthread > 1.  The likelihood is
  # a sum over independent pair observations, so it partitions cleanly; the
  # posterior is unchanged, only the wall time.
  thr <- if (nthread > 1) threading(nthread) else NULL
  # output_dir makes cmdstan stream each chain's draws to CSV *as it samples*,
  # so a job killed mid-run still leaves usable partial draws on disk rather
  # than nothing.  Without it cmdstanr writes to a tempdir that is discarded.
  odir <- paste0(outpre, "_chains")
  dir.create(odir, showWarnings = FALSE, recursive = TRUE)
  brm(form, data = d, family = gaussian(), prior = prior,
      chains = chains, iter = iter, warmup = floor(iter / 2),
      cores = chains, seed = 42, threads = thr,
      control = list(adapt_delta = 0.9, max_treedepth = 12),
      silent = 1, refresh = 100,
      output_dir = odir, output_basename = basename(outpre), ...)
}

# Write the cheap numeric summary before the (large, slow) .rds, so results
# survive even if serialising the fit object fails or is interrupted.
save_fit <- function(m, pre) {
  write.csv(as.data.frame(posterior_summary(m)), paste0(pre, "_summary.csv"))
  saveRDS(m, paste0(pre, ".rds"))
}

if (variant %in% c("twofit_rec", "twofit_unrec")) {
  # One arm per invocation so the scheduler can run both concurrently.
  arm <- if (variant == "twofit_rec") 1 else 0
  lab <- if (arm == 1) "rec" else "unrec"
  f <- bf(as.formula(paste("abs_diff ~ sem_dissim_z + vis_dissim_z +", rand)))
  d <- df[df$recognized == arm, ]
  cat("== twofit", lab, "n =", nrow(d), "\n"); flush.console()
  m <- fit_one(d, f, pri)
  save_fit(m, outpre)
  # slope draws, so the two arms can be differenced downstream on the posterior
  dr <- as_draws_df(m)[, c("b_sem_dissim_z", "b_vis_dissim_z")]
  write.csv(dr, paste0(outpre, "_draws.csv"), row.names = FALSE)

} else if (variant == "joint") {
  # distributional model: residual SD free to differ between conditions
  f <- bf(as.formula(paste("abs_diff ~ sem_dissim_z * recognized +",
                           "vis_dissim_z * recognized +", rand)),
          sigma ~ recognized)
  pri_j <- c(prior(normal(0, 1), class = "b"),
             prior(student_t(3, 0, 2.5), class = "sd"),
             prior(normal(0, 1), class = "b", dpar = "sigma"))
  cat("== joint (sigma ~ recognized) n =", nrow(df), "\n"); flush.console()
  m <- fit_one(df, f, pri_j)
  save_fit(m, outpre)

} else if (variant %in% c("fammean", "famhigh")) {
  d <- df
  if (variant == "famhigh") d <- df[df$fam_min > 5, ]
  if (nrow(d) < 1000 || sd(d$fam_mean) < 1e-8) {
    stop("moderator has no variance in this subset — not estimable (expected for US)")
  }
  d$fam_mean_z <- as.numeric(scale(d$fam_mean))
  f <- bf(as.formula(paste("abs_diff ~ sem_dissim_z * fam_mean_z +",
                           "vis_dissim_z * fam_mean_z +", rand)))
  cat("==", variant, "n =", nrow(d),
      " fam_mean range", range(d$fam_mean), "\n"); flush.console()
  m <- fit_one(d, f, pri)
  save_fit(m, outpre)

} else {
  stop("unknown variant: ", variant)
}

cat("done ->", outpre, "\n")
