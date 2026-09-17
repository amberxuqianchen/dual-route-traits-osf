# Multi-membership maximal ladder (brms / cmdstan).
#
# Fixes the last structural objection to the glmmTMB maximal models: there,
# an identity appearing in both slots of a pair received TWO independent random
# effects, from two separate factors with separate variances.  Because pairs
# were built with triu ordering, identity 0 only ever appeared in slot 1 and
# identity 49 only in slot 2, and the same identity's familiarity slopes across
# slots correlated r = -.11.
#
# mm(target1, target2) draws both members from ONE pool: each identity has one
# effect, estimated from all of its pairs, and slot ordering stops existing.
# Note this is SIMPLER than the glmmTMB maximal model, not more complex --
# one 6-term covariance for identities (21 params) rather than two (42).
#
# Our covariates are pair-level (identical for both members of a row), so plain
# mm() suffices; mmc() would only be needed for member-varying covariates.
#
# Ladder, reduced identity-side first (same logic as the frequentist version):
#   B0  (full 6 | participant) + (full 6 | mm)
#   B1  (full 6 | participant) + (1 + FAM | mm)
#   B2  (1 + sem + vis + FAM | participant) + (1 + FAM | mm)
#   B3  (1 + FAM | participant) + (1 + FAM | mm)
#
# Bayesian analogue of "convergent and non-singular": all Rhat < 1.01, no
# divergent transitions, and bulk/tail ESS above 400.  Levels are compared by
# LOO rather than AIC.
#
# Usage: Rscript brms_multimembership_ladder.R <pairs.csv> <level> <out_prefix>
#                [chains] [iter] [threads] [subsample_n]

suppressPackageStartupMessages({library(brms); library(cmdstanr)})
options(brms.backend = "cmdstanr")

a       <- commandArgs(trailingOnly = TRUE)
inp     <- a[1]
level   <- a[2]
outpre  <- a[3]
chains  <- if (length(a) >= 4) as.integer(a[4]) else 4
iter    <- if (length(a) >= 5) as.integer(a[5]) else 1000
nthread <- if (length(a) >= 6) as.integer(a[6]) else 1
subn    <- if (length(a) >= 7) as.integer(a[7]) else 0

df <- read.csv(inp)
if (subn > 0 && subn < nrow(df)) {
  set.seed(42)
  df <- df[sample(nrow(df), subn), ]
  cat("subsampled to", nrow(df), "rows\n")
}
df$participant <- factor(df$participant)
df$target1 <- factor(df$target1)
df$target2 <- factor(df$target2)
# one shared level set, so mm() indexes a single identity pool
lev <- sort(union(levels(df$target1), levels(df$target2)))
df$target1 <- factor(df$target1, levels = lev)
df$target2 <- factor(df$target2, levels = lev)

fam_sd  <- sd(df$fam_mean)
df$FAM  <- as.numeric(scale(df$fam_mean))
df$FD   <- as.numeric(scale(df$fam_diff))
df$SXF  <- df$sem_dissim_z * df$FAM
df$VXF  <- df$vis_dissim_z * df$FAM

FULL <- "1 + sem_dissim_z + vis_dissim_z + FAM + SXF + VXF"
res <- list(
  B0 = sprintf("(%s | participant) + (%s | mm(target1, target2))", FULL, FULL),
  B1 = sprintf("(%s | participant) + (1 + FAM | mm(target1, target2))", FULL),
  B2 = "(1 + sem_dissim_z + vis_dissim_z + FAM | participant) + (1 + FAM | mm(target1, target2))",
  B3 = "(1 + FAM | participant) + (1 + FAM | mm(target1, target2))"
)
stopifnot(level %in% names(res))

form <- bf(as.formula(paste(
  "abs_diff ~ sem_dissim_z + vis_dissim_z + FAM + SXF + VXF + FD +", res[[level]])))

pri <- c(prior(normal(0, 1), class = "b"),
         prior(student_t(3, 0, 2.5), class = "sd"),
         prior(student_t(3, 0, 2.5), class = "sigma"),
         prior(lkj(2), class = "cor"))

cat("[", basename(outpre), "]", level, " n =", nrow(df), "\n"); flush.console()
odir <- paste0(outpre, "_chains")
dir.create(odir, showWarnings = FALSE, recursive = TRUE)
t0 <- Sys.time()
m <- brm(form, data = df, family = gaussian(), prior = pri,
         chains = chains, iter = iter, warmup = floor(iter / 2),
         cores = chains, seed = 42,
         threads = if (nthread > 1) threading(nthread) else NULL,
         control = list(adapt_delta = 0.9, max_treedepth = 12),
         silent = 1, refresh = 50,
         output_dir = odir,
         output_basename = basename(outpre))
el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

# summary first (cheap), then the fit object
s <- posterior_summary(m)
write.csv(as.data.frame(s), paste0(outpre, "_summary.csv"))
saveRDS(m, paste0(outpre, ".rds"))

np   <- nuts_params(m)
ndiv <- sum(np$Value[np$Parameter == "divergent__"])
rh   <- max(brms::rhat(m), na.rm = TRUE)
ess  <- min(posterior::ess_bulk(as_draws_matrix(m)), na.rm = TRUE)
key  <- c("b_SXF", "b_VXF")
kk   <- s[rownames(s) %in% key, , drop = FALSE]

diag <- data.frame(level = level, n = nrow(df), secs = el,
                   divergent = ndiv, max_rhat = rh, min_ess = ess,
                   fam_sd = fam_sd,
                   SXF = kk["b_SXF", "Estimate"],
                   SXF_lo = kk["b_SXF", "Q2.5"], SXF_hi = kk["b_SXF", "Q97.5"],
                   SXF_raw = kk["b_SXF", "Estimate"] / fam_sd,
                   VXF = kk["b_VXF", "Estimate"],
                   VXF_lo = kk["b_VXF", "Q2.5"], VXF_hi = kk["b_VXF", "Q97.5"])
write.csv(diag, paste0(outpre, "_diag.csv"), row.names = FALSE)

cat(sprintf("  %.0fs  divergent=%d  max_rhat=%.4f  min_ess=%.0f\n", el, ndiv, rh, ess))
cat(sprintf("  SXF %+.5f [%+.5f, %+.5f]   VXF %+.5f [%+.5f, %+.5f]\n",
            diag$SXF, diag$SXF_lo, diag$SXF_hi, diag$VXF, diag$VXF_lo, diag$VXF_hi))
cat("done ->", outpre, "\n")
