# Is the M2 result an artefact of how targets were assigned to positions?
#
# Pairs were built with np.triu_indices, so target1 is ALWAYS the lower-indexed
# identity.  That makes the two grouping factors systematically different:
# identity 0 only ever appears as target1, identity 49 only as target2, and the
# same identity's random effects in the two factors correlate only r=.41
# (intercept) and r=-.11 (familiarity slope).
#
# Swapping the two positions within a row changes nothing substantive: the
# outcome |y_i - y_j| and the model dissimilarities are symmetric in i and j.
# So any change in the fixed effects under random re-assignment is attributable
# purely to the random-effect structure's position dependence.
#
# For each seed the model is refit with positions randomised, at both M0
# (intercepts only) and M2 (with target slopes).  If the M2 semantic sign is a
# real identity-level effect it should be stable across seeds; if it tracks the
# systematic ordering it should move.
#
# Usage: Rscript glmmtmb_position_randomisation.R <pairs.csv> <seed> <out.csv> [label]

suppressPackageStartupMessages(library(glmmTMB))

args  <- commandArgs(trailingOnly = TRUE)
inp   <- args[1]
seed  <- as.integer(args[2])
outp  <- args[3]
label <- if (length(args) >= 4) args[4] else basename(inp)

df <- read.csv(inp)
df$fam_mean_z <- as.numeric(scale(df$fam_mean))
df$fam_diff_z <- as.numeric(scale(df$fam_diff))

if (seed > 0) {
  set.seed(seed)
  swap <- runif(nrow(df)) < 0.5
  a <- df$target1; b <- df$target2
  df$target1[swap] <- b[swap]
  df$target2[swap] <- a[swap]
}
df$participant <- factor(df$participant)
df$target1 <- factor(df$target1)
df$target2 <- factor(df$target2)

fixed <- paste("abs_diff ~ sem_dissim_z * fam_mean_z +",
               "vis_dissim_z * fam_mean_z + fam_diff_z")
specs <- list(
  M0 = "(1 | participant) + (1 | target1) + (1 | target2)",
  M2 = paste("(1 + fam_mean_z | participant) +",
             "(1 + fam_mean_z | target1) + (1 + fam_mean_z | target2)")
)

cat("[", label, "] seed =", seed, " n =", nrow(df), "\n"); flush.console()
out <- list(label = label, seed = seed, n = nrow(df))
for (nm in names(specs)) {
  m <- tryCatch(glmmTMB(as.formula(paste(fixed, "+", specs[[nm]])), data = df),
                error = function(e) NULL)
  ok <- !is.null(m) && is.finite(logLik(m))
  g <- function(t) {
    if (!ok) return(NA)
    co <- summary(m)$coefficients$cond
    alt <- paste(rev(strsplit(t, ":")[[1]]), collapse = ":")
    rn <- if (t %in% rownames(co)) t else alt
    if (!(rn %in% rownames(co))) NA else co[rn, "Estimate"]
  }
  out[[paste0("sem_", nm)]] <- g("sem_dissim_z:fam_mean_z")
  out[[paste0("vis_", nm)]] <- g("vis_dissim_z:fam_mean_z")
  out[[paste0("aic_", nm)]] <- if (ok) AIC(m) else NA
  cat(sprintf("  %s sem %+.4f  vis %+.4f  AIC %.0f\n", nm,
              out[[paste0("sem_", nm)]], out[[paste0("vis_", nm)]],
              out[[paste0("aic_", nm)]])); flush.console()
}
write.csv(as.data.frame(out), outp, row.names = FALSE)
cat("Saved ->", outp, "\n")
