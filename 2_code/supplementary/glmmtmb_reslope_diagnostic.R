# Do we need a random slope for the familiarity moderator?
#
# Run BEFORE the main variants, so the random-effect structure is chosen from
# evidence rather than assumed.  The moderator (mean familiarity, or arm) varies
# within participant and within target, so a random slope is defensible in
# principle — but it doubles the covariance parameters per grouping factor and
# can fail to identify, so it should be justified per design.
#
# Nested comparison, by likelihood-ratio test and AIC:
#
#   M0  (1 | ppt) + (1 | t1) + (1 | t2)                      intercepts only
#   M1  (1 + mod | ppt) + (1 | t1) + (1 | t2)                + slope by perceiver
#   M2  (1 + mod | ppt) + (1 + mod | t1) + (1 + mod | t2)    + slope by target
#
# Fitted with ML (REML=FALSE) so the fixed-effect structure is held constant and
# the LRTs are valid for random-effect comparisons of nested models.
#
# Usage: Rscript glmmtmb_reslope_diagnostic.R <pairs.csv> <moderator> <out.csv> [label]
#   moderator: "fam_mean_z"  (continuous variants) or "arm" (binarised variant)

suppressPackageStartupMessages(library(glmmTMB))

args  <- commandArgs(trailingOnly = TRUE)
inp   <- args[1]
mod   <- args[2]
outp  <- args[3]
label <- if (length(args) >= 4) args[4] else basename(inp)

df <- read.csv(inp)
df$participant <- factor(df$participant)
df$target1 <- factor(df$target1)
df$target2 <- factor(df$target2)

if (mod == "arm") {
  df$arm <- factor(ifelse(df$recognized == 1, "rec", "unrec"),
                   levels = c("unrec", "rec"))
  fixed <- "abs_diff ~ sem_dissim_z * arm + vis_dissim_z * arm"
} else {
  df$fam_mean_z <- as.numeric(scale(df$fam_mean))
  df$fam_diff_z <- as.numeric(scale(df$fam_diff))
  fixed <- paste("abs_diff ~ sem_dissim_z * fam_mean_z +",
                 "vis_dissim_z * fam_mean_z + fam_diff_z")
}

res_of <- function(re) as.formula(paste(fixed, "+", re))

specs <- list(
  M0 = "(1 | participant) + (1 | target1) + (1 | target2)",
  M1 = sprintf("(1 + %s | participant) + (1 | target1) + (1 | target2)", mod),
  M2 = sprintf("(1 + %s | participant) + (1 + %s | target1) + (1 + %s | target2)",
               mod, mod, mod)
)

cat("[", label, "] moderator =", mod, " n =", nrow(df), "\n"); flush.console()

fits <- list(); times <- c()
for (nm in names(specs)) {
  t0 <- Sys.time()
  f <- tryCatch(glmmTMB(res_of(specs[[nm]]), data = df, REML = FALSE),
                error = function(e) {cat("   ", nm, "ERROR:", conditionMessage(e), "\n"); NULL})
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  times[nm] <- el
  ok <- !is.null(f) && is.finite(logLik(f))
  cat(sprintf("   %s  %6.0fs  converged=%s  AIC=%s\n", nm, el, ok,
              if (ok) sprintf("%.1f", AIC(f)) else "NA")); flush.console()
  fits[[nm]] <- if (ok) f else NULL
}

lrt <- function(a, b) {
  if (is.null(fits[[a]]) || is.null(fits[[b]])) return(c(NA, NA, NA))
  an <- tryCatch(anova(fits[[a]], fits[[b]]), error = function(e) NULL)
  if (is.null(an)) return(c(NA, NA, NA))
  c(an$Chisq[2], an$`Chi Df`[2], an$`Pr(>Chisq)`[2])
}
a1 <- lrt("M0", "M1"); a2 <- lrt("M1", "M2")

getb <- function(f, nm) {
  if (is.null(f)) return(NA)
  co <- summary(f)$coefficients$cond
  alt <- paste(rev(strsplit(nm, ":")[[1]]), collapse = ":")
  rn <- if (nm %in% rownames(co)) nm else alt
  if (!(rn %in% rownames(co))) NA else co[rn, "Estimate"]
}
int_sem <- if (mod == "arm") "sem_dissim_z:armrec" else "sem_dissim_z:fam_mean_z"
int_vis <- if (mod == "arm") "vis_dissim_z:armrec" else "vis_dissim_z:fam_mean_z"

res <- data.frame(
  label = label, moderator = mod, n = nrow(df),
  AIC_M0 = if (is.null(fits$M0)) NA else AIC(fits$M0),
  AIC_M1 = if (is.null(fits$M1)) NA else AIC(fits$M1),
  AIC_M2 = if (is.null(fits$M2)) NA else AIC(fits$M2),
  lrt01_chisq = a1[1], lrt01_df = a1[2], lrt01_p = a1[3],
  lrt12_chisq = a2[1], lrt12_df = a2[2], lrt12_p = a2[3],
  semXmod_M0 = getb(fits$M0, int_sem), semXmod_M1 = getb(fits$M1, int_sem),
  semXmod_M2 = getb(fits$M2, int_sem),
  visXmod_M0 = getb(fits$M0, int_vis), visXmod_M1 = getb(fits$M1, int_vis),
  visXmod_M2 = getb(fits$M2, int_vis),
  secs_M0 = times["M0"], secs_M1 = times["M1"], secs_M2 = times["M2"]
)
write.csv(res, outp, row.names = FALSE)

cat(sprintf("\n  LRT M0->M1  chi2=%.1f df=%s p=%.3g\n", a1[1], a1[2], a1[3]))
cat(sprintf("  LRT M1->M2  chi2=%.1f df=%s p=%.3g\n", a2[1], a2[2], a2[3]))
cat(sprintf("  semXmod:  M0 %+.4f | M1 %+.4f | M2 %+.4f\n",
            res$semXmod_M0, res$semXmod_M1, res$semXmod_M2))
cat(sprintf("  visXmod:  M0 %+.4f | M1 %+.4f | M2 %+.4f\n",
            res$visXmod_M0, res$visXmod_M1, res$visXmod_M2))
cat("Saved ->", outp, "\n")
