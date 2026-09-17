# Maximal random-effects structure with a pre-specified reduction ladder.
#
# Following Barr et al. (2013) for "maximal justified by design" and Bates et
# al. (2015) / Matuschek et al. (2017) for principled reduction when the
# maximal model is unidentifiable.  The fixed structure is held constant
# throughout; only random terms are reduced.
#
# Design justification for the maximal structure: sem, vis, fam and both focal
# interactions all vary WITHIN participant and WITHIN identity, so slopes are
# warranted for every one of them at every grouping factor.  Barr's argument
# matters here specifically: omitting the sem:fam / vis:fam slopes inflates the
# Type I error rate of exactly the fixed effects under test.
#
# Ladder (identity terms reduced first — 50 identities cannot support what 325+
# participants can; steps are FIXED IN ADVANCE so "reduce until convergent"
# does not become a garden of forking paths):
#
#   L0  full 6-term unstructured at ppt, target1, target2
#   L1  identity terms -> diagonal (correlations zeroed)
#   L2  identity terms -> (1 + fam) only, ppt still full
#   L3  ppt -> diagonal, identity (1 + fam)
#   L4  ppt -> (1 + sem + vis + fam), identity (1 + fam)
#   L5  ppt -> (1 + fam), identity (1 + fam)          [ = the earlier M2 ]
#
# Selection: the most complex level that CONVERGES and is NON-SINGULAR.
# Singular = any RE sd below 1e-4 or any |correlation| above 0.999.
#
# Interaction terms are precomputed as explicit columns so the random-effects
# syntax is unambiguous and the fixed/random specifications match exactly.
#
# Usage: Rscript glmmtmb_maximal_ladder.R <pairs.csv> <out.csv> [label] [max_level]

suppressPackageStartupMessages(library(glmmTMB))

a <- commandArgs(trailingOnly = TRUE)
inp <- a[1]; outp <- a[2]
lab <- if (length(a) >= 3) a[3] else basename(inp)
maxlev <- if (length(a) >= 4) as.integer(a[4]) else 5
# moderator: "fam" = graded mean familiarity (V1/V3); "arm" = binarised
# popular/unpopular contrast on same-arm pairs only (V2)
mod <- if (length(a) >= 5) a[5] else "fam"

df <- read.csv(inp)
df$participant <- factor(df$participant)
df$target1 <- factor(df$target1)
df$target2 <- factor(df$target2)

if (mod == "arm") {
  # binarised contrast: 1 = both targets recognised, 0 = neither (mixed dropped
  # upstream).  Fixed effects and dispformula match the M2 run exactly so the
  # nested LRT against it is valid.
  fam_sd <- 1
  df$FAM <- as.numeric(df$recognized)
  df$SXF <- df$sem_dissim_z * df$FAM
  df$VXF <- df$vis_dissim_z * df$FAM
  fixed  <- "abs_diff ~ sem_dissim_z + vis_dissim_z + FAM + SXF + VXF"
  dispf  <- ~FAM
} else {
  fam_sd <- sd(df$fam_mean)
  df$FAM <- as.numeric(scale(df$fam_mean))  # z for conditioning; raw = beta/fam_sd
  df$FD  <- as.numeric(scale(df$fam_diff))
  df$SXF <- df$sem_dissim_z * df$FAM
  df$VXF <- df$vis_dissim_z * df$FAM
  fixed  <- "abs_diff ~ sem_dissim_z + vis_dissim_z + FAM + SXF + VXF + FD"
  dispf  <- ~1
}
FULL  <- "1 + sem_dissim_z + vis_dissim_z + FAM + SXF + VXF"

ladder <- list(
  L0 = sprintf("(%s | participant) + (%s | target1) + (%s | target2)", FULL, FULL, FULL),
  L1 = sprintf("(%s | participant) + diag(%s | target1) + diag(%s | target2)", FULL, FULL, FULL),
  L2 = sprintf("(%s | participant) + (1 + FAM | target1) + (1 + FAM | target2)", FULL),
  L3 = sprintf("diag(%s | participant) + (1 + FAM | target1) + (1 + FAM | target2)", FULL),
  L4 = "(1 + sem_dissim_z + vis_dissim_z + FAM | participant) + (1 + FAM | target1) + (1 + FAM | target2)",
  L5 = "(1 + FAM | participant) + (1 + FAM | target1) + (1 + FAM | target2)"
)

is_singular <- function(m) {
  vc <- VarCorr(m)$cond
  bad <- FALSE
  for (g in names(vc)) {
    s <- attr(vc[[g]], "stddev")
    r <- attr(vc[[g]], "correlation")
    if (any(!is.finite(s)) || any(s < 1e-4)) bad <- TRUE
    if (!is.null(r)) {
      rr <- r[upper.tri(r)]
      if (any(!is.finite(rr)) || any(abs(rr) > 0.999)) bad <- TRUE
    }
  }
  bad
}

grab <- function(m, t) {
  co <- summary(m)$coefficients$cond
  if (!(t %in% rownames(co))) return(rep(NA, 4))
  co[t, c("Estimate", "Std. Error", "z value", "Pr(>|z|)")]
}

rows <- list(); fits <- list()
cat("[", lab, "]", mod, " n =", nrow(df), " fam sd =", sprintf("%.3f", fam_sd), "\n"); flush.console()
for (nm in names(ladder)[1:(maxlev + 1)]) {
  t0 <- Sys.time()
  m <- tryCatch(glmmTMB(as.formula(paste(fixed, "+", ladder[[nm]])),
                        dispformula = dispf, data = df),
                error = function(e) {cat("   ", nm, "ERROR:", conditionMessage(e), "\n"); NULL},
                warning = function(w) NULL)
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  conv <- !is.null(m) && is.finite(logLik(m))
  sing <- if (conv) is_singular(m) else NA
  s <- if (conv) grab(m, "SXF") else rep(NA, 4)
  v <- if (conv) grab(m, "VXF") else rep(NA, 4)
  cat(sprintf("   %s %7.0fs conv=%-5s sing=%-5s  SXF %+.5f (p=%.3g)  VXF %+.5f\n",
              nm, el, conv, sing, s[1], s[4], v[1])); flush.console()
  rows[[nm]] <- data.frame(label = lab, moderator = mod, level = nm,
    n = nrow(df), secs = el,
    converged = conv, singular = sing,
    logLik = if (conv) as.numeric(logLik(m)) else NA,
    AIC = if (conv) AIC(m) else NA,
    semXfam_z = s[1], semXfam_se = s[2], semXfam_zval = s[3], semXfam_p = s[4],
    semXfam_raw = s[1] / fam_sd,
    visXfam_z = v[1], visXfam_se = v[2], visXfam_zval = v[3], visXfam_p = v[4],
    visXfam_raw = v[1] / fam_sd, fam_sd = fam_sd)
  if (conv) fits[[nm]] <- m
  if (conv && !sing) { cat("   -> SELECTED", nm, "(most complex convergent, non-singular)\n"); break }
}
res <- do.call(rbind, rows)
write.csv(res, outp, row.names = FALSE)

ok <- res[res$converged %in% TRUE & res$singular %in% FALSE, ]
if (nrow(ok)) {
  sel <- ok[1, ]
  cat(sprintf("\nSELECTED %s: semXfam = %+.5f (z-scale) / %+.5f (raw), p = %.3g\n",
              sel$level, sel$semXfam_z, sel$semXfam_raw, sel$semXfam_p))
} else cat("\nNo convergent non-singular model in the ladder.\n")
cat("Saved ->", outp, "\n")
