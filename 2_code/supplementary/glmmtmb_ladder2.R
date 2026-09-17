# Maximal random-effects ladder, reduction ordered target-first.
#
# Barr et al. (2013) for "maximal justified by design"; Bates et al. (2015) /
# Matuschek et al. (2017) for principled reduction when it is unidentifiable.
# Fixed effects are held constant throughout; only random terms are reduced.
#
# Reduction order (fixed IN ADVANCE, so "reduce until convergent" is not a
# garden of forking paths).  Identities are the thin factor -- 50 levels cannot
# support what 325+ participants can -- so they are stripped first, and the
# PARTICIPANT interaction slopes are kept to the very last step.  That ordering
# is deliberate: sem:FAM and vis:FAM are the fixed effects under test, and
# dropping their participant slopes is what inflates their Type I error, so
# those terms survive every reduction that precedes them.
#
#   level  participant                      target1 / target2
#   L0     1 + S + V + F + SxF + VxF        1 + S + V + F + SxF + VxF
#   L1     1 + S + V + F + SxF + VxF        1 + S + V + F            (drop tgt interactions)
#   L2     1 + S + V + F + SxF + VxF        1 + S + V                (drop tgt F)
#   L3     1 + S + V + F + SxF + VxF        1                        (drop tgt S,V)
#   L4     1 + S + V + F                    1                        (drop ppt interactions)
#   L5     1 + S + V                        1                        (drop ppt F)
#   L6     1                                1                        (drop ppt S,V)  = M0
#
# EVERY level is fitted, not just up to the first success, so the full nested
# LRT chain is available.  Selection reports the most complex level that both
# converges and is non-singular, per trait.
#
# Usage: Rscript glmmtmb_ladder2.R <pairs.csv> <out.csv> [label] [moderator]
#   moderator: "fam" (graded mean familiarity) | "arm" (binarised popular/unpopular)

suppressPackageStartupMessages(library(glmmTMB))

a   <- commandArgs(trailingOnly = TRUE)
inp <- a[1]; outp <- a[2]
lab <- if (length(a) >= 3) a[3] else basename(inp)
mod <- if (length(a) >= 4) a[4] else "fam"

df <- read.csv(inp)
df$participant <- factor(df$participant)
df$target1 <- factor(df$target1)
df$target2 <- factor(df$target2)

if (mod == "arm") {
  fam_sd <- 1
  df$FAM <- as.numeric(df$recognized)
  fixed  <- "abs_diff ~ sem_dissim_z + vis_dissim_z + FAM + SXF + VXF"
  # Homoscedastic residual, matching the graded version so the two moderator
  # codings differ ONLY in the coding.  An arm-specific residual SD is real
  # (sigma ratios .85-1.08, all LRTs significant) but moves the interaction
  # coefficients by under 4%, so comparability is worth more than modelling it.
  dispf  <- ~1
} else {
  fam_sd <- sd(df$fam_mean)
  df$FAM <- as.numeric(scale(df$fam_mean))
  df$FD  <- as.numeric(scale(df$fam_diff))
  fixed  <- "abs_diff ~ sem_dissim_z + vis_dissim_z + FAM + SXF + VXF + FD"
  dispf  <- ~1
}
df$SXF <- df$sem_dissim_z * df$FAM
df$VXF <- df$vis_dissim_z * df$FAM

P_FULL <- "1 + sem_dissim_z + vis_dissim_z + FAM + SXF + VXF"
P_noI  <- "1 + sem_dissim_z + vis_dissim_z + FAM"
P_noF  <- "1 + sem_dissim_z + vis_dissim_z"
P_int  <- "1"
T_FULL <- "1 + sem_dissim_z + vis_dissim_z + FAM + SXF + VXF"
T_noI  <- "1 + sem_dissim_z + vis_dissim_z + FAM"
T_noF  <- "1 + sem_dissim_z + vis_dissim_z"
T_int  <- "1"

tt <- function(x) sprintf("(%s | target1) + (%s | target2)", x, x)
ladder <- list(
  L0 = paste0("(", P_FULL, " | participant) + ", tt(T_FULL)),
  L1 = paste0("(", P_FULL, " | participant) + ", tt(T_noI)),
  L2 = paste0("(", P_FULL, " | participant) + ", tt(T_noF)),
  L3 = paste0("(", P_FULL, " | participant) + ", tt(T_int)),
  L4 = paste0("(", P_noI, " | participant) + ", tt(T_int)),
  L5 = paste0("(", P_noF, " | participant) + ", tt(T_int)),
  L6 = paste0("(", P_int, " | participant) + ", tt(T_int))
)
# number of covariance parameters per level (p terms -> p(p+1)/2 per factor)
np <- function(p) p * (p + 1) / 2
K <- c(L0 = np(6) + 2 * np(6), L1 = np(6) + 2 * np(4), L2 = np(6) + 2 * np(3),
       L3 = np(6) + 2 * np(1), L4 = np(4) + 2 * np(1), L5 = np(3) + 2 * np(1),
       L6 = np(1) + 2 * np(1))

is_singular <- function(m) {
  vc <- VarCorr(m)$cond; bad <- FALSE
  for (g in names(vc)) {
    s <- attr(vc[[g]], "stddev"); r <- attr(vc[[g]], "correlation")
    if (any(!is.finite(s)) || any(s < 1e-4)) bad <- TRUE
    if (!is.null(r)) { rr <- r[upper.tri(r)]
      if (any(!is.finite(rr)) || any(abs(rr) > 0.999)) bad <- TRUE }
  }
  bad
}
grab <- function(m, t) {
  co <- summary(m)$coefficients$cond
  if (!(t %in% rownames(co))) rep(NA, 4) else co[t, c("Estimate","Std. Error","z value","Pr(>|z|)")]
}

cat("[", lab, "]", mod, " n =", nrow(df), "\n"); flush.console()
rows <- list()
for (nm in names(ladder)) {
  t0 <- Sys.time()
  m <- tryCatch(glmmTMB(as.formula(paste(fixed, "+", ladder[[nm]])),
                        dispformula = dispf, data = df),
                error = function(e) {cat("   ", nm, "ERROR:", conditionMessage(e), "\n"); NULL},
                warning = function(w) NULL)
  el   <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  conv <- !is.null(m) && is.finite(logLik(m))
  sing <- if (conv) is_singular(m) else NA
  s <- if (conv) grab(m, "SXF") else rep(NA, 4)
  v <- if (conv) grab(m, "VXF") else rep(NA, 4)
  rows[[nm]] <- data.frame(label = lab, moderator = mod, level = nm, k_re = K[[nm]],
    n = nrow(df), secs = el, converged = conv, singular = sing,
    logLik = if (conv) as.numeric(logLik(m)) else NA,
    AIC = if (conv) AIC(m) else NA,
    semXfam = s[1], semXfam_se = s[2], semXfam_z = s[3], semXfam_p = s[4],
    semXfam_raw = s[1] / fam_sd,
    visXfam = v[1], visXfam_se = v[2], visXfam_z = v[3], visXfam_p = v[4],
    visXfam_raw = v[1] / fam_sd, fam_sd = fam_sd)
  cat(sprintf("   %s k=%2d %6.0fs conv=%-5s sing=%-5s  SXF %+.5f (p=%.3g)  VXF %+.5f (p=%.3g)\n",
              nm, K[[nm]], el, conv, sing, s[1], s[4], v[1], v[4])); flush.console()
}
res <- do.call(rbind, rows)

# nested LRT between consecutive levels (each level is nested in the one above)
res$lrt_chisq <- NA; res$lrt_df <- NA; res$lrt_p <- NA
for (i in 2:nrow(res)) {
  if (isTRUE(res$converged[i]) && isTRUE(res$converged[i - 1])) {
    chi <- 2 * (res$logLik[i - 1] - res$logLik[i])
    dfree <- res$k_re[i - 1] - res$k_re[i]
    res$lrt_chisq[i] <- chi; res$lrt_df[i] <- dfree
    res$lrt_p[i] <- pchisq(chi, dfree, lower.tail = FALSE)
  }
}
write.csv(res, outp, row.names = FALSE)

ok <- res[res$converged %in% TRUE & res$singular %in% FALSE, ]
if (nrow(ok)) {
  sel <- ok[1, ]
  cat(sprintf("\nSELECTED %s  semXfam %+.5f (raw %+.5f) p=%.3g | visXfam %+.5f p=%.3g\n",
              sel$level, sel$semXfam, sel$semXfam_raw, sel$semXfam_p, sel$visXfam, sel$visXfam_p))
} else cat("\nNo convergent non-singular level.\n")
cat("Saved ->", outp, "\n")
