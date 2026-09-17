# Maximal ladder WITH the familiarity-difference control in the random slopes.
#
# glmmtmb_ladder2.R carried FD (|f_i - f_j|, z-scored) as a fixed main effect
# only.  By the same design argument that motivates every other slope here, FD
# varies within perceiver and within identity, so it warrants a random slope
# too -- and leaving it out means any perceiver- or identity-level heterogeneity
# in how familiarity IMBALANCE affects trait distance is unmodelled, and can be
# absorbed by the neighbouring FAM and interaction terms.
#
# Graded moderator only.  The binarised design has no FD term to give a slope:
# in the same-arm subset US familiarity difference is identically zero, and the
# CN version omits it, so there is nothing to vary.
#
#   level  participant                          target1 / target2
#   L0     1 + S + V + F + FD + SxF + VxF       1 + S + V + F + FD + SxF + VxF
#   L1     full                                 1 + S + V + F + FD     (drop tgt interactions)
#   L2     full                                 1 + S + V + F          (drop tgt FD)
#   L3     full                                 1 + S + V              (drop tgt F)
#   L4     full                                 1                      (drop tgt S,V)
#   L5     1 + S + V + F + FD                   1                      (drop ppt interactions)
#   L6     1 + S + V + F                        1                      (drop ppt FD)
#   L7     1 + S + V                            1                      (drop ppt F)
#   L8     1                                    1                      (drop ppt S,V) = M0
#
# Same reduction order on both sides (interactions -> FD -> F -> S,V), targets
# first, so the participant interaction slopes -- the Barr protection for the
# fixed effects under test -- survive every step that precedes L5.
#
# Every level is fitted so the full nested LRT chain is available.
#
# Usage: Rscript glmmtmb_ladder3.R <pairs.csv> <out.csv> [label]

suppressPackageStartupMessages(library(glmmTMB))

a   <- commandArgs(trailingOnly = TRUE)
inp <- a[1]; outp <- a[2]
lab <- if (length(a) >= 3) a[3] else basename(inp)

df <- read.csv(inp)
df$participant <- factor(df$participant)
df$target1 <- factor(df$target1)
df$target2 <- factor(df$target2)

fam_sd <- sd(df$fam_mean)
df$FAM <- as.numeric(scale(df$fam_mean))
df$FD  <- as.numeric(scale(df$fam_diff))
df$SXF <- df$sem_dissim_z * df$FAM
df$VXF <- df$vis_dissim_z * df$FAM
fixed  <- "abs_diff ~ sem_dissim_z + vis_dissim_z + FAM + SXF + VXF + FD"

FULL7 <- "1 + sem_dissim_z + vis_dissim_z + FAM + FD + SXF + VXF"
noI   <- "1 + sem_dissim_z + vis_dissim_z + FAM + FD"
noFD  <- "1 + sem_dissim_z + vis_dissim_z + FAM"
noF   <- "1 + sem_dissim_z + vis_dissim_z"
INT   <- "1"

tt <- function(x) sprintf("(%s | target1) + (%s | target2)", x, x)
pp <- function(x) sprintf("(%s | participant)", x)
ladder <- list(
  L0 = paste(pp(FULL7), "+", tt(FULL7)),
  L1 = paste(pp(FULL7), "+", tt(noI)),
  L2 = paste(pp(FULL7), "+", tt(noFD)),
  L3 = paste(pp(FULL7), "+", tt(noF)),
  L4 = paste(pp(FULL7), "+", tt(INT)),
  L5 = paste(pp(noI),   "+", tt(INT)),
  L6 = paste(pp(noFD),  "+", tt(INT)),
  L7 = paste(pp(noF),   "+", tt(INT)),
  L8 = paste(pp(INT),   "+", tt(INT))
)
np <- function(p) p * (p + 1) / 2
K <- c(L0 = np(7) + 2*np(7), L1 = np(7) + 2*np(5), L2 = np(7) + 2*np(4),
       L3 = np(7) + 2*np(3), L4 = np(7) + 2*np(1), L5 = np(5) + 2*np(1),
       L6 = np(4) + 2*np(1), L7 = np(3) + 2*np(1), L8 = np(1) + 2*np(1))

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

cat("[", lab, "] n =", nrow(df), " fam_sd =", sprintf("%.3f", fam_sd), "\n"); flush.console()
rows <- list()
for (nm in names(ladder)) {
  t0 <- Sys.time()
  m <- tryCatch(glmmTMB(as.formula(paste(fixed, "+", ladder[[nm]])), data = df),
                error = function(e) {cat("   ", nm, "ERROR:", conditionMessage(e), "\n"); NULL},
                warning = function(w) NULL)
  el   <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  conv <- !is.null(m) && is.finite(logLik(m))
  sing <- if (conv) is_singular(m) else NA
  s <- if (conv) grab(m, "SXF") else rep(NA, 4)
  v <- if (conv) grab(m, "VXF") else rep(NA, 4)
  d <- if (conv) grab(m, "FD")  else rep(NA, 4)
  rows[[nm]] <- data.frame(label = lab, level = nm, k_re = K[[nm]], n = nrow(df),
    secs = el, converged = conv, singular = sing,
    logLik = if (conv) as.numeric(logLik(m)) else NA,
    AIC = if (conv) AIC(m) else NA,
    semXfam = s[1], semXfam_se = s[2], semXfam_p = s[4], semXfam_raw = s[1]/fam_sd,
    visXfam = v[1], visXfam_se = v[2], visXfam_p = v[4], visXfam_raw = v[1]/fam_sd,
    FD_beta = d[1], FD_p = d[4], fam_sd = fam_sd)
  cat(sprintf("   %s k=%2d %6.0fs conv=%-5s sing=%-5s  SXF %+.5f (p=%.3g)  VXF %+.5f (p=%.3g)\n",
              nm, K[[nm]], el, conv, sing, s[1], s[4], v[1], v[4])); flush.console()
}
res <- do.call(rbind, rows)
res$lrt_chisq <- NA; res$lrt_df <- NA; res$lrt_p <- NA
for (i in 2:nrow(res)) {
  if (isTRUE(res$converged[i]) && isTRUE(res$converged[i-1])) {
    chi <- 2*(res$logLik[i-1] - res$logLik[i]); dfree <- res$k_re[i-1] - res$k_re[i]
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
