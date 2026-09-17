# Diagnostics + variance components for the SELECTED maximal familiarity LMMs.
#
# glmmtmb_ladder2.R wrapped every fit in tryCatch(..., warning = function(w) NULL),
# so any fit emitting any warning was discarded and logged as converged = FALSE.
# That conflates genuine non-convergence with incidental warnings -- visible in the
# ladder output as non-monotonic failures (a simpler level failing where a more
# complex one converged).  Here warnings are RECORDED, never fatal, and the real
# diagnostics are extracted explicitly.
#
# Three fits per outcome, all with identical data and optimizer defaults:
#   selected  : the level chosen by the ladder, full fixed effects
#   nointer   : same random structure, SXF + VXF dropped from the FIXED part only
#               -> 2-df likelihood-ratio test of the focal interaction, replacing
#                  the Wald z (df = Inf) currently reported
#   L0        : only when the selected level is not L0, to test whether the
#               reduction was actually required or was a swallowed warning
#
# Graded moderator (fam) only -- the main-text analysis.
#
# Usage: Rscript glmmtmb_diagnostics.R <pairs.csv> <out_prefix> <label> <level>
#   writes <out_prefix>_summary.csv and <out_prefix>_varcorr.csv

suppressPackageStartupMessages(library(glmmTMB))

a    <- commandArgs(trailingOnly = TRUE)
inp  <- a[1]; outp <- a[2]; lab <- a[3]; sel_lvl <- a[4]

df <- read.csv(inp)
df$participant <- factor(df$participant)
df$target1 <- factor(df$target1)
df$target2 <- factor(df$target2)

fam_sd <- sd(df$fam_mean)
df$FAM <- as.numeric(scale(df$fam_mean))
df$FD  <- as.numeric(scale(df$fam_diff))
df$SXF <- df$sem_dissim_z * df$FAM
df$VXF <- df$vis_dissim_z * df$FAM

FIX_FULL  <- "abs_diff ~ sem_dissim_z + vis_dissim_z + FAM + SXF + VXF + FD"
FIX_NOINT <- "abs_diff ~ sem_dissim_z + vis_dissim_z + FAM + FD"

P_FULL <- "1 + sem_dissim_z + vis_dissim_z + FAM + SXF + VXF"
P_noI  <- "1 + sem_dissim_z + vis_dissim_z + FAM"
P_noF  <- "1 + sem_dissim_z + vis_dissim_z"
T_FULL <- P_FULL
T_noI  <- P_noI
T_noF  <- P_noF

tt <- function(x) sprintf("(%s | target1) + (%s | target2)", x, x)
ladder <- list(
  L0 = paste0("(", P_FULL, " | participant) + ", tt(T_FULL)),
  L1 = paste0("(", P_FULL, " | participant) + ", tt(T_noI)),
  L2 = paste0("(", P_FULL, " | participant) + ", tt(T_noF)),
  L3 = paste0("(", P_FULL, " | participant) + ", tt("1")),
  L4 = paste0("(", P_noI,  " | participant) + ", tt("1")),
  L5 = paste0("(", P_noF,  " | participant) + ", tt("1")),
  L6 = paste0("(1 | participant) + ", tt("1"))
)

# ── Fit, capturing warnings instead of aborting on them ───────────────────────

fit_one <- function(fixed, lvl) {
  warns <- character(0)
  m <- withCallingHandlers(
    tryCatch(glmmTMB(as.formula(paste(fixed, "+", ladder[[lvl]])),
                     dispformula = ~1, data = df),
             error = function(e) {
               warns <<- c(warns, paste("ERROR:", conditionMessage(e))); NULL }),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  list(m = m, warns = warns)
}

# ── Explicit diagnostics ──────────────────────────────────────────────────────

diag_row <- function(f, fitname, lvl) {
  m <- f$m
  wtxt <- paste(unique(f$warns), collapse = " || ")
  if (is.null(m)) {
    return(data.frame(label = lab, fit = fitname, level = lvl, n = nrow(df),
                      secs = NA, conv_code = NA, conv_msg = NA, pdHess = NA,
                      max_abs_grad = NA, hess_cond = NA, logLik = NA, AIC = NA,
                      npar = NA, sigma = NA,
                      sem = NA, sem_se = NA, sem_z = NA, sem_p = NA,
                      vis = NA, vis_se = NA, vis_z = NA, vis_p = NA,
                      fam_sd = fam_sd, warnings = wtxt,
                      stringsAsFactors = FALSE))
  }
  g   <- m$sdr$gradient.fixed
  cf  <- m$sdr$cov.fixed
  kap <- NA
  if (!is.null(cf) && all(is.finite(cf))) {
    ev <- eigen(cf, symmetric = TRUE, only.values = TRUE)$values
    # cov.fixed is the inverse Hessian, so its condition number is the Hessian's
    if (all(ev > 0)) kap <- max(ev) / min(ev)
  }
  co <- summary(m)$coefficients$cond
  gr <- function(t, j) if (t %in% rownames(co)) co[t, j] else NA

  data.frame(label = lab, fit = fitname, level = lvl, n = nrow(df),
    secs = NA,
    conv_code = m$fit$convergence,
    conv_msg = if (is.null(m$fit$message)) NA else m$fit$message,
    pdHess = isTRUE(m$sdr$pdHess),
    max_abs_grad = if (is.null(g)) NA else max(abs(g)),
    hess_cond = kap,
    logLik = as.numeric(logLik(m)), AIC = AIC(m),
    npar = attr(logLik(m), "df"), sigma = sigma(m),
    sem = gr("SXF", "Estimate"), sem_se = gr("SXF", "Std. Error"),
    sem_z = gr("SXF", "z value"), sem_p = gr("SXF", "Pr(>|z|)"),
    vis = gr("VXF", "Estimate"), vis_se = gr("VXF", "Std. Error"),
    vis_z = gr("VXF", "z value"), vis_p = gr("VXF", "Pr(>|z|)"),
    fam_sd = fam_sd, warnings = wtxt, stringsAsFactors = FALSE)
}

# ── Variance components, long format ─────────────────────────────────────────

vc_rows <- function(f, fitname, lvl) {
  m <- f$m
  if (is.null(m)) return(NULL)
  out <- list()
  vc <- VarCorr(m)$cond
  for (g in names(vc)) {
    s <- attr(vc[[g]], "stddev")
    out[[length(out) + 1]] <- data.frame(
      label = lab, fit = fitname, level = lvl, group = g, type = "sd",
      term1 = names(s), term2 = NA_character_, value = as.numeric(s),
      stringsAsFactors = FALSE)
    r <- attr(vc[[g]], "correlation")
    if (!is.null(r) && nrow(r) > 1) {
      ij <- base::which(upper.tri(r), arr.ind = TRUE)
      out[[length(out) + 1]] <- data.frame(
        label = lab, fit = fitname, level = lvl, group = g, type = "cor",
        term1 = rownames(r)[ij[, 1]], term2 = colnames(r)[ij[, 2]],
        value = r[ij], stringsAsFactors = FALSE)
    }
  }
  out[[length(out) + 1]] <- data.frame(
    label = lab, fit = fitname, level = lvl, group = "Residual", type = "sd",
    term1 = "(Observation)", term2 = NA_character_, value = sigma(m),
    stringsAsFactors = FALSE)
  do.call(rbind, out)
}

# ── Run ──────────────────────────────────────────────────────────────────────

jobs <- list(c("selected", sel_lvl), c("nointer", sel_lvl))
if (sel_lvl != "L0") jobs[[3]] <- c("L0", "L0")

srows <- list(); vrows <- list()
for (j in jobs) {
  fitname <- j[1]; lvl <- j[2]
  fixed <- if (fitname == "nointer") FIX_NOINT else FIX_FULL
  cat(sprintf("[%s] %s @ %s ...\n", lab, fitname, lvl)); flush.console()
  t0 <- Sys.time()
  f  <- fit_one(fixed, lvl)
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  r  <- diag_row(f, fitname, lvl); r$secs <- el
  srows[[fitname]] <- r
  vrows[[fitname]] <- vc_rows(f, fitname, lvl)
  cat(sprintf("[%s] %s @ %s  %.0fs conv=%s pdHess=%s logLik=%s\n",
              lab, fitname, lvl, el, r$conv_code, r$pdHess,
              formatC(r$logLik, format = "f", digits = 1))); flush.console()
}

summ <- do.call(rbind, srows)

# 2-df LRT of the fixed interaction terms, selected vs nointer
sel <- summ[summ$fit == "selected", ]; noi <- summ[summ$fit == "nointer", ]
summ$int_lrt_chisq <- NA; summ$int_lrt_df <- NA; summ$int_lrt_p <- NA
if (nrow(sel) && nrow(noi) && is.finite(sel$logLik) && is.finite(noi$logLik)) {
  chi <- 2 * (sel$logLik - noi$logLik)
  i <- summ$fit == "selected"
  summ$int_lrt_chisq[i] <- chi
  summ$int_lrt_df[i]    <- 2
  summ$int_lrt_p[i]     <- pchisq(chi, 2, lower.tail = FALSE)
  cat(sprintf("[%s] interaction LRT chi2(2) = %.2f  p = %.4g\n",
              lab, chi, pchisq(chi, 2, lower.tail = FALSE)))
}

write.csv(summ, paste0(outp, "_summary.csv"), row.names = FALSE)
vv <- do.call(rbind, vrows)
if (!is.null(vv)) write.csv(vv, paste0(outp, "_varcorr.csv"), row.names = FALSE)
cat("[", lab, "] done\n")
