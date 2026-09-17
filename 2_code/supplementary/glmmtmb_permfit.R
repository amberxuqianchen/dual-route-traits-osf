# One chunk of max-T permutations for a single outcome.
#
# Permutation scheme: donor reassignment of familiarity profiles across
# participants.  For permutation b, a permutation pi of the FULL participant
# index space is drawn (seeded on b alone, so every trait in a dataset uses the
# SAME pi -- required for max|z| to be taken over a coherent family).  Row
# (p, i, j) then takes its familiarity from donor pi(p) for the same target pair.
# This holds the semantic/visual pair structure, each participant's own ratings,
# and the pair-level dependence fixed, while breaking the familiarity-to-rating
# link.  It tests the FAM main effect and both interactions jointly -- there is
# no exact permutation null for an interaction alone in the presence of main
# effects.
#
# Rows whose donor lacks a familiarity rating for either target are dropped, so n
# varies slightly across permutations; n is recorded per fit.
#
# The random-effects structure is the level SELECTED for the observed fit, so the
# null statistic is generated under the same specification as the observed one.
# Cold start (glmmTMB defaults) -- no warm starting, so each permuted fit is
# independent of the observed solution.
#
# Appends one row per permutation, so the chunk is resumable at permutation
# granularity.
#
# Usage: Rscript glmmtmb_permfit.R <pairs.csv> <fam.csv> <level> <out.csv> \
#                                  <b_start> <b_end> <seed>

suppressPackageStartupMessages(library(glmmTMB))

a       <- commandArgs(trailingOnly = TRUE)
pairs_f <- a[1]; fam_f <- a[2]; lvl <- a[3]; out_f <- a[4]
b0      <- as.integer(a[5]); b1 <- as.integer(a[6]); seed <- as.integer(a[7])

df0 <- read.csv(pairs_f)
df0$pidx <- as.integer(df0$participant) + 1L      # 1-based row into FAM
df0$t1   <- as.integer(df0$target1) + 1L
df0$t2   <- as.integer(df0$target2) + 1L

FAM_MAT <- as.matrix(read.csv(fam_f, header = FALSE))   # n_participants x n_targets
N_ALL   <- nrow(FAM_MAT)

P_FULL <- "1 + sem_dissim_z + vis_dissim_z + FAM + SXF + VXF"
P_noI  <- "1 + sem_dissim_z + vis_dissim_z + FAM"
P_noF  <- "1 + sem_dissim_z + vis_dissim_z"
tt <- function(x) sprintf("(%s | target1) + (%s | target2)", x, x)
ladder <- list(
  L0 = paste0("(", P_FULL, " | participant) + ", tt(P_FULL)),
  L1 = paste0("(", P_FULL, " | participant) + ", tt(P_noI)),
  L2 = paste0("(", P_FULL, " | participant) + ", tt(P_noF)),
  L3 = paste0("(", P_FULL, " | participant) + ", tt("1")),
  L4 = paste0("(", P_noI,  " | participant) + ", tt("1")),
  L5 = paste0("(", P_noF,  " | participant) + ", tt("1")),
  L6 = paste0("(1 | participant) + ", tt("1"))
)
form <- as.formula(paste(
  "abs_diff ~ sem_dissim_z + vis_dissim_z + FAM + SXF + VXF + FD +",
  ladder[[lvl]]))

done <- integer(0)
if (file.exists(out_f)) {
  done <- tryCatch(read.csv(out_f)$b, error = function(e) integer(0))
}

for (b in b0:b1) {
  if (b %in% done) next

  # pi depends only on b, so it is identical for every trait in this dataset
  set.seed(seed + b)
  pimap <- sample.int(N_ALL)

  donor <- pimap[df0$pidx]
  fi <- FAM_MAT[cbind(donor, df0$t1)]
  fj <- FAM_MAT[cbind(donor, df0$t2)]
  ok <- !is.na(fi) & !is.na(fj)

  d <- df0[ok, ]
  fm <- (fi[ok] + fj[ok]) / 2
  fd <- abs(fi[ok] - fj[ok])
  d$FAM <- as.numeric(scale(fm))
  d$FD  <- as.numeric(scale(fd))
  d$SXF <- d$sem_dissim_z * d$FAM
  d$VXF <- d$vis_dissim_z * d$FAM
  d$participant <- factor(d$participant)
  d$target1 <- factor(d$target1)
  d$target2 <- factor(d$target2)

  t0 <- Sys.time()
  warns <- character(0)
  m <- withCallingHandlers(
    tryCatch(glmmTMB(form, dispformula = ~1, data = d),
             error = function(e) { warns <<- c(warns, "ERROR"); NULL }),
    warning = function(w) { warns <<- c(warns, conditionMessage(w))
                            invokeRestart("muffleWarning") })
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  # Save estimate, SE and z for every fixed effect, not just the focal z.  This
  # run is far too expensive to repeat for a missing column: keeping the full
  # coefficient vector means the family definition (per-channel vs combined),
  # the statistic, and any main-effect null can all be recomputed at collate time
  # without refitting.  The two route main effects are also a validity check on
  # the permutation scheme -- it permutes familiarity only, so sem/vis main
  # effects should stay close to their observed values across permutations.
  CO <- if (is.null(m)) NULL else summary(m)$coefficients$cond
  g3 <- function(t) {
    if (is.null(CO) || !(t %in% rownames(CO))) return(c(NA_real_, NA_real_, NA_real_))
    unname(CO[t, c("Estimate", "Std. Error", "z value")])
  }
  TERMS <- c(sem = "SXF", vis = "VXF", fam = "FAM",
             semmain = "sem_dissim_z", vismain = "vis_dissim_z", fd = "FD")
  vals <- lapply(TERMS, g3)

  row <- data.frame(
    b = b, n = nrow(d), secs = el,
    conv_code = if (is.null(m)) NA_integer_ else m$fit$convergence,
    pdHess = if (is.null(m)) NA else isTRUE(m$sdr$pdHess),
    logLik = if (is.null(m)) NA_real_ else as.numeric(logLik(m)),
    sigma = if (is.null(m)) NA_real_ else sigma(m),
    warned = length(warns) > 0)
  for (nm in names(TERMS)) {
    row[[nm]]              <- vals[[nm]][1]
    row[[paste0(nm, "_se")]] <- vals[[nm]][2]
    row[[paste0(nm, "_z")]]  <- vals[[nm]][3]
  }

  write.table(row, out_f, sep = ",", row.names = FALSE,
              col.names = !file.exists(out_f), append = file.exists(out_f))
  cat(sprintf("b=%d n=%d %.0fs conv=%s pd=%s semXfam_z=%.3f visXfam_z=%.3f | main sem_z=%.1f vis_z=%.1f\n",
              b, nrow(d), el, row$conv_code, row$pdHess,
              row$sem_z, row$vis_z, row$semmain_z, row$vismain_z))
}
