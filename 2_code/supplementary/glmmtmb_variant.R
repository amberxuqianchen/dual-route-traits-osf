# The three familiarity variants, all fitted with the M2 random-effect structure.
#
#   V1  all_*.csv  + fam_mean_z   every pair, graded familiarity moderator
#   V2  bn_*.csv   + arm          same-arm pairs only, binarised moderator
#   V3  bn_*.csv   + fam_mean_z   same-arm pairs only, graded moderator
#
# V2 vs V3 isolate the moderator coding on an identical sample; V1 vs V3 isolate
# the sample on an identical moderator.
#
# Random effects are M2 throughout:
#
#   (1 + mod | participant) + (1 + mod | target1) + (1 + mod | target2)
#
# chosen by likelihood-ratio test in glmmtmb_reslope_diagnostic.R, where M2 beat
# the intercepts-only structure by 900-4800 AIC across every outcome tested.
# This matters substantively, not just for fit: under intercepts-only the
# semantic interaction is negative, and under M2 it reverses sign.
#
# fam_diff_z is a control in the graded variants so the moderator reflects how
# familiar a pair is overall rather than how unequally familiar it is.
# The binarised variant additionally gets dispformula = ~arm, so each arm keeps
# its own residual SD.
#
# Usage: Rscript glmmtmb_variant.R <pairs.csv> <moderator> <out.csv> [label]

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
  disp  <- ~arm
  int_s <- "sem_dissim_z:armrec"; int_v <- "vis_dissim_z:armrec"
} else {
  df$fam_mean_z <- as.numeric(scale(df$fam_mean))
  df$fam_diff_z <- as.numeric(scale(df$fam_diff))
  fixed <- paste("abs_diff ~ sem_dissim_z * fam_mean_z +",
                 "vis_dissim_z * fam_mean_z + fam_diff_z")
  disp  <- ~1
  int_s <- "sem_dissim_z:fam_mean_z"; int_v <- "vis_dissim_z:fam_mean_z"
}

re <- sprintf("(1 + %s | participant) + (1 + %s | target1) + (1 + %s | target2)",
              mod, mod, mod)
form <- as.formula(paste(fixed, "+", re))

cat("[", label, "] mod =", mod, " n =", nrow(df), "\n"); flush.console()
t0 <- Sys.time()
m <- tryCatch(glmmTMB(form, dispformula = disp, data = df),
              error = function(e) {cat("ERROR:", conditionMessage(e), "\n"); NULL})
el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
ok <- !is.null(m) && is.finite(logLik(m))
cat(sprintf("  %.0fs converged=%s\n", el, ok))
if (!ok) quit(status = 1)

co <- summary(m)$coefficients$cond
grab <- function(nm) {
  alt <- paste(rev(strsplit(nm, ":")[[1]]), collapse = ":")
  rn <- if (nm %in% rownames(co)) nm else alt
  if (!(rn %in% rownames(co))) rep(NA, 4)
  else co[rn, c("Estimate", "Std. Error", "z value", "Pr(>|z|)")]
}
s <- grab(int_s); v <- grab(int_v)
sm <- grab("sem_dissim_z"); vm <- grab("vis_dissim_z")

res <- data.frame(
  label = label, moderator = mod, n = nrow(df), converged = ok, secs = el,
  AIC = AIC(m),
  sem_main = sm[1], vis_main = vm[1],
  semXmod_beta = s[1], semXmod_se = s[2], semXmod_z = s[3], semXmod_p = s[4],
  visXmod_beta = v[1], visXmod_se = v[2], visXmod_z = v[3], visXmod_p = v[4]
)
write.csv(res, outp, row.names = FALSE)
cat(sprintf("  semXmod %+.4f (p=%.3g) | visXmod %+.4f (p=%.3g)\n",
            s[1], s[4], v[1], v[4]))
cat("Saved ->", outp, "\n")
