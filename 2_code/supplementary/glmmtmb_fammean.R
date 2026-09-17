# Continuous-familiarity variant: all pairs, mean familiarity as moderator.
#
#   abs_diff ~ (sem + vis) * fam_mean_z + fam_diff_z
#              + (1|participant) + (1|target1) + (1|target2)
#
# fam_mean_z is the pair's mean familiarity (the moderator of interest);
# fam_diff_z is |f_i - f_j|, entered as a control so that the moderator reflects
# how familiar the pair is overall rather than how *unequally* familiar it is.
# Without that control the two are confounded: a pair averaging 4 could be two
# 4s or a 1 and a 7, and those are different situations.
#
# Unlike the binarised design this keeps every pair, including mixed ones, which
# are the most informative observations for a graded moderator.
#
# dispformula is left at ~1: with a continuous moderator there is no arm to give
# its own residual variance.
#
# Usage: Rscript glmmtmb_fammean.R <all_pairs.csv> <out.csv> [label]

suppressPackageStartupMessages(library(glmmTMB))

args  <- commandArgs(trailingOnly = TRUE)
inp   <- args[1]
outp  <- args[2]
label <- if (length(args) >= 3) args[3] else basename(inp)

df <- read.csv(inp)
df$participant <- factor(df$participant)
df$target1 <- factor(df$target1)
df$target2 <- factor(df$target2)
df$fam_mean_z <- as.numeric(scale(df$fam_mean))
df$fam_diff_z <- as.numeric(scale(df$fam_diff))

if (!is.finite(sd(df$fam_mean)) || sd(df$fam_mean) < 1e-8) {
  stop("fam_mean has no variance — moderator not estimable")
}

form <- abs_diff ~ sem_dissim_z * fam_mean_z + vis_dissim_z * fam_mean_z +
  fam_diff_z + (1 | participant) + (1 | target1) + (1 | target2)

cat("[", label, "] n =", nrow(df),
    " fam_mean sd =", sprintf("%.3f", sd(df$fam_mean)), "\n"); flush.console()
t0 <- Sys.time()
m <- glmmTMB(form, data = df)
ok <- is.finite(logLik(m))
cat(sprintf("  %.0fs converged=%s\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs")), ok))

co <- summary(m)$coefficients$cond
grab <- function(nm) {
  alt <- paste(rev(strsplit(nm, ":")[[1]]), collapse = ":")
  rn <- if (nm %in% rownames(co)) nm else alt
  if (!(rn %in% rownames(co))) return(rep(NA, 4))
  co[rn, c("Estimate", "Std. Error", "z value", "Pr(>|z|)")]
}
s <- grab("sem_dissim_z:fam_mean_z")
v <- grab("vis_dissim_z:fam_mean_z")
sm <- grab("sem_dissim_z"); vm <- grab("vis_dissim_z")
fd <- grab("fam_diff_z"); fm <- grab("fam_mean_z")

res <- data.frame(
  label = label, n = nrow(df), converged = ok,
  sem_main = sm[1], vis_main = vm[1],
  fam_mean_beta = fm[1], fam_diff_beta = fd[1],
  semXfam_beta = s[1], semXfam_se = s[2], semXfam_z = s[3], semXfam_p = s[4],
  visXfam_beta = v[1], visXfam_se = v[2], visXfam_z = v[3], visXfam_p = v[4]
)
write.csv(res, outp, row.names = FALSE)
cat(sprintf("  semXfam %+.4f (p=%.3g) | visXfam %+.4f (p=%.3g)\n",
            s[1], s[4], v[1], v[4]))
cat("Saved ->", outp, "\n")
