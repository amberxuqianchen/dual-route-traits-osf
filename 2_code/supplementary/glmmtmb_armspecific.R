# Joint model that exactly nests the two-fit specification.
#
# The two-fit contrast (beta_rec - beta_unrec) is tested by assuming the two
# arm estimates are independent.  They are not: both arms contain the same
# participants and the same target identities, so their random-effect estimates
# are correlated and the true Var(delta) is
#
#     Var(b_rec) + Var(b_unrec) - 2*Cov(b_rec, b_unrec)
#
# with the covariance unknown.  This model estimates both arms in ONE likelihood
# with everything free to differ by arm:
#
#   abs_diff ~ 0 + arm + arm:sem + arm:vis
#              + (0 + arm | participant) + (0 + arm | target1) + (0 + arm | target2)
#   dispformula = ~arm
#
# Each grouping factor gets an unstructured 2x2 covariance, so the cross-arm
# correlation is estimated rather than assumed, and dispformula gives each arm
# its own residual SD.  The contrast is then a linear combination of fixed
# effects whose SE uses the actual covariance from vcov().
#
# Reports both the naive (independence) SE and the correct one, so the size of
# the assumption's effect is visible.
#
# Usage: Rscript glmmtmb_armspecific.R <pairs.csv> <out.csv> [label]

suppressPackageStartupMessages(library(glmmTMB))

args  <- commandArgs(trailingOnly = TRUE)
inp   <- args[1]
outp  <- args[2]
label <- if (length(args) >= 3) args[3] else basename(inp)

df <- read.csv(inp)
df$arm <- factor(ifelse(df$recognized == 1, "rec", "unrec"),
                 levels = c("unrec", "rec"))
df$participant <- factor(df$participant)
df$target1 <- factor(df$target1)
df$target2 <- factor(df$target2)

form <- abs_diff ~ 0 + arm + arm:sem_dissim_z + arm:vis_dissim_z +
  (0 + arm | participant) + (0 + arm | target1) + (0 + arm | target2)

cat("[", label, "] n =", nrow(df),
    " rec =", sum(df$arm == "rec"), " unrec =", sum(df$arm == "unrec"), "\n")
cat("fitting joint arm-specific model ...\n"); flush.console()
t0 <- Sys.time()
m <- glmmTMB(form, dispformula = ~arm, data = df)
cat(sprintf("  %.0fs  converged=%s\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs")),
            is.finite(logLik(m))))

co <- summary(m)$coefficients$cond
V  <- vcov(m)$cond

contrast <- function(nm_rec, nm_unrec) {
  if (!(nm_rec %in% rownames(co)) || !(nm_unrec %in% rownames(co))) {
    return(rep(NA, 7))
  }
  br <- co[nm_rec, "Estimate"]; bu <- co[nm_unrec, "Estimate"]
  vr <- V[nm_rec, nm_rec];      vu <- V[nm_unrec, nm_unrec]
  cv <- V[nm_rec, nm_unrec]
  d  <- br - bu
  se_correct <- sqrt(vr + vu - 2 * cv)     # uses the estimated covariance
  se_naive   <- sqrt(vr + vu)              # what the two-fit Wald assumes
  z <- d / se_correct
  c(br, bu, d, se_correct, se_naive, z, 2 * pnorm(-abs(z)))
}

# glmmTMB names interaction terms as "armrec:sem_dissim_z" etc.
nm <- function(arm, v) paste0("arm", arm, ":", v)
sem <- contrast(nm("rec", "sem_dissim_z"), nm("unrec", "sem_dissim_z"))
vis <- contrast(nm("rec", "vis_dissim_z"), nm("unrec", "vis_dissim_z"))

# residual SD per arm (log link on dispersion)
dco <- summary(m)$coefficients$disp
sd_unrec <- exp(dco[1, "Estimate"])
sd_rec   <- exp(sum(dco[, "Estimate"]))

# cross-arm correlation of the random intercepts, per grouping factor
vc <- VarCorr(m)$cond
cors <- sapply(names(vc), function(g) {
  cm <- attr(vc[[g]], "correlation")
  if (is.null(cm) || nrow(cm) < 2) NA else cm[1, 2]
})

res <- data.frame(
  label = label, n = nrow(df),
  sem_beta_rec = sem[1], sem_beta_unrec = sem[2], sem_delta = sem[3],
  sem_se_correct = sem[4], sem_se_naive = sem[5], sem_z = sem[6], sem_p = sem[7],
  vis_beta_rec = vis[1], vis_beta_unrec = vis[2], vis_delta = vis[3],
  vis_se_correct = vis[4], vis_se_naive = vis[5], vis_z = vis[6], vis_p = vis[7],
  sd_rec = sd_rec, sd_unrec = sd_unrec,
  r_participant = cors["participant"], r_target1 = cors["target1"],
  r_target2 = cors["target2"]
)
write.csv(res, outp, row.names = FALSE)

cat("\nSEM  rec", sprintf("%.4f", sem[1]), " unrec", sprintf("%.4f", sem[2]),
    " delta", sprintf("%+.4f", sem[3]),
    "\n     SE correct", sprintf("%.4f", sem[4]),
    " vs naive", sprintf("%.4f", sem[5]),
    sprintf(" (ratio %.2f)", sem[4] / sem[5]),
    "  p =", format.pval(sem[7], digits = 3), "\n")
cat("VIS  rec", sprintf("%.4f", vis[1]), " unrec", sprintf("%.4f", vis[2]),
    " delta", sprintf("%+.4f", vis[3]),
    "\n     SE correct", sprintf("%.4f", vis[4]),
    " vs naive", sprintf("%.4f", vis[5]),
    sprintf(" (ratio %.2f)", vis[4] / vis[5]),
    "  p =", format.pval(vis[7], digits = 3), "\n")
cat("cross-arm RE correlations:",
    paste(sprintf("%s=%.3f", names(cors), cors), collapse = "  "), "\n")
cat("Saved ->", outp, "\n")
