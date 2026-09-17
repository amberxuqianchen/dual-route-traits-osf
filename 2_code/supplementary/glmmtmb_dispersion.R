# Heteroscedastic test for the recognition-split interaction model.
#
# lme4 forces a single residual variance across both recognition conditions.
# glmmTMB's dispformula lets the residual SD depend on condition, while keeping
# the crossed random effects that nlme cannot fit.  Comparing the two by
# likelihood ratio tests whether the equal-variance assumption is tenable, and
# refitting the fixed effects under the heteroscedastic model shows whether the
# interaction estimates depend on it.
#
# Usage: Rscript glmmtmb_dispersion.R <pairs.csv> <out.csv>

suppressPackageStartupMessages({
  library(glmmTMB)
})

args <- commandArgs(trailingOnly = TRUE)
inp <- args[1]
outp <- args[2]

df <- read.csv(inp)
df$participant <- factor(df$participant)
df$target1 <- factor(df$target1)
df$target2 <- factor(df$target2)
df$recognized_f <- factor(df$recognized, levels = c(0, 1),
                          labels = c("unrec", "rec"))

form <- abs_diff ~ sem_dissim_z * recognized + vis_dissim_z * recognized +
  (1 | participant) + (1 | target1) + (1 | target2)

cat("fitting homoscedastic ...\n"); flush.console()
t0 <- Sys.time()
m0 <- glmmTMB(form, data = df, REML = TRUE)
cat(sprintf("  %.0fs\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

cat("fitting dispformula = ~recognized_f ...\n"); flush.console()
t0 <- Sys.time()
m1 <- glmmTMB(form, data = df, dispformula = ~recognized_f, REML = TRUE)
cat(sprintf("  %.0fs\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))

get <- function(m, nm) {
  co <- summary(m)$coefficients$cond
  alt <- paste(rev(strsplit(nm, ":")[[1]]), collapse = ":")
  rn <- if (nm %in% rownames(co)) nm else alt
  if (!(rn %in% rownames(co))) return(c(NA, NA, NA))
  c(co[rn, "Estimate"], co[rn, "Std. Error"], co[rn, "Pr(>|z|)"])
}

lr <- tryCatch(anova(m0, m1), error = function(e) NULL)
chisq <- if (is.null(lr)) NA else lr$Chisq[2]
lrp <- if (is.null(lr)) NA else lr$`Pr(>Chisq)`[2]

# dispformula is on the log-SD scale: intercept = unrec, +coef = rec
dco <- summary(m1)$coefficients$disp
sd_unrec <- exp(dco[1, "Estimate"])
sd_rec <- exp(dco[1, "Estimate"] + dco[2, "Estimate"])

s0 <- get(m0, "sem_dissim_z:recognized"); v0 <- get(m0, "vis_dissim_z:recognized")
s1 <- get(m1, "sem_dissim_z:recognized"); v1 <- get(m1, "vis_dissim_z:recognized")

res <- data.frame(
  sem_beta_homo = s0[1], sem_se_homo = s0[2], sem_p_homo = s0[3],
  sem_beta_het  = s1[1], sem_se_het  = s1[2], sem_p_het  = s1[3],
  vis_beta_homo = v0[1], vis_se_homo = v0[2], vis_p_homo = v0[3],
  vis_beta_het  = v1[1], vis_se_het  = v1[2], vis_p_het  = v1[3],
  sd_rec = sd_rec, sd_unrec = sd_unrec, sd_ratio = sd_rec / sd_unrec,
  lrt_chisq = chisq, lrt_p = lrp,
  AIC_homo = AIC(m0), AIC_het = AIC(m1),
  n = nrow(df)
)
write.csv(res, outp, row.names = FALSE)
cat("\n"); print(round(res[, c("sem_beta_homo","sem_beta_het","vis_beta_homo",
                               "vis_beta_het","sd_ratio","lrt_chisq","lrt_p")], 4))
cat("\nSaved ->", outp, "\n")
