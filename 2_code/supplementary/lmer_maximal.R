# Maximal random-effects model in lme4 / lmerTest.
#
#   abs_diff ~ sem + vis + FAM + SXF + VXF + FD
#     + (1 + sem + vis + FAM + SXF + VXF | participant)
#     + (1 + sem + vis + FAM + SXF + VXF | target1)
#     + (1 + sem + vis + FAM + SXF + VXF | target2)
#
# Same structure as the glmmTMB L0 fits, refitted in lme4 so that (a) REML is
# used, which is the appropriate criterion when the fixed effects are held
# constant, and (b) Satterthwaite denominator df are available -- the focal
# interactions lean on variation across 50 identities, so the Wald z used by
# glmmTMB (df = Inf) is anticonservative for this design.
#
# Satterthwaite requires the Hessian of the variance parameters, which is
# expensive at this scale, so it is attempted with a time budget and the script
# falls back to Wald z if it does not return.
#
# Singularity is checked with isSingular() rather than a hand-rolled threshold.
#
# Usage: Rscript lmer_maximal.R <pairs.csv> <out.csv> [label]

suppressPackageStartupMessages({library(lme4); library(lmerTest)})

a <- commandArgs(trailingOnly = TRUE)
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

FULL <- "1 + sem_dissim_z + vis_dissim_z + FAM + SXF + VXF"
form <- as.formula(sprintf(
  "abs_diff ~ sem_dissim_z + vis_dissim_z + FAM + SXF + VXF + FD + (%s | participant) + (%s | target1) + (%s | target2)",
  FULL, FULL, FULL))

cat("[", lab, "] n =", nrow(df), " fam_sd =", sprintf("%.3f", fam_sd), "\n"); flush.console()
t0 <- Sys.time()
m <- lmerTest::lmer(form, data = df, REML = TRUE,
                    control = lmerControl(optimizer = "bobyqa",
                                          optCtrl = list(maxfun = 2e5),
                                          calc.derivs = FALSE))
fit_s <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
sing  <- lme4::isSingular(m)
cat(sprintf("  fit %.0fs  singular=%s\n", fit_s, sing)); flush.console()

# Satterthwaite with a budget; fall back to Wald if too slow
ddf_used <- "Satterthwaite"
co <- tryCatch({
  setTimeLimit(elapsed = 1800, transient = TRUE)
  on.exit(setTimeLimit(elapsed = Inf, transient = TRUE), add = TRUE)
  summary(m, ddf = "Satterthwaite")$coefficients
}, error = function(e) {
  cat("  Satterthwaite unavailable:", conditionMessage(e), "-> Wald\n")
  ddf_used <<- "Wald"
  NULL
})
if (is.null(co)) {
  s <- summary(as(m, "lmerMod"))$coefficients          # Estimate, SE, t
  p <- 2 * pnorm(-abs(s[, "t value"]))
  co <- cbind(s, df = Inf, `Pr(>|t|)` = p)
  colnames(co)[1:3] <- c("Estimate", "Std. Error", "t value")
}
sat_s <- as.numeric(difftime(Sys.time(), t0, units = "secs")) - fit_s

g <- function(t) if (t %in% rownames(co)) co[t, ] else rep(NA, ncol(co))
s_ <- g("SXF"); v_ <- g("VXF")
gv <- function(x, nm) if (nm %in% names(x)) unname(x[nm]) else NA

res <- data.frame(
  label = lab, n = nrow(df), fit_secs = fit_s, ddf_secs = sat_s,
  singular = sing, ddf = ddf_used, fam_sd = fam_sd,
  semXfam = gv(s_, "Estimate"), semXfam_se = gv(s_, "Std. Error"),
  semXfam_df = gv(s_, "df"), semXfam_t = gv(s_, "t value"),
  semXfam_p = gv(s_, "Pr(>|t|)"), semXfam_raw = gv(s_, "Estimate") / fam_sd,
  visXfam = gv(v_, "Estimate"), visXfam_se = gv(v_, "Std. Error"),
  visXfam_df = gv(v_, "df"), visXfam_t = gv(v_, "t value"),
  visXfam_p = gv(v_, "Pr(>|t|)"), visXfam_raw = gv(v_, "Estimate") / fam_sd)
write.csv(res, outp, row.names = FALSE)
cat(sprintf("  semXfam %+.5f (df=%.0f, p=%.3g) | visXfam %+.5f (df=%.0f, p=%.3g) [%s]\n",
            res$semXfam, res$semXfam_df, res$semXfam_p,
            res$visXfam, res$visXfam_df, res$visXfam_p, ddf_used))
cat("Saved ->", outp, "\n")
