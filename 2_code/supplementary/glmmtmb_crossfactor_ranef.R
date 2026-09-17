# Cross-factor correlation of the identity random effects, for the multi-membership defence.
#
# The two targets in a pair are entered as separate grouping factors (target1, target2)
# because glmmTMB cannot express multi-membership, so an identity appearing in both
# positions receives two independent random-effect vectors.  If those two vectors are
# capturing the same underlying identity, they should be correlated across identities.
# diagnostics_varcorr.csv holds within-factor variance components only, so this has to
# come from the predicted random effects.
#
# Fits the selected model for one outcome, then correlates ranef()$target1 against
# ranef()$target2 term by term, over the identities appearing in both positions.
#
# Usage: Rscript glmmtmb_crossfactor_ranef.R <pairs.csv> <out.csv> <label> <level>

suppressPackageStartupMessages(library(glmmTMB))

a   <- commandArgs(trailingOnly = TRUE)
inp <- a[1]; outp <- a[2]; lab <- a[3]; lvl <- a[4]

df <- read.csv(inp)
df$participant <- factor(df$participant)
df$target1 <- factor(df$target1)
df$target2 <- factor(df$target2)

df$FAM <- as.numeric(scale(df$fam_mean))
df$FD  <- as.numeric(scale(df$fam_diff))
df$SXF <- df$sem_dissim_z * df$FAM
df$VXF <- df$vis_dissim_z * df$FAM

FIX <- "abs_diff ~ sem_dissim_z + vis_dissim_z + FAM + SXF + VXF + FD"

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

cat(sprintf("[%s] fitting %s ...\n", lab, lvl)); flush.console()
t0 <- Sys.time()
m  <- glmmTMB(as.formula(paste(FIX, "+", ladder[[lvl]])), dispformula = ~1, data = df)
el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("[%s] fitted in %.0fs, conv=%s pdHess=%s\n",
            lab, el, m$fit$convergence, isTRUE(m$sdr$pdHess))); flush.console()

re <- ranef(m)$cond
r1 <- re$target1; r2 <- re$target2
ids <- intersect(rownames(r1), rownames(r2))
r1 <- r1[ids, , drop = FALSE]; r2 <- r2[ids, , drop = FALSE]

out <- do.call(rbind, lapply(colnames(r1), function(tm) {
  ct <- cor.test(r1[[tm]], r2[[tm]])
  data.frame(label = lab, level = lvl, term = tm, n_identities = length(ids),
             r = unname(ct$estimate), ci_lo = ct$conf.int[1], ci_hi = ct$conf.int[2],
             p = ct$p.value,
             sd_pos1 = sd(r1[[tm]]), sd_pos2 = sd(r2[[tm]]),
             secs = el, stringsAsFactors = FALSE)
}))
write.csv(out, outp, row.names = FALSE)
print(out)
cat("[", lab, "] done\n")
