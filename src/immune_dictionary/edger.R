args <- commandArgs(trailingOnly = TRUE)
work <- args[1]
n_genes <- as.integer(args[2])
n_samples <- as.integer(args[3])
workers <- as.integer(args[4])
suppressPackageStartupMessages(library(edgeR))
counts <- matrix(readBin(file.path(work, "counts.bin"), "integer", n = n_genes * n_samples,
                         size = 4, endian = "little"), nrow = n_genes)
meta <- read.delim(file.path(work, "samples.tsv"))
groups <- split(meta, factor(meta$condition_index, levels = sort(unique(meta$condition_index))))
fit_one <- function(g) {
    y <- DGEList(counts = counts[, g$column, drop = FALSE])
    g$condition <- factor(g$condition, levels = c("pbs", "treated"))
    design <- model.matrix(~ factor(batch) + condition, g)
    residual_df <- nrow(design) - qr(design)$rank
    if (qr(design)$rank != ncol(design) || residual_df <= 0) stop("Non-estimable treatment effect")
    keep <- filterByExpr(y, design = design)
    codes <- raw(n_genes)
    effects <- data.frame(gene_index = integer(), log2FC = numeric(), PValue = numeric(), FDR = numeric())
    if (any(keep)) {
        y <- calcNormFactors(y[keep, , keep.lib.sizes = FALSE], method = "TMM")
        y <- estimateDisp(y, design)
        tab <- glmQLFTest(glmQLFit(y, design), coef = "conditiontreated")$table
        fdr <- p.adjust(tab$PValue, method = "BH")
        significant <- is.finite(fdr) & fdr <= 0.05
        status <- rep(1L, nrow(tab))
        status[significant] <- ifelse(tab$logFC[significant] < 0, 2L,
                                      ifelse(tab$logFC[significant] > 0, 3L, 4L))
        codes[keep] <- as.raw(status)
        effects <- data.frame(gene_index = which(keep)[significant] - 1L,
                              log2FC = tab$logFC[significant], PValue = tab$PValue[significant],
                              FDR = fdr[significant])
    }
    list(codes = codes, residual_df = residual_df, effects = effects)
}
fits <- parallel::mclapply(groups, fit_one, mc.cores = workers)
if (any(vapply(fits, inherits, logical(1), "try-error"))) stop("An edgeR condition failed")
handle <- file(file.path(work, "codes.bin"), "wb")
for (fit in fits) writeBin(fit$codes, handle)
close(handle)
designs <- data.frame(condition_index = as.integer(names(fits)),
                      residual_df = vapply(fits, function(x) x$residual_df, numeric(1)))
write.table(designs, file.path(work, "design.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
effects <- do.call(rbind, lapply(names(fits), function(k) {
    x <- fits[[k]]$effects
    x$condition_index <- rep(as.integer(k), nrow(x))
    x
}))
write.table(effects, file.path(work, "effects.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
capture.output(sessionInfo(), file = file.path(work, "session.txt"))
