#!/usr/bin/env Rscript

# Formal mixed-type MICE engine for proper_mice_impute() (Python orchestrator).
#
# Reads:  <run_dir>/input.csv  and  <run_dir>/mice_spec.json
# Writes: imputed_001.csv ... imputed_m.csv, methods.csv, predictor_matrix.csv,
#         logged_events.csv, r_session.json, chain_diagnostics.png
#
# Each incomplete variable is imputed with a model matched to its declared kind:
#   continuous -> pmm   count -> pmm   binary -> logreg
#   nominal    -> polyreg            ordinal -> polr
# One mice() FCS chain produces m completed datasets and preserves uncertainty.
# The script fails loudly rather than silently switching methods.

suppressWarnings(suppressMessages({
  library(mice)
  library(jsonlite)
}))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("usage: run_mice.R <run_dir>")
}
run_dir <- args[[1]]

input_path <- file.path(run_dir, "input.csv")
spec_path <- file.path(run_dir, "mice_spec.json")
if (!file.exists(input_path)) stop(paste("missing input.csv in", run_dir))
if (!file.exists(spec_path)) stop(paste("missing mice_spec.json in", run_dir))

spec <- jsonlite::fromJSON(spec_path, simplifyVector = TRUE, simplifyDataFrame = FALSE)

# jsonlite collapses 1-element vectors; normalise the structures we rely on.
as_chr_vec <- function(x) if (is.null(x)) character(0) else as.character(unlist(x))
columns <- as_chr_vec(spec$columns)
row_id_col <- as.character(spec$row_id_col)
non_predictor_cols <- as_chr_vec(spec$non_predictor_cols)
vars_with_missing <- as_chr_vec(spec$vars_with_missing)

raw <- read.csv(
  input_path,
  stringsAsFactors = FALSE,
  na.strings = c("", "NA"),
  check.names = FALSE,
  colClasses = "character"
)

# Order columns: row id first, then the declared imputation columns.
ordered_cols <- c(row_id_col, columns)
missing_in_csv <- setdiff(ordered_cols, names(raw))
if (length(missing_in_csv) > 0) {
  stop(paste("input.csv missing columns:", paste(missing_in_csv, collapse = ", ")))
}
mice_data <- raw[, ordered_cols, drop = FALSE]

# ---- Type conversion per declared kind ----------------------------------
get_levels <- function(col) {
  lv <- spec$levels[[col]]
  if (is.null(lv)) NULL else as.character(unlist(lv))
}

for (col in columns) {
  r_type <- as.character(spec$r_types[[col]])
  values <- mice_data[[col]]

  if (identical(r_type, "numeric")) {
    mice_data[[col]] <- suppressWarnings(as.numeric(values))

  } else if (identical(r_type, "factor") || identical(r_type, "ordered")) {
    levels_decl <- get_levels(col)
    if (is.null(levels_decl) || length(levels_decl) == 0) {
      stop(paste("column", col, "declared categorical but has no levels"))
    }
    kind <- as.character(spec$kinds[[col]])
    if (identical(kind, "binary") && length(levels_decl) != 2) {
      stop(paste("binary column", col, "must have exactly two levels; got",
                 paste(levels_decl, collapse = ", ")))
    }
    observed <- unique(values[!is.na(values)])
    illegal <- setdiff(observed, levels_decl)
    if (length(illegal) > 0) {
      stop(paste("column", col, "has observed levels outside declared set:",
                 paste(illegal, collapse = ", ")))
    }
    mice_data[[col]] <- factor(
      values,
      levels = levels_decl,
      ordered = identical(r_type, "ordered")
    )
  } else {
    stop(paste("column", col, "has unknown r_type", r_type))
  }
}

# Row id stays numeric, never imputed, never a predictor.
mice_data[[row_id_col]] <- suppressWarnings(as.numeric(mice_data[[row_id_col]]))

# ---- Methods vector ------------------------------------------------------
all_cols <- names(mice_data)
methods <- setNames(rep("", length(all_cols)), all_cols)
valid_methods <- c("pmm", "logreg", "polyreg", "polr")
type_for_method <- list(
  pmm = c("numeric"),
  logreg = c("factor"),
  polyreg = c("factor"),
  polr = c("ordered")
)
for (col in columns) {
  meth <- spec$methods[[col]]
  meth <- if (is.null(meth)) "" else as.character(meth)
  methods[[col]] <- meth
  if (nzchar(meth)) {
    if (!(meth %in% valid_methods)) {
      stop(paste("column", col, "requests unknown method", meth))
    }
    r_type <- as.character(spec$r_types[[col]])
    ok_types <- type_for_method[[meth]]
    if (!(r_type %in% ok_types)) {
      stop(paste("method", meth, "incompatible with type", r_type,
                 "for column", col))
    }
  }
}
methods[[row_id_col]] <- ""

# ---- Predictor matrix ----------------------------------------------------
predictor_matrix <- mice::make.predictorMatrix(mice_data)
diag(predictor_matrix) <- 0
for (col in non_predictor_cols) {
  if (col %in% rownames(predictor_matrix)) predictor_matrix[col, ] <- 0
  if (col %in% colnames(predictor_matrix)) predictor_matrix[, col] <- 0
}

write.csv(
  data.frame(column = names(methods), method = unname(methods),
             stringsAsFactors = FALSE),
  file.path(run_dir, "methods.csv"),
  row.names = FALSE
)
write.csv(
  as.data.frame(predictor_matrix),
  file.path(run_dir, "predictor_matrix.csv"),
  row.names = TRUE
)

# ---- Run formal mixed-type MICE -----------------------------------------
# Surface a method change as a hard error rather than silently continuing.
imp <- withCallingHandlers(
  mice::mice(
    data = mice_data,
    m = as.integer(spec$m),
    maxit = as.integer(spec$max_iter),
    method = methods,
    predictorMatrix = predictor_matrix,
    seed = as.integer(spec$seed),
    printFlag = FALSE
  ),
  warning = function(w) {
    msg <- conditionMessage(w)
    if (grepl("changed", msg, ignore.case = TRUE) &&
        grepl("method", msg, ignore.case = TRUE)) {
      stop(paste("mice changed an imputation method:", msg))
    }
    invokeRestart("muffleWarning")
  }
)

# ---- Logged events -------------------------------------------------------
logged <- imp$loggedEvents
if (is.null(logged) || nrow(logged) == 0) {
  logged <- data.frame(
    it = integer(0), im = integer(0), dep = character(0),
    meth = character(0), out = character(0), stringsAsFactors = FALSE
  )
}
write.csv(logged, file.path(run_dir, "logged_events.csv"), row.names = FALSE)

# ---- Export completed datasets ------------------------------------------
m <- as.integer(spec$m)
for (i in seq_len(m)) {
  completed <- mice::complete(imp, i)
  out_path <- file.path(run_dir, sprintf("imputed_%03d.csv", i))
  write.csv(completed, out_path, row.names = FALSE, na = "")
}

# ---- Chain diagnostics ---------------------------------------------------
png(file.path(run_dir, "chain_diagnostics.png"), width = 1000, height = 800)
tryCatch(
  print(plot(imp)),
  error = function(e) {
    plot.new()
    title(paste("chain diagnostics unavailable:", conditionMessage(e)))
  }
)
dev.off()

# ---- Session metadata ----------------------------------------------------
session <- list(
  r_version = R.version.string,
  mice_version = as.character(utils::packageVersion("mice")),
  jsonlite_version = as.character(utils::packageVersion("jsonlite")),
  methods = as.list(methods),
  m = m,
  maxit = as.integer(spec$max_iter),
  seed = as.integer(spec$seed),
  logged_events_count = nrow(logged)
)
writeLines(
  jsonlite::toJSON(session, auto_unbox = TRUE, pretty = TRUE),
  file.path(run_dir, "r_session.json")
)

cat(sprintf(
  "run_mice.R OK — m=%d maxit=%d incomplete_vars=%d logged_events=%d\n",
  m, as.integer(spec$max_iter), length(vars_with_missing), nrow(logged)
))
