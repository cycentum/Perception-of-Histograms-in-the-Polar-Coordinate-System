# # Load necessary library
library(tibble)
library(lme4)
library(tidyverse)


args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
	stop("No arguments provided. Please provide analysisType and formulaName.")
}
analysisType <- args[1]
formulaName <- args[2]

dirProject <- args[3]
ratioType <- args[4]

dirAnalysis <- file.path(paste0(dirProject, "/exp"), paste0("Analysis_", analysisType))
if (ratioType == "true") {
	dirModel <- file.path(dirAnalysis, paste0("LMEM_", formulaName))
} else if (ratioType == "expected") {
	dirModel <- file.path(dirAnalysis, "LMEM_expectedRatio", paste0("LMEM_", formulaName))
}

if (!dir.exists(dirModel)) {
	dir.create(dirModel, recursive = TRUE)
}
print("=====================================")
print(dirModel)

# # # Load data from a tab separated file into a tibble
if (ratioType == "true") {
	filenameData="Data.tsv"
} else if (ratioType == "expected") {
	filenameData="Data_expectedRatio.tsv"
}
data <- read.table(file.path(dirAnalysis, filenameData), sep = "\t", header = TRUE) %>% as_tibble()

# # # Print the first few rows of the dataset
# print(head(data))
# print(tail(data))

print("=====================================")
print(summary(data))
print("=====================================")

as_factor <- function(data) {
	data$sessionID <- as.factor(data$sessionID)
	data$eqType <- as.factor(data$eqType)
	data$coordType <- as.factor(data$coordType)
	data$numBin <- as.factor(data$numBin)
	if ("distributionName" %in% colnames(data)) {
		data$distributionName <- as.factor(data$distributionName)
	} else if ("peak_to_trough" %in% colnames(data)) {
		data$peak_to_trough <- as.factor(data$peak_to_trough)
	}
	return(data)
}
data <- as_factor(data)

# print(head(data))
# print(tail(data))
print("=====================================")
print(summary(data))
print("=====================================")

if(formulaName == "chosenRatio_FixEqCoord_RanSessionBinDist") {
	f <- chosenRatio_log ~ 0 + trueRatio_log:eqType:coordType + 
		(0 + trueRatio_log:coordType | sessionID) +
		(0 + trueRatio_log:eqType:coordType | numBin) + (0 + trueRatio_log:eqType:coordType | distributionName)

} else if(formulaName == "absError_FixEqCoord_RanSessionBinDistTrue") {
	f <- absError ~ eqType*coordType + 
		(coordType | sessionID) + 
		(eqType*coordType | numBin) + (eqType*coordType | distributionName) + (eqType*coordType | trueRatio_log)

} else if(formulaName == "chosenRatio_FixEqCoord_RanSessionBinPtt") {
	f <- chosenRatio_log ~ 0 + trueRatio_log:eqType:coordType + 
		(0 + trueRatio_log:coordType | sessionID) +
		(0 + trueRatio_log:eqType:coordType | numBin) + (0 + trueRatio_log:eqType:coordType | peak_to_trough)

} else if(formulaName == "absError_FixEqCoord_RanSessionBinPttTrue") {
	f <- absError ~ eqType*coordType + 
		(coordType | sessionID) + 
		(eqType*coordType | numBin) + (eqType*coordType | peak_to_trough) + (eqType*coordType | trueRatio_log)
}
print(f)
print("=====================================")


fileModel <- file.path(dirModel, "model.rds")
if (file.exists(fileModel)) {
	model <- readRDS(file = fileModel)
} else {
	model <- lmer(f, data = data)
	saveRDS(model, file = fileModel)
}
print(summary(model))
print("=====================================")

if (ratioType=="expected") {
	fixef_model=fixef(model)
	write.table(as.data.frame(fixef_model), file = file.path(dirModel, "fixef.tsv"), sep = "\t", row.names = TRUE, quote = FALSE)
}


nsim <- 1600

fileSeed <- file.path(dirModel, "seed_forSimulate.txt")
if (file.exists(fileSeed)) {
	seed <- as.integer(readLines(fileSeed)[1])
	print(paste("Seed", seed, "loaded from", fileSeed, sep=" "))
} else {
	seed <- as.integer(runif(1, 1, 2^31))
	write(seed, file = fileSeed)
	print(paste("Seed", seed, "saved to", fileSeed, sep=" "))
}
print("=====================================")

fileSimulated <- file.path(dirModel, "simulated.rds")
if (file.exists(fileSimulated)) {
	simulated <- readRDS(file = fileSimulated)
} else {
	simulated <- simulate(model, nsim = nsim, seed = seed)
	saveRDS(simulated, file = fileSimulated)
}


dirModelSimulated <- file.path(dirModel, "model_simulated")
if (!dir.exists(dirModelSimulated)) {
	dir.create(dirModelSimulated)
}
for(i in 1:nsim) {
	fileModelSimulated <- file.path(dirModelSimulated, paste0(i, ".rds"))
	if (!file.exists(fileModelSimulated)) {
		print(paste("refitting", i, sep=" "))
		new_data <- data
		if (startsWith(formulaName, "chosenRatio")) {		
			new_data$chosenRatio_log <- simulated[, i]
		} else if (startsWith(formulaName, "absError")) {
			new_data$absError <- simulated[, i]
		}
		model_simulated=lmer(f, data = new_data)
		saveRDS(model_simulated, file = fileModelSimulated)
	}
}


if (startsWith(formulaName, "chosenRatio")) {
	fileFixef <- file.path(dirModel, "fixef_all.rds")
	if (file.exists(fileFixef)) {
		fixef_all <- readRDS(file = fileFixef)
	} else {
		fixef_all <- tibble()
		for(i in 1:nsim) {
			print(paste("loading", i, "for fixef", sep=" "))
			fileModelSimulated <- file.path(dirModelSimulated, paste0(i, ".rds"))
			model_simulated <- readRDS(file = fileModelSimulated)
			fixef_all <- bind_rows(fixef_all, fixef(model_simulated))
		}

		saveRDS(fixef_all, file = fileFixef)
	}

	ci_all <- tibble()
	for (col in colnames(fixef_all)) {
		boot_out <- list()
		boot_out$t <- matrix(fixef_all[[col]], ncol = 1)
		boot_out$t0 <- as.vector(fixef(model)[[col]])
		boot_out$R <- nsim

		ci99 <- boot::boot.ci(boot_out, index=1, conf = 1-0.01, type = "basic")
		ci95 <- boot::boot.ci(boot_out, index=1, conf = 1-0.05, type = "basic")
		ci99_4 <- boot::boot.ci(boot_out, index=1, conf = 1-0.01/4, type = "basic")
		ci95_4 <- boot::boot.ci(boot_out, index=1, conf = 1-0.05/4, type = "basic")
		ci_all <- bind_rows(ci_all, tibble(
			name = col, 
			ci99_lower = ci99$basic[4], 
			ci99_upper = ci99$basic[5], 
			ci95_lower = ci95$basic[4], 
			ci95_upper = ci95$basic[5],
			ci99_4_lower = ci99_4$basic[4], 
			ci99_4_upper = ci99_4$basic[5], 
			ci95_4_lower = ci95_4$basic[4], 
			ci95_4_upper = ci95_4$basic[5],
			center = boot_out$t0,
			nsim=nsim
		))
	}

	print(ci_all)
	write.table(ci_all, file = file.path(dirModel, "ci_fixef.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)


	#CI of difference
	ci_diff_all <- tibble()
	for(col0 in colnames(fixef_all)) {
		for(col1 in colnames(fixef_all)) {
			if (which(colnames(fixef_all) == col0) >= which(colnames(fixef_all) == col1)) {
				next
			}
			boot_out <- list()
			boot_out$t <- matrix(fixef_all[[col1]] - fixef_all[[col0]], ncol = 1)
			boot_out$t0 <- as.vector(fixef(model)[[col1]]) - as.vector(fixef(model)[[col0]])
			boot_out$R <- nsim

			ci99 <- boot::boot.ci(boot_out, index=1, conf = 1-0.01, type = "basic")
			ci95 <- boot::boot.ci(boot_out, index=1, conf = 1-0.05, type = "basic")
			ci99_6 <- boot::boot.ci(boot_out, index=1, conf = 1-0.01/6, type = "basic")
			ci95_6 <- boot::boot.ci(boot_out, index=1, conf = 1-0.05/6, type = "basic")
			ci_diff_all <- bind_rows(ci_diff_all, tibble(
				name0 = col0, 
				name1 = col1, 
				ci99_lower = ci99$basic[4], 
				ci99_upper = ci99$basic[5], 
				ci95_lower = ci95$basic[4], 
				ci95_upper = ci95$basic[5],
				ci99_6_lower = ci99_6$basic[4], 
				ci99_6_upper = ci99_6$basic[5], 
				ci95_6_lower = ci95_6$basic[4], 
				ci95_6_upper = ci95_6$basic[5],
				center = boot_out$t0,
				nsim=nsim
			))
		}
	}

	print(ci_diff_all)
	write.table(ci_diff_all, file = file.path(dirModel, "ci_fixef_diff.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
}

if (startsWith(formulaName, "chosenRatio")) {
	data_forPrediction <- read.table(file.path(dirAnalysis, "Data_forPrediction.tsv"), sep = "\t", header = TRUE) %>% as_tibble()
} else if (startsWith(formulaName, "absError")) {
	data_forPrediction <- read.table(file.path(dirAnalysis, "Data_forPrediction_absError.tsv"), sep = "\t", header = TRUE) %>% as_tibble()
}
data_forPrediction <- as_factor(data_forPrediction)

predicted <- predict(model, newdata = data_forPrediction, re.form = NA, allow.new.levels = FALSE)
data_forPrediction$center <- predicted

filePredicted <- file.path(dirModel, "predicted_simulated.rds")
if(file.exists(filePredicted)) {
	predicted_list <- readRDS(file = filePredicted)
} else {
	predicted_list <- list()
	for(i in 1:nsim) {
		print(paste("loading", i, "for prediction", sep=" "))
		model_simulated <- readRDS(file = file.path(dirModelSimulated, paste0(i, ".rds")))
		predicted_simulated <- predict(model_simulated, newdata = data_forPrediction, re.form = NA, allow.new.levels = FALSE)
		predicted_list[[i]] <- predicted_simulated
	}
	saveRDS(predicted_list, file = filePredicted)
}

data_forPrediction <- data_forPrediction %>% mutate(ci99_lower = NA, ci99_upper = NA, ci95_lower = NA, ci95_upper = NA, ci99_4_lower = NA, ci99_4_upper = NA, ci95_4_lower = NA, ci95_4_upper = NA, nsim=NA)
for (i in 1:length(predicted)) {
	boot_out <- list()
	boot_out$t <- matrix(unlist(lapply(predicted_list, function(x) x[i])), ncol = 1)
	boot_out$t0 <- predicted[i]
	boot_out$R <- nsim

	ci99 <- boot::boot.ci(boot_out, index = 1, conf = 1-0.01, type = "basic")
	ci95 <- boot::boot.ci(boot_out, index = 1, conf = 1-0.05, type = "basic")
	ci99_4 <- boot::boot.ci(boot_out, index = 1, conf = 1-0.01/4, type = "basic")
	ci95_4 <- boot::boot.ci(boot_out, index = 1, conf = 1-0.05/4, type = "basic")
	
	data_forPrediction$ci99_lower[i] <- ci99$basic[4]
	data_forPrediction$ci99_upper[i] <- ci99$basic[5]
	data_forPrediction$ci95_lower[i] <- ci95$basic[4]
	data_forPrediction$ci95_upper[i] <- ci95$basic[5]
	data_forPrediction$ci99_4_lower[i] <- ci99_4$basic[4]
	data_forPrediction$ci99_4_upper[i] <- ci99_4$basic[5]
	data_forPrediction$ci95_4_lower[i] <- ci95_4$basic[4]
	data_forPrediction$ci95_4_upper[i] <- ci95_4$basic[5]
	data_forPrediction$nsim[i] <- nsim
}

write.table(data_forPrediction, file = file.path(dirModel, "ci_prediction.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)


#Ci of difference
if (startsWith(formulaName, "absError")) {
	ci_diff_all <- tibble()
	for (r0 in 1:(nrow(data_forPrediction) - 1)) {
		for (r1 in (r0 + 1):nrow(data_forPrediction)) {
			boot_out <- list()
			boot_out$t <- matrix(unlist(lapply(predicted_list, function(x) x[r1] - x[r0])), ncol = 1)
			boot_out$t0 <- predicted[r1] - predicted[r0]
			boot_out$R <- nsim

			ci99 <- boot::boot.ci(boot_out, index = 1, conf = 1-0.01, type = "basic")
			ci95 <- boot::boot.ci(boot_out, index = 1, conf = 1-0.05, type = "basic")
			ci99_6 <- boot::boot.ci(boot_out, index = 1, conf = 1-0.01/6, type = "basic")
			ci95_6 <- boot::boot.ci(boot_out, index = 1, conf = 1-0.05/6, type = "basic")

			ci_diff_all <- bind_rows(ci_diff_all, tibble(
				name0 = paste(data_forPrediction$eqType[r0], data_forPrediction$coordType[r0], sep = "_"),
				name1 = paste(data_forPrediction$eqType[r1], data_forPrediction$coordType[r1], sep = "_"),
				ci99_lower = ci99$basic[4],
				ci99_upper = ci99$basic[5],
				ci95_lower = ci95$basic[4],
				ci95_upper = ci95$basic[5],
				ci99_6_lower = ci99_6$basic[4],
				ci99_6_upper = ci99_6$basic[5],
				ci95_6_lower = ci95_6$basic[4],
				ci95_6_upper = ci95_6$basic[5],
				center = boot_out$t0,
				nsim = nsim
			))
		}
	}

	print(ci_diff_all)
	write.table(ci_diff_all, file = file.path(dirModel, "ci_prediction_diff.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
}		

q()