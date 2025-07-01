import numpy as np
import zipfile
import os
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from collections import defaultdict, Counter
from pathlib import Path
import json
import re
import itertools
import sys	
import pickle
import scipy.stats

from analysis_common import plot_person, add_excel, plot_expectedNormalized, str_significance, str_significance_pair

sys.path.append(str(Path(__file__).parent.parent))
from utils_plot import defaultColors
from utils_numpy import getAssertUnique, getAssertSingle

def main_extract_finalTrial():
	sessionID_maxTrial=defaultdict(int)
	sessionID_name={}

	with zipfile.ZipFile(FILE_RAW, 'r') as zf:
		for name in zf.namelist():
			if not name.endswith(".json"): continue
			assert name.startswith(f"{eqtid}/")
			sessionID=name.split("/")[1]
			trial=int(name.split("/")[2].split(".")[0])
			
			if trial<sessionID_maxTrial[sessionID]: continue
			sessionID_maxTrial[sessionID]=trial
			sessionID_name[sessionID]=name

		for sessionID, name in sessionID_name.items():
			print(name)
			with zf.open(name) as source_file:
				with zipfile.ZipFile(FILE_FINAL, 'a') as target_zip:
					target_zip.writestr(name, source_file.read())


def parseJSON_session(raw):
	windowW_max=0
	windowH_max=0
	for trial in raw:
		if "timeHistory" in trial:
			for th in trial["timeHistory"]:
				if "windowSize" in th:
					windowW_max=max(windowW_max, th["windowSize"]["innerWidth"])
					windowH_max=max(windowH_max, th["windowSize"]["innerHeight"])
		
		if trial["trial_type"]=="browser-check": #browser-check does not have trialName 
			browserInfo=trial
			continue 

		if "trialName" in trial and trial["trialName"]=="Init":
			expType=trial["sessionParam"]["expType"]
			assert expType=="singlePeak"
			eqType=trial["sessionParam"]["eqType"]
			posFixed=trial["sessionParam"]["posFixed"]
			sessionID=trial["sessionID"]
			timeStrBegin=sessionID.split("_")[1]
	
	data={
		"sessionID": sessionID,
		"personID": sessionID.split("_")[0],
		"timeStrBegin": timeStrBegin,
		"posFixed": posFixed,
		"eqType": eqType,
		"windowSize": (windowW_max, windowH_max),
		"browserInfo": browserInfo,
	}
	return data


def parseJSON_trial(raw):
	data=[]

	for trial in raw:
		if trial["trial_type"]=="browser-check": #browser-check does not have trialName 
			browserInfo=trial
			continue 
		
		elif "trialName" in trial and trial["trialName"]=="Init":
			expType=trial["sessionParam"]["expType"]
			assert expType=="singlePeak"
			eqType=trial["sessionParam"]["eqType"]
			posFixed=trial["sessionParam"]["posFixed"]
			sessionID=trial["sessionID"]
			timeStrBegin=sessionID.split("_")[1]

		elif "trialName" in trial and trial["trialName"]=="Stim" and trial["trialInfo"]["blockType"]=="main":
			coordType=trial["trialInfo"]["coordType"]
			trueRatio=trial["trialInfo"]["splitRatio"]
			imageScale=trial["trialInfo"]["svgScale"]

			svgFilename=trial["trialInfo"]["paths"].split("/")[-1]
			match=PATTERN_SVGFILENAME.match(svgFilename)
			numBin, distributionName, trueRatio=match.groups()

			numBin=int(numBin)
			trueRatio=float(trueRatio)
			imageScale=float(imageScale)

			chosenOption=trial["chosenOptionHistory"][-1]
			chosenRatio=chosenOption/100

			# Append the current trial's data to the DataFrame
			data.append({
				"sessionID": sessionID,
				"posFixed": posFixed,
				"eqType": eqType,
				"coordType": coordType,
				"numBin": numBin,
				"distributionName": distributionName,
				"trueRatio": trueRatio,
				"chosenRatio": chosenRatio,
			})
	return data


def loadData():
	data_session_all=[]
	data_trial_all=[]
	with zipfile.ZipFile(FILE_FINAL, 'r') as zf:
		for name in zf.namelist():
			if not name.endswith(".json"): continue
			assert name.startswith(f"{eqtid}/")

			with zf.open(name) as source_file:
				raw=json.load(source_file)
				
				data_trial=parseJSON_trial(raw)
				data_trial_all.extend(data_trial)

				data_session=parseJSON_session(raw)
				data_session_all.append(data_session)
	
	df_trial_all=DataFrame(data_trial_all)
	df_session_all=DataFrame(data_session_all)
	return df_trial_all, df_session_all


def main_plot_validRatio():
	df_trial_all, df_session_all=loadData()

	unique_sessionIDs = df_session_all["sessionID"].unique()
	unique_personIDs = df_session_all["personID"].unique()
	print(len(unique_sessionIDs), len(unique_personIDs))
	assert len(unique_sessionIDs)==len(unique_personIDs)

	NUM_TRIAL=32

	unique_sessionIDs = df_trial_all["sessionID"].unique()
	print(f"Unique session IDs: {unique_sessionIDs}")
	ratioValid=np.empty(len(unique_sessionIDs))
	for s,sessionID in enumerate(unique_sessionIDs):
		df_sessionID = df_trial_all[df_trial_all["sessionID"]==sessionID]
		print(f"Session ID: {sessionID}")

		trueRatio_all=df_sessionID["trueRatio"].to_numpy()
		chosenRatio_all=df_sessionID["chosenRatio"].to_numpy()
		validMask=(trueRatio_all>1) & (chosenRatio_all>1) | (trueRatio_all<1) & (chosenRatio_all<1)
		numValid=np.count_nonzero(validMask)
		ratioValid[s]=numValid/NUM_TRIAL
	
	threshold_ratioValid=0.8
	print((ratioValid>=threshold_ratioValid).mean())
	print((ratioValid>=threshold_ratioValid).sum())
	print(len(unique_sessionIDs))

	plt.hist(ratioValid, bins=np.linspace(0,1,33))
	plt.show()


def extract_validSessionID(df_trial_all, df_session_all, threshold_ratioValid):
	NUM_TRIAL=32

	unique_sessionIDs = df_trial_all["sessionID"].unique()
	ratioValid=np.empty(len(unique_sessionIDs))
	for s,sessionID in enumerate(unique_sessionIDs):
		df_sessionID = df_trial_all[df_trial_all["sessionID"]==sessionID]
		# print(f"Session ID: {sessionID}")

		trueRatio_all=df_sessionID["trueRatio"].to_numpy()
		chosenRatio_all=df_sessionID["chosenRatio"].to_numpy()
		validMask=(trueRatio_all>1) & (chosenRatio_all>1) | (trueRatio_all<1) & (chosenRatio_all<1)
		numValid=np.count_nonzero(validMask)
		ratioValid[s]=numValid/NUM_TRIAL
	
	valid_sessionIDs=unique_sessionIDs[ratioValid>=threshold_ratioValid]

	df_trial_all=df_trial_all[df_trial_all["sessionID"].isin(valid_sessionIDs)]
	df_session_all=df_session_all[df_session_all["sessionID"].isin(valid_sessionIDs)]

	return df_trial_all, df_session_all


def main_ratio_posFixed():
	df_trial_all=loadData()

	df_trial_all=extract_validSessionID(df_trial_all, 0.8)
	unique_sessionIDs = df_trial_all["sessionID"].unique()
	count_posFixed=Counter()
	for s,sessionID in enumerate(unique_sessionIDs):
		df_sessionID = df_trial_all[df_trial_all["sessionID"]==sessionID]
		posFixed=df_sessionID["posFixed"].values[0]
		count_posFixed[posFixed]+=1
	
	print(count_posFixed)


def main_screenSize():
	df_trial_all, df_session_all=loadData()
	df_trial_all, df_session_all=extract_validSessionID(df_trial_all, df_session_all, 0.8)
	
	size_all=np.array(df_session_all["windowSize"].to_list())
	fig,ax=plt.subplots(subplot_kw={'aspect': 'equal'})
	# w=size_all[:,0]+np.random.uniform(-1,1,len(size_all))*10
	# h=size_all[:,1]+np.random.uniform(-1,1,len(size_all))*10
	# ax.plot(w, h, ".")

	# f_size=lambda num: (num**0.5)*6
	f_size=lambda num: (np.log10(num)+1)*6

	size_unique=np.unique(size_all, axis=0)
	for size in size_unique:
		mask=(size_all==size).all(axis=1)
		num=mask.sum()
		ax.plot(size[0], size[1], "o", markersize=f_size(num), color=(0,0,0,0.5))
		print('size["', size, '"]=', num, sep="")

	ax.set_xlabel("Width (px)")
	ax.set_ylabel("Height (px)")

	ax.plot(2200, 800, "o", markersize=f_size(1), color=(0,0,0,0.5))
	ax.plot(2200, 800, "o", markersize=f_size(10), color=(0,0,0,0.5))
	ax.plot(2200, 800, "o", markersize=f_size(100), color=(0,0,0,0.5))

	plt.show()


def add_absError(df_trial_all):
	trueRatio_all=df_trial_all["trueRatio"].to_numpy()
	chosenRatio_all=df_trial_all["chosenRatio"].to_numpy()
	trueRatio_log=np.log2(trueRatio_all)
	chosenRatio_log=np.log2(chosenRatio_all)
	absError_all=abs(chosenRatio_log-trueRatio_log)/abs(trueRatio_log)

	df_trial_all["trueRatio_log"] = trueRatio_log
	df_trial_all["chosenRatio_log"] = chosenRatio_log
	df_trial_all["absError"] = absError_all


def main_plot_chosenRatio_FixEqCoord():
	df_trial_all, df_session_all=loadData()
	df_trial_all, df_session_all=extract_validSessionID(df_trial_all, df_session_all, 0.8)

	formulaName="chosenRatio_FixEqCoord_RanSessionBinDist"

	df_ci_prediction = pd.read_csv(DIR_ANALYSIS/f"LMEM_{formulaName}"/"ci_prediction.tsv", sep="\t")
	print("nsim", df_ci_prediction["nsim"].unique())

	df_expectedRatio = pd.read_csv(DIR_ANALYSIS/"expectedRatio.tsv", sep="\t")

	fig,axs=plt.subplots(1,4,figsize=(8,4), sharex=True, sharey=True, subplot_kw={'aspect': 'equal'})
	for i,((ct,coordType),(et,eqType)) in enumerate(itertools.product(enumerate(("Polar", "Cart")), enumerate(("Len", "Area")))):
		ax=axs.flatten()[i]

		ax.set_title(f"{eqType} {coordType}")

		x_all=[]
		y_all=[]
		for trueRatio in (0.25, 0.5, 2.0, 4.0):
			df=df_trial_all[(df_trial_all["eqType"]==eqType) & (df_trial_all["coordType"]==coordType) & (df_trial_all["trueRatio"]==trueRatio)]
			chosenRatio = df["chosenRatio"].to_numpy()

			log_chosenRatio=np.log2(chosenRatio)
			log_trueRatio=np.log2(trueRatio)
			x_all.append(log_trueRatio)
			y_all.append(log_chosenRatio)

		ax.violinplot(y_all, x_all, widths=0.8, showextrema=False)

		ax.axline((0, 0), slope=1, color="black", linestyle="--")
		
		df_ci=df_ci_prediction[(df_ci_prediction["eqType"]==eqType) & (df_ci_prediction["coordType"]==coordType) & (df_ci_prediction["numBin"]==6) & (df_ci_prediction["distributionName"]=="vonMises")]
		x=df_ci["trueRatio_log"].to_numpy()
		assert (x==np.sort(x)).all()
		for p,perc in enumerate(("ci99_4", "ci95_4")):
			y=np.empty((2, len(x)))
			for lui,lu in enumerate(("lower", "upper")):
				y[lui]=df_ci[f"{perc}_{lu}"].to_numpy()
			ax.fill_between(x, y[0], y[1], alpha=0.2, color=defaultColors(et), edgecolor=None)
		
		y=df_ci[f"center"].to_numpy()
		ax.plot(x, y, color=defaultColors(et))

		if coordType=="Polar":
			log_trueRatio_expectedRatio=[]
			for trueRatio in (0.25, 0.5, 2.0, 4.0):
				log_trueRatio=np.log2(trueRatio)
				trueRatio_for_expectedRatio = 1/trueRatio if trueRatio< 1 else trueRatio
				if eqType=="Len":
					eqType_for_expectedRatio="area"
				elif eqType=="Area":
					eqType_for_expectedRatio="length"
				expectedRatio=df_expectedRatio[(df_expectedRatio["eqType"]==eqType) & (df_expectedRatio["coordType"]==coordType) & (df_expectedRatio["splitRatio"]==trueRatio_for_expectedRatio)][f"ratio_{eqType_for_expectedRatio}"].to_numpy()
				expectedRatio_forPlot=1/expectedRatio if trueRatio<1 else expectedRatio
				log_expectedRatio_forPlot=np.log2(expectedRatio_forPlot)
				log_trueRatio_expectedRatio.append(np.stack((np.full_like(log_expectedRatio_forPlot, log_trueRatio), log_expectedRatio_forPlot), axis=1))

				m=np.mean(log_expectedRatio_forPlot)
				sd=np.std(log_expectedRatio_forPlot)
				ax.errorbar(log_trueRatio, m, yerr=sd, fmt="_k")

			log_trueRatio_expectedRatio=np.concatenate(log_trueRatio_expectedRatio, axis=0)

			slope = np.sum(log_trueRatio_expectedRatio[:, 0] * log_trueRatio_expectedRatio[:, 1]) / np.sum(log_trueRatio_expectedRatio[:, 0] ** 2)
			ax.axline((0, 0), slope=slope, color="black", linestyle=":")

	fig,ax=plt.subplots(subplot_kw={'aspect': 'equal'})
	ax.axline((0, 0), slope=1, color="black", linestyle="--")
	for i,((ct,coordType),(et,eqType)) in enumerate(itertools.product(enumerate(("Polar", "Cart")), enumerate(("Len", "Area")))):
		df_ci=df_ci_prediction[(df_ci_prediction["eqType"]==eqType) & (df_ci_prediction["coordType"]==coordType) & (df_ci_prediction["numBin"]==6) & (df_ci_prediction["distributionName"]=="vonMises")]
		x=df_ci["trueRatio_log"].to_numpy()
		assert (x==np.sort(x)).all()
		for p,perc in enumerate(("ci99_4", "ci95_4")):
			y=np.empty((2, len(x)))
			for lui,lu in enumerate(("lower", "upper")):
				y[lui]=df_ci[f"{perc}_{lu}"].to_numpy()
			ax.fill_between(x, y[0], y[1], alpha=0.2, color=defaultColors(i))
		
		y=df_ci[f"center"].to_numpy()
		ax.plot(x, y, color=defaultColors(i), label=f"{eqType} {coordType}")

	ax.legend()

	df_ci_slope = pd.read_csv(DIR_ANALYSIS/f"LMEM_{formulaName}"/"ci_fixef.tsv", sep="\t")
	print("nsim", df_ci_slope["nsim"].unique())

	fig,ax=plt.subplots(figsize=(2,2))
	ax.axhline(1, color="black", linestyle="--")
	for i,((ct,coordType),(et,eqType)) in enumerate(itertools.product(enumerate(("Polar", "Cart", )), enumerate(("Len", "Area")))):
		df_ci=df_ci_slope[(df_ci_slope["name"]==f"trueRatio_log:eqType{eqType}:coordType{coordType}")]
		ci_all={}
		for p,perc in enumerate(("ci99_4", "ci95_4")):
			y=np.empty((2, ))
			for lui,lu in enumerate(("lower", "upper")):
				y[lui]=getAssertSingle(df_ci[f"{perc}_{lu}"].to_numpy())
				ci_all[perc.split("_")[0], lu]=y[lui]
			ax.fill_between((i-0.4, i+0.4), y[0], y[1], alpha=0.2, color=defaultColors(et))
		
		print(eqType, coordType, str_significance(ci_all), sep="\t")

		y=df_ci[f"center"].to_numpy()
		ax.plot((i-0.4, i+0.4), (y,y), color=defaultColors(et), label=f"{eqType} {coordType}")
	ax.set_yscale("log")
	ax.plot(np.full(np.arange(0.7, 1.5, 0.1).shape, 0), np.arange(0.7, 1.5, 0.1), ".k")
	ax.legend()

	df_ci_slope_diff = pd.read_csv(DIR_ANALYSIS/f"LMEM_{formulaName}"/"ci_fixef_diff.tsv", sep="\t")
	print("nsim", df_ci_slope_diff["nsim"].unique())

	fig,ax=plt.subplots()
	ax.axhline(0, color="black", linestyle="--")
	for i,((i1,((ct1,coordType1),(et1,eqType1))),(i0,((ct0,coordType0),(et0,eqType0)))) in enumerate(itertools.combinations(enumerate(itertools.product(enumerate(("Polar", "Cart", )), enumerate(("Len", "Area")))), 2)):
		df_ci=df_ci_slope_diff[(df_ci_slope_diff["name0"]==f"trueRatio_log:eqType{eqType0}:coordType{coordType0}") & (df_ci_slope_diff["name1"]==f"trueRatio_log:eqType{eqType1}:coordType{coordType1}")]
		ci_all={}
		for p,perc in enumerate(("ci99_6", "ci95_6")):
			y=np.empty((2, ))
			for lui,lu in enumerate(("lower", "upper")):
				y[lui]=getAssertSingle(df_ci[f"{perc}_{lu}"].to_numpy())
				ci_all[perc.split("_")[0], lu]=y[lui]
			ax.fill_between((i-0.4, i+0.4), y[0], y[1], alpha=0.2, color=defaultColors(i))
		
		print(f"{eqType1:_<4}{coordType1:_<5}", str_significance_pair(ci_all)[0], f"{eqType0:_<4}{coordType0:_<5}", str_significance_pair(ci_all)[1])

		y=df_ci[f"center"].to_numpy()
		ax.plot((i-0.4, i+0.4), (y,y), color=defaultColors(i), label=f"{eqType1}{coordType1}-{eqType0}{coordType0}")

	ax.legend()

	plt.show()


def main_plot_absError_FixEqCoord():
	df_trial_all, df_session_all=loadData()
	df_trial_all, df_session_all=extract_validSessionID(df_trial_all, df_session_all, 0.8)
	add_absError(df_trial_all)

	formulaName="absError_FixEqCoord_RanSessionBinDistTrue"

	df_ci_prediction = pd.read_csv(DIR_ANALYSIS/f"LMEM_{formulaName}"/"ci_prediction.tsv", sep="\t")
	print("nsim", df_ci_prediction["nsim"].unique())

	x_all=[]
	y_all=[]
	fig,ax=plt.subplots(figsize=(2,2), num=formulaName)
	for i,((ct,coordType),(et,eqType)) in enumerate(itertools.product(enumerate(("Polar", "Cart", )), enumerate(("Len", "Area")))):
		df_ci=df_ci_prediction[(df_ci_prediction["eqType"]==eqType) & (df_ci_prediction["coordType"]==coordType)]
		for p,perc in enumerate(("ci99_4", "ci95_4")):
			y=np.empty((2, ))
			for lui,lu in enumerate(("lower", "upper")):
				y[lui]=getAssertSingle(df_ci[f"{perc}_{lu}"].to_numpy())
			ax.fill_between((i-0.4, i+0.4), y[0], y[1], alpha=0.2, color=defaultColors(et))
		
		y=getAssertSingle(df_ci[f"center"].to_numpy())
		ax.plot((i-0.4, i+0.4), (y,y), color=defaultColors(et), label=f"{eqType} {coordType}")

		df=df_trial_all[(df_trial_all["eqType"]==eqType) & (df_trial_all["coordType"]==coordType)]
		absError = df["absError"].to_numpy()
		x_all.append(i)
		y_all.append(absError)

	ax.violinplot(y_all, x_all, widths=0.8, showextrema=False)
	ax.legend()

	df_ci_prediction_diff = pd.read_csv(DIR_ANALYSIS/f"LMEM_{formulaName}"/"ci_prediction_diff.tsv", sep="\t")
	print("nsim", df_ci_prediction_diff["nsim"].unique())

	fig,ax=plt.subplots(num=formulaName+"_diff")
	ax.axhline(0, color="black", linestyle="--")
	for i,((i1,((ct1,coordType1),(et1,eqType1))),(i0,((ct0,coordType0),(et0,eqType0)))) in enumerate(itertools.combinations(enumerate(itertools.product(enumerate(("Polar", "Cart", )), enumerate(("Len", "Area")))), 2)):
		name0=eqType0+"_"+coordType0
		name1=eqType1+"_"+coordType1
		df_ci=df_ci_prediction_diff[(df_ci_prediction_diff["name0"]==name0) & (df_ci_prediction_diff["name1"]==name1)]
		if len(df_ci)==0:
			name0, name1=name1, name0
			df_ci=df_ci_prediction_diff[(df_ci_prediction_diff["name0"]==name0) & (df_ci_prediction_diff["name1"]==name1)]
		ci_all={}
		for p,perc in enumerate(("ci99_6", "ci95_6")):
			y=np.empty((2, ))
			for lui,lu in enumerate(("lower", "upper")):
				y[lui]=getAssertSingle(df_ci[f"{perc}_{lu}"].to_numpy())
				ci_all[perc.split("_")[0], lu]=y[lui]
			ax.fill_between((i-0.4, i+0.4), y[0], y[1], alpha=0.2, color=defaultColors(i))
		
		print(f"{eqType1:_<4}{coordType1:_<5}", str_significance_pair(ci_all)[0], f"{eqType0:_<4}{coordType0:_<5}", str_significance_pair(ci_all)[1])

		y=getAssertSingle(df_ci[f"center"].to_numpy())
		ax.plot((i-0.4, i+0.4), (y,y), color=defaultColors(i), label=f"{eqType1}{coordType1}-{eqType0}{coordType0}")

	ax.legend()

	plt.show()


def main_write_forR():
	df_trial_all, df_session_all=loadData()
	df_trial_all, df_session_all=extract_validSessionID(df_trial_all, df_session_all, 0.8)

	add_absError(df_trial_all)

	df_trial_all.to_csv(DIR_ANALYSIS/"Data.tsv", index=False, sep="\t")

	grouped=df_trial_all.groupby(["eqType", "coordType", "numBin", "distributionName", ])
	df_forPrediction=[]
	for (eqType, coordType, numBin, distributionName),df in grouped:
		trueRatio_log=np.linspace(-2.5, 2.5, 65)
		for tr in trueRatio_log:
			df_forPrediction.append({
				"sessionID": "-",
				"eqType": eqType,
				"coordType": coordType,
				"numBin": numBin,
				"distributionName": distributionName,
				"trueRatio_log": tr,
			})
	df_forPrediction=DataFrame(df_forPrediction)
	df_forPrediction.to_csv(DIR_ANALYSIS/"Data_forPrediction.tsv", index=False, sep="\t")


def main_plot_person():
	df_trial_all, df_session_all=loadData()

	print(len(df_session_all))
	unique_sessionIDs = df_session_all["sessionID"].unique()
	print(len(unique_sessionIDs))
	unique_personIDs = df_session_all["personID"].unique()
	print(len(unique_personIDs))

	df_trial_all, df_session_all=extract_validSessionID(df_trial_all, df_session_all, 0.8)
	
	df_session_all=add_excel(df_session_all, FILE_EXCEL)
	print(len(df_session_all))
	print(df_session_all["personID"].to_numpy()[np.nonzero(np.isnan(df_session_all["sc1"].to_numpy()))[0]])
	print(df_session_all[df_session_all["personID"]=="OCZ5oTJxqBzz"]["sc1"])
	print(df_session_all[df_session_all["personID"]=="OCZ5oTJxqBzz"]["sc2_1"])

	plot_person(df_session_all)


def add_expectedRatio(df_trial_all):
	'''
	NOT in place
	'''

	df_expectedRatio = pd.read_csv(DIR_ANALYSIS/"expectedRatio.tsv", sep="\t")

	df_expectedRatio["splitRatio"]=df_expectedRatio["splitRatio"]
	df_expectedRatio_inverse=df_expectedRatio.copy()
	df_expectedRatio_inverse["splitRatio"]=1/df_expectedRatio_inverse["splitRatio"]
	df_expectedRatio_inverse["ratio_length"]=1/df_expectedRatio_inverse["ratio_length"]
	df_expectedRatio_inverse["ratio_area"]=1/df_expectedRatio_inverse["ratio_area"]
	df_expectedRatio = pd.concat([df_expectedRatio, df_expectedRatio_inverse], ignore_index=True)

	df_expectedRatio.rename(columns={"splitRatio": "trueRatio"}, inplace=True)
	df_expectedRatio.rename(columns={"binSize": "numBin"}, inplace=True)
	df_expectedRatio.rename(columns={"ratio_length": "expectedRatio_Len"}, inplace=True)
	df_expectedRatio.rename(columns={"ratio_area": "expectedRatio_Area"}, inplace=True)

	df_trial_all = pd.merge(
		df_trial_all,
		df_expectedRatio,
		on=["numBin", "distributionName", "trueRatio", "eqType", "coordType"],
		how="inner"
	)

	for eqType_expected in ("Len", "Area"):
		df_trial_all[f"expectedRatio_{eqType_expected}_log"] = np.log2(df_trial_all[f"expectedRatio_{eqType_expected}"].to_numpy())

	mask=df_trial_all["coordType"]=="Polar"
	chosenRatio_log=df_trial_all.loc[mask, "chosenRatio_log"].to_numpy()
	expectedRatio_Len_log=df_trial_all.loc[mask, "expectedRatio_Len_log"].to_numpy()
	expectedRatio_Area_log=df_trial_all.loc[mask, "expectedRatio_Area_log"].to_numpy()
	expectedNormalized = (chosenRatio_log-expectedRatio_Len_log)/(expectedRatio_Area_log-expectedRatio_Len_log) #Linear transform of chosenRatio_log; expectedRatio_Len_log=0; expectedRatio_Area_log=1
	expectedNormalized=2*expectedNormalized-1 #expectedNormalized=-1; expectedNormalized=+1
	df_trial_all.loc[mask, "expectedNormalized"] = expectedNormalized

	return df_trial_all


def main_plot_expectedNormalized():
	df_trial_all, df_session_all=loadData()
	df_trial_all, df_session_all=extract_validSessionID(df_trial_all, df_session_all, 0.8)
	add_absError(df_trial_all)
	df_trial_all=add_expectedRatio(df_trial_all)

	plot_expectedNormalized(df_trial_all, 4, 1, np.arange(0, 14, 4))


def main_plot_expectedNormalized_trivial():
	fig,axs=plt.subplots(1,2, figsize=(6,10/7), tight_layout=True)
	for et,eqType in enumerate(("Len", "Area")):
		ax=axs[et]
		ax.set_title(eqType)
		for v in (-1,+1):
			ax.axvline(v, color=(0.5,)*3, linestyle="--")

		ax.set_xlim(-5, 5)
		
		x=np.linspace(-5, 5, 100)
		y=[2,1][et]*scipy.stats.norm(-1,0.2).pdf(x)+[1,2][et]*scipy.stats.norm(1,0.2).pdf(x)
		ax.fill_between(x, y, color="black")
	plt.show()


def main():
	main_extract_finalTrial()
	
	# main_plot_validRatio()
	# main_ratio_posFixed()
	# main_screenSize()

	# main_plot_person()

	# main_plot_chosenRatio_FixEqCoord()
	# main_plot_absError_FixEqCoord()

	# main_write_forR()

	# main_plot_expectedNormalized()
	# main_plot_expectedNormalized_trivial()


if __name__=="__main__":
	DIR_PROJECT=Path(__file__).parent.parent.parent
	DIR_EXP=DIR_PROJECT/"exp"
	expType="singlePeak"
	DIR_RESULT=DIR_EXP/f"Result_{expType}"
	eqtid="eqtid"
	FILE_RAW=DIR_RESULT/f"{eqtid}.zip"
	DIR_ANALYSIS=DIR_EXP/f"Analysis_{expType}"
	FILE_FINAL=DIR_ANALYSIS/"Raw_finalTrial.zip"

	FILE_EXCEL=DIR_EXP/"Participants.xlsx"

	PATTERN_SVGFILENAME=re.compile(r"Bin(\d+)_(.+)_Split(.+)_.+\.svg")

	main()