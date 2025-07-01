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
import scipy.stats
import pickle

from analysis_common import plot_person, add_excel, plot_expectedNormalized, str_significance, str_significance_pair

sys.path.append(str(Path(__file__).parent.parent))
from utils_plot import defaultColors
from utils_numpy import getAssertUnique, getAssertSingle

def main_extract_finalTrial():
	sessionID_maxTrial=defaultdict(int)
	sessionID_name={}

	FILE_FINAL.parent.mkdir(parents=True, exist_ok=True)

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
			assert trial["sessionParam"]["expType"]==expType
			eqType=trial["sessionParam"]["eqType"]
			posFixed=trial["sessionParam"]["posFixed"]
			binSize=trial["sessionParam"]["binSize"]
			peak_to_trough=trial["sessionParam"]["peak_to_trough"]
			setIndex=trial["sessionParam"]["setIndex"]
			sessionID=trial["sessionID"]
			person, timeStrBegin=sessionID.split("_")
	
	data={
		"sessionID": sessionID,
		"personID": person,
		"timeStrBegin": timeStrBegin,
		"posFixed": posFixed,
		"eqType": eqType,
		"numBin": int(binSize),
		"peak_to_trough": float(peak_to_trough),
		"setIndex": int(setIndex),
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
			assert trial["sessionParam"]["expType"]==expType
			eqType=trial["sessionParam"]["eqType"]
			posFixed=trial["sessionParam"]["posFixed"]
			binSize=trial["sessionParam"]["binSize"]
			peak_to_trough=trial["sessionParam"]["peak_to_trough"]
			setIndex=trial["sessionParam"]["setIndex"]
			sessionID=trial["sessionID"]
			person, timeStrBegin=sessionID.split("_")

		elif "trialName" in trial and trial["trialName"]=="Stim" and trial["trialInfo"]["blockType"]=="main":
			coordType=trial["trialInfo"]["coordType"]
			trueRatio=trial["trialInfo"]["splitRatio"]
			imageScale=trial["trialInfo"]["svgScale"]

			svgFilename=trial["trialInfo"]["paths"]
			match=PATTERN_SVGFILENAME.match(svgFilename)
			binSize_, peak_to_trough_, eqType_, posFixed_, setIndex_, trueRatio_, coordType_eqType, instance=match.groups()
			
			assert binSize_==binSize
			assert peak_to_trough_==peak_to_trough
			assert eqType_==eqType
			assert posFixed_==posFixed
			assert setIndex_==setIndex
			assert float(trueRatio_)==float(trueRatio)
			assert coordType_eqType==f"{coordType}{eqType}"

			chosenOption=trial["chosenOptionHistory"][-1]
			chosenRatio=chosenOption/100

			trialIndex_jspsych=trial["trial_index"]

			# Append the current trial's data to the DataFrame
			data.append({
				"sessionID": sessionID,
				"personID": person,
				"posFixed": posFixed,
				"eqType": eqType,
				"coordType": coordType,
				"numBin": int(binSize),
				"peak_to_trough": float(peak_to_trough),
				"setIndex": int(setIndex),
				"instance": int(instance),
				"trueRatio": float(trueRatio),
				"chosenRatio": float(chosenRatio),
				"trialIndex_jspsych": int(trialIndex_jspsych),
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
	df_trial_all, df_session_all=extract_validSessionID(df_trial_all, df_session_all, 0.0)

	unique_sessionIDs = df_session_all["sessionID"].unique()
	unique_personIDs = df_session_all["personID"].unique()
	print(len(unique_sessionIDs), len(unique_personIDs))
	assert len(unique_sessionIDs)==len(unique_personIDs)

	NUM_TRIAL=32

	unique_sessionIDs = df_trial_all["sessionID"].unique()
	# print(f"Unique session IDs: {unique_sessionIDs}")
	ratioValid=np.empty(len(unique_sessionIDs))
	for s,sessionID in enumerate(unique_sessionIDs):
		df_sessionID = df_trial_all[df_trial_all["sessionID"]==sessionID]
		# print(f"Session ID: {sessionID}")

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

	unique_personIDs = df_session_all["personID"].unique()
	
	#remove duplicated sessions in a single person
	sessionID_noDuplicate=set()
	duplicated_person=[]
	for personID in unique_personIDs:
		df_person = df_session_all[df_session_all["personID"]==personID]
		timeStrBegin_all = df_person["timeStrBegin"].to_list()
		timeStrBegin_all=sorted(timeStrBegin_all)

		timeStrBegin_maxTrial=np.empty(len(timeStrBegin_all), int)
		for ts,timeStrBegin in enumerate(timeStrBegin_all):
			df_trial = df_trial_all[(df_trial_all["personID"]==personID) & (df_trial_all["sessionID"]==f"{personID}_{timeStrBegin}")]
			trialIndex_max=df_trial["trialIndex_jspsych"].max()
			timeStrBegin_maxTrial[ts]=trialIndex_max
		selectedIndex=np.argmax(timeStrBegin_maxTrial) #if tie, first one is selected by argmax -> earlier trial is selected because timeStrBegin_all is sorted

		sessionID_selected = f"{personID}_{timeStrBegin_all[selectedIndex]}"
		sessionID_noDuplicate.add(sessionID_selected)

		if len(df_person)>1:
			duplicated_person.append((personID, len(df_person)))

	print(len(duplicated_person), duplicated_person)
	print("sessionID_noDuplicate", len(sessionID_noDuplicate))

	#remove duplicated sessions in a single set
	combinations = df_session_all[["numBin", "peak_to_trough", "eqType", "posFixed", "setIndex"]].drop_duplicates()
	duplicated_set_count={}
	for i,combination in combinations.iterrows():
		numBin, peak_to_trough, eqType, posFixed, setIndex = combination
		df_set = df_session_all[(df_session_all["numBin"]==numBin) & (df_session_all["peak_to_trough"]==peak_to_trough) & (df_session_all["eqType"]==eqType) & (df_session_all["posFixed"]==posFixed) & (df_session_all["setIndex"]==setIndex)]
		sessionID_inSet = df_set["sessionID"].to_list()
		sessionID_inSet = list(filter(lambda sessionID: sessionID in sessionID_noDuplicate, sessionID_inSet))
		if len(sessionID_inSet)<=1: continue

		sessionID_inSet=sorted(sessionID_inSet, key=lambda sessionID: int(sessionID.split("_")[1])) #sort by timeStrBegin in case of tie in maxTrial
		# print(combination, sessionID_inSet)
		# print("len(sessionID_inSet)", len(sessionID_inSet))
		duplicated_set_count[tuple(combination)]=len(sessionID_inSet)

		sessionID_maxTrial=np.empty(len(sessionID_inSet), int)
		for si,sessionID in enumerate(sessionID_inSet):
			df_trial = df_trial_all[(df_trial_all["sessionID"]==sessionID)]
			trialIndex_max=df_trial["trialIndex_jspsych"].max()
			sessionID_maxTrial[ts]=trialIndex_max
		selectedIndex=np.argmax(timeStrBegin_maxTrial) #if tie, first one is selected by argmax -> earlier trial is selected because sessionID_inSet is sorted by timeStrBegin
		
		for si,sessionID in enumerate(sessionID_inSet):
			if si==selectedIndex: continue
			sessionID_noDuplicate.remove(sessionID)

	print("duplicated_set_count", len(duplicated_set_count), min(duplicated_set_count.values()), max(duplicated_set_count.values()))
	unique_sessionIDs = np.array(sorted(sessionID_noDuplicate))

	#check if there is no duplicated setIndex
	exist=set()
	for sessionID in unique_sessionIDs:
		numBin, peak_to_trough, eqType, posFixed, setIndex=df_session_all[df_session_all["sessionID"]==sessionID][["numBin", "peak_to_trough", "eqType", "posFixed", "setIndex"]].values[0]
		assert (numBin, peak_to_trough, eqType, posFixed, setIndex) not in exist
		exist.add((numBin, peak_to_trough, eqType, posFixed, setIndex))

	ratioValid=np.empty(len(unique_sessionIDs))
	for s,sessionID in enumerate(unique_sessionIDs):
		df_sessionID = df_trial_all[df_trial_all["sessionID"]==sessionID]
		assert len(df_sessionID)<=NUM_TRIAL
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


def main_save_validData():
	df_trial_all, df_session_all=loadData()
	df_trial_all, df_session_all=extract_validSessionID(df_trial_all, df_session_all, 0.8)

	fileValidData=DIR_ANALYSIS/"ValidData.pkl"
	with open(fileValidData, "wb") as f:
		pickle.dump((df_trial_all, df_session_all), f)


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
		print('size["', size, '"]+=', num, sep="")

	ax.set_xlabel("Width (px)")
	ax.set_ylabel("Height (px)")

	ax.plot(3000, 1000, "o", markersize=f_size(1), color=(0,0,0,0.5))
	ax.plot(3000, 1000, "o", markersize=f_size(10), color=(0,0,0,0.5))
	ax.plot(3000, 1000, "o", markersize=f_size(100), color=(0,0,0,0.5))

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

	formulaName="chosenRatio_FixEqCoord_RanSessionBinPtt"

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
		
		df_ci=df_ci_prediction[(df_ci_prediction["eqType"]==eqType) & (df_ci_prediction["coordType"]==coordType) & (df_ci_prediction["numBin"]==6) & (df_ci_prediction["peak_to_trough"]==1.5)]
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
	for i,((ct,coordType),(et,eqType)) in enumerate(itertools.product(enumerate(("Polar", "Cart", )), enumerate(("Len", "Area")))):
		df_ci=df_ci_prediction[(df_ci_prediction["eqType"]==eqType) & (df_ci_prediction["coordType"]==coordType) & (df_ci_prediction["numBin"]==6) & (df_ci_prediction["peak_to_trough"]==1.5)]
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
	ax.plot(np.full(np.arange(0.8, 1.3, 0.1).shape, 0), np.arange(0.8, 1.3, 0.1), ".k")
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

	formulaName="absError_FixEqCoord_RanSessionBinPttTrue"

	df_ci_prediction = pd.read_csv(DIR_ANALYSIS/f"LMEM_{formulaName}"/"ci_prediction.tsv", sep="\t")
	print("nsim", df_ci_prediction["nsim"].unique())

	numBin_all=sorted(df_session_all["numBin"].unique())
	print(numBin_all)
	ptt_all=sorted(df_session_all["peak_to_trough"].unique())
	print(ptt_all)

	x_all=[]
	y_all=[]
	fig,ax=plt.subplots(figsize=(2,2))
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

	grouped=df_trial_all.groupby(["eqType", "coordType", "numBin", "peak_to_trough"])
	df_forPrediction=[]
	for (eqType, coordType, numBin, peak_to_trough),df in grouped:
		trueRatio_log=np.linspace(-2.5, 2.5, 65)
		for tr in trueRatio_log:
			df_forPrediction.append({
				"sessionID": "-",
				"eqType": eqType,
				"coordType": coordType,
				"numBin": numBin,
				"peak_to_trough": peak_to_trough,
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

	print("engaging", len(df_session_all))
	
	df_session_all=add_excel(df_session_all, FILE_EXCEL)

	print(df_session_all["personID"].to_numpy()[np.nonzero(np.isnan(df_session_all["sc1"].to_numpy()))[0]])
	print("sc1", df_session_all[df_session_all["personID"]=="b9Ajo0eVGuGi"]["sc1"])
	print("sc2_1", df_session_all[df_session_all["personID"]=="b9Ajo0eVGuGi"]["sc2_1"])

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
		on=["numBin", "peak_to_trough", "trueRatio", "eqType", "coordType", "setIndex", "posFixed", "instance"],
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
	df_trial_all.loc[mask, "expectedNormalized"] = expectedNormalized #in-place set

	return df_trial_all


def main_plot_expectedNormalized():
	df_trial_all, df_session_all=loadData()
	df_trial_all, df_session_all=extract_validSessionID(df_trial_all, df_session_all, 0.8)
	add_absError(df_trial_all)
	df_trial_all=add_expectedRatio(df_trial_all)

	plot_expectedNormalized(df_trial_all, 1, 0.3, None)


def main_figHTML_stim():
	dirFig=Path(__file__).parent/"Figures"/"Paper"
	fileTemplate=dirFig/f"Stim_template.html"
	fileHTML=dirFig/f"Stim_{expType}.html"
	with open(fileTemplate, "r") as f_t:
		with open(fileHTML, "w") as f_h:
			for line in f_t:
				f_h.write(line)
				if line.startswith("<body>"):
					f_h.write("<table border>")

					f_h.write("<tr>")
					f_h.write("<th>Left-to-right ratio</th>")
					for text in ("0.5 or 2.0", "0.25 or 4.0"):
						f_h.write(f"<th colspan='16'>{text}</th>")
					f_h.write("</tr>")

					f_h.write("<tr>")
					f_h.write("<th>Peak-to-trough ratio</th>")
					for i in range(2):
						for text in [1.5, 2.0, 3.0, 5.0]:
							f_h.write(f"<th colspan='4'>{text:.1f}</th>")
					f_h.write("</tr>")

					f_h.write("<tr>")
					f_h.write("<th>Bin size</th>")
					for i in range(4*2):
						for text in [6, 10, 14, 18]:
							f_h.write(f"<th colspan='1'>{text}</th>")
					f_h.write("</tr>")

					for coordType, eqType, coordTypeText in [("Polar", "Len", "Polar (equal-length)"), ("Polar", "Area", "Polar (equal-area)"), ("Cart", "Len", "Cartesian")]:
						f_h.write("<tr>\n")
						f_h.write(f"<th>{coordTypeText}</th>\n")
						for trueRatio, ptt, binSize in itertools.product((2.0, 4.0), (1.5, 2.0, 3.0, 5.0), (6, 10, 14, 18)):
							f_h.write(f'<td><img src="../../../www/Stim_random/Bin{binSize}_PtF{ptt:.1f}/{eqType}_left/Set0/main/Split{trueRatio:.1f}_{coordType}{eqType}_Ins0.svg"></td>\n')
						f_h.write("</tr>")

					f_h.write("</table>")


def main():
	# main_extract_finalTrial()
	
	# main_plot_validRatio()

	main_screenSize()

	# main_plot_person()

	# main_write_forR()

	# main_plot_chosenRatio_FixEqCoord()
	# main_plot_absError_FixEqCoord()

	# main_plot_expectedNormalized()

	# main_figHTML_stim()


if __name__=="__main__":
	DIR_PROJECT=Path(__file__).parent.parent.parent
	DIR_EXP=DIR_PROJECT/"exp"
	expType="random"
	DIR_RESULT=DIR_EXP/f"Result_{expType}"
	eqtid="eqtid"
	FILE_RAW=DIR_RESULT/f"{eqtid}.zip"
	DIR_ANALYSIS=DIR_EXP/f"Analysis_{expType}"
	FILE_FINAL=DIR_ANALYSIS/"Raw_finalTrial.zip"

	FILE_EXCEL=DIR_EXP/"Participants.xlsx"

	PATTERN_SVGFILENAME=re.compile(r"Stim_random\/Bin(\d+)_PtF(.+)\/(.+)_(.+)\/Set(\d+)\/main\/Split(.+)_(.+)_Ins(\d+).svg")

	main()