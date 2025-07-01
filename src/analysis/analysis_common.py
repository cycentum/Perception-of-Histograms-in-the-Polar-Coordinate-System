from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def add_excel(df_session_all, fileExcel):
	df_excel=pd.read_excel(fileExcel, sheet_name=1)

	df_excel = df_excel[df_excel['id'].isin(df_session_all['personID'])]

	df_excel.rename(columns={'id': 'personID'}, inplace=True)

	df_session_all = df_session_all.merge(df_excel, on='personID', how='left')
	return df_session_all


def plot_person(df_session_all):
	gender_all=df_session_all[~np.isnan(df_session_all["sc1"])]["sc1"].to_numpy()
	count_gender = Counter(gender_all)
	print(count_gender)
	print(count_gender[1], "males,", count_gender[2], "females")

	fig,axs=plt.subplots(1,2, figsize=(6,3), tight_layout=True)
	ax=axs[0]
	ax.bar(count_gender.keys(), count_gender.values(), color="k")
	ax.set_xticks((1,2),("M","F"))
	ax.set_xlabel("Gender")
	ax.set_ylabel("Count")

	age_all=df_session_all[~np.isnan(df_session_all["sc1"])]["sc2_1"].to_numpy()
	print(age_all.min(), age_all.max())

	# count=list(Counter(age_all).items())
	# count=sorted(count, key=lambda x:x[0])
	# for k,v in count:
	# 	print(k,v)
	count=Counter(age_all)
	age=np.array(sorted(count.keys()))
	num=np.array([count[a] for a in age])
	for v,n in ((20,"twenties"), (30,"thirties"), (40, "forties")):
		print(v, num[(age>=v)&(age<v+10)].sum(), n)

	ax=axs[1]
	# ax.hist(age_all, bins=np.arange(20,51,5), color="k")
	ax.hist(age_all, bins=np.arange(20,51,1), color="k")
	ax.set_xlabel("Age")
	ax.set_ylabel("Count")

	plt.show()


def plot_expectedNormalized(df_trial_all, markersize, elinewidth, hist_yticks):
	coordType="Polar"

	expectedNormalized = df_trial_all["expectedNormalized"].to_numpy()
	bins=np.arange(np.floor(np.nanmin(expectedNormalized)), np.ceil(np.nanmax(expectedNormalized)), 1.0)

	fig,axs=plt.subplots(2,1, figsize=(8,8), sharex=True, sharey=True)
	for et,eqType in enumerate(("Len", "Area")):
		ax=axs[et]
		ax.set_title(eqType)
		for v in (-1,+1):
			ax.axvline(v, color=(0.5,)*3, linestyle="--")
		
		df=df_trial_all[(df_trial_all["coordType"]==coordType) & (df_trial_all["eqType"]==eqType)]
		expectedNormalized = df["expectedNormalized"].to_numpy()
		hist,bins=np.histogram(expectedNormalized, bins=bins, density=True)
		binCenter=(bins[:-1]+bins[1:])/2
		ax.bar(binCenter, hist, width=np.diff(bins))

	fig,axs=plt.subplots(2,4, figsize=(20,8), sharex=True, sharey=True)
	for et,eqType in enumerate(("Len", "Area")):
		for tr,trueRatio in enumerate((0.25, 0.5, 2.0, 4.0)):
			ax=axs[et,tr]
			ax.set_title(f"{eqType} {trueRatio}")
			for v in (-1,+1):
				ax.axvline(v, color=(0.5,)*3, linestyle="--")
			
			df=df_trial_all[(df_trial_all["coordType"]==coordType) & (df_trial_all["eqType"]==eqType) & (df_trial_all["trueRatio"]==trueRatio)]
			expectedNormalized = df["expectedNormalized"].to_numpy()
			hist,bins=np.histogram(expectedNormalized, bins=bins, density=True)
			binCenter=(bins[:-1]+bins[1:])/2
			ax.bar(binCenter, hist, width=np.diff(bins))

	means={}
	fig,axs=plt.subplots(2,2, figsize=(6,10), sharex=True, sharey="row", gridspec_kw={"height_ratios":[1, 6]}, tight_layout=True)
	for et,eqType in enumerate(("Len", "Area")):
		ax=axs[1, et]
		# ax.set_title(eqType)
		for v in (-1,+1):
			ax.axvline(v, color=(0.5,)*3, linestyle="--")
		
		df=df_trial_all[(df_trial_all["coordType"]==coordType) & (df_trial_all["eqType"]==eqType)]
		df_grouped=df.groupby("sessionID")
		m=df_grouped["expectedNormalized"].mean()
		sd=df_grouped["expectedNormalized"].std()
		index=m.argsort()
		m=m[index]
		sd=sd[index]
		y=np.arange(len(m))
		ax.errorbar(m, y, xerr=sd, fmt=".k", markersize=markersize, elinewidth=elinewidth)

		means[eqType]=m

	mi=np.min([np.min(means[eqType]) for eqType in ("Len", "Area")])
	ma=np.max([np.max(means[eqType]) for eqType in ("Len", "Area")])
	binWidth=0.5
	bins=np.arange(np.floor(mi), np.ceil(ma)+1, binWidth)+binWidth/2

	for et,eqType in enumerate(("Len", "Area")):
		ax=axs[0, et]
		ax.set_title(eqType)
		for v in (-1,+1):
			ax.axvline(v, color=(0.5,)*3, linestyle="--")
		
		hist,_=np.histogram(means[eqType], bins=bins)
		print("hist.max()", hist.max())
		binCenter=(bins[:-1]+bins[1:])/2
		ax.bar(binCenter, hist, width=np.diff(bins), color="k")
		if hist_yticks is not None:
			ax.set_yticks(hist_yticks)
		
	plt.show()


def str_significance(ci_all):
	if ci_all["ci99", "lower"]>1:
		text=f"> 1; p < 0.01"
	elif ci_all["ci95", "lower"]>1:
		text=f"> 1; p < 0.05"
	elif ci_all["ci99", "upper"]<1:
		text=f"< 1; p < 0.01"
	elif ci_all["ci95", "upper"]<1:
		text=f"< 1; p < 0.05"
	else:
		text="ns"
	return text


def str_significance_pair(ci_all):
	if ci_all["ci99", "lower"]>0:
		text=f">", f"p < 0.01"
	elif ci_all["ci95", "lower"]>0:
		text=f">", f"p < 0.05"
	elif ci_all["ci99", "upper"]<0:
		text=f"<", f"p < 0.01"
	elif ci_all["ci95", "upper"]<0:
		text=f"<", f"p < 0.05"
	else:
		text="ns", ""
	return text

