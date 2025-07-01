from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import vonmises
from scipy.stats import wrapcauchy
import scipy.spatial
import scipy.optimize
import scipy.stats
from builtins import enumerate
import itertools
from pathlib import Path
from io import BytesIO
from PIL import Image
from operator import itemgetter
import sys
import pickle
import json
import socket
import argparse
import csv

from xml.etree.ElementTree import Element
from xml.etree import ElementTree

from svgwrite import Drawing
from svgwrite.mixins import Transform
from svgwrite.base import BaseElement, Metadata, Title

from utils_distribution import normalizeDensity
from utils_math import cart_to_pol, pol_to_cart, PI2
from utils_numpy import roundedArange, triangularArange, assertSymm
from utils_svg import rgbToHex, read_viewboxSize
import utils_plot
from utils import copyByte, load_or_dump_rng, write_json_as_jsVar

from make_stim_singlePeak import Bins, calc_splitRatio, convertDensity_onCoord, Param_Visualization, CanvParam, make_drawing, plotSVG, plotSVG_tutorial_border, plotSVG_tutorial_len, plotSVG_tutorial_text, _saveSVG


class Param_Density_Random():
	def __init__(self, peak_to_trough):
		assert peak_to_trough>1
		self.peak_to_trough=peak_to_trough
	

	def _make_half(self, bins, rng):
		assert bins.size%2==0
		density=rng.random(bins.size//2)
		assert density.ptp()>0
		density = (density - density.min()) / density.ptp()
		density=density*(self.peak_to_trough-1)+1
		density/=density.sum()
		return density

	def calc(self, bins, rng, splitRatio):
		density0=self._make_half(bins, rng)
		density1=self._make_half(bins, rng)
		density=np.concatenate([density0, density1*splitRatio])
		density=normalizeDensity(density, bins)
		
		return density
	

def load_textSymbol(name):
	path=DIR_WWW/"Symbols"/f"{name}.svg"
	size=read_viewboxSize(path)
	relPath=f"../../../../../../Symbols/{name}.svg#text"
	symbolInfo=type('', (), {'size': size, "path": relPath})()
	return symbolInfo


def draw(stimType, binSize, peak_to_trough, splitRatio, eqType, instance, rng, dirSet):
	bins=Bins.make_rad(binSize)

	param_density=Param_Density_Random(peak_to_trough)

	density=param_density.calc(bins, rng, splitRatio)
	splitRatio_actual=(density[binSize//2:]*bins.width[binSize//2:]).sum()/(density[:binSize//2]*bins.width[:binSize//2]).sum()
	print("splitRatio_actual", splitRatio_actual)
	
	canvParam=CanvParam(512)

	colors={"white":1, "black":0, "grayLight":0.75, "grayDark":0.5}
	for cn,c in colors.items(): colors[cn]=rgbToHex((c,c,c))

	for coordType in ("Cart", "Polar"):
		param_vis=Param_Visualization(eqType, coordType)

		print(stimType, binSize, splitRatio, param_vis, color="green")

		density_onCoord=convertDensity_onCoord(density, param_vis)
		height_canv, bins_canv=canvParam.toCanv(density_onCoord, bins, param_vis.coordType)

		if stimType=="main":
			state_all=[None,]
		elif stimType=="tutorial":
			state_all=["Orig", "Shape"]
			posFixed=dirSet.parent.name.split("_")[1]
			flipStr={"left":"F", "right":"T"}[posFixed]
			for LR in ("L", "R"):
				state_all.append(f"Text{LR}_Flip{flipStr}")
		
		for state in (state_all):
			drawing, drawingParam=make_drawing(param_vis, canvParam)
			plotSVG(drawing, drawingParam, height_canv, bins_canv, param_vis, colors["black"], None)

			if state=="Shape":
				plotSVG_tutorial_border(drawing, drawingParam, height_canv, bins_canv, param_vis, canvParam, colors["white"], None)
			
			elif state is not None and state.startswith("Text"):
				targetSplitStr=state.split("_")[0][-1:]
				targetSplit={"L":0, "R":1}[targetSplitStr]
				targetSplit=set([targetSplit])

				if param_vis.eqType=="Len":
					plotSVG_tutorial_len(drawing, drawingParam, height_canv, bins_canv, param_vis, canvParam, targetSplit)
				elif param_vis.eqType=="Area":
					plotSVG(drawing, drawingParam, height_canv, bins_canv, param_vis, colors["grayDark"], targetSplit)
					plotSVG_tutorial_border(drawing, drawingParam, height_canv, bins_canv, param_vis, canvParam, colors["black"], targetSplit)

				symbolInfo=load_textSymbol("text_"+str(param_vis.eqType))
				smallerSplit={False:0, True:1}[splitRatio<1]
				flip={"F":False, "T":True}[state[-1:]]
				plotSVG_tutorial_text(drawing, drawingParam, height_canv, bins_canv, param_vis, symbolInfo, targetSplit, smallerSplit, flip)
			
			if state is None:
				dirSVG=dirSet/stimType
			else:
				dirSVG=dirSet/stimType/state
			fileSVG=dirSVG/f"Split{splitRatio}_{param_vis}_Ins{instance}.svg"
			_saveSVG(drawing, fileSVG)


def main_draw():
	rng=np.random.default_rng()
	dirRNG=DIR_STIM/"rng_random"
	dirRNG.mkdir(parents=True, exist_ok=True)

	numSet=20
	setIndex0=12
	fileNumSet=DIR_STIM/"numSet.php"
	with open(fileNumSet, "w") as f: f.write(f"<?php $numSet={numSet}; ?>\n")
	# write_json_as_jsVar(fileNumSet, "numSet", numSet)

	for setIndex in range(numSet):
		if setIndex<setIndex0: continue

		fileRNG=dirRNG/f"Set{setIndex}.pkl"
		rng=load_or_dump_rng(fileRNG, rng)

		binSize_all=(6, 10, 14, 18)
		peak_to_trough_all=(1.5, 2.0, 3.0, 5.0)
		for binSize, peak_to_trough in itertools.product(binSize_all, peak_to_trough_all):
			for eqType, posFixed in itertools.product(("Len", "Area"), ("left", "right")):
				dirSet=DIR_STIM/f"Bin{binSize}_PtF{peak_to_trough}"/f"{eqType}_{posFixed}"/f"Set{setIndex}"
				dirSet.mkdir(parents=True, exist_ok=True)
				for stimType in ("main", "tutorial", ):
			
					if stimType=="main":
						splitRatio_all=(0.25, 0.5, 2.0, 4.0)
						numInstance=4
					elif stimType=="tutorial":
						splitRatio_all=(0.2, 5.0,)
						numInstance=1
					
					for splitRatio, instance in itertools.product(splitRatio_all, range(numInstance)):
						draw(stimType, binSize, peak_to_trough, splitRatio, eqType, instance, rng, dirSet)


def main_expectedRatio():
	expectedRatio=[]

	numSet=20
	dirRNG=DIR_STIM/"rng_random"
	for setIndex in range(numSet):
		fileRNG=dirRNG/f"Set{setIndex}.pkl"
		with open(fileRNG, "rb") as f: rng=pickle.load(f)

		binSize_all=(6, 10, 14, 18)
		peak_to_trough_all=(1.5, 2.0, 3.0, 5.0)
		for binSize, peak_to_trough in itertools.product(binSize_all, peak_to_trough_all):
			for eqType, posFixed in itertools.product(("Len", "Area"), ("left", "right")):
				for stimType in ("main", ):
			
					splitRatio_all=(2.0, 4.0)
					numInstance=4
					
					for splitRatio, instance in itertools.product(splitRatio_all, range(numInstance)):
						bins=Bins.make_rad(binSize)

						param_density=Param_Density_Random(peak_to_trough)

						density=param_density.calc(bins, rng, splitRatio)

						for coordType in ("Cart", "Polar"):
							param_vis=Param_Visualization(eqType, coordType)

							density_onCoord=convertDensity_onCoord(density, param_vis)

							length=density_onCoord

							if param_vis.coordType=="Cart":
								area=density_onCoord*bins.width
							elif param_vis.coordType=="Polar":
								area=density_onCoord**2*bins.width/2

							ratio_length=length[binSize//2:].sum()/length[:binSize//2].sum()
							ratio_area=area[binSize//2:].sum()/area[:binSize//2].sum()
							print(binSize, peak_to_trough, splitRatio, instance, param_vis, color="green")
							print(ratio_length)
							print(ratio_area)
							print()
							expectedRatio.append((setIndex, binSize, peak_to_trough, posFixed, splitRatio, instance, param_vis.eqType, param_vis.coordType, ratio_length, ratio_area))
	
	expType="random"
	DIR_ANALYSIS=DIR_EXP/f"Analysis_{expType}"
	file=DIR_ANALYSIS/f"expectedRatio.tsv"
	with open(file, 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow(["setIndex", "binSize", "peak_to_trough", "posFixed", "splitRatio", "instance", "eqType", "coordType", "ratio_length", "ratio_area"])
		writer.writerows(expectedRatio)


def main():
	main_draw()

	main_expectedRatio()


if __name__=="__main__":
	DIR_WWW=Path(__file__).parent/"www"
	DIR_STIM=DIR_WWW/"Stim_random"
	DIR_EXP=Path(__file__).parent.parent/"exp"

	main()
