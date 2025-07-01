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


class CanvParam:
	def __init__(self, canvSize):
		self.canvSize=canvSize


	@property
	def polar_rMax(self):
		return self.canvSize/2
	

	@property
	def polar_size(self):
		return self.canvSize


	@property
	def cart_height(self):
		return self.canvSize
	

	@property
	def cart_width(self):
		return self.canvSize


	def toCanv(self, height, bins, coordType):
		height_normalized=height/height.max()
		if coordType=="Cart":
			bins_canv=bins.new_transformed((0, self.cart_width))
			height_canv=height_normalized*self.cart_height
			return height_canv, bins_canv
		
		elif coordType=="Polar":
			height_canv=height_normalized*self.polar_rMax
			return height_canv, bins


	def canvSize(self, coordType):
		if coordType=="Cart":
			width=self.cart_width
			height=self.cart_height
		
		elif coordType=="Polar":
			size=self.polar_rMax*2
			width=size
			height=size
		
		return type('', (), {"width":width, "height":height})()


def make_arc(cx, cy, radius, start_rad, end_rad):
	x1=cx+radius*np.cos(start_rad)
	y1=cy+radius*np.sin(start_rad)
	x2=cx+radius*np.cos(end_rad)
	y2=cy+radius*np.sin(end_rad)
	large_arc_flag=0
	sweep_flag=0
	command=f"A {radius},{radius} 0 {large_arc_flag},{sweep_flag} {x2},{y2}"
	return (x1,y1),command


def make_drawing(param_vis, canvParam):
	gap=canvParam.canvSize/2**7

	transform="scaleY(-1)"

	if param_vis.coordType=="Cart":
		viewBox=(0, 0, canvParam.cart_width+gap, canvParam.cart_height)
	elif param_vis.coordType=="Polar":
		viewBox=(-canvParam.polar_rMax, -canvParam.polar_rMax, canvParam.polar_size+gap, canvParam.polar_size)
	viewBoxStr=" ".join(map(str, viewBox))

	drawing=Drawing(style="transform:"+transform, debug=False, viewBox=viewBoxStr)
	
	group=drawing.g(id="main")
	drawing.add(group)

	drawingParam={"gap":gap, }

	return drawing, drawingParam


def plotSVG(drawing, drawingParam, height, bins, param_vis, fillColor, targetSplit):
	black=rgbToHex((0,0,0))
	white=rgbToHex((1,1,1))
	
	borderWidth=1/2**7
	borderColor=rgbToHex((0.75, 0.75, 0.75))

	gap=drawingParam["gap"]
	
	group=_getChildById(drawing, "main")
	
	group_bins=drawing.g(fill=fillColor, stroke_width=0)
	group.add(group_bins)

	if targetSplit is None: targetSplit=(0,1)

	for split in targetSplit:
		if split==0:
			edge_half=bins.edge[:bins.size//2+1]
			height_half=height[:bins.size//2]
		elif split==1:
			edge_half=bins.edge[bins.size//2:]
			height_half=height[bins.size//2:]

		if param_vis.coordType=="Cart":
			points=[]
			points.append((edge_half[0], 0))
			for b,(b0,b1,h) in enumerate(zip(edge_half[:-1], edge_half[1:], height_half)):
				points.append((b0,h))
				points.append((b1,h))
			points.append((edge_half[-1], 0))
			points=np.array(points)

			if split==1:
				points[:,0]+=gap

			commands=[]
			for i,(x,y) in enumerate(points):
				if i==0:
					c="M"
				else:
					c="L"
				commands.append(f"{c}{x},{y}")

		elif param_vis.coordType=="Polar":
			commands=[]
			edge=-edge_half+np.pi/2

			if split==0: cx=0
			elif split==1: cx=gap

			for b,(b0,b1,h) in enumerate(zip(edge[:-1], edge[1:], height_half)):
				(x,y), arc=make_arc(cx, 0, h, b0, b1)
				if b==0:
					c="M"
				else:
					c="L"

				commands.append(f"{c}{x},{y}")
				commands.append(f"{arc}")
		
		commands.append("Z")
		path=" ".join(commands)

		group_bins.add(drawing.path(d=path))


def _getChildById(group, ID):
	for e in group.elements:
		if hasattr(e, "attribs") and "id" in e.attribs and e.attribs["id"]==ID:
			return e
	return None


def plotSVG_tutorial_border(drawing, drawingParam, height, bins, param_vis, canvParam, borderColor, targetSplit):
	strokeWidth=canvParam.canvSize/2**8

	gap=drawingParam["gap"]
	
	group=_getChildById(drawing, "main")
	group_border=drawing.g(stroke_width=strokeWidth, stroke=borderColor, fill="none", stroke_linecap="round")
	group.add(group_border)

	if targetSplit is None: targetSplit=(0,1)

	if param_vis.coordType=="Cart":
		for b,(b0,w,h) in enumerate(zip(bins.edge[:-1], bins.width, height)):
			split=b//(bins.size//2)
			if split not in targetSplit: continue

			left=b0
			width=w
			if split==1:
				left+=gap
			border=drawing.rect(insert=(left,0), size=(width, h), class_=f"sp{split}")
			group_border.add(border)

	elif param_vis.coordType=="Polar":
		edge=-bins.edge+np.pi/2
		for b,(b0,b1,h) in enumerate(zip(edge[:-1], edge[1:], height)):
			split=b//(bins.size//2)
			if split not in targetSplit: continue

			if split==0: cx=0
			elif split==1: cx=gap
			(x,y),arc=make_arc(cx, 0, h, b0, b1)
			path=f"M{cx},0 L{x},{y} {arc} L{cx},{0}"
			border=drawing.path(d=path, class_=f"sp{split}")
			group_border.add(border)


def plotSVG_tutorial_len(drawing, drawingParam, height, bins, param_vis, canvParam, targetSplit):
	strokeWidth=canvParam.canvSize/2**8
	strokeColor=rgbToHex((0.75,0.75,0.75))

	gap=drawingParam["gap"]
	
	group=_getChildById(drawing, "main")
	group_len=drawing.g(stroke_width=strokeWidth, stroke=strokeColor, fill="none")
	group.add(group_len)

	if targetSplit is None: targetSplit=(0,1)

	if param_vis.coordType=="Cart":
		for b,(bc,h) in enumerate(zip(bins.center, height)):
			split=b//(bins.size//2)
			if split not in targetSplit: continue

			line=drawing.line(start=(bc,0), end=(bc,h), class_=f"sp{split}")
			group_len.add(line)

	elif param_vis.coordType=="Polar":
		binCenter=-bins.center+np.pi/2
		for b,(bc,h) in enumerate(zip(binCenter, height)):
			split=b//(bins.size//2)
			if split not in targetSplit: continue

			if split==0: cx=0
			elif split==1: cx=gap

			topX=cx+h*np.cos(bc)
			topY=h*np.sin(bc)
			line=drawing.line(start=(cx,0), end=(topX,topY), class_=f"sp{split}")
			group_len.add(line)


def plotSVG_tutorial_text(drawing, drawingParam, height, bins, param_vis, textSymbol, targetSplit, smallerSplit, flip):
	gap=drawingParam["gap"]
	
	group=_getChildById(drawing, "main")
	group_text=drawing.g()
	group.add(group_text)

	scale=1/2
	if flip:
		transform=f"scale({-scale},{scale}) translate({-textSymbol.size[0]},0) " #flip text
	else:
		transform=f"scale({scale})"

	drawing.defs.add(drawing.use(href=textSymbol.path, id="t", transform=transform))

	textSize=np.array(textSymbol.size)*scale
	textSizeHalf=textSize/2

	if targetSplit is None: targetSplit=(0,1)

	if param_vis.coordType=="Cart":
		for b,(bc,h) in enumerate(zip(bins.center, height)):
			split=b//(bins.size//2)
			if split not in targetSplit: continue

			textCenterX=bc
			if split==1:
				textCenterX+=gap
			
			textCenterY=h/2
			use=drawing.use(href="#t", x=textCenterX-textSizeHalf[0], y=textCenterY-textSizeHalf[1], class_=f"sp{split}")
			group_text.add(use)

	elif param_vis.coordType=="Polar":
		binCenter=-bins.center+np.pi/2
		for b,(bc,w,h) in enumerate(zip(binCenter, bins.width, height)):
			split=b//(bins.size//2)
			if split not in targetSplit: continue

			if split==0: cx=0
			elif split==1: cx=gap

			if param_vis.eqType=="Len":
				if split==smallerSplit:
					sig={0:-1, 1:+1}[split]
					textR=h+sig*textSizeHalf[0]*np.cos(bc)
				else:
					textR=h/2
			elif param_vis.eqType=="Area":
				textR=4/3*h*np.sin(w/2)/w #center of gravity
			textCenterX=cx+textR*np.cos(bc)
			textCenterY=textR*np.sin(bc)

			use=drawing.use(href="#t", x=textCenterX-textSizeHalf[0], y=textCenterY-textSizeHalf[1], class_=f"sp{split}")
			group_text.add(use)


class Bins:
	def __init__(self, edge, width, center):
		self.size=len(width)
		assert len(edge)==self.size+1
		assert len(center)==self.size
		self.edge=edge
		self.width=width
		self.center=center


	def new_transformed(self, range):
		mi=self.edge.min()
		ptp=self.edge.ptp()
		ptp_new=range[1]-range[0]
		edge_new=(self.edge-mi)/ptp*ptp_new+range[0]
		width_new=self.width/ptp*ptp_new
		center_new=(self.center-mi)/ptp*ptp_new+range[0]
		return Bins(edge_new, width_new, center_new)


	@staticmethod
	def make_rad(binSize):
		if binSize%2==0:
			edge=np.arange(-binSize//2, binSize//2+1)/(binSize//2)*np.pi
			center=(np.arange(-binSize//2, binSize//2)+1/2)/(binSize//2)*np.pi
		# else:
		# 	center=np.arange(-(binSize-1), binSize-1+1, 2)/binSize*np.pi
		# 	edge=np.arange(-binSize, binSize+1, 2)/binSize*np.pi

		# assert binSize%2==0
		# edge=(np.arange(binSize+1)*4-3*binSize)/binSize*np.pi/2
		# center=( (2*np.arange(binSize)+1)*2-3*binSize )/2/binSize*np.pi

		width=PI2/binSize
		width=width*np.ones(binSize, float)
		return Bins(edge, width, center)
	

class Param_Visualization:
	def __init__(self, eqType, coordType):
		'''
		eqType: (if coordType==Cart) None; else Len | Area;
		coordType: Cart | Polar
		'''
		assert coordType in ("Cart" , "Polar")
		if eqType is not None: assert eqType in ("Len", "Area")
		self.eqType=eqType
		self.coordType=coordType
		
		
	def __str__(self):
		if self.eqType is None:
			return f"{self.coordType}"
		else:
			return f"{self.coordType}{self.eqType}"
	

	@staticmethod
	def makeAll(stimType):
		param_all=[]
		for eqType,coordType in itertools.product(("Len", "Area"), ("Cart", "Polar",), ):
			param_all.append(Param_Visualization(eqType, coordType))
		return param_all


def add_metadata(drawing, data, ID):
	metadata=Element("data")
	drawing.set_metadata(metadata)
	metadata.attrib["id"]=ID
	metadata.text=json.dumps(data, indent=1)


def cdf_to_density(cdf, bins):
	cdf_edge=cdf(bins.edge)
	density=(cdf_edge[1:]-cdf_edge[:-1])/bins.width
	return density


def force_symm(density):
	binSize=len(density)
	halfSize=(binSize-2)//2
	troughIndex=halfSize//2
	peakIndex=troughIndex+1+halfSize
	halfIndex0=troughIndex+1+np.arange(halfSize)
	halfIndex1=np.concatenate((np.arange(troughIndex-1, -1, -1), np.arange(binSize-1, peakIndex, -1)))

	half0=density[halfIndex0]
	half1=density[halfIndex1]
	half=(half0+half1)/2
	
	density_symm=density.copy()
	density_symm[halfIndex0]=half
	density_symm[halfIndex1]=half

	# assertSymm(density_symm, 1)
	return density_symm


class Param_Density_vonMises():
	def __init__(self, angleDeg, concentration, weight):
		assert len(angleDeg)==len(concentration)
		self.numMode=len(angleDeg)
		self.angleDeg=angleDeg #deg
		self.concentration=concentration
		if weight is None:
			self.weight=np.ones(self.numMode)
		else:
			self.weight=weight
		

	def calc(self, bins):
		numMode=self.numMode
		angleRad=self.angleRad
		concentration=self.concentration
		weight=self.weight
		
		cdf=lambda x: sum([w*vonmises.cdf(x, loc=a+np.pi/2, kappa=c) for (a,c,w) in zip(angleRad, concentration, weight)])/sum(weight) #peak at pi/2
	
		density=cdf_to_density(cdf, bins)
		density=force_symm(density)
		
		return density
	
	
	@property
	def angleRad(self):
		return tuple(np.deg2rad(np.array(self.angleDeg)))
	
		
	def __str__(self):
		# return f"Mod{self.numMode}"+"".join([f"Ang{a}Conc{c}" for (a,c) in zip(self.angleDeg, self.concentration)])
		# assert self.numMode==1
		# return f"Con{self.concentration[0]}"
		return f"Mod{self.numMode}Con{self.concentration[0]}"
	
		
	@staticmethod
	def make(mode, concentration):
		if mode==1:
			angleDeg_all=(0,)
			concentration_all=(concentration,)
		elif mode==2:
			angleDeg_all=(90, 270)
			concentration_all=(concentration, concentration)

		return Param_Density_vonMises(angleDeg_all, concentration_all, None)


class WrappedLaplace:
	@staticmethod
	def cdf_loc0(x, lamb):
		assert 0<lamb
		assert ((x>=0) & (x<=PI2)).all()

		e2PiLambda=np.exp(2*np.pi*lamb)
		eLambdaX=np.exp(lamb*x)
		
		cdf=(e2PiLambda +eLambdaX*(1 -eLambdaX - e2PiLambda) ) / eLambdaX / 2 / (1- e2PiLambda)
		return cdf


	@staticmethod
	def cdf(x, lamb, loc=0):
		assert (x==np.sort(x)).all()
		assert x[-1]-x[0]<=PI2
		assert x[0]<=loc<=x[-1]

		x_from_loc=x-loc
		x_nonneg=x_from_loc.copy()
		x_nonneg[x_nonneg<0]+=PI2
		cdf_nonneg=WrappedLaplace.cdf_loc0(x_nonneg, lamb)
		cdf_nonneg[x_from_loc>=0]+=1
		cdf_nonneg-=cdf_nonneg[0]
		return cdf_nonneg


class Param_Density_wrappedLaplace():
	def __init__(self, angleDeg, lamb, weight):
		assert len(angleDeg)==len(lamb)
		self.numMode=len(angleDeg)
		self.angleDeg=angleDeg #deg
		self.lamb=lamb
		if weight is None:
			self.weight=np.ones(self.numMode)
		else:
			self.weight=weight
	

	def calc(self, bins):
		cdf=lambda x: sum([w*WrappedLaplace.cdf(x, loc=a+np.pi/2, lamb=la) for (a,la,w) in zip(self.angleRad, self.lamb, self.weight)])/sum(self.weight) #peak at pi/2
	
		density=cdf_to_density(cdf, bins)
		density=force_symm(density)
		
		return density
	
	
	@property
	def angleRad(self):
		return tuple(np.deg2rad(np.array(self.angleDeg)))
	
		
	def __str__(self):
		# return f"Mod{self.numMode}"+"".join([f"Ang{a}Conc{c}" for (a,c) in zip(self.angleDeg, self.concentration)])
		# assert self.numMode==1
		# return f"Con{self.concentration[0]}"
		return f"Mod{self.numMode}Lamb{self.lamb[0]}"
	
		
	@staticmethod
	def make(mode, concentration):
		if mode==1:
			angleDeg_all=(0,)
			concentration_all=(concentration,)
		elif mode==2:
			angleDeg_all=(90, 270)
			concentration_all=(concentration, concentration)

		return Param_Density_wrappedLaplace(angleDeg_all, concentration_all, None)


def _add_metadata(drawing, param):
	for name in ("trueOption", ):
		add_metadata(drawing, param[name], name)


def _saveSVG(drawing, fileSVG):
	fileSVG.parent.mkdir(exist_ok=True, parents=True)
	with open(fileSVG, "w", encoding="utf8") as f:
		drawing.write(f, pretty=True)


def calc_splitRatio(density, bins):
	binSize=bins.size
	splitRatio=(density[binSize//2:]*bins.width[binSize//2:]).sum()/(density[:binSize//2]*bins.width[:binSize//2]).sum()
	return splitRatio
	


def loss_splitRatio(param_density, bins, target):
	density=param_density.calc(bins)
	splitRatio=calc_splitRatio(density, bins)
	return (splitRatio-target)**2


def loss_splitRatio_vomMises(concentration, bins, target):
	param_density=Param_Density_vonMises.make(1, concentration)
	loss=loss_splitRatio(param_density, bins, target)
	return loss


def loss_splitRatio_wrappedLaplace(lamb, bins, target):
	param_density=Param_Density_wrappedLaplace.make(1, lamb)
	loss=loss_splitRatio(param_density, bins, target)
	return loss


def search_distribution_param(bins, splitRatio, lossFunc, bounds):
	opt=scipy.optimize.minimize_scalar(lossFunc, args=(bins,splitRatio), bounds=bounds, method="bounded")
	concentration=opt.x
	return concentration


def search_vonMises_concentration(bins, splitRatio):
	lossFunc=loss_splitRatio_vomMises
	bounds=(1e-1, 1e+1)
	concentration=search_distribution_param(bins, splitRatio, lossFunc, bounds)
	return concentration


def search_wrappedLaplace_lamb(bins, splitRatio):
	lossFunc=loss_splitRatio_wrappedLaplace
	bounds=(1e-1, 1e+1)
	concentration=search_distribution_param(bins, splitRatio, lossFunc, bounds)
	return concentration


def convertDensity_onCoord(density, param_vis):
	if param_vis.coordType=="Cart" or param_vis.eqType=="Len":
		height=density
	else:
		height=(2*density)**0.5

	return height


def load_textSymbol(name):
	path=DIR_WWW/"Symbols"/f"{name}.svg"
	size=read_viewboxSize(path)
	relPath=f"../../../Symbols/{name}.svg#text"
	symbolInfo=type('', (), {'size': size, "path": relPath})()
	return symbolInfo


def draw(stimType, binSize, distributionName, splitRatio):
	bins=Bins.make_rad(binSize)

	if splitRatio<1:
		splitRatio_distribution=1/splitRatio
	else:
		splitRatio_distribution=splitRatio

	if distributionName=="vonMises":
		concentration=search_vonMises_concentration(bins, splitRatio_distribution)
		param_density=Param_Density_vonMises.make(1, concentration)
	elif distributionName=="wrappedLaplace":
		lamb=search_wrappedLaplace_lamb(bins, splitRatio_distribution)
		param_density=Param_Density_wrappedLaplace.make(1, lamb)

	density=param_density.calc(bins)

	if splitRatio<1:
		density=np.flip(density)

	splitRatio_actual=(density[binSize//2:]*bins.width[binSize//2:]).sum()/(density[:binSize//2]*bins.width[:binSize//2]).sum()
	print("splitRatio_actual", splitRatio_actual)
	
	canvParam=CanvParam(512)

	colors={"white":1, "black":0, "grayLight":0.75, "grayDark":0.5}
	for cn,c in colors.items(): colors[cn]=rgbToHex((c,c,c))

	params_vis=Param_Visualization.makeAll(stimType)
	for param_vis in params_vis:
		print(stimType, binSize, splitRatio, param_vis, color="green")
		density_onCoord=convertDensity_onCoord(density, param_vis)
		height_canv, bins_canv=canvParam.toCanv(density_onCoord, bins, param_vis.coordType)

		if stimType=="main":
			state_all=[None,]
		elif stimType=="tutorial":
			state_all=["Orig", "Shape", "TextL_FlipF", "TextL_FlipT", "TextR_FlipF", "TextR_FlipT"]
		
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
				dirSVG=DIR_STIM/stimType
			else:
				dirSVG=DIR_STIM/stimType/state
			fileSVG=dirSVG/f"Bin{binSize}_{distributionName}_Split{splitRatio}_{param_vis}.svg"
			_saveSVG(drawing, fileSVG)


def main_draw():
	distributionName_all=("vonMises", "wrappedLaplace")
	for stimType in ("main", "tutorial", ):
		binSize_all, splitRatio_all=makeParams(stimType)
		for binSize, distributionName, splitRatio in itertools.product(binSize_all, distributionName_all, splitRatio_all):
			draw(stimType, binSize, distributionName, splitRatio)


def makeParams(stimType):
	if stimType=="main":
		binSize_all=(6,10)
		splitRatio_all=(0.25, 0.5, 2.0, 4.0)
	elif stimType=="tutorial":
		binSize_all=(6, 10,)
		splitRatio_all=(0.2, 5.0,)
	
	return binSize_all, splitRatio_all


def main_expectedRatio():
	expectedRatio=[]
	distributionName_all=("vonMises", "wrappedLaplace")
	for stimType in ("main", ):
		binSize_all, splitRatio_all=makeParams(stimType)
		for binSize, distributionName, splitRatio in itertools.product(binSize_all, distributionName_all, splitRatio_all):
			if splitRatio<1: continue

			bins=Bins.make_rad(binSize)

			splitRatio_distribution=splitRatio

			if distributionName=="vonMises":
				concentration=search_vonMises_concentration(bins, splitRatio_distribution)
				param_density=Param_Density_vonMises.make(1, concentration)
			elif distributionName=="wrappedLaplace":
				lamb=search_wrappedLaplace_lamb(bins, splitRatio_distribution)
				param_density=Param_Density_wrappedLaplace.make(1, lamb)

			density=param_density.calc(bins)

			params_vis=Param_Visualization.makeAll(stimType)
			for param_vis in params_vis:
				# print(stimType, binSize, splitRatio, param_vis, color="green")
				density_onCoord=convertDensity_onCoord(density, param_vis)

				length=density_onCoord

				if param_vis.coordType=="Cart":
					area=density_onCoord*bins.width
				elif param_vis.coordType=="Polar":
					area=density_onCoord**2*bins.width/2

				ratio_length=length[binSize//2:].sum()/length[:binSize//2].sum()
				ratio_area=area[binSize//2:].sum()/area[:binSize//2].sum()
				print(binSize, distributionName, splitRatio, param_vis, color="green")
				print(ratio_length)
				print(ratio_area)
				print()
				expectedRatio.append((binSize, distributionName, splitRatio, param_vis.eqType, param_vis.coordType, ratio_length, ratio_area))
	
	expType="singlePeak"
	DIR_ANALYSIS=DIR_EXP/f"Analysis_{expType}"
	file=DIR_ANALYSIS/f"expectedRatio.tsv"
	with open(file, 'w', newline='') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow(["binSize", "distributionName", "splitRatio", "eqType", "coordType", "ratio_length", "ratio_area"])
		writer.writerows(expectedRatio)


def main():
	main_draw()

	main_expectedRatio()


if __name__=="__main__":
	DIR_WWW=Path(__file__).parent.parent/"www"
	DIR_STIM=DIR_WWW/"Stim_singlePeak"
	DIR_EXP=Path(__file__).parent.parent/"exp"
	
	main()
