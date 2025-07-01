import numpy as np
from numpy import uint8
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image


def figToImage(dpi, fig):
	bytes_w=BytesIO()
	
	if fig is None:
		plt.savefig(bytes_w, dpi=dpi)
	else:
		fig.savefig(bytes_w, dpi=dpi)
	
	bytes_r=BytesIO(bytes_w.getvalue())
	img=Image.open(bytes_r)
	return img


def _extract_foreground(img_arr, backgroundColor):
	back=(img_arr==backgroundColor).all(axis=2)
	
	back0=back.all(axis=1)
	fore0Index=np.where(~back0)[0]
	if len(fore0Index)>0:
		img_arr=img_arr[fore0Index[0]:fore0Index[-1]+1]
		
	back1=back.all(axis=0)
	fore1Index=np.where(~back1)[0]
	if len(fore1Index)>0:
		img_arr=img_arr[:, fore1Index[0]:fore1Index[-1]+1]
	
	return img_arr


def show_figForeground(dpi, backgroundColor, fig):
	img=figToImage(dpi, fig).convert("RGB")
	
	img_arr=np.asarray(img).copy()
	
	if backgroundColor is None:
		backgroundColor=(255,255,255)
	if not isinstance(backgroundColor, np.ndarray):
		backgroundColor=np.array(backgroundColor, uint8)
	
	img_arr=_extract_foreground(img_arr, backgroundColor)
	
	Image.fromarray(img_arr, "RGB").show()


def defaultColors(index=None, alpha=1):
	'''
	@param alpha: [0,1]
	'''
	colorStr = plt.rcParams['axes.prop_cycle'].by_key()['color']
	if index is None:
		colors=[]
		for cs in colorStr:
			r=int(cs[1:3],16)/255
			g=int(cs[3:5],16)/255
			b=int(cs[5:7],16)/255
			colors.append((r,g,b,alpha))
		return colors

	else:
		cs=colorStr[index%len(colorStr)]
		r=int(cs[1:3],16)/255
		g=int(cs[3:5],16)/255
		b=int(cs[5:7],16)/255
		return(r,g,b,alpha)

