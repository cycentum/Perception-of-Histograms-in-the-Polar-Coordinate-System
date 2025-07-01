import math
import numpy as np
from xml.etree import ElementTree


def make_arc(cx, cy, radius, start_angle, end_angle, oppositeDirection=False, angleType="deg", returnEnd=False):
	'''
	start_angle: The starting angle of the arc in degrees
	end_angle: The ending angle of the arc in degrees
	'''

	if oppositeDirection:
		start_angle, end_angle=end_angle, start_angle
	
	# Convert angles from degrees to radians
	if angleType=="deg":
		start_angle_rad = math.radians(start_angle)
		end_angle_rad = math.radians(end_angle)
	elif angleType=="rad":
		start_angle_rad = start_angle
		end_angle_rad = end_angle
		start_angle=math.degrees(start_angle_rad)
		end_angle=math.degrees(end_angle_rad)
	
	# Calculate the start and end points
	x1 = cx + radius * math.cos(start_angle_rad)
	y1 = cy + radius * math.sin(start_angle_rad)
	x2 = cx + radius * math.cos(end_angle_rad)
	y2 = cy + radius * math.sin(end_angle_rad)
	
	# Determine if the arc should be greater than 180 degrees
	large_arc_flag = 1 if (end_angle - start_angle) % 360 > 180 else 0
	# Sweep flag is 1 for a clockwise direction
	sweep_flag = 1
	
	if oppositeDirection:
		large_arc_flag=1-large_arc_flag
		sweep_flag=1-sweep_flag
	
	# Create the SVG path data for the arc
	arc_command = f"A {radius},{radius} 0 {large_arc_flag},{sweep_flag} {x2},{y2}"
	if returnEnd:
		return x1, y1, x2, y2, arc_command
	else:
		return x1, y1, arc_command


def make_sector(cx, cy, radius, start_angle, end_angle):
	'''
	start_angle: The starting angle of the arc in degrees
	end_angle: The ending angle of the arc in degrees
	'''
	x1, y1, arc=make_arc(cx, cy, radius, start_angle, end_angle)
	
	# Create the SVG path data for the arc
	pathData = f"M {cx},{cy} L {x1},{y1} {arc} Z"
	
	return pathData


def make_outerSector(cx, cy, radius_in, radius_out, start_angle, end_angle):
	x1_in,y1_in,arc_in=make_arc(cx, cy, radius_in, start_angle, end_angle, oppositeDirection=True)
	x1_out,y1_out,arc_out=make_arc(cx, cy, radius_out, start_angle, end_angle)
	
	pathData=f"M{x1_in},{y1_in} {arc_in} L{x1_out},{y1_out} {arc_out} Z"
	return pathData


def rgbToHex(rgb):
	rgb=np.array(rgb)
	rgb=np.round(rgb*255).astype(int)
	s=[]
	for c in rgb:
		s_=f"{c:02x}"
		s.append(s_)
	s="#"+"".join(s)
	return s


def brokenLine(coordinates):

	# Start with the 'M' command to move to the first point
	d_list = [f"M {coordinates[0][0]},{coordinates[0][1]}"]

	# Add the 'L' commands to draw lines to subsequent points
	for x, y in coordinates[1:]:
		d_list.append(f" L {x},{y}")

	d="".join(d_list)

	return d


def read_viewboxSize(fileSVG):
	tree=ElementTree.parse(fileSVG)
	viewBox=tree.getroot().attrib["viewBox"]
	viewBox=[float(x) for x in viewBox.split()]
	size=viewBox[2:]
	return size
	

