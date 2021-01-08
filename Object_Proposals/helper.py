'''
Author - Rohan Karnawat
Description - Utility functions used by selective search and edge box algorithm
'''
import os
import re
import cv2
from copy import deepcopy as dc

def gtboxes(path):
	'''Parses xml and returns a list of Ground Truth bounding box coordinates'''
	boxes = []
	f = open(path, 'r')
	xml = f.read().replace('\n', '').replace('\t', '')
	pattern = re.compile('<bndbox>'+'<xmin>[0-9]*</xmin>'+
		'<ymin>[0-9]*</ymin>'+'<xmax>[0-9]*</xmax>'+
		'<ymax>[0-9]*</ymax>'+'</bndbox>')
	boxes_pattern = pattern.findall(xml)
	pattern = re.compile('[0-9]+')
	for box in boxes_pattern:
		boxes.append([int(coord) for coord in pattern.findall(box)])
	return boxes  

def area(box):
	'''Computes area of rectangular box'''
	return abs(box[3]-box[1])*abs(box[2]-box[0])

def overlap(box1, box2):
	'''Computes area of overlapping region (intersection) between two boxes'''
	left = max(box1[0], box2[0])
	top = max(box1[1], box2[1])
	right = min(box1[2], box2[2])
	bottom = min(box1[3], box2[3])
	return 0 if(left >= right or top >= bottom) else (right - left) * (bottom - top)


def display(img):
	'''Opens a window to display image, press any key to continue'''
	cv2.imshow('Image', img)
	cv2.waitKey()
	cv2.destroyWindow('Image')

def iou(pred_b, gt_b):
	'''Computes Intersection over Union and return best boxes (iou>=0.5) and recall'''
	boxes, R = [], None
	clean = []
	num_gt = 0
	for gt in gt_b:
		matched = False
		for pred in pred_b:
			I = overlap(pred, gt)
			U = area(pred)+area(gt)-I
			IOU = float(I)/float(U)
			if IOU>0.5:
				boxes.append(pred)
				if not matched:
					clean.append(pred)
				matched = True
		if matched:
			num_gt += 1

	R = float(num_gt)/float(len(gt_b))
	return boxes, round(R, 3), clean

def draw(img, boxes, color):
	'''Returns a copy of the image with boxes. Color is of form (R,G,B)'''
	im = dc(img)
	for x in boxes:
		im = cv2.rectangle(im, (x[0],x[1]),(x[2],x[3]), color, 1)
	return im

