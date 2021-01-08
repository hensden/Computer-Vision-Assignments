'''
Author: Rohan Karnawat
Description: Implementation of Selective Search for
             generating object proposal bounding boxes
'''
import os
import cv2
import argparse
import numpy as np
from helper import *

class SelectiveSearch:
	'''
	Selective Search class implements the core algorithm
	'''
	def __init__(self, args):
		'''Class object initialized with arguments'''
		self.A = args
		self.algo = None

	def setup(self):
		'''Preliminary parameters and paths are set/created'''
		os.makedirs(self.A.out, exist_ok=True)
		self.algo = self.A.strategy

	def __core(self, image):
		'''Selective Search Algorithm: returns a list of proposed bounding box'''
		
		# Create SS Object for Segmentation
		ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
		ss_graph = cv2.ximgproc.segmentation.createGraphSegmentation()
		
		# Create SS strategy objects
		color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
		texture = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
		fill = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
		size = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
		multi = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(color, fill, texture, size)
		
		# Clear old data
		ss.clearImages()
		ss.clearStrategies()
		ss.clearGraphSegmentations()
		
		# Add image
		ss.addImage(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		# Add graph segmentation
		ss.addGraphSegmentation(ss_graph)
		
		# Add strategy based on the argument chosen
		if self.algo=='color':
			ss.addStrategy(color)
		elif self.algo=='texture':
			ss.addStrategy(texture)
		else:
			ss.addStrategy(multi)
			# ss.switchToSelectiveSearchQuality()
		# Run search and get bounding boxes
		bboxes = ss.process()
		
		return [[box[0], box[1], box[0]+box[2], box[1]+box[3]] for box in bboxes]

	def search(self):
		'''Controller function of the class'''
		for img_name in os.listdir(self.A.src):
			# check if jpg
			if 'jpg' not in img_name:
				continue
			
			# read image
			img = cv2.imread(os.path.join(self.A.src, img_name))
			#display(img)
			
			# selective search core, call private method to get proposals
			pred_b = self.__core(img)
			num_boxes = len(pred_b)
			
			# read ground truth
			gt_b = gtboxes(os.path.join(self.A.gt, img_name.split('.')[0]+'.xml'))
			
			# evaluate iou
			best_b, R, clean_b = iou(pred_b, gt_b)
			
			# draw all bboxes and best bboxes on the image
			img_gtbox = draw(img, gt_b, (0,255,0))
			img_allbox = draw(img, pred_b[:100], (0,0,0)) # draw only 100 boxes
			img_goodbox = draw(img_gtbox, best_b, (0,0,255))
			img_cleanbox = draw(img_gtbox, clean_b, (0,0,255))
			
			# save annotated images
			outpath = os.path.join(self.A.out, img_name.split('.')[0])
			cv2.imwrite(outpath+'_all_'+self.algo+'.jpg', img_allbox)
			cv2.imwrite(outpath+'_good_'+self.algo+'.jpg', img_goodbox)
			cv2.imwrite(outpath+'_clean_'+self.algo+'.jpg', img_cleanbox)
			cv2.imwrite(outpath+'_gt.jpg', img_gtbox)
			
			# show img if argument is true
			if(self.A.show_img):
				display(img_gtbox)
				display(img_goodbox)
				display(img_cleanbox)
			
			# Print final result		
			print('strategy:{}, img:{}, recall:{}, total_boxes:{}'.format(self.algo, img_name, R, len(pred_b)))


def main():
	'''Main function'''
	# Parsing arguments
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--src', type=str, default='HW2_Data/JPEGImages', help='Path to directory containing source images --src <path>')
	parser.add_argument('--gt',  type=str, default='HW2_Data/Annotations', help='Path to directory containing ground truth info. --gt <path>')
	parser.add_argument('--out', type=str, default=None, help='Path to directory to save output. --out <path>')
	parser.add_argument('--strategy', type=str, default=None, help='--strategy color|texture|multi')
	parser.add_argument('--show_img', help='Flag to display output', action="store_true")
	args = parser.parse_args()
	# Create edgebox instance and launch algorithm
	ss = SelectiveSearch(args)
	ss.setup()
	ss.search()

if __name__=='__main__':
	main()
