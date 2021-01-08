'''
Author: Rohan Karnawat
Description: Implementation of EdgeBox algorithm for
             generating object proposal bounding boxes
'''
import os
import cv2
import argparse
import numpy as np
from helper import *
# from plotter import *


class EdgeBoxes:
	'''
	EdgeBoxes class implements the core algorithm
	'''
	def __init__(self, args):
		'''Class object initialized with arguments'''
		self.A = args 
		self.model = None
		self.alpha = None
		self.beta = None
		self.save = False

	def setup(self):
		'''Preliminary parameters and paths are set/created'''
		if self.A.out is not None:
			os.makedirs(self.A.out, exist_ok=True)
			self.save = True
		self.model = 'model.yml'
		self.alpha = [0.2,0.5,0.8]
		self.beta = [0.2,0.5,0.8]


	def __core(self, image, alpha, beta):
		'''EdgeBox Algorithm: returns a list of proposed bounding box'''
		# Structured edge detection object
		edge_object = cv2.ximgproc.createStructuredEdgeDetection(self.model)
		# Get edges
		## Convert from BGR to RGB
		rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		## FC32 data format needed
		f_img = np.array(rgb_img).astype(np.float32)
		## Normalize to 0-1 and get edges and orientation
		edges = edge_object.detectEdges(f_img/255.0)
		orientation = edge_object.computeOrientation(edges)
		## Non maximal suppression
		edges = edge_object.edgesNms(edges, orientation)

		# Get edge boxes with set alpha and beta
		eb_object = cv2.ximgproc.createEdgeBoxes()
		eb_object.setAlpha(alpha)
		eb_object .setBeta(beta)
		boxes, scores = eb_object.getBoundingBoxes(edges, orientation)

		return [[box[0], box[1], box[0]+box[2], box[1]+box[3]] for box in boxes]



	def search(self):
		'''Controller function of the class'''
		for img_name in os.listdir(self.A.src):
			# check if jpg
			if 'jpg' not in img_name:
				continue
			# read image
			img = cv2.imread(os.path.join(self.A.src, img_name))
			
			# read ground truth and save boxes
			gt_b = gtboxes(os.path.join(self.A.gt, img_name.split('.')[0]+'.xml'))
			img_gtbox = draw(img, gt_b, (0,255,0))

			# create output directory
			if self.save:
				outpath = os.path.join(self.A.out, img_name.split('.')[0])
				cv2.imwrite(outpath+'_gt.jpg', img_gtbox)

			print('Processing image: {}'.format(img_name))
			abrecall = []
			abnum = []
			# Iterate over all alpha-beta pair variations
			for A in self.alpha:
				for B in self.beta:
					print('alpha: {}; beta: {}'.format(A, B), end='===>')
					
					# Call private method to get proposals
					pred_b = self.__core(img, A, B)
					num_boxes = len(pred_b)
					
					# evaluate iou
					best_b, R, clean_b = iou(pred_b, gt_b)
					abrecall.append([A, B, R])
					abnum.append([A, B, num_boxes])
					# draw all bboxes and best bboxes on the image
					img_allbox = draw(img, pred_b[:100], (0,0,0))
					img_goodbox = draw(img_gtbox, best_b, (0,0,255))
					img_cleanbox = draw(img_gtbox, clean_b, (0,0,255))
					
					# save annotated images
					if self.save:
						cv2.imwrite(outpath+'_'+str(A)+'_'+str(B)+'_'+'all.jpg', img_allbox)
						cv2.imwrite(outpath+'_'+str(A)+'_'+str(B)+'_'+'best.jpg', img_goodbox)
						cv2.imwrite(outpath+'_'+str(A)+'_'+str(B)+'_'+'clean.jpg', img_cleanbox)
			
					# show img if argument is true
					if(self.A.show_img):
						display(img_gtbox)
						display(img_goodbox)
						display(img_cleanbox)
					# Print final result
					print('img:{}, recall:{}, total_proposal_boxes:{};'.format(img_name, R, num_boxes))

			abr = np.array(abrecall)
			abn = np.array(abnum)
			#plotter(abr, img_name, 'r')
			#plotter(abn, img_name, 'n')
			


def main():
	'''Main function'''
	# Parsing arguments
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--src', type=str, default='HW2_Data/JPEGImages', help='Path to directory containing source images --src <path>')
	parser.add_argument('--gt',  type=str, default='HW2_Data/Annotations', help='Path to directory containing ground truth info. --gt <path>')
	parser.add_argument('--out', type=str, default=None, help='Path to directory to save output. --out <path>')
	parser.add_argument('--show_img', help='Flag to display output', action="store_true")
	args = parser.parse_args()
	# Create edgebox instance and launch algorithm
	eb = EdgeBoxes(args)
	eb.setup()
	eb.search()

if __name__=='__main__':
	main()
