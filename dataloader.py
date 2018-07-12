import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from collections import Counter

def _isArrayLike(obj):
	return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class COCO:
	def __init__(self, annotation_file, image_folder, question_file):
		"""
		Constructor of COCO helper class.
		"""
		# load dataset
		self.dataset,self.qns,self.imgs = dict(),dict(),dict()
		self.imgToAnns, self.imgToQns = defaultdict(list), defaultdict(list)
		
		print('loading annotations into memory...')
		tic = time.time()
		dataset = json.load(open(annotation_file, 'r'))
		assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
		print('Done (t={:0.2f}s)'.format(time.time()- tic))
		self.dataset = dataset
		self.image_folder = image_folder
		questionSet = json.load(open(question_file, 'r'))
		assert type(questionSet)==dict, 'question file format {} not supported'.format(type(questionSet))
		self.questionSet = questionSet
		self.createIndex()

	def createIndex(self):
		# create index
		print('creating index...')
		qns, imgs = {}, {}
		imgToAnns,imgToQns = defaultdict(list), defaultdict(list)
		if 'annotations' in self.dataset:
			for ann in self.dataset['annotations']:
				imgToAnns[ann['image_id']].append(ann)
				imgs[ann['image_id']] = 'COCO_' + self.image_folder + '_%12d.JPG' % (ann['image_id'])
				for qn in self.questionSet['questions']:
					if qn['question_id']==ann['question_id']:
						imgToQns[ann['image_id']].append((ann['question_id'], qn['question']))
						qns[ann['question_id']] = qn['question']
						break

		print('index created!')

		# create class members
		self.imgToAnns = imgToAnns
		self.imgs = imgs
		self.imgToQns = imgToQns
		self.qns = qns

	def info(self):
		"""
		Print information about the annotation file.
		"""
		for key, value in self.dataset['info'].items():
			print('{}: {}'.format(key, value))

	def getAnns(self, imgIds=[], qnIds=[]):
		"""
		Get anns that satisfy given filter conditions. default skips that filter
		"""
		imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
		qnIds = qnIds if _isArrayLike(qnIds) else [qnIds]

		if len(imgIds) == len(qnIds) == 0:
			anns = self.dataset['annotations']
		else:
			if len(imgIds) > 0:
				lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
				anns = list(itertools.chain.from_iterable(lists))
			else:
				anns = self.dataset['annotations']
			anns = anns if len(qnIds)  == 0 else [ann for ann in anns if ann['question_id'] in qnIds]
		return anns

	def getImgIds(self, imgIds=[], qnIds=[]):
		'''
		Get img ids that satisfy given filter conditions.
		'''
		imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
		qnIds = qnIds if _isArrayLike(qnIds) else [qnIds]

		if len(imgIds) == len(qnIds) == 0:
			ids = self.imgs.keys()
		else:
			ids = set(imgIds)
			for i, qnId in enumerate(qnIds):
				if i == 0 and len(ids) == 0:
					ids = set(qnId[:-3])
				else:
					ids &= set(qnId[:-3])
		return list(ids)

	def loadQns(self, ids=[]):
		"""
		Load qns with the specified ids.
		:param ids (int array)       : integer ids specifying qns
		:return: qns (str array)     : loaded qn strings
		"""
		if _isArrayLike(ids):
			return [self.qns[id] for id in ids]
		elif type(ids) == int:
			return [self.qns[ids]]

	def loadImgs(self, ids=[]):
		"""
		Load imgs with the specified ids.
		"""
		if _isArrayLike(ids):
			return [Image.open(os.path.join(self.image_folder, self.imgs[id])) for id in ids]
		elif type(ids) == int:
			return [Image.open(os.path.join(self.image_folder, self.imgs[ids]))]

	def loadRes(self, resFile):
		"""
		Load result file and return a result api object.
		:param   resFile (str)     : file name of result file
		:return: res (obj)         : result api object
		"""
		res = COCO()

		print('Loading and preparing results...')
		if type(resFile) == str or type(resFile) == unicode:
			anns = json.load(open(resFile))
		elif type(resFile) == np.ndarray:
			anns = self.loadNumpyAnnotations(resFile)
		else:
			anns = resFile
		assert type(anns) == list, 'results in not an array of objects'
		annsImgIds = [ann['image_id'] for ann in anns]
		assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
			   'Results do not correspond to current coco set'

		res.dataset['annotations'] = anns
		res.createIndex()
		return res

	def loadNumpyAnnotations(self, data):
		"""
		Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
		:param  data (numpy.ndarray)
		:return: annotations (python nested list)
		"""
		print('Converting ndarray to lists...')
		assert(type(data) == np.ndarray)
		print(data.shape)
		assert(data.shape[1] == 7)
		N = data.shape[0]
		ann = []
		for i in range(N):
			if i % 1000000 == 0:
				print('{}/{}'.format(i,N))
			ann += [{
				'image_id'  : int(data[i, 0]),
				'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
				'score' : data[i, 5],
				'category_id': int(data[i, 6]),
				}]
		return ann

	def build_vocab(threshold):
		"""Build a simple vocabulary wrapper."""
		counter = Counter()
		ids = self.qns.keys()
		for i, id in enumerate(ids):
			question = str(self.qns[id])
			tokens = nltk.tokenize.word_tokenize(question.lower())
			counter.update(tokens)

			if (i+1) % 1000 == 0:
				print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

		# If the word frequency is less than 'threshold', then the word is discarded.
		words = [word for word, cnt in counter.items() if cnt >= threshold]

		# Create a vocab wrapper and add some special tokens.
		vocab = Vocabulary()
		vocab.add_word('<pad>')
		vocab.add_word('<start>')
		vocab.add_word('<end>')
		vocab.add_word('<unk>')

		# Add the words to the vocabulary.
		for i, word in enumerate(words):
			vocab.add_word(word)
		return vocab


class Vocabulary(object):
	"""Simple vocabulary wrapper."""
	def __init__(self):
		self.word2idx = {}
		self.idx2word = {}
		self.idx = 0

	def add_word(self, word):
		if not word in self.word2idx:
			self.word2idx[word] = self.idx
			self.idx2word[self.idx] = word
			self.idx += 1

	def __call__(self, word):
		if not word in self.word2idx:
			return self.word2idx['<unk>']
		return self.word2idx[word]

	def __len__(self):
		return len(self.word2idx)

	
class CocoDataset(data.Dataset):
	"""COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
	def __init__(self, root, anns_json, qns_json, vocab_path, transform=None):
		"""Set the path for images, annotations, and questions.
		
		Args:
			root: image directory.
			anns_json: coco annotation file path.
			qns_json: coco question file path.
			transform: image transformer.
		"""
		self.root = root
		self.coco = COCO(anns_json, root, qns_json)
		self.transform = transform
		self.vocab = self.coco.build_vocab(2)
		with open(vocab_path, 'wb') as f:
			pickle.dump(vocab, f)
		print("Total vocabulary size: {}".format(len(vocab)))
		print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

	def __getitem__(self, index):
		"""Returns one data pair (image-question and answer)."""
		coco = self.coco
		answer = coco.dataset['annotations'][index]['multiple_choice_answer']
		qn_id = coco.dataset['annotations'][index]['question_id']
		img_id = coco.dataset['annotations'][index]['image_id']
		img = coco.loadImgs(img_id)[0]

		image = img.convert('RGB')
		if self.transform is not None:
			image = self.transform(image)

		# Convert question (string) to word ids.
		tokens = nltk.tokenize.word_tokenize(str(coco.qns[qn_id]).lower())
		question = []
		question.append(self.vocab('<start>'))
		question.extend([self.vocab(token) for token in tokens])
		question.append(self.vocab('<end>'))
		target = torch.Tensor(question)
		return image, target

	def __len__(self):
		return len(self.coco.dataset['annotations'])


def collate_fn(data):
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
	"""Returns torch.utils.data.DataLoader for custom coco dataset."""
	# COCO caption dataset
	coco = CocoDataset(root=root,
					   json=json,
					   vocab=vocab,
					   transform=transform)
	
	# Data loader for COCO dataset
	# This will return (images, captions, lengths) for each iteration.
	# images: a tensor of shape (batch_size, 3, 224, 224).
	# captions: a tensor of shape (batch_size, padded_length).
	# lengths: a list indicating valid length for each caption.  length is
	# (batch_size).
	data_loader = torch.utils.data.DataLoader(dataset=coco, 
											  batch_size=batch_size,
											  shuffle=shuffle,
											  num_workers=num_workers,
											  collate_fn=collate_fn)
	return data_loader
