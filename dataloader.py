import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import time
import os
import pickle
import numpy as np
import nltk
import json
import itertools
import argparse
from PIL import Image
from collections import defaultdict, Counter

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class COCO:
    def __init__(self, annotation_file, image_folder, question_file, index_file):
        """
        Constructor of COCO helper class.
        """
        # load dataset
        self.dataset,self.qns,self.imgs = dict(),dict(),dict()
        self.imgToAnns, self.imgToQns = defaultdict(list), defaultdict(list)
        
        print('loading annotations into memory...')
        tic = time.time()
        dataset = json.load(open(annotation_file, 'r'))
        assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))
        self.dataset = dataset
        self.image_folder = image_folder
        questionSet = json.load(open(question_file, 'r'))
        assert type(questionSet) == dict, 'question file format {} not supported'.format(type(questionSet))
        self.questionSet = questionSet
        self.createIndex(index_file)

    def createIndex(self, index_file):
        if not os.path.isfile(index_file):
            # create index
            print('creating index...')
            qns, imgs = {}, {}
            imgToAnns,imgToQns = defaultdict(list), defaultdict(list)
            if 'annotations' in self.dataset:
                for ann in self.dataset['annotations']:
                    imgToAnns[ann['image_id']].append(ann)
                    imgs[ann['image_id']] = 'COCO_' + self.image_folder.split('/')[-1] + '_%012d.jpg' % (ann['image_id'])
                    for qn in self.questionSet['questions']:
                        if qn['question_id'] == ann['question_id']:
                            imgToQns[ann['image_id']].append((ann['question_id'], qn['question']))
                            qns[ann['question_id']] = qn['question']
                            break

            print('index created!')

            # create class members
            self.imgToAnns = imgToAnns
            self.imgs = imgs
            self.imgToQns = imgToQns
            self.qns = qns

            index_data = {'imgToAnns': imgToAnns, 'imgs': imgs, 'imgToQns': imgToQns, 'qns': qns}
            with open('index_data.json', 'w') as f:
                json.dump(index_data, f)

        else:
            # load index
            print('loading index into memory...')
            index_data = json.load(open(index_file, 'r'))
            print('index loaded!')

            self.imgToAnns = index_data['imgToAnns']
            self.imgs = index_data['imgs']
            self.imgToQns = index_data['imgToQns']
            self.qns = index_data['qns']


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
        imgIds = [imgIds] if type(imgIds) == str or type(imgIds) == int else imgIds
        qnIds = [qnIds] if type(qnIds) == str or type(qnIds) == int else qnIds

        if len(imgIds) == len(qnIds) == 0:
            anns = self.dataset['annotations']
        else:
            if len(imgIds) > 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(qnIds) == 0 else [ann for ann in anns if ann['question_id'] in qnIds]
        return anns

    def getImgIds(self, imgIds=[], qnIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        '''
        imgIds = [imgIds] if type(imgIds) == str or type(imgIds) == int else imgIds
        qnIds = [qnIds] if type(qnIds) == str or type(qnIds) == int else qnIds

        if len(imgIds) == len(qnIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, qnId in enumerate(qnIds):
                if i == 0 and len(ids) == 0:
                    ids = set(str(qnId)[:-3])
                else:
                    ids &= set(str(qnId)[:-3])
        return list(ids)

    def loadQns(self, ids=[]):
        """
        Load qns with the specified ids.
        :param ids (str array)       : str ids specifying qns
        :return: qns (str array)     : loaded qn strings
        """
        if type(ids) == str or type(ids) == int:
            return [self.qns[str(ids)]]
        elif _isArrayLike(ids):
            return [self.qns[str(id)] for id in ids]

    def loadImgs(self, ids=[]):
        """
        Load imgs with the specified ids.
        """
        if type(ids) == str or type(ids) == int:
            return [Image.open(os.path.normpath(os.path.join(self.image_folder, self.imgs[str(ids)])))]
        elif _isArrayLike(ids):
            return [Image.open(os.path.normpath(os.path.join(self.image_folder, self.imgs[str(id)]))) for id in ids]

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

    def build_vocab(self, threshold):
        """Build a simple vocabulary wrapper."""
        counter = Counter()
        ids = self.qns.keys()
        for i, id in enumerate(ids):
            question = str(self.qns[id])
            tokens = nltk.tokenize.word_tokenize(question.lower())
            counter.update(tokens)
        
        words = [word for word, cnt in counter.items() if cnt >= threshold]

        vocab_questions = Vocabulary()
        vocab_questions.add_word('<pad>')
        vocab_questions.add_word('<start>')
        vocab_questions.add_word('<end>')
        vocab_questions.add_word('<unk>')

        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            vocab_questions.add_word(word)

        counter = Counter()
        anns = self.dataset['annotations']
        for ann in anns:
            answers = ann['answers']
            for answer in answers:
                tokens = nltk.tokenize.word_tokenize(answer['answer'].lower())
                counter.update(tokens)

        words = [word for word, cnt in counter.items() if cnt >= threshold]

        vocab_answers = Vocabulary()
        vocab_answers.add_word('<pad>')
        vocab_answers.add_word('<start>')
        vocab_answers.add_word('<end>')
        vocab_answers.add_word('<unk>')

        # Add the words to the vocabulary.
        for i, word in enumerate(words):
            vocab_answers.add_word(word)

        return vocab_questions, vocab_answers


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
    def __init__(self, root, anns_json, qns_json, index_file, vocab_file, transform=None):
        self.root = root
        self.coco = COCO(anns_json, root, qns_json, index_file)
        self.transform = transform
        if not os.path.isfile(vocab_file):
            self.qn_vocab, self.ans_vocab = self.coco.build_vocab(2)
            with open('vocab.pkl', 'wb') as f:
                pickle.dump((self.qn_vocab, self.ans_vocab), f)
            print("Saved the vocabulary wrapper to 'vocab.pkl'")
        else:
            self.qn_vocab, self.ans_vocab = pickle.load(open(vocab_file, 'rb'))

            print("Qn vocab size: {}".format(len(self.qn_vocab)))
            print("Ans vocab size: {}".format(len(self.ans_vocab)))
            print("Total vocab size: {}".format(len(self.qn_vocab) + len(self.ans_vocab)))

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
        tokens = nltk.tokenize.word_tokenize(str(coco.qns[str(qn_id)]).lower())
        question = []
        question.append(self.qn_vocab('<start>'))
        question.extend([self.qn_vocab(token) for token in tokens])
        question.append(self.qn_vocab('<end>'))
        question = torch.Tensor(question)

        # Convert answer (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(answer).lower())
        answer = []
        answer.append(self.ans_vocab('<start>'))
        answer.extend([self.ans_vocab(token) for token in tokens])
        answer.append(self.ans_vocab('<end>'))
        target = torch.Tensor(answer)
        return (image, question), target

    def __len__(self):
        return len(self.coco.dataset['annotations'])


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples ((image, question), target).
    """
    # Sort a data list by qn length (descending order).
    data.sort(key=lambda x: len(x[0][1]), reverse=True)

    images = [item[0][0] for item in data]
    questions = [item[0][1] for item in data]
    targets = [item[1] for item in data]

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge questions (from tuple of 1D tensor to 2D tensor).
    qn_lengths = [len(qn) for qn in questions]
    qns = torch.zeros(len(questions), max(qn_lengths)).long()
    for i, qn in enumerate(questions):
        end = qn_lengths[i]
        qns[i, :end] = qn[:end]

    # Merge targets (from tuple of 1D tensor to 2D tensor).
    ans_lengths = [len(ans) for ans in targets]
    answers = torch.zeros(len(targets), max(ans_lengths)).long()
    for i, ans in enumerate(targets):
        end = ans_lengths[i]
        answers[i, :end] = ans[:end]
        
    return (images, qns), answers, (qn_lengths, ans_lengths)


def get_loader(root, anns_json, qns_json, batch_size, index_file, vocab_file, transform=None, shuffle=False, num_workers=0):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    coco = CocoDataset(root=root,
                       anns_json=anns_json,
                       qns_json=qns_json,
                       index_file=index_file,
                       vocab_file=vocab_file,
                       transform=transform)
    
    # Data loader for COCO dataset
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--index_file', type=str, default="")
    argparser.add_argument('--vocab_file', type=str, default="")
    args = argparser.parse_args()
    get_loader('data/train2014',
                'data/v2_mscoco_train2014_annotations.json',
                'data/v2_OpenEnded_mscoco_train2014_questions.json',
                128,
                args.index_file,
                args.vocab_file)
