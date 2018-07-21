import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    coco = CocoDataset(root='data/train2014',
                       anns_json=args.anns_json,
                       qns_json=args.qns_json,
                       vocab_file=args.vocab_path,
                       index_file=args.index_file,
                       transform=transform)

    # Data loader for COCO dataset
    data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    cnn_model = CNN().eval()
    lstmqn = LSTM(len(coco.qn_vocab).eval()

    lstmqn.load_state_dict(torch.load(args.lstmqn_path))
    lstmqn.eval()

    for i, ((images, qns), targets, (qn_lengths, ans_lengths)) in enumerate(data_loader):
            images = images.to(device)
            qns = qns.to(device)
            ft_output = cnn_model(images)
            lstm_output = lstmqn(qns, qn_lengths)
            concat_ft = torch.cat((ft_output,lstm_output), 1)
            concat_dim = concat_ft.shape[1]
            concat = Concat(concat_dim)
            concat.load_state_dict(torch.load(args.concat_path))
            concat.eval()
            outputs = concat(concat_ft)
            answer = output.data.topk(top_k, dim=1)
            answer = vocab.idx2word[answer]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=229 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/train2014', help='directory for resized images')
    parser.add_argument('--anns_json', type=str, default='data/v2_mscoco_train2014_annotations.json', help='path for train annotation json file')
    parser.add_argument('--qns_json', type=str, default='data/v2_OpenEnded_mscoco_train2014_questions.json', help='path for qns')
    parser.add_argument('--index_file', type=str, default='index_data.json', help='path for index file')
    parser.add_argument('--lstmqn_path', type=str, default='models/lstmqn-5-3000.ckpt', help='path for trained lstmqn model')
    parser.add_argument('--concat_path', type=str, default='models/concat-5-3000.ckpt', help='path for trained concat model')
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=300, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=1024, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    main(args)
            
            
            
            
                  
    
     
