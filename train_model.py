import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from dataloader import *
from net import Net
from torch.autograd import Variable
from torchvision import transforms


use_cuda = torch.cuda.is_available()


def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    transform = transforms.Compose([transforms.CenterCrop(args.crop_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    coco = CocoDataset(root=args.image_dir,
                       anns_json=args.anns_json,
                       qns_json=args.qns_json,
                       vocab_file=args.vocab_path,
                       index_file=args.index_file,
                       transform=transform)

    # Data loader for COCO dataset
    data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    net = nn.DataParallel(Net(len(coco.qn_vocab)))
    if use_cuda:
        net = net.cuda()
    net.train()

    criterion = nn.CrossEntropyLoss()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    total_step = len(data_loader)

    img_params = {'volatile': False}
    target_params = {'requires_grad': False}

    #Train the models
    for epoch in range(args.num_epochs):
        for i, ((images, qns), targets, qn_lengths) in enumerate(data_loader):
            if use_cuda:
                images, qns, targets, qn_lengths = images.cuda(), qns.cuda(), targets.cuda(), qn_lengths.cuda()

            images, qns, targets = Variable(images, **img_params), Variable(qns), Variable(targets, **target_params)

            #Forward, backward, and optimize
            outputs = net(images, qns, qn_lengths)
            m = nn.Linear(targets.shape[1], 1)
            loss = criterion(outputs, m(targets))

            net.zero_grad()
            loss.backward()
            optimizer.step()

            # Print info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

            # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                torch.save(net.state_dict(), os.path.join(args.model_path, 'net-{}-{}.ckpt'.format(epoch + 1, i + 1)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_path', type=str, default="models/", help='path for saving trained models')
    argparser.add_argument('--crop_size', type=int, default=448, help='size for cropping images')
    argparser.add_argument('--vocab_path', type=str, default="vocab.pkl", help='path to saved vocabulary file')
    argparser.add_argument('--image_dir', type=str, default="data/train2014", help='image directory')
    argparser.add_argument('--anns_json', type=str, default="data/v2_mscoco_train2014_annotations.json", help='path to annotations json')
    argparser.add_argument('--qns_json', type=str, default="data/v2_OpenEnded_mscoco_train2014_questions.json", help='path to qns json')
    argparser.add_argument('--index_file', type=str, default="index_data.json", help='path to index file')
    argparser.add_argument('--log_step', type=int , default=10, help='step size for printing info')
    argparser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')

    # Model parameters
    argparser.add_argument('--embed_size', type=int , default=300, help='dimension of word embedding vectors')
    argparser.add_argument('--hidden_size', type=int , default=1024, help='dimension of lstm hidden states')
    argparser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    argparser.add_argument('--num_epochs', type=int, default=5)
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--num_workers', type=int, default=0)
    argparser.add_argument('--learning_rate', type=float, default=0.001)
    
    args = argparser.parse_args()
    print(args)
    main(args)
