import argparse
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    data_loader = get_loader(args.anns_json, args.qns_json, vocab_path, transform = transform, args.batch_size, shuffle=True, num_workers=args.num_workers) 

    #Build models
    cnn = CNN()
    lstmqn = LSTMQuestion(args.vocab_size).to(device)
    concat = Concat(args.concat_size).to(device)

    #Loss and optimizer
    criterion = nn.CrossEntropyLoss
    params = list(concat.parameters()) + list(cnn.parameters()) + list(lstm.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    #Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images,annotations,questions,lengths) in enumerate(dataloader):
            
            images = images.to(device)
            question = questions.to(device)
            targets = annotations.to(device)

            #Forward, backward and optimize
            
            lstm_output = lstmqn(question, length)
            outputs = concat(ft_output,lstm_output)
            outputs = outputs[0]
            loss = criterion(outputs,targets)
            cnn.zero_grad()
            lstmqn.zero_grad()
            concat.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(lstmqn.state_dict(), os.path.join(
                    args.model_path, 'lstmqn-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(concat.state_dict(), os.path.join(
                    args.model_path, 'concat-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(concat.state_dict(), os.path.join(
                    args.model_path, 'cnn-{}-{}.ckpt'.format(epoch+1, i+1)))
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=229 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/train2014', help='directory for resized images')
    parser.add_argument('--anns_json', type=str, default='data/v2_mscoco_train2014_annotations.json', help='path for train annotation json file')
    parser.add_argument('--qns_json', type=str, default='data/v2_OpenEnded_mscoco_train2014_questions.json', help='path for qns')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=300, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=1024, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)

            
            

        
