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

            
            

        
