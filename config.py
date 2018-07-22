settings = {
    'crop_size': 448,
    'vocab_path': 'vocab.pkl',
    'image_dir': 'data/train2014',
    'anns_json': 'data/v2_mscoco_train2014_annotations.json',
    'qns_json': 'data/v2_OpenEnded_mscoco_train2014_questions.json',
    'index_file': 'index_data.json', 
    'lstmqn_path': 'models/lstmqn-5-3000.ckpt', 
    'concat_path': 'models/concat-5-3000.ckpt',
    
    # Model parameters
    'embed_size': 'dimension of word embedding vectors',
    'hidden_size': 1024,
    'num_layers':1, 
    'batch_size': 16,
    'num_workers': 0
}
