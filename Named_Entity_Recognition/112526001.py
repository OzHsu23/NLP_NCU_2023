import re
import torch
import math
import numpy as np
from functools import partial


import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import gensim
from gensim.models import KeyedVectors


#### Function ####
def read_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return f.readlines()

def get_stopword():
    global stopword
    stopword = [stop.replace('\n', '') for stop in read_file("./stopword.txt")]

def extract_data_label(input_list):
    '''extract data and label from input_list,
        if data in stopword, pass it, and lower the other data.
    ''' 
    data_list = []
    label_list = []
    for item in input_list:
        match = re.search(r'([^\t\n]+)\t([^\t\n]+)\n', item)
        if match:
            if match.group(1) in stopword:
                continue
            else:
                data_list.append((match.group(1)).lower())
                label_list.append(match.group(2))
        else:
            if item == '\n':
                data_list.append('\n')
                label_list.append('\n')
            else:
                print(f'Error: no data and label found: {item}')
    return data_list, label_list

def extract_data(input_list):
    '''extract data from input_list
    ''' 
    data_list = []
    for item in input_list:
        match = re.search(r'([^\t\n]+)\n', item)
        if match:
            data_list.append((match.group(1)))
        else:
            if item == '\n':
                data_list.append('\n')
            else:
                print(f'Error: no data found: {item}')
    return data_list

def extract_data_label_benchmark(input_list):
    '''extract data and label from input_list,
        and lower data.
    ''' 
    data_list = []
    txt_list = []
    for item in input_list:
        match = re.search(r'([^\t\n]+)\t([^\t\n]+)\n', item)
        if match:
            data_list.append(match.group(1))
            txt_list.append(match.group(1)+'\t'+match.group(2))
        else:
            if item == '\n':
                data_list.append('\n')
                txt_list.append('\n')
            else:
                print(f'Error: no data and label found: {item}')
    return data_list, txt_list

def compose_sentences(data_list):
    sentence = []
    sentences = []
    for word in data_list:
        if word != '\n':
            sentence.append(word)
        else:
            sentences.append(sentence)
            sentence = []
    return sentences

### Build vocabulary table and convert token to idx for training###
def word2index(vocabulary):
    return {word: idx for idx, word in enumerate(vocabulary)}

def vocabulary_key_index(embedding_model):
    return embedding_model.key_to_index

def build_label_vocabulary(train_label_list):
    label_vocabulary = set(label for label in train_label_list)
    label_vocabulary.remove('\n')
    # label_vocabulary.add("<pad>")
    return sorted(label_vocabulary)

def token2index(data_list, word_index, use_lower=False):
    if use_lower:
        data_list = [word.lower() for word in data_list]
    #unkown word use rare case <span> to instead #twitter-25
    return [word_index[word] if word in word_index else word_index["<span>"] for word in data_list]

def index2token(index_list, vocabulary):
    return [vocabulary[idx] for idx in index_list]

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def customed_preprocess(batch, word2index, label2index):
    x, y = zip(*batch) #x:tuple
    x = [token2index(sentence, word2index) for sentence in x]
    y = [token2index(sentence, label2index) for sentence in y]
    
    #pad x to same size in the batch, and convert x long tensor for rnn.pad_sequencce format
    #pad use the rare case <lol> instead of <pad>
    pad_token_idx = word2index['<lol>'] #twitter-25
    x = [torch.LongTensor(sentence) for sentence in x]
    x_pad = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=pad_token_idx)
    
    pad_label_idx = label2index['O']
    y = [torch.LongTensor(sentence) for sentence in y]
    y_pad = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=pad_label_idx)
    return x_pad, y_pad

def customed_test_preprocess(batch, word2index):
    x = batch
    x = [token2index(sentence, word2index, use_lower=True) for sentence in x]
    x = [torch.LongTensor(sentence) for sentence in x]

    is_stopword_list = [[True if word in stopword else False for word in sentence] for sentence in batch]
    lengths = [len(sentence) for sentence in x]
    pad_token_idx = word2index['<lol>'] #twitter-25
    x_pad = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=pad_token_idx)
    return x_pad, batch, lengths, is_stopword_list

def evaluate(model, loader):
    val_num, val_correct = 0,0
    total_val_loss, total_val_acc = 0, 0
    for batch_inputs, batch_labels in loader:
        with torch.no_grad():
            outputs = model(batch_inputs)
            loss = loss_function(outputs, batch_labels)
            total_val_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            val_correct += (predicted == batch_labels).sum().item()
            val_num += math.prod(batch_labels.shape)
    total_val_loss = total_val_loss / len(loader)
    total_val_acc = val_correct / val_num
    return total_val_loss, total_val_acc

def train(loss_function, optimizer, model, loader, epochs, val_loader=None, scheduler=None, model_path="./weights/model_test.pth"):
    for epoch in range(epochs):
        train_num, train_correct = 0, 0
        total_loss, total_acc = 0, 0
        for batch_inputs, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = loss_function(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            #summary
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            train_correct += (predicted == batch_labels).sum().item()
            train_num += math.prod(batch_labels.shape)
        total_loss = total_loss / len(loader)
        total_loss = total_loss
        total_acc = train_correct / train_num
        if val_loader and (epoch+1) % 5 == 0:
            model = model.eval()
            total_val_loss, total_val_acc = evaluate(model, val_loader)
            model = model.train()
            print(f'epoch: {epoch}, train-loss: {total_loss:.5f}, train-acc: {total_acc:.3f}, val_loss: {total_val_loss:.5f}, val_acc: {total_val_acc:.3f}')
        else:
            print(f'epoch: {epoch}, train-loss: {total_loss:.5f}, train-acc: {total_acc:.3f}, lr: {optimizer.param_groups[0]["lr"]:.5f}')
        scheduler.step()
        torch.save(model.state_dict(), model_path)

def inference(model, test_tokens):
    model = model.eval()
    result = []
    with torch.no_grad():
        outputs = model(test_tokens)
        batch_predict = torch.argmax(outputs, dim=1)
    result += to_numpy(batch_predict).tolist()
    return result

def result_parsing(data_list, result, lengths, is_stopword_list):
    text = ""
    index_label_table = {index:label for label, index in label_index_table.items()}
    idx = 0
    for i,batch_result in enumerate(result):
        for j,word_result in enumerate(batch_result):
            if j >= lengths[i]:
                #pass the <pad>
                break
            predict_name = index_label_table.get(word_result)

            if is_stopword_list[i][j] == True:
                predict_name = 'O'
            text += f'{data_list[i][j]}\t{predict_name}\n'
        text += '\n'
    return text


def loss_function(input, target):
    """
    focal loss
    input: [N, C]
    target: [N, ]
    """
    gamma = 2
    logpt = torch.log(input)
    pt = torch.exp(logpt)
    logpt = (1-pt)**gamma * logpt
    loss = F.nll_loss(logpt, target)
    return loss


def loss_function_normal(batch_outputs, batch_labels):
    log_softmax_outputs = torch.log(batch_outputs)
    loss = F.nll_loss(log_softmax_outputs, batch_labels)
    return loss

def loss_function_class_weight(batch_outputs, batch_labels):
    log_softmax_outputs = torch.log(batch_outputs)
    class_weights = torch.FloatTensor([100 for i in range(20)] +[1] )
    loss = F.nll_loss(log_softmax_outputs, batch_labels, weight=class_weights)
    return loss

def get_word_index_table(train_file_path, embedding_vector, test_mode=False):
    global word_index_table, label_index_table
    #extract data and label
    train_all_list = read_file(train_file_path)
    train_data_list, train_label_list = extract_data_label(train_all_list)
    assert len(train_data_list) == len(train_label_list)
    
    #build data and label index table
    # vocabulary = build_vocabulary(train_data_list)
    # word_index_table = word2index(vocabulary)
    word_index_table = vocabulary_key_index(embedding_vector)
    
    label_vocabulary = build_label_vocabulary(train_label_list)
    print('label_vocabulary: ', label_vocabulary)
    print('len(label_vocabulary): ', len(label_vocabulary))
    label_index_table = word2index(label_vocabulary)
    if not test_mode:
        return train_data_list, train_label_list, word_index_table, label_index_table
    else:
        return word_index_table, label_index_table

def get_train_loader(train_file_path, embedding_vector, batch_size):
    train_data_list, train_label_list, word_index_table, label_index_table = get_word_index_table(train_file_path, embedding_vector)
    data_content = compose_sentences(train_data_list)
    label_content = compose_sentences(train_label_list)
    index_content = [token2index(sentence, word_index_table) for sentence in data_content]
    #build dataset and dataloader
    data = list(zip(data_content, label_content))
    train_dataset = NerDataset(data)
    collate_fn = partial(customed_preprocess, word2index=word_index_table, label2index=label_index_table)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader

def get_val_loader(val_file_path, batch_size):
    val_all_list = read_file(val_file_path)
    val_data_list,val_label_list = extract_data_label(val_all_list)
    assert len(val_data_list) == len(val_label_list)
    val_data_content = compose_sentences(val_data_list)
    val_label_content = compose_sentences(val_label_list)
    val_data = list(zip(val_data_content, val_label_content))
    val_dataset = NerDataset(val_data)
    collate_fn = partial(customed_preprocess, word2index=word_index_table, label2index=label_index_table)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return val_loader

def get_test_token_and_data(test_file_path):
    test_all_list = read_file(test_file_path)
    test_data_list = extract_data(test_all_list)
    test_data_content = compose_sentences(test_data_list)
    all_tokens, all_data, lengths, is_stopword_list = customed_test_preprocess(test_data_content, word2index=word_index_table)
    return all_tokens, all_data, lengths, is_stopword_list

def get_test_token_and_data_benchmark(test_file_path):
    test_all_list = read_file(test_file_path)
    test_data_list, test_txt_list = extract_data_label_benchmark(test_all_list)
    test_data_content = compose_sentences(test_data_list)
    test_txt_list_content = compose_sentences(test_txt_list)
    all_tokens, all_data, lengths, is_stopword_list = customed_test_preprocess(test_data_content, word2index=word_index_table)
    return all_tokens, test_txt_list_content, lengths, is_stopword_list

def train_flow(train_file_path, val_file_path, batch_size, epochs, learning_rate, model_hyperparameters, model_path):
    #get glove embedding model
    embedding_model = KeyedVectors.load_word2vec_format("./glove-twitter-25.txt")
    #get stopword
    get_stopword()

    #data preprocess and build dataloader
    loader = get_train_loader(train_file_path, embedding_model, batch_size)
    val_loader = get_val_loader(val_file_path, batch_size)

    #build model, optimizer, scheduler
    vocab_size = len(word_index_table)
    output_size = len(label_index_table)
    model = LSTM_NER(model_hyperparameters, vocab_size, output_size, embedding_model)

    # model.load_state_dict(torch.load(r"./weights/model_test_latest_0448.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    #start training
    train(loss_function, optimizer, model, loader, epochs, val_loader=val_loader, scheduler=scheduler, model_path=model_path)

def test_flow(train_file_path, test_file_path, model_path, model_hyperparameters, save_file_path):
    #get glove embedding model
    embedding_model = KeyedVectors.load_word2vec_format("./glove-twitter-25.txt")

    #get stopword
    get_stopword()

    # data preprocess and build dataloader
    word_index_table, label_index_table =  get_word_index_table(train_file_path, embedding_model, test_mode=True)

    if 'test' in test_file_path:
        test_tokens, test_datas, lengths, is_stopword_list = get_test_token_and_data(test_file_path)
    else:
        test_tokens, test_datas, lengths, is_stopword_list = get_test_token_and_data_benchmark(test_file_path)

    #build model and load weights
    vocab_size = len(word_index_table)
    output_size = len(label_index_table)
    model = LSTM_NER(model_hyperparameters, vocab_size, output_size, embedding_model)
    model.load_state_dict(torch.load(model_path))
    
    #inference
    result = inference(model, test_tokens)
    
    #result parsing and save
    text = result_parsing(test_datas, result, lengths, is_stopword_list)
    with open(save_file_path, 'w', encoding="utf-8") as f:
        f.write(text)

### Classes ###
class NerDataset(Dataset):
    def __init__(self, data:list):
        self.x = [sample[0] for sample in data]
        self.y= [sample[1] for sample in data]
    def __getitem__(self,index:int):
        return self.x[index], self.y[index]
    def __len__(self):
        return len(self.x)


class LSTM_NER(nn.Module):
    def __init__(self, hyperparameters, vocab_size, output_size, embedding_model):
        super(LSTM_NER, self).__init__()

        """ Instance variables """
        self.embed_dim = hyperparameters["embed_dim"]
        self.hidden_dim = hyperparameters["hidden_dim"]
        self.freeze_embeddings = hyperparameters["freeze_embeddings"]
        self.bidirectional = hyperparameters["bidirectional"]

        self.dropout = nn.Dropout(0.5)
        """ Embedding Layer
        """
        self.output_size = output_size
        # oz defined
        # self.embeds = nn.Embedding(vocab_size, self.embed_dim, padding_idx=0)
        # if self.freeze_embeddings:
        #     self.embed_layer.weight.requires_grad = False
        
        # Load word2vec pre-train model
        # model = KeyedVectors.load_word2vec_format("./glove-wiki-gigaword-25.txt")
        self.embedding_model = embedding_model
        weights = torch.FloatTensor(self.embedding_model.vectors)
        print('vocab_size, embedding_dims: ', weights.shape)

        self.embeds = nn.Embedding.from_pretrained(weights, padding_idx=0)
        if self.freeze_embeddings:
            print('--freeze_embeddings--')
            self.embeds.requires_grad = False
        
        """ LSTM Layer
        """
        num_lstm_layers = 2
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim, num_layers=num_lstm_layers, batch_first=True, bidirectional=self.bidirectional)
        
        if self.bidirectional:
            D = 2
        else:
            D = 1
            
        """ Output Layer
        """
        self.output_layer = nn.Linear(D*self.hidden_dim, self.output_size)
        self.fc1 = nn.Linear(D*self.hidden_dim, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.output_size)
        

        """ Probabilities
        """
        if self.output_size == 1:
            self.probabilities = nn.Sigmoid()
        else:
            self.probabilities = nn.Softmax(dim=-1)

    def forward(self, inputs):
        """
        Let B:= batch_size
            D:= self.embed_dim
            H:= self.hidden_dim

        inputs: a (B, L) tensor of token indices
        """
        B, L = inputs.shape
        embeds = self.embeds(inputs)

        hidden = None
        lstm_output, hidden= self.lstm(embeds, hidden)

        #v1
        output = self.output_layer(lstm_output)
        #v2
        #use dropout to prevent overfitting
        # output = self.dropout(output)
        #v3
        # output1= self.fc1(lstm_output)
        # residual = output1 + embeds
        # output = self.fc2(residual)
        output = self.probabilities(output)
        #for cross-entropy format, cannot use view.(B, self.output_size, -1)
        output = output.permute(0, 2, 1)
        return output


if __name__ == '__main__':
    ### Global variables ###
    model_hyperparameters = {
        "embed_dim": 25,
        "hidden_dim": 25,
        "freeze_embeddings": True,
        "bidirectional":True,
    }

    task_mode = "test" #train or test

    if task_mode == "train":
        ### Train Global variables ###
        train_file_path, val_file_path = './train.txt', './dev.txt'
        batch_size = 32 
        epochs = 10 #1000
        learning_rate = 1e-2
        model_path = r"./weights/model_latest_focal.pth"

        train_flow(train_file_path, val_file_path, batch_size, epochs, learning_rate, model_hyperparameters, model_path)

    elif task_mode == "test":
        ### Test Global variables ###
        train_file_path, test_file_path = './train.txt', './test-submit.txt' #test.txt
        batch_size = 1
        model_path = r"./weights/model_latest_focal.pth"
        save_file_path = './test-submit_output.txt'

        test_flow(train_file_path, test_file_path, model_path, model_hyperparameters, save_file_path)
    else:
        assert "Error: task_mode not in ['train', 'test']"
    