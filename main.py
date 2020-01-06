import konlpy
import pandas as pd
import numpy as np
from konlpy.tag import *
from collections import Counter
from utils.preprocessing import corpus_span, padding
import time
from torch import nn
import torch
from models.model import *
import matplotlib.pyplot as plt
from torchsummary import summary
import pickle

PAD = 0
stopwords = ['XXX', '.', '-', '(', ')', ':', '!', '?', ')-', '.-', 'ㅡ', 'XXXXXX', '..', '.('] #필요없는 단어 리스트


def evaluate(data, corpus, model_path, max_len, stop_words, tokenizer = Kkma(), PAD = 0):

    test =pd.read_csv(test_path)

    idx = test.iloc[:,0]
    test_data = test.iloc[:,2]

    _, tokens = corpus_span(test_data, tokenizer = Kkma(), corpus = corpus, stop_words = stop_words)

    pad_tokens, _ = padding(tokenized_data = tokens, PAD = PAD, evaluation = True, max_len = max_len)

    model = torch.load(model_path)
    model.eval()

    y_pred = model(pad_tokens)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    y_pred = pd.DataFrame(y_pred.detach().numpy(), columns = ["smishing"])
    y_pred = pd.concat([idx,y_pred], axis = 1)

    y_pred.to_csv("./submission.csv", index = False)


def data_gen(data_path, stop_words, tokenizer = Kkma(), size = 2000):

    train_data = pd.read_csv(data_path)
    y = train_data.iloc[:, 3]

    norm_seed = np.random.choice(y[y == 0].index, size // 2)
    abnorm_seed = np.random.choice(y[y == 1].index, size // 2)

    target = np.concatenate([train_data.iloc[norm_seed,3], train_data.iloc[abnorm_seed,3]])
    dataset = np.concatenate([train_data.iloc[norm_seed,2],train_data.iloc[abnorm_seed,2]])

    print(f"target 0 : {len(target[target == 0])} target 1 : {len(target[target == 1])}")\
    
    target = torch.Tensor(target)

    corpus, tokens = corpus_span(data = dataset, tokenizer = tokenizer , stop_words = stop_words)

    '''
    #Save corpus
    with open("./corpus.pickle", "wb") as handle:
        pickle.dump(corpus, handle, protocol = pickle.HIGHEST_PROTOCOL)
    '''


    #pad_tokens : (total_size , max_len)
    pad_tokens, max_len = padding(tokens, PAD = PAD)
    

    # data 섞기
    total_size = len(pad_tokens)
    perm = np.random.permutation(total_size)
    pad_tokens = pad_tokens[perm]
    target = target[perm]

    print(f"random_test : {target[np.random.choice(len(target), 10)]}")

    # train, validation split
    train_val_ratio = 0.9
    tmp = int(total_size * train_val_ratio)

    x_train = pad_tokens[:tmp]
    x_val = pad_tokens[tmp:]

    y_train = target[:tmp].unsqueeze(1)
    y_val = target[tmp:].unsqueeze(1)

    train_data = {"x_train" : x_train, 
                "y_train" : y_train,
                "x_val" : x_val, 
                "y_val" : y_val}

    return train_data, corpus, max_len

def train(model, device, criterion, optimizer, epochs, batch_size, **train_data):
    x_train = train_data["x_train"]
    y_train = train_data["y_train"]
    x_val = train_data["x_val"].to(device)
    y_val = train_data["y_val"].to(device)

    model.to(device)
    model.train()

    trn_loss_list = []
    val_loss_list = []
    trn_loss = 0.
    # Training loop
    ts = time.time()

    best_score = 1e-10
    for epoch in range(epochs):
        # 1) Forward pass: Compute predicted y by passing x to the model
        for i in range(len(x_train) // batch_size):
            seed = np.random.choice(len(x_train), batch_size)
            x_data = x_train[seed].to(device)

            y_data = y_train[seed].to(device)
            

            y_data = y_data.unsqueeze(1)

            y_pred = model(x_data)

            #print(y_data.shape, y_pred.shape)

            # 2) Compute and print loss
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()

            loss = criterion(y_pred, y_data)
            loss.backward()
            optimizer.step()

            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred < 0.5] = 0

            #print(f"Iteration score for test : {len(y_pred[y_pred == y_data])/ len(y_data)}")
            trn_loss += loss.item()

            del x_data, y_data

        val_output = model(x_val)
        val_loss = criterion(val_output, y_val)

        val_output[val_output > 0.5] = 1
        val_output[val_output < 0.5] = 0

        val_score = len(val_output[val_output == y_val]) / len(y_val)


        #Save model
        best_model = (val_score > best_score)
        if best_model:
            best_score = val_score
            torch.save(model, "./best_model.pt")

        print(f"Epoch: {epoch} | Loss: {trn_loss / 100}")
        print(f"Validation loss: {val_loss} | Score: {val_score}")

        trn_loss_list.append(trn_loss/100)
        val_loss_list.append(val_loss/100)
        
        trn_loss = 0.

    print("time spend %d"%(time.time() - ts))


    plt.figure(figsize = (10,8))
    plt.plot(np.linspace(1,epochs,epochs), trn_loss_list)
    plt.plot(np.linspace(1,epochs,epochs), val_loss_list)
    plt.show()

def main():

    np.random.seed(941017)
    size = 2000
    hidden_layer = 128
    num_layer = 3

    print("\n ==============================> Training Start <=============================")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
    print(torch.cuda.is_available())
    if torch.cuda.device_count() >= 1:
        print(f"\n ====> Training Start with GPU Number : {torch.cuda.device_count()} GPU Name: {torch.cuda.get_device_name(device=None)}")
    else:
        print(f"\n ====> Training Start with CPU Number : {torch.cuda.device_count()} CPU Name: {torch.cuda.get_device_name(device=None)}")


    train_path = "./data/train.csv"
    stopwords = ['XXX', '.', '-', '(', ')', ':', '!', '?', ')-', '.-', 'ㅡ', 'XXXXXX', '..', '.('] #필요없는 단어 리스트

    train_data, corpus, max_len = data_gen(data_path = train_path, stop_words = stopwords, tokenizer = Kkma(), size = size)
    
    args = {"corpus" : corpus, "max_len" : max_len, "hidden_layer" : hidden_layer}

    with open("./args.pickle", "wb") as handle:
        pickle.dump(args, handle, protocol = pickle.HIGHEST_PROTOCOL)


    #backpropagataion parameters
    epochs = 7
    batch_size = 150
    learning_rate= 0.001

    model = RNNLM(vocab_size = len(corpus), embed_size = 300, hidden_layer = hidden_layer, max_len = max_len, n_layers = num_layer, bidirectional = True)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-8)

    # 모델의 state_dict 출력
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    train(model = model, device = device, criterion = criterion, optimizer = optimizer, epochs = epochs, batch_size = batch_size, **train_data)


if __name__ == "__main__":
    #main()
    test_path = "./data/public_test.csv"
    with open("./args.pickle", "rb") as handle:
        args = pickle.load(handle)
    
    corpus = args["corpus"]
    max_len = args["max_len"]
    evaluate(data = test_path, corpus = corpus, model_path = "./best_model.pt", max_len = max_len, stop_words = stopwords)