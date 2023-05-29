import numpy as np 
import tqdm
import pickle

# from preprocess import get_values_for_parameter,prepare_dataset,filter_dataset,save_vocab_file






class model:
    def __init__(self,dataset, vector_size: int, lr: float):
        self.lr = lr
        self.vector_size = vector_size
        self.dataset = dataset
        self.vocab_size = np.amax(dataset)+1
        # print(f"minimum value in a is {np.amin(dataset)}")
        self.w1 = np.random.uniform(low=-0.8,high=0.8,size = (self.vocab_size,self.vector_size))
        self.w2 = np.random.uniform(low=-0.8,high=0.8,size =(self.vector_size,self.vocab_size))
        
        
    def softmax(self,x):
        # print(f"x = {x}")
        # x = x.T
        # print(np.exp(x) / np.sum(np.exp(x), axis=0))
            # print(x)
            # exit(0)
        y = np.exp(x.T) / np.sum(np.exp(x), axis=1)
        return y.T
    
    def dataset_dot_w1(self, dataset):
        # print(f"shape of w1 is {self.w1.shape}")
        self.z1 = np.zeros((len(dataset),self.vector_size),dtype="int")
        for j, (x,y) in enumerate(dataset):
            for i in range(self.w1.shape[1]):
                # try:
                    self.z1[j][i] = self.w1[x][i] + self.w1[y][i]
                # except:
                #     print(f"i is {i} and j is {j} and x is {x} and y is {y} shape of w1 is {self.w1.shape} and shape of z1 is {self.z1.shape}")
                #     exit(0)
    def forward(self,dataset):        
        # self.z1 = np.dot(self.dataset,self.w1)
        self.dataset_dot_w1(dataset)
        self.z2 = np.dot(self.z1,self.w2)
        self.z3 = self.softmax(self.z2)
        return self.z3
    
    def get_dataset_arr(self,data):
        temp = np.zeros((len(data),self.vocab_size),dtype="bool")
        for ind,(x,y) in enumerate(data):
            temp[ind][x] = 1
            temp[ind][y] = 1
        return temp
    
    def backward(self,dataset, outputs):
        dataset = self.get_dataset_arr(dataset)
        self.dz2 = outputs - dataset
        self.dw2 = np.dot(self.z1.T,self.dz2)
        self.dz1 = np.dot(self.dz2,self.w2.T)
        self.dw1 = np.dot(dataset.T,self.dz1)
        self.w1 -= self.lr * self.dw1
        self.w2 -= self.lr * self.dw2 
     
    def divide_list_into_n_chunks(self,n):
        i = 0
        for i in range(0,len(self.dataset) - 2 * len(self.dataset)//n,len(self.dataset)//n):
            yield self.dataset[i:i+len(self.dataset)//n]
        yield self.dataset[i+len(self.dataset)//n:]
     
        
    def train_for_n_epoch(self,number_of_epochs, number_of_batch = 2):
        # print(len(self.dataset))
        # exit(0)
        for i in tqdm.tqdm(range(number_of_epochs)):
            for _,data in tqdm.tqdm(enumerate(self.divide_list_into_n_chunks(n = number_of_batch))):
                outputs = self.forward(data)
                print(outputs)
                self.backward(dataset= data, outputs = outputs)
            with open("model.pkl","wb") as f:
                pickle.dump(self.w1,f)
                pickle.dump(self.w2,f)

    # def encode_data(self,data,vocab_size):
    #     encoded_data = np.zeros((vocab_size,1))
    #     encoded_data[data] = 1
    #     return encoded_data
    


# def encode_data(data,vocab_size):
#     encoded_data = np.zeros((vocab_size,1))
#     encoded_data[data] = 1
#     return encoded_data


def encode_complete_dataset(data,vocab_size):
    encoded_data = np.zeros((len(data),vocab_size),dtype="bool")
    for i in range(len(data)):
        # print(data[i][0])
        encoded_data[i,data[i]] = 1
    return encoded_data




def get_words_vocab(word_vocab_file):
    return open(word_vocab_file,"r",encoding="utf-8").read().split("\n")


if __name__ == "__main__":
    
    config = {
        "learning_rate": 0.01,
        "vector_size": 10,
        "number_of_batch":1000,
        "epochs": 10,
    }

    dataset = pickle.load(open("dataset.pkl","rb"))
    
    # dataset = [[1,2],[1,3],[2,3],[4,1],[4,2],[5,4],[5,3],[5,1]]
    
    print(len(dataset))
    
    model = model(dataset = dataset, lr=config['learning_rate'], vector_size=config['vector_size'])
    model.train_for_n_epoch(number_of_epochs=config['epochs'],number_of_batch=config['number_of_batch'])
