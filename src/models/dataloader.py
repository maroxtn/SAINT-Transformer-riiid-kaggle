import torch

class TextLoader(torch.utils.data.Dataset):
    def __init__(self, data):
        self.x, self.y, self.ts, self.user_answer = [], [], [], []
        
        for line in data:
            x, y, ts, user_answer = line
            
            self.x.append(x)
            self.y.append(y)      
            self.ts.append(ts)
            self.user_answer.append(user_answer)

    def __getitem__(self, index):
        return (torch.LongTensor(self.x[index]), 
                torch.LongTensor(self.y[index]), 
                torch.LongTensor(self.ts[index]),
               torch.LongTensor(self.user_answer[index]))

    def __len__(self):
        return len(self.x)