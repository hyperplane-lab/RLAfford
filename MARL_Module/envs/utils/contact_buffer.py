import torch

class ContactBuffer() :

    def __init__(self, buffer_size, content_dim=3, device=torch.device('cpu')) :

        self.buffer_size = buffer_size
        self.content_dim = content_dim
        self.device = device
        self.buffer = torch.zeros((buffer_size, content_dim), device=device)
        self.top = 0
    
    def insert(self, batch) :

        batch_size = batch.shape[0]
        start_random_insert = batch_size

        if self.top+batch_size <= self.buffer_size :
            self.buffer[self.top:self.top+batch_size].copy_(batch)
            self.top += batch_size
        elif self.top < self.buffer_size :
            avl_len = self.buffer_size - self.top
            self.buffer[self.top:self.buffer_size].copy_(batch[:avl_len])
            start_random_insert = avl_len
            self.top += avl_len
        else :
            start_random_insert = 0
        
        num_insert = batch_size - start_random_insert
        if num_insert > 0 :
            i,idx = torch.sort(self.buffer[:, -1])
            i,rank = torch.sort(idx)
            possibility = torch.softmax(rank.float(), dim=-1)
            replace_idx = torch.multinomial(possibility, num_insert)
            self.buffer[replace_idx] = batch[start_random_insert:]
    
    def all(self) :

        return self.buffer[:self.top, :]
        
    def print(self):

        print(self.buffer[:self.top])
    
    def save(self, path) :

        torch.save(self.buffer[:self.top], path)

if __name__ == "__main__" :

    buffer = ContactBuffer(10, 3, torch.device('cpu'))
    buffer.insert(torch.tensor([[1,1,1],[2,2,2],[3,3,3]]).float())
    buffer.print()
    buffer.insert(torch.tensor([[4,4,4],[5,5,5],[6,6,6],[7,7,7]]).float())
    buffer.print()
    buffer.insert(torch.tensor([[8,8,8],[9,9,9],[10,10,10],[11,11,11]]).float())
    buffer.print()
    buffer.insert(torch.tensor([[12,12,12],[13,13,13]]).float())
    buffer.print()