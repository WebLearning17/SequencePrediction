import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim


CONTEXT_SIZE = 2 #表示想由前面的几个单词来预测这个单词
EMBEDDING_DIM = 10 #表示词嵌入的维度

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()


#单词序列分为三元祖，每个三元组前两个用于传入，后一个用于预测的结果
trigram = [((test_sentence[i], test_sentence[i+1]), test_sentence[i+2])
           for i in range(len(test_sentence) - 2)]

vocb = set(test_sentence)
word_to_ix = {word : i for i, word in enumerate(vocb)}
idx_to_word = {word_to_ix[word] : word for word in word_to_ix}


class NgramModel(nn.Module):
    def __init__(self, vocb_size, context_size, n_dim):
        super().__init__()
        self.n_word = vocb_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        log_prob = F.log_softmax(out, 1)
        return log_prob

net = NgramModel(len(vocb), CONTEXT_SIZE, EMBEDDING_DIM)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-5)

epoches = 200
for epoch in range(epoches) :
    train_loss = 0
    for word, label in trigram :
        word = Variable(torch.LongTensor([word_to_ix[i] for i in word]))
        label = Variable(torch.LongTensor([word_to_ix[label]]))
        out = net(word)
        loss = criterion(out, label)
        train_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0 :
        print('epoch: {}, Loss : {:.6f}'.format(epoch + 1, train_loss / len(trigram)))


#进行预测
net = net.eval()
word, label = trigram[19]
print('input: {}'.format(word))
print('input: {}'.format(label), end ="\n\n")

word = Variable(torch.LongTensor([word_to_ix[i] for i in word]))
out = net(word)

print("out:",out)

pred_label_idx = out.max(1)[1].data[0]
print(pred_label_idx)

predict_word = idx_to_word[int(pred_label_idx)]
print('real word is "{}", predicted word is "{}"'.format(label, predict_word))