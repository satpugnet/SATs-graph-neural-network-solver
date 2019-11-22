from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# print(dataset[0])
# print(dataset.num_node_features)
# print(dataset[0].edge_index)
# print(dir(dataset[0]))
# dataset[0].edge_index = 2 # torch.tensor([[2, 3], [4, 7]], dtype=torch.long)
# print(dataset[0].edge_index)

data = Data(x=torch.tensor([[2, 3, 4], [4, 7, 5], [4, 7, 5]], dtype=torch.float), y=torch.tensor([1], dtype=torch.long),
            edge_index=torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.long),
            edge_attr=torch.tensor([[4, 2], [3, 2], [1, 2]], dtype=torch.long))

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, len(data.y))
        self.fc2 = nn.Linear(len(data.y), 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc2(x)
        x = global_add_pool(x, torch.tensor([0, 0, 0]))

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
# data = dataset[0].to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    print(epoch)
    optimizer.zero_grad()
    out = model(data)
    print(out)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = float(pred.eq(data.y).sum().item())
acc = correct / sum(data.y)
print('Accuracy: {:.4f}'.format(acc))