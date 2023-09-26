import torch
import torch.nn as nn
import torch.nn.functional as F

def smish(x):
    return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))

class GatedLinearUnit(nn.Module):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(units, units)
        self.sigmoid = nn.Linear(units, units)
    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.sigmoid(x))

class GatedResidualNetwork(nn.Module):
    def __init__(self, units, dropout_rate):
        super(GatedResidualNetwork, self).__init__()
        self.units = units
        self.relu_dense = nn.Linear(units, units)
        self.linear_dense = nn.Linear(units, units)
        self.dropout = nn.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = nn.LayerNorm(units)
        self.project = nn.Linear(units, units)
    def forward(self, x):
        x1 = smish(self.relu_dense(x))
        x1 = self.linear_dense(x)
        x1 = self.dropout(x1)
        if x1.shape[1] != self.units:
            x = self.project(x)
        x1 = x + self.gated_linear_unit(x1)
        x1 = self.layer_norm(x1)
        return x1

class VariableSelection(nn.Module):
    def __init__(self, num_features, units, dropout_rate):
        super(VariableSelection, self).__init__()
        self.grns = nn.ModuleList([GatedResidualNetwork(units, dropout_rate) for _ in range(num_features)])
        self.grn_concat = GatedResidualNetwork(units * num_features, dropout_rate)
        self.softmax = nn.Linear(units * num_features, num_features)
    def forward(self, x):
        inputs = torch.cat(x, dim=1)
        v = self.grn_concat(inputs)
        v = F.softmax(self.softmax(v), dim=1).unsqueeze(2)
        x_list = [self.grns[idx](x[idx]) for idx in range(len(x))]
        x_stack = torch.stack(x_list, dim=1)
        outputs = torch.squeeze(torch.bmm(v, x_stack), dim=1)
        return outputs

class VariableSelectionFlow(nn.Module):
    def __init__(self, num_features, units, dropout_rate, dense_units=None):
        super(VariableSelectionFlow, self).__init__()
        self.variableselection = VariableSelection(num_features, units, dropout_rate)
        self.dense_units = dense_units
        if dense_units:
            self.dense_list = nn.ModuleList([nn.Linear(units, dense_units) for _ in range(num_features)])
    def forward(self, x):
        if self.dense_units:
            split_input = torch.split(x, 1, dim=1)
            l = [F.linear(dense(split_input[i])) for i, dense in enumerate(self.dense_list)]
        else:
            l = torch.split(x, 1, dim=1)
        return self.variableselection(l)
