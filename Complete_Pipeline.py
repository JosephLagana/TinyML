

# LET'S INSTALL EVERYTHING WE NEED.
# 
#
# %%capture
# 
!git clone https://github.com/SamsungLabs/zero-cost-nas
!pip install -Uqq ipdb
!pip install -U fvcore
!pip install psutil gputil
!pip install pynvml
!pip install pyvww
!pip install thop
!pip install timm
!pip install torchsummaryX
!pip install zero-cost-nas/


# LET'S IMPORT EVERYTHING WE NEED TO.

import copy
import gc
import math
import numpy as np
import pandas as pd
import pyvww
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import types
from foresight.pruners.p_utils import *
from foresight.pruners import measures
from fvcore.nn import FlopCountAnalysis
from io import StringIO
from PIL import Image
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from thop import profile
from torch.utils.data import Dataset, DataLoader
from torchsummaryX import summary
from torchvision import transforms


Commented out IPython magic to ensure Python compatibility.
# upload the dataset.
# move this peace of code to work only with gdrive to retrive the csv file

%%capture

!mkdir path-to-mscoco-dataset
!mkdir path-to-mscoco-dataset/all2017
!mkdir path-to-mscoco-dataset/annotations
!mkdir scripts
from google.colab import drive
drive.mount('/content/gdrive')


Commented out IPython magic to ensure Python compatibility.
%%capture

!unzip /content/gdrive/MyDrive/dataset.zip
!unzip /content/gdrive/MyDrive/annotations.zip
!unzip /content/gdrive/MyDrive/scripts.zip
!find /content -name "*.jpg" -type f | xargs -I '{}' mv '{}' /content/path-to-mscoco-dataset/all2017
!mv /content/*.json /content/path-to-mscoco-dataset/annotations
!mv /content/*.py /content/scripts
!mv /content/mscoco_minival_ids.txt /content/scripts
!rmdir /content/dataset

!python scripts/create_coco_train_minival_split.py \
  --train_annotations_file=path-to-mscoco-dataset/annotations/instances_train2017.json \
  --val_annotations_file=path-to-mscoco-dataset/annotations/instances_val2017.json \
--output_dir=path-to-mscoco-dataset/annotations/

!python scripts/create_visualwakewords_annotations.py \
  --train_annotations_file=path-to-mscoco-dataset/annotations/instances_maxitrain.json \
  --val_annotations_file=path-to-mscoco-dataset/annotations/instances_minival.json \
  --output_dir=path-to-mscoco-dataset/annotations/ \
  --threshold=0.005 \
  --foreground_class="person"


# fuction to reduce image size: it was written by joseph and improved by matteo.

def joseph(img):
  preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # ricalcolare medie e sd di ciasun canale
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  data = preprocess(img)
  return data

# let's use pytorch to create the dataset

train = pyvww.pytorch.VisualWakeWordsClassification(
    root = "path-to-mscoco-dataset/all2017",
    annFile = "path-to-mscoco-dataset/annotations/instances_train.json",
    transform = joseph
)

test = pyvww.pytorch.VisualWakeWordsClassification(
    root = "path-to-mscoco-dataset/all2017",
    annFile = "path-to-mscoco-dataset/annotations/instances_val.json",
    transform = joseph
)

# this time is better to do that after the loading of the dataset.
# the uploading process is in fact slower when using the gpu.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

loss_function = torch.nn.CrossEntropyLoss()

# The final neural network should satisfy the following constraints:
# <= 2.5M parameters;
# <= 200M Flops;
# >= 80% of accuracy.

# very useful function for creating cnn:

def ensure_divisible(number, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_num = max(min_value, int(number + divisor / 2) // divisor * divisor)
    if new_num < 0.9 * number:
        new_num += divisor
    return new_num

# MOBILENETV2

class Bottleneck(nn.Module):
    # The basic unit of MobileNetV2, including Linear Bottlenecks and Inverted Residuals
    def __init__(self, in_channels_num, out_channels_num, stride, expansion_factor):
        super(Bottleneck, self).__init__()
        # Number of channels for Depthwise Convolution input/output
        DW_channels_num = round(in_channels_num * expansion_factor)
        # Whether to use residual structure or not
        self.use_residual = (stride == 1 and in_channels_num == out_channels_num)

        if expansion_factor == 1:
            # Without expansion, the first depthwise convolution is omitted
            self.conv = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels=in_channels_num, out_channels=in_channels_num, kernel_size=3, stride=stride, padding=1, groups=in_channels_num, bias=False),
                nn.BatchNorm2d(num_features=in_channels_num),
                nn.ReLU6(inplace=True),
                # Linear-PW
                nn.Conv2d(in_channels=in_channels_num, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels_num)
            )
        else:
            # With expansion
            self.conv = nn.Sequential(
                # Pointwise Convolution for expansion
                nn.Conv2d(in_channels=in_channels_num, out_channels=DW_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=DW_channels_num),
                nn.ReLU6(inplace=True),
                # Depthwise Convolution
                nn.Conv2d(in_channels=DW_channels_num, out_channels=DW_channels_num, kernel_size=3, stride=stride, padding=1, groups=DW_channels_num, bias=False),
                nn.BatchNorm2d(num_features=DW_channels_num),
                nn.ReLU6(inplace=True),
                # Linear-PW
                nn.Conv2d(in_channels=DW_channels_num, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels_num)
            )

    def forward(self, x):
        if self.use_residual:
            return self.conv(x) + x
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, depths = [1, 2, 3, 4, 3, 3, 1],
                 expansion_factor = [1, 6, 6, 6, 6, 6, 6],
                 output_channels = [16, 24, 32, 64, 96, 160, 320],
                 classes_num=1000, input_size=224, width_multiplier=1.0):
        super(MobileNetV2, self).__init__()
        first_channel_num = 32
        last_channel_num = 1280
        divisor = 8
        stride = [1, 2, 2, 2, 1, 2, 1]
        '''
           original bottleneck settings:
           # t: expansion factor,
           # c: number of output channels,
           # n: repeat times,
           # s: stride
           # t,   c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        '''
        bottleneck_setting = zip(expansion_factor, output_channels, depths,
                                 stride)
        # feature extraction part
        # input layer
        input_channel_num = ensure_divisible(first_channel_num * width_multiplier, divisor)
        last_channel_num = ensure_divisible(last_channel_num * width_multiplier, divisor) if width_multiplier > 1 else last_channel_num
        self.network = []
        first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=input_channel_num, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=input_channel_num),
            nn.ReLU6(inplace=True)
        )
        self.network.append(first_layer)
        # Overlay of multiple bottleneck structures
        # Join the layers of the network sequentially
        for t, c, n, s in bottleneck_setting:
            output_channel_num = ensure_divisible(c * width_multiplier, divisor)
            for i in range(n):
                if i == 0:
                    # The first layer of each bottleneck performs the convolution with stride>=1
                    self.network.append(Bottleneck(in_channels_num=input_channel_num,
                                                   out_channels_num=output_channel_num,
                                                   stride=s, expansion_factor=t))
                    input_channel_num = output_channel_num
                else:
                    # The later layers of the bottleneck perform the convolution with stride=1
                    self.network.append(Bottleneck(in_channels_num=input_channel_num,
                                                   out_channels_num=output_channel_num,
                                                   stride=1, expansion_factor=t))
        # The last several layers
        self.network.append(
            nn.Sequential(
                nn.Conv2d(in_channels=input_channel_num, out_channels=last_channel_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=last_channel_num),
                nn.ReLU6(inplace=True)
            )
        )
        self.network.append(
            nn.AvgPool2d(kernel_size=input_size//32, stride=1)
        )
        self.network = nn.Sequential(*self.network)

        # Classification part
        self.classifier = nn.Linear(last_channel_num, classes_num)

        # Initialize the weights
        self._initialize_weights()

    def forward(self, x):
        x = self.network(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# compute value for each net using jacob, synflow and snip metrics

measure_names = ('jacob_cov', 'snip', 'synflow')

def no_op(self,x):
    return x

def copynet(self, bn):
    net = copy.deepcopy(self)
    if bn==False:
        for l in net.modules():
            if isinstance(l,nn.BatchNorm2d) or isinstance(l,nn.BatchNorm1d) :
                l.forward = types.MethodType(no_op, l)
    return net

def find_measures_arrays(net_orig, trainloader, dataload_info, device, measure_names=None, loss_fn=F.cross_entropy):
    if measure_names is None:
        measure_names = measures.available_measures

    dataload, num_imgs_or_batches, num_classes = dataload_info

    if not hasattr(net_orig,'get_prunable_copy'):
        net_orig.get_prunable_copy = types.MethodType(copynet, net_orig)

    #move to cpu to free up mem
    torch.cuda.empty_cache()
    net_orig = net_orig.cpu()
    torch.cuda.empty_cache()

    #given 1 minibatch of data
    if dataload == 'random':
        inputs, targets = get_some_data(trainloader, num_batches=num_imgs_or_batches, device=device)
    elif dataload == 'grasp':
        inputs, targets = get_some_data_grasp(trainloader, num_classes, samples_per_class=num_imgs_or_batches, device=device)
    else:
        raise NotImplementedError(f'dataload {dataload} is not supported')

    done, ds = False, 1
    measure_values = {}

    while not done:
        try:
            for measure_name in measure_names:
                if measure_name not in measure_values:
                    val = measures.calc_measure(measure_name, net_orig, device, inputs, targets, loss_fn=loss_fn, split_data=ds)
                    measure_values[measure_name] = val

            done = True
        except RuntimeError as e:
            if 'out of memory' in str(e):
                done=False
                if ds == inputs.shape[0]//2:
                    raise ValueError(f'Can\'t split data anymore, but still unable to run. Something is wrong')
                ds += 1
                while inputs.shape[0] % ds != 0:
                    ds += 1
                torch.cuda.empty_cache()
                print(f'Caught CUDA OOM, retrying with data split into {ds} parts')
            else:
                raise e

    net_orig = net_orig.to(device).train()
    return measure_values

def find_measures(net_orig,                  # neural network
                  dataloader,                # a data loader (typically for training data)
                  dataload_info,             # a tuple with (dataload_type = {random, grasp}, number_of_batches_for_random_or_images_per_class_for_grasp, number of classes)
                  device,                    # GPU/CPU device used
                  loss_fn=F.cross_entropy,   # loss function to use within the zero-cost metrics
                  measure_names=None,        # an array of measure names to compute, if left blank, all measures are computed by default
                  measures_arr=None):        # [not used] if the measures are already computed but need to be summarized, pass them here

    #Given a neural net
    #and some information about the input data (dataloader)
    #and loss function (loss_fn)
    #this function returns an array of zero-cost proxy metrics.

    def sum_arr(arr):
        sum = 0.
        for i in range(len(arr)):
            sum += torch.sum(arr[i])
        return sum.item()

    if measures_arr is None:
        measures_arr = find_measures_arrays(net_orig, dataloader, dataload_info,
                                            device, loss_fn=loss_fn,
                                            measure_names=measure_names)

    measures = {}
    for k,v in measures_arr.items():
        if k=='jacob_cov':
            measures[k] = v
        else:
            measures[k] = sum_arr(v)

    return measures

# SEARCH-SPACE FOR MOBILENETV2:

# for each block we define some options for the number of output channels.
# reading lines: same net (iteration of the search), different blocks;
# reading columns: same block, different nets.

# original channel list:
# [16, 24, 32, 64, 96, 160, 320]

structures = [
    [1, 3, 2, 4, 3, 2, 1],
    [1, 2, 3, 4, 3, 3, 1],
    [1, 2, 2, 3, 3, 4, 1],
    [1, 3, 3, 3, 4, 2, 1],
    [1, 4, 4, 4, 3, 2, 1],
    [1, 4, 3, 2, 3, 4, 1],
    [1, 2, 4, 3, 4, 2, 1],
    [1, 2, 1, 2, 3, 4, 1]
]

expansion_factors = [
    [1, 6, 6, 6, 6, 6, 6],
    [1, 2, 2, 2, 4, 4, 4],
    [1, 2, 2, 4, 4, 6, 6],
    [1, 2, 4, 4, 4, 6, 6],
    [2, 2, 2, 2, 4, 4, 6],
    [2, 4, 4, 4, 4, 4, 6],
    [2, 4, 4, 6, 6, 6, 6],
    [2, 3, 2, 4, 3, 1, 1]
]

output_channels = [
    [ 4,  4, 16, 64, 256, 256, 256],
    [ 4,  8, 12, 16,  24,  48,  96],
    [ 8, 12, 16, 32,  48,  96, 160],
    [ 8, 16, 16, 24,  32,  48,  56],
    [16, 24, 32, 64,  96, 160, 320],
    [16, 16, 24, 24,  64,  64, 128],
    [24, 32, 40, 80, 160, 160, 320],
    [24, 24, 64, 64, 128, 128, 160]
]

# last step: create the dataframe we will save everything into.

totale_reti = (len(output_channels) * len(expansion_factors) * len(structures))
print("Totale reti: ", totale_reti)

# # NET GENERATOR FOR MOBILENETV2
# 
# # we have to run this cell only if we don't have any net: after the net
# # generation has been completed, comment everything.
# 
# %%capture
# 
# """
# 
# df = pd.DataFrame([], columns=['structure', 'expansion_factors', 'channels',
#                                'score_jacob', 'score_synflow', 'score_snip',
#                                'score_AVG', 'param','flops'])
# dataloader = DataLoader(train, batch_size = 2, shuffle = True)
# feature, labels = next(iter(dataloader))
# i = 0
# 
# for structure in structures:
#     for expansion_factor in expansion_factors:
#         for lista_canali in output_channels:
#             df.loc[i, 'structure'] = structure
#             df.loc[i, 'expansion_factors'] = expansion_factor
#             df.loc[i, 'channels'] = lista_canali
#             net = MobileNetV2(structure, expansion_factor, lista_canali)
#             vector = find_measures(net,
#                                     dataloader,
#                                     ('random', 2, 2),
#                                     # a tuple with:
#                                     # (dataload_type = {random, grasp},
#                                     #  number_of_batches_for_random,
#                                     #  number of classes)
#                                     device,
#                                     measure_names = measure_names)
#             df.loc[i, 'score_jacob'] = vector['jacob_cov']
#             df.loc[i, 'score_synflow'] = vector['synflow']
#             df.loc[i, 'score_snip'] = vector['snip']
#             feature = feature.to(device)
#             FLOPs, params = profile(net, inputs=(feature,), verbose=0)
#             df.loc[i, 'param'] = params/1000000
#             df.loc[i, 'flops'] = FLOPs/1000000
#             i += 1
#             del net
#             gc.collect()
#             torch.cuda.empty_cache()
# 
# """
#

# run that everytime after the first net generation has been completed

f = pd.read_csv('/content/gdrive/MyDrive/dataframe.csv', index_col = False,
                  usecols = ['score_jacob', 'score_synflow', 'score_snip',
                             'score_AVG', 'param', 'flops'])

prova = pd.DataFrame(index = (range(totale_reti)),
                     columns =
                     ['structure', 'expansion_factors', 'channels'])

i = 0
for structure in structures:
    for expansion_factor in expansion_factors:
        for lista_canali in output_channels:
            prova.loc[i, 'structure'] = structure
            prova.loc[i, 'expansion_factors'] = expansion_factor
            prova.loc[i, 'channels'] = lista_canali
            i += 1

df = pd.concat([prova, f], axis = 1)

# what are the mean and standard deviation for each measure? we need that for
# the standardization of the scores and to define the vote measure

jacob_mean = df['score_jacob'].mean()
synflow_mean = df['score_synflow'].mean()
snip_mean = df['score_snip'].mean()

jacob_sd = df['score_jacob'].std()
synflow_sd = df['score_synflow'].std()
snip_sd = df['score_snip'].std()

print("Jacob   -> mean: {:.2f}, std: {:.5f}".format(jacob_mean, jacob_sd))
print("Synflow -> mean: {:.2f}, std: {:.2f}".format(synflow_mean, synflow_sd))
print("Snip    -> mean: {:.2f}, std: {:.2f}".format(snip_mean, snip_sd))

# best overall net each measure

max_score_jacob = df['score_jacob'].max()
max_index_jacob = df['score_jacob'].values.argmax()
best_net_jacob = df.loc[max_index_jacob]

max_score_synflow = df['score_synflow'].max()
max_index_synflow = df['score_synflow'].values.argmax()
best_net_synflow = df.loc[max_index_synflow]

max_score_snip = df['score_snip'].max()
max_index_snip = df['score_snip'].values.argmax()
best_net_snip = df.loc[max_index_snip]

print("Jacob -> best score: {:.2f}, best net index: {}".format(max_score_jacob,
                                                               max_index_jacob))
print("Best Jacob net: \n")
print(best_net_jacob)
print("\n")

print("Synflow -> best score: {:.2f}, best net index: {}".format(max_score_synflow,
                                                                 max_index_synflow))
print("Best Synflow net: \n")
print(best_net_synflow)
print("\n")


print("Snip -> best score: {:.2f}, best net index: {}".format(max_score_snip,
                                                              max_index_snip))
print("Best Snip net: \n")
print(best_net_snip)
print("\n")

"""

# save the dataframe

df.to_csv("dataframe.csv")

"""

# standardization cycle

for i in range(totale_reti):
    df.loc[i, 'score_jacob'] = (df.loc[i, 'score_jacob'] - jacob_mean) / jacob_sd
    df.loc[i, 'score_synflow'] = (df.loc[i, 'score_synflow'] - synflow_mean) / synflow_sd
    df.loc[i, 'score_snip'] = (df.loc[i, 'score_snip'] - snip_mean) / snip_sd

# vote measure

for i in range(totale_reti):
    df.loc[i, 'score_AVG'] = (df.loc[i, 'score_jacob'] +
                              df.loc[i, 'score_synflow'] +
                              df.loc[i, 'score_snip']) / 3

# standardized dataframe

df

# how many nets respect the constraints on flops and parameters after the
# mutation?

df1 = pd.DataFrame(index = (range(totale_reti)),
                   columns =
                 ['structure', 'expansion_factors',  'channels', 'score_jacob',
                  'score_synflow', 'score_snip', 'score_AVG', 'param', 'flops'])
k = 0
for i in range(len(df)):
    if (df.loc[i, 'param'] <= 2.5 and df.loc[i, 'flops'] <= 200):
        df1.loc[k] = df.loc[i]
        k += 1
df1 = df1[0:k]
df1

print("Totale reti che rispettano i vincoli: ", k)

# from now on we will only consider the vote measure

# best avg overall net:

max_score_avg = df['score_AVG'].max()
max_index_avg = df['score_AVG'].values.argmax()
best_net_avg = df.loc[max_index_avg]

print("Best AVG measure overall net: \n")
print("Vote -> best score: {:.2f}, best net index: {}".format(max_score_avg,
                                                              max_index_avg))
print("Best AVG net: \n")
print(best_net_avg)
print("\n")

# best avg net respecting constraints

max_score_avg = df1['score_AVG'].max()
max_index_avg = df1['score_AVG'].values.argmax()
best_net_avg = df1.loc[max_index_avg]

print("Best AVG measure net respecting constraints: \n")
print("Vote -> best score: {:.2f}, best net index: {}".format(max_score_avg,
                                                              max_index_avg))
print("Best AVG net: \n")
print(best_net_avg)

def mutate(input, dataloader, feature, labels, loss_function):
    nuovo = pd.DataFrame(columns =
                        ['structure', 'expansion_factors',
                         'channels', 'score_jacob', 'score_synflow',
                         'score_snip', 'score_AVG', 'param', 'flops'])
    choices = ['structure', 'expansion_factors', 'channels']
    choice = random.choice(choices)
    structure = input['structure'].copy()
    factors = input['expansion_factors'].copy()
    channels = input['channels'].copy()
    if choice == 'structure':
        index = random.choice(range(len(structure)))
        new = random.choice(range(1, 4 + 1))
        while new == structure[index]:
            new = random.choice(range(1, 4 + 1))
        structure[index] = new
    elif choice == 'expansion_factors':
        index = random.choice(range(len(factors)))
        new = random.choice(range(1, 7))
        while new == structure[index]:
            new = random.choice(range(1, 7))
        factors[index] = new
    elif choice == 'channels':
        index = random.choice(range(len(channels)))
        a = random.choice(range(9))
        new = 2 ** a
        while new <= channels[index]:
            new += 2
        for j in range(index, len(channels)):
            channels[j] = new
    nuovo.loc[0, 'structure'] = structure
    nuovo.loc[0, 'expansion_factors'] = factors
    nuovo.loc[0, 'channels'] = channels
    net = MobileNetV2(structure, factors, channels)
    net = net.to(device)
    vector = find_measures(net, dataloader, ('random', 2, 2), device,
                           loss_fn = loss_function,
                           measure_names = measure_names)
    nuovo.loc[0, 'score_jacob'] = vector['jacob_cov']
    nuovo.loc[0, 'score_synflow'] = vector['synflow']
    nuovo.loc[0, 'score_snip'] = vector['snip']

    # standardization:

    nuovo.loc[0, 'score_jacob'] = (nuovo.loc[0, 'score_jacob'] - jacob_mean) / jacob_sd
    nuovo.loc[0, 'score_synflow'] = (nuovo.loc[0, 'score_synflow'] - synflow_mean) / synflow_sd
    nuovo.loc[0, 'score_snip'] = (nuovo.loc[0, 'score_snip'] - snip_mean) / snip_sd
    nuovo.loc[0, 'score_AVG'] = (nuovo.loc[0, 'score_jacob']   +
                                 nuovo.loc[0, 'score_synflow'] +
                                 nuovo.loc[0, 'score_snip']) / 3
    feature = feature.to(device)
    FLOPs, params = profile(net, inputs=(feature,), verbose=0)
    nuovo.loc[0, 'param'] = params/1000000
    nuovo.loc[0, 'flops'] = FLOPs/1000000
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return nuovo

# MUTATION CYCLE

cycles = 500
sample_dim = 20
dataloader = DataLoader(train, batch_size = 2, shuffle = False)
feature, labels = next(iter(dataloader))
history = df

for c in range(cycles):
    sample = pd.DataFrame(columns =
                          ['structure', 'expansion_factors', 'channels',
                           'score_jacob', 'score_synflow', 'score_snip',
                           'score_AVG', 'param', 'flops'])
    for s in range(sample_dim):
        index = random.choice(range(len(df)))
        sample.loc[s] = df.loc[index]
    massimo = sample['score_AVG'].values.argmax()
    figlio_di_massimo = mutate(df.loc[massimo], dataloader,
                               feature, labels, loss_function)
    df.drop(index = 0, inplace = True)
    df = pd.concat([df1, figlio_di_massimo], ignore_index = True)
    history = pd.concat([history, figlio_di_massimo], ignore_index = True)

max_score = history['score_AVG'].max()
massimo = history['score_AVG'].values.argmax()
best_net = history.loc[massimo]

# best unconstrained net

print("Senza vincoli: \n")
print("Score più alto: ", max_score)
print("Indice della rete con score più alto: ", massimo)
print(best_net)

# how many nets respect the constraints on flops and parameters after the
# mutation?

df2 = pd.DataFrame(index = (range(totale_reti)),
                   columns =
                 ['structure', 'expansion_factors',  'channels', 'score_jacob',
                  'score_synflow', 'score_snip', 'score_AVG', 'param', 'flops'])
k = 0
for i in range(len(history)):
    if (history.loc[i, 'param'] <= 2.5 and history.loc[i, 'flops'] <= 200):
        df2.loc[k] = history.loc[i]
        k += 1
df2 = df2[0:k]

# best constrained net

max_score_vinc = df2['score_AVG'].max()
massimo_vinc = df2['score_AVG'].values.argmax()
best_net_vinc = df2.loc[massimo_vinc]

print("Con i vincoli: \n")
print("Score più alto: ", max_score_vinc)
print("Indice della rete con score più alto: ", massimo_vinc)
print(best_net_vinc)

batch_size = 64
learning_rate = 0.1
weight_decay = 0.000001
momentum = 0.9
epochs = 40
net = MobileNetV2(best_net_vinc['structure'],
                  best_net_vinc['expansion_factors'],
                  best_net_vinc['channels'])
train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)
optimizer = torch.optim.SGD(net.parameters(),
                            lr = learning_rate,
                            weight_decay = weight_decay,
                            momentum = momentum)

"""

# adjustable learning rate: if you want to use it, comment line number 2 in this
# cell and uncomment line number 58 of training cell

import torch.optim.lr_scheduler as lr_scheduler

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

"""

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

max_score_vinc = df2['score_AVG'].max()
massimo_vinc = df2['score_AVG'].values.argmax()
best_net_vinc = df2.loc[massimo_vinc]

print("Con i vincoli: \n")
print("Score più alto: ", max_score_vinc)
print("Indice della rete con score più alto: ", massimo_vinc)
print(best_net_vinc)
net = MobileNetV2(best_net_vinc['structure'],
                  best_net_vinc['expansion_factors'],
                  best_net_vinc['channels'])

optimizer = torch.optim.SGD(net.parameters(),
                            lr = learning_rate,
                            weight_decay = weight_decay,
                            momentum = momentum)

# load last check-point (only if needed)

checkpoint = torch.load('/content/gdrive/MyDrive/NNcheckpoint.pth')
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epochs = checkpoint['epoch']

# LET'S TRAIN AND TEST THE BEST MODEL WE FOUND.

net = net.to(device)
net.half()

for e in range(epochs):

      # train loop

      # we put 0. to specify we want to work with floats from the get-go:

      samples = 0.
      cumulative_loss = 0.
      cumulative_accuracy = 0.

      # strictly needed if network contains layers which has different behaviours
      # between train and test
      # (it is net.eval() for the test data).

      net.train()
      for inputs, targets in train_loader:

          # load data into gpu:

          inputs, targets = inputs.to(device), targets.to(device)

          # remember: we are using floats with 16 digits.

          inputs = inputs.half()

          # forward pass:

          outputs = net(inputs)

          # apply the loss:

          loss = loss_function(outputs, targets)

          # backward pass:

          loss.backward()

          # update parameters:

          optimizer.step()

          # reset the optimizer:

          optimizer.zero_grad()

          # compute loss and accuracy:

          samples += inputs.shape[0]
          cumulative_loss += loss.item()
          _, predicted = outputs.max(1)
          cumulative_accuracy += predicted.eq(targets).sum().item()

      #scheduler.step()
      train_loss.append(cumulative_loss / samples)
      train_accuracy.append(cumulative_accuracy / samples * 100)

      # test loop:

      samples = 0.
      cumulative_loss = 0.
      cumulative_accuracy = 0.
      net.eval()
      with torch.no_grad():
          for inputs, targets in test_loader:
              inputs, targets = inputs.to(device), targets.to(device)
              inputs = inputs.half()
              outputs = net(inputs)
              loss = loss_function(outputs, targets)
              _, predicted = outputs.max(1)
              samples += inputs.shape[0]
              cumulative_loss += loss.item()
              cumulative_accuracy += predicted.eq(targets).sum().item()
      test_loss.append(cumulative_loss / samples)
      test_accuracy.append(cumulative_accuracy / samples * 100)

      print("Epoch: {:d}".format(e+1))
      print("\t Training loss {:.5f}, Training accuracy {:.2f}".format(train_loss[e],
      train_accuracy[e]))
      print("\t Test loss {:.5f}, Test accuracy {:.2f}".format(test_loss[e],
      test_accuracy[e]))
      print('-----------------------------------------------------')

torch.save({
    'epoch': epochs,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'NNcheckpoint.pth')

del net
gc.collect()
torch.cuda.empty_cache()

print(train_loss)
