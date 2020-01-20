import argparse
import math
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from PIL import Image

from models.enet import ENet
import transforms as ext_transforms
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_false', help="If True, showcase how this model works; else, return latency")
parser.add_argument('--iter_batch', action='store_true', help="If True, iterate from batch size of 1 to maximum batch size; else, only run for current batch size")
parser.add_argument('--plot', action='store_true', help="If True, plot the latency result")
parser.add_argument('-i', '--iter', type=int, default=100, help="Number of iterations to run for latency")
parser.add_argument('-b', '--batch_size', type=int, default=1, help="Batch size for inference")
args = parser.parse_args()

# Set for GPU
device = torch.device('cuda')

# Load the sample data
data_dir = "../data/cityscapes"
image_path = "berlin_000000_000019_leftImg8bit.png"
image_path = os.path.join(data_dir, image_path)
sample_image = Image.open(image_path)
print("Original sample image dimension:", sample_image.size)



# Preprocess the image per model requirement and load onto the GPU
height, width = 512, 1024
image_transform = transforms.Compose(
    [transforms.Resize((height, width)),
     transforms.ToTensor()])
sample_image = image_transform(sample_image).to(device)
print("Preprocessed sample image dimension:", sample_image.shape)


# Load the required parameters for inference
color_encoding = OrderedDict([
        ('unlabeled', (0, 0, 0)),
        ('road', (128, 64, 128)),
        ('sidewalk', (244, 35, 232)),
        ('building', (70, 70, 70)),
        ('wall', (102, 102, 156)),
        ('fence', (190, 153, 153)),
        ('pole', (153, 153, 153)),
        ('traffic_light', (250, 170, 30)),
        ('traffic_sign', (220, 220, 0)),
        ('vegetation', (107, 142, 35)),
        ('terrain', (152, 251, 152)),
        ('sky', (70, 130, 180)),
        ('person', (220, 20, 60)),
        ('rider', (255, 0, 0)),
        ('car', (0, 0, 142)),
        ('truck', (0, 0, 70)),
        ('bus', (0, 60, 100)),
        ('train', (0, 80, 100)),
        ('motorcycle', (0, 0, 230)),
        ('bicycle', (119, 11, 32))
])
num_classes = len(color_encoding)
model = ENet(num_classes).to(device)

# Load the pre-trained weights
model_path = "./save/ENet_Cityscapes/ENet"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
print('Model loaded successfully!')

# Run the inference
# If args.test, then showcase how this model works
if not args.test:
    model.eval()
    sample_image = torch.unsqueeze(sample_image, 0)
    with torch.no_grad():
        output = model(sample_image)
    print("Model output dimension:", output.shape)

    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(output.data, 1)

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(color_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(sample_image.data.cpu(), color_predictions)
# Run several iterations for each batch size to determine the 
else:
    model.eval()
    with torch.no_grad():
        if args.iter_batch:
            batch_size = [int(2**i) for i in range(int(math.log2(args.batch_size)+1))]
        else:
            batch_size = [args.batch_size]
        means = []
        stds = []
        percentile_90 = []
        percentile_99 = []
        fps = []
        for bs in batch_size:
            print("Batch size: {}".format(bs))
            batched_image = torch.stack([sample_image]*bs, 0)
            latencies = np.zeros(args.iter)
            
            # Warm up round
            for _ in range(5):
                # start = time.time()
                output = model(batched_image)
                # end = time.time()
                # print("Cold start latency: {:.3f} ms".format((end-start)*1000))
            for i in range(args.iter):
                start = time.time()
                output = model(batched_image)
                end = time.time()
                latencies[i] = end - start

            latencies.sort()
            mean_latency = np.mean(latencies) * 1000
            std_latency = np.std(latencies) * 1000
            p90 = latencies[int(args.iter * 0.9 - 1)] * 1000
            p99 = latencies[int(args.iter * 0.99 - 1)] * 1000
            # print("Latency Total: mean: {:.3f} ms, std: {:.3f} ms".format(mean_latency, std_latency))
            print("Latency: mean: {:.3f}ms ({:.2f} FPS), std: {:.3f}ms, P90: {:.3f}ms, P99: {:.3f}ms".format(
                mean_latency/bs, 1000/mean_latency*bs, std_latency/bs, p90/bs, p99/bs))
            means.append(mean_latency/bs)
            stds.append(std_latency/bs)
            fps.append(1000/mean_latency*bs)
            percentile_90.append(p90/bs)
            percentile_99.append(p99/bs)

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("PyTorch-ENet Latency Test on Cityscapes Dataset", fontsize='xx-large', fontweight='bold')
    axs = fig.subplots(2, 1)
    axs[0].errorbar(batch_size, means, stds, c='b')
    axs[0].set_xlabel('Batch Size'); 
    axs[0].set_ylabel('Latency (ms)', c='b')
    axs[0].set_ylim(0, 30)
    axs[0].set_xscale('log', basex=2); axs[0].xaxis.set_major_formatter(ScalarFormatter()); axs[0].set_xticks(batch_size)
    # axs[0].set_title("Latency vs Batch Size")
    axs[0].grid(True)
    axs[0].yaxis.set_major_locator(MultipleLocator(5))
    axs[0].tick_params(axis='y', labelcolor='b')
    for x, y in zip(batch_size, means):
        axs[0].annotate('{:.1f}'.format(y), xy=(x, y))

    ax_fps = axs[0].twinx()
    ax_fps.plot(batch_size, fps, c='r', marker='o')
    # ax_fps.set_xlabel('Batch Size')
    ax_fps.set_ylabel('FPS', c='r')
    ax_fps.set_ylim(0, 150)
    ax_fps.yaxis.set_major_locator(MultipleLocator(30))
    ax_fps.tick_params(axis='y', labelcolor='r')
    # ax_fps.set_xscale('log', basex=2); ax_fps.xaxis.set_major_formatter(ScalarFormatter()); ax_fps.set_xticks(batch_size)
    # ax_fps.grid(True)
    for x, y in zip(batch_size, fps):
        ax_fps.annotate('{:.1f}'.format(y), xy=(x, y))
    
    labels = [str(bs) for bs in batch_size]
    x = np.arange(len(labels))
    print(labels, x)
    width = 0.2
    rects1 = axs[1].bar(x - width, means, width, label='mean')
    rects2 = axs[1].bar(x,         percentile_90, width, label='P90')
    rects3 = axs[1].bar(x + width, percentile_99, width, label='P99')
    axs[1].set_ylabel('Latency (ms)')
    axs[1].set_xlabel('Batch Size')
    # axs[1].set_title("Latency across percentile")
    # axs[1].set_xscale('log', basex=2); axs[0].xaxis.set_major_formatter(ScalarFormatter())
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels)
    axs[1].set_ylim(0, 20)
    axs[1].legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            axs[1].annotate('{:.1f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    # fig.tight_layout()
    
    if args.plot:
        plt.show()
    else:
        plt.savefig('enet')
