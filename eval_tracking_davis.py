# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Some parts are taken from https://github.com/Liusifei/UVC
"""
import os
import copy
import glob
import queue
from urllib.request import urlopen
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import utils
import vision_transformer as vits


@torch.no_grad()
def eval_video_tracking_davis(args, model, frame_list, video_dir, first_seg, seg_ori, color_palette):
    """
    Evaluate tracking on a video given first frame & segmentation
    """
    video_folder = os.path.join(args.output_dir, video_dir.split('/')[-1])
    os.makedirs(video_folder, exist_ok=True)

    # The queue stores the n preceeding frames
    que = queue.Queue(args.n_last_frames)

    # first frame
    frame1 = read_frame(frame_list[0], args.input_size, args.patch_size)

    # saving first segmentation
    out_path = os.path.join(video_folder, "00000.png")
    imwrite_indexed(out_path, seg_ori, color_palette)
    mask_neighborhood = None
    for cnt in tqdm(range(1, len(frame_list))):
        frame_tar = read_frame(frame_list[cnt], args.input_size, args.patch_size)

        # we use the first segmentation and the n previous ones
        used_frames = [frame1] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

        frame_tar_avg, mask_neighborhood = label_propagation(args, model, frame_tar, used_frames, used_segs, mask_neighborhood)

        # pop out oldest frame if neccessary
        if que.qsize() == args.n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([frame_tar, seg])

        # upsampling & argmax
        frame_tar_avg = torch.nn.functional.interpolate(frame_tar_avg, scale_factor=args.patch_size, mode='bilinear')[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        # saving to disk
        frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
        frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg", ".png")
        imwrite_indexed(os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette)


def restrict_neighborhood(h, w):
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * args.size_mask_neighborhood + 1):
                for q in range(2 * args.size_mask_neighborhood + 1):
                    if i - args.size_mask_neighborhood + p < 0 or i - args.size_mask_neighborhood + p >= h:
                        continue
                    if j - args.size_mask_neighborhood + q < 0 or j - args.size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - args.size_mask_neighborhood + p, j - args.size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)


def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt,:,:] = mask_cnt
    return mask


def label_propagation(args, model, frame_tar, list_frames, list_segs, mask_neighborhood=None):
    """
    propagate segs of frames in list_frames to frame_tar
    """
    # get feature maps
    input_frames = torch.stack([frame_tar] + list_frames)
    curr_i = 0
    _, w0, h0 = frame_tar.shape
    while curr_i < input_frames.shape[0]:
        inp = input_frames[curr_i: curr_i + args.bs].cuda()
        out = model.get_intermediate_layers(inp, n=1)[0]
        out = out[:, 1:, :]  # we discard the [CLS] token
        w, h = int(w0 / model.patch_embed.patch_size), int(h0 / model.patch_embed.patch_size)
        dim = out.shape[-1]
        out = out.reshape(out.shape[0], w, h, dim)
        out = out.permute(0, 3, 1, 2)

        if curr_i == 0:
            feature_maps = out
        else:
            feature_maps = torch.cat((feature_maps, out))
        curr_i += args.bs
        del out  # free memory

    feat_tar = feature_maps[0].reshape(feature_maps.shape[1], -1).T # h*w x dim

    ncontext = feature_maps.shape[0] - 1
    feat_sources = feature_maps[1:].reshape(ncontext, dim, -1) # nmb_context x dim x h*w

    feat_tar = nn.functional.normalize(feat_tar, dim=1, p=2)
    feat_sources = nn.functional.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1) # nmb_context x h*w (tar: query) x h*w (source: keys)

    if args.size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

    aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)
    tk_val, _ = torch.topk(aff, dim=0, k=args.topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, w, h = segs.shape
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, w, h)
    return seg_tar, mask_neighborhood
 

def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
      raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if(n_dims is None):
        n_dims = int(y_tensor.max()+ 1)
    _,h,w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h,w,n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)


def read_frame_list(video_dir):
    frame_list = [img for img in glob.glob(os.path.join(video_dir,"*.jpg"))]
    frame_list = sorted(frame_list)
    return frame_list


def read_seg(seg_path, size, downsampling_ratio):
    seg = Image.open(seg_path)
    w, h = seg.size
    if len(size) == 1:
        if h > w:
            tw, th = size[0], (size[0] * h) / w
        else:
            th, tw = size[0], (size[0] * w) / h
    else:
        tw, th = size
    seg = torch.tensor(np.asarray(seg).copy())
    seg = nn.functional.interpolate(seg[None, None], scale_factor=(tw/w, th/h), mode="nearest")[0]
    tw, th = int(tw - tw % args.patch_size), int(th - th % args.patch_size)
    seg = seg[:, :th, :tw]
    small_seg = nn.functional.interpolate(seg[None], scale_factor=1/downsampling_ratio, mode="nearest")[0]
    return to_one_hot(small_seg), seg.squeeze().numpy()


def read_frame(frame_path, size, downsampling_ratio):
    with open(frame_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % downsampling_ratio, img.shape[2] - img.shape[2] % downsampling_ratio
    img = img[:, :w, :h]
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with video object segmentation on DAVIS 2017')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='deit_small', type=str,
        choices=['deit_tiny', 'deit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--output_dir', default=".", help='Path where to save segmentations')
    parser.add_argument("--input_size", type=int, nargs="+", default=[480], help="size of shorter image side")
    parser.add_argument('--data_path', default='/path/to/davis/', type=str)
    parser.add_argument("--n_last_frames", type=int, default=7, help="number of preceeding frames")
    parser.add_argument("--size_mask_neighborhood", default=12, type=int,
        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--topk", type=int, default=5, help="accumulate label from top k neighbors")
    parser.add_argument("--bs", type=int, default=6, help="Batch size, try to reduce if OOM")
    args = parser.parse_args()

    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # building network
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    color_palette = []
    for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
        color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
    color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1,3)

    video_list = open(os.path.join(args.data_path, "ImageSets/2017/val.txt")).readlines()
    for i, video_name in enumerate(video_list):
        video_name = video_name.strip()
        print(f'[{i}/{len(video_list)}] Begin to segmentate video {video_name}.')
        video_dir = os.path.join(args.data_path, "JPEGImages/480p/", video_name)
        frame_list = read_frame_list(video_dir)
        seg_path = frame_list[0].replace("JPEGImages", "Annotations").replace("jpg", "png")
        first_seg, seg_ori = read_seg(seg_path, args.input_size, args.patch_size)
        eval_video_tracking_davis(args, model, frame_list, video_dir, first_seg, seg_ori, color_palette)
