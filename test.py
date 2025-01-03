import os
import time
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from coder import Coder
from model.Network import Network
from utils.pc_error import pc_error
from utils.data_utils import load_sparse_tensor, write_ply_ascii_geo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def test(currfile, prevfile, ckptdir, outply, outbin, resultdir, res=1024):

    # load data
    start_time = time.time()
    current_frame = load_sparse_tensor(currfile, device)

    print('\n\nCurrent File: ', currfile, '\n')

    # load model
    model = Network().to(device)

    for idx, ckp in enumerate(ckptdir):
        print('='*10, idx+1, '='*10)
        # output file_nmae
        dest_file = os.path.join(outply, ckp, os.path.split(currfile)[-1].split('.')[0]) + '.ply'
        filename_bin = os.path.join(outbin, ckp, os.path.split(currfile)[-1].split('.')[0])

        # Getting the previous file
        if not prevfile:
            previous_file = sorted(glob.glob(os.path.join(outply, ckp)+'/**.ply'))[-1]
        else:
            previous_file = prevfile
        previous_frame = load_sparse_tensor(previous_file, device)

        print('Previous File: ', previous_file)
        print('Destination File: ', dest_file)
        print('Previous Frame Shape: ', previous_frame.shape[0])
        print('Current Frame Shape: ', current_frame.shape[0])

        # load checkpoints
        assert os.path.exists(ckptdir[ckp])
        ckpt = torch.load(ckptdir[ckp])
        model.entropy_bottleneck.load_state_dict(ckpt['entropy_bottleneck'])
        model.predictor.load_state_dict(ckpt['predictor'])
        model.encoder.load_state_dict(ckpt['encoder'])
        model.decoder.load_state_dict(ckpt['decoder'])
        print('load checkpoint from \t', ckptdir[ckp])
        coder = Coder(model=model, filename=filename_bin)

        # encode
        start_time = time.time()
        _ = coder.encode(current_frame, previous_frame, postfix='')
        print('Enc Time:\t', round(time.time() - start_time, 3), 's')
        time_enc = round(time.time() - start_time, 3)

        # decode
        start_time = time.time()
        x_dec = coder.decode(previous_frame, postfix='')
        print('Dec Time:\t', round(time.time() - start_time, 3), 's')
        time_dec = round(time.time() - start_time, 3)


        # bitrate
        bits = np.array([os.path.getsize(filename_bin + postfix)*8 \
                                for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])
        bpps = (bits/len(current_frame)).round(3)
        print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))

        # distortion
        start_time = time.time()
        write_ply_ascii_geo(dest_file, x_dec.C.detach().cpu().numpy()[:,1:])
        print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

        start_time = time.time()
        pc_error_metrics = pc_error(currfile, dest_file, res=res, normal=False, show=False)
        print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
        print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])

        # save results
        results = pc_error_metrics
        results["num_points(input)"] = len(current_frame)
        results["num_points(output)"] = len(x_dec)
        results["resolution"] = res
        results["bits"] = sum(bits).round(3)
        results["bits"] = sum(bits).round(3)
        results["bpp"] = sum(bpps).round(3)
        results["bpp(coords)"] = bpps[0]
        results["bpp(feats)"] = bpps[1]
        results["time(enc)"] = time_enc
        results["time(dec)"] = time_dec
        if idx == 0:
            all_results = results.copy(deep=True)
        else:
            all_results = all_results.append(results, ignore_index=True)


    csv_name = os.path.join(resultdir, os.path.split(currfile)[-1].split('.')[0]+'.csv')
    all_results.to_csv(csv_name, index=False)
    print('Wrile results to: \t', csv_name)

    return all_results


def average_csv(csvs):
    input_pds = [pd.read_csv(x) for x in csvs]

    rates = len(input_pds[1].index)
    metrics = input_pds[1].columns
    # print(rates, metrics)

    for rate in range(rates):
        # print(rate)
        results = {}
        for i, metric in enumerate(metrics):
            results[metric] = []
            for j, input_pd in enumerate(input_pds):
                results[metric].append(input_pd[metric][rate])
            # mean
            results[metric] = np.mean(results[metric])

        results = pd.DataFrame([results])
        if rate == 0:
            all_results = results.copy(deep=True)
        else:
            all_results = all_results.append(results, ignore_index=True)

    return all_results


def plot_panda_csv(df, x_axis, y_axis, dest_file):
    name = dest_file.split('/')[-1].split('.')[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.plot(np.array(df[x_axis][:]), np.array(df[y_axis][:]), label="D1", marker='x', color='red')
    plt.title(name)
    plt.xlabel('bpp')
    plt.ylabel('PSNR')
    plt.grid(ls='-.')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join(dest_file), dpi=300)

    return


if __name__ == '__main__':
    '''
    THIS CODE WOULD ASSUME THAT THE FIRST FRAME (I-FRAME) IS THE ORIGINAL (LOSSLESSLY ENCODED) FRAME AND ALL
    THE REST ARE ENCODED USING P-FRAME.
    '''

    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results", default='Results/exercise')
    parser.add_argument("--test_folder", default='/dataset/Owlii_10bit/exercise/')
    parser.add_argument("--first_frame", default='Results/I-Frame/output/outply/exercise_vox11_00000001_r1_dec.ply')
    parser.add_argument("--no_frames", type=int, default=100, help='Number of frames')
    parser.add_argument("--res", type=int, default=1024, help='resolution')
    args = parser.parse_args()

    ckptdir = {'r1': './pretrained/r1.pth','r2': './pretrained/r2.pth',
               'r3': './pretrained/r3.pth','r4': './pretrained/r4.pth',
               'r5': './pretrained/r5.pth','r6': './pretrained/r6.pth',
               'r7': './pretrained/r7.pth'}


    resultdir = os.path.join(args.results, 'results')
    figdir = os.path.join(args.results, 'figs')
    if not os.path.exists(resultdir): os.makedirs(resultdir)
    if not os.path.exists(figdir): os.makedirs(figdir)

    outply = os.path.join(args.results, 'output/outply')
    outbin = os.path.join(args.results, 'output/binary')
    for ckpt in ckptdir:
        oply = os.path.join(outply, ckpt)
        obin = os.path.join(outbin, ckpt)
        if not os.path.exists(oply): os.makedirs(oply)
        if not os.path.exists(obin): os.makedirs(obin)


    files = sorted(glob.glob(args.test_folder+'**.ply'))[:args.no_frames]
    for currfile in files[1:]:
        print(currfile)

        if currfile==files[1]:
            # prevfile = files[0]
            prevfile = args.first_frame
        else:
            prevfile = None

        all_results = test(currfile, prevfile, ckptdir, outply, outbin, resultdir, res=args.res)

        # plot RD-curve
        filename = os.path.split(currfile)[-1][:-4]
        plot_panda_csv(all_results, 'bpp', 'mseF,PSNR (p2point)', os.path.join(figdir, filename+'.jpg'))


    ## Find Average of the results.
    filename = os.path.basename(files[0]).split('_')[0]
    csv_files = sorted(glob.glob(os.path.join(resultdir, '**.csv')))
    avg_csv = average_csv(csv_files)
    dest_csv = filename + '.csv'
    dest_fig = filename + '.jpg'
    avg_csv.to_csv(os.path.join(args.results, dest_csv), index=False)
    plot_panda_csv(avg_csv, 'bpp', 'mseF,PSNR (p2point)', os.path.join(args.results, dest_fig))
