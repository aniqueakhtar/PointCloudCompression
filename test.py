import torch
import numpy as np
import os
from model.Network import Network
from coder import Coder
import time
from utils.data_utils import load_sparse_tensor_downsample, load_sparse_tensor
from utils.data_utils import write_ply_ascii_geo
from utils.pc_error import pc_error
import glob
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def test(files, ckptdir_list, outdir, resultdir, res=1024, ds=1):
    
    for i in range(1,len(files)):
        current_file = files[i]
        # load data
        start_time = time.time()
        # current_frame = load_sparse_tensor_downsample(current_file, device, ds=ds)
        current_frame = load_sparse_tensor(current_file, device)
        
        print('\n\nCurrent File: ', current_file, '\n')
        # Where to keep temporary bin files
        filename = os.path.join('./tmp', os.path.split(current_file)[-1].split('.')[0])
        
        # load model
        model = Network().to(device)
        
        for idx, ckptdir in enumerate(ckptdir_list):
            # postfix: rate index
            postfix_idx = '_'+os.path.split(ckptdir)[-1].split('.')[0]
            print('='*10, postfix_idx, '='*10)
            dest_file = os.path.join(outdir, os.path.split(current_file)[-1].split('.')[0]) + postfix_idx + '.ply'
            
            # Getting the previous file
            if i==1:
                previous_file = files[0]
                previous_frame = load_sparse_tensor_downsample(previous_file, device, ds=ds)
            else:
                previous_file = os.path.join(outdir, os.path.split(files[i-1])[-1].split('.')[0]) + postfix_idx + '.ply'
                previous_frame = load_sparse_tensor_downsample(previous_file, device, ds=1)
            
            print('Previous File: ', previous_file) 
            print('Destination File: ', dest_file)
            print('Previous Frame Shape: ', previous_frame.shape[0])
            print('Current Frame Shape: ', current_frame.shape[0])
            
            # load checkpoints
            assert os.path.exists(ckptdir)
            ckpt = torch.load(ckptdir)
            model.entropy_bottleneck.load_state_dict(ckpt['entropy_bottleneck'])
            model.predictor.load_state_dict(ckpt['predictor'])
            model.encoder.load_state_dict(ckpt['encoder'])
            model.decoder.load_state_dict(ckpt['decoder'])
            print('load checkpoint from \t', ckptdir)
            coder = Coder(model=model, filename=filename)
            
            # encode
            start_time = time.time()
            _ = coder.encode(current_frame, previous_frame, postfix=postfix_idx)
            print('Enc Time:\t', round(time.time() - start_time, 3), 's')
            time_enc = round(time.time() - start_time, 3)
            
            # decode
            start_time = time.time()
            x_dec = coder.decode(previous_frame, postfix=postfix_idx)
            print('Dec Time:\t', round(time.time() - start_time, 3), 's')
            time_dec = round(time.time() - start_time, 3)
            
            
            # bitrate
            bits = np.array([os.path.getsize(filename + postfix_idx + postfix)*8 \
                                    for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])
            bpps = (bits/len(current_frame)).round(3)
            print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))
            
            # distortion
            start_time = time.time()
            write_ply_ascii_geo(dest_file, x_dec.C.detach().cpu().numpy()[:,1:])
            GTfile = 'tmp/'+'GT.ply'
            write_ply_ascii_geo(GTfile, current_frame.C.detach().cpu().numpy()[:,1:])
            print('Write PC Time:\t', round(time.time() - start_time, 3), 's')
            
            start_time = time.time()
            pc_error_metrics = pc_error(GTfile, dest_file, res=res, normal=False, show=False)
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
            
            os.system('rm ./tmp/*.bin')
        
        csv_name = os.path.join(resultdir, os.path.split(current_file)[-1].split('.')[0]+'.csv')
        all_results.to_csv(csv_name, index=False)
        print('Wrile results to: \t', csv_name)
        
        fig, ax = plt.subplots(figsize=(7, 4))
        plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2point)"][:]), 
                label="D1", marker='x', color='red')
        # plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2plane)"][:]), 
        #         label="D2", marker='x', color='blue')
        filename = os.path.split(current_file)[-1][:-4]
        plt.title(filename)
        plt.xlabel('bpp')
        plt.ylabel('PSNR')
        plt.grid(ls='-.')
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(figdir, filename+'.jpg'))
    
    return 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--outdir", default='../Results/P-Frame/outply')
    parser.add_argument("--resultdir", default='../Results/P-Frame/results')
    parser.add_argument("--figdir", default='../Results/P-Frame/figs')
    parser.add_argument("--test_folder", default='../Results/Test_Files')
    args = parser.parse_args()
    
    if not os.path.exists('./tmp'): os.makedirs('./tmp')
    ckptdir_list = ['../pretrained/P-Frame/r1.pth', '../pretrained/P-Frame/r2.pth', '../pretrained/P-Frame/r3.pth',
                    '../pretrained/P-Frame/r4.pth', '../pretrained/P-Frame/r5.pth', '../pretrained/P-Frame/r6.pth', '../pretrained/P-Frame/r7.pth']
    
    test_fold = ['basketball', 'dancer', 'exercise', 'model', 'redandblack', 'soldier']
    
    res = 1024
    ds = 1
    
    for seq_name in test_fold:
        files = sorted(glob.glob(args.test_folder+ '/' + seq_name +'/**.ply'))
        
        outdir = os.path.join(args.outdir, seq_name)
        resultdir = os.path.join(args.resultdir, seq_name)
        figdir = os.path.join(args.figdir, seq_name)
        
        if not os.path.exists(outdir): os.makedirs(outdir)
        if not os.path.exists(resultdir): os.makedirs(resultdir)
        if not os.path.exists(figdir): os.makedirs(figdir)
        os.system('rm ./tmp/*.bin')
        
        test(files, ckptdir_list, outdir, resultdir, res=res, ds=ds)
        
