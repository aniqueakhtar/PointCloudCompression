import os, time
import numpy as np
import torch
import MinkowskiEngine as ME
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils.data_utils import array2vector, istopk, sort_spare_tensor, load_sparse_tensor, scale_sparse_tensor
from utils.data_utils import write_ply_ascii_geo, read_ply_ascii_geo, load_sparse_tensor_downsample, isin

from utils.gpcc import gpcc_encode, gpcc_decode
from utils.pc_error import pc_error
from model.Network import Network


class CoordinateCoder():
    """encode/decode coordinates using gpcc
    """
    def __init__(self, filename):
        self.filename = filename
        self.ply_filename = filename + '.ply'

    def encode(self, coords, postfix=''):
        coords = coords.numpy().astype('int')
        write_ply_ascii_geo(filedir=self.ply_filename, coords=coords)
        gpcc_encode(self.ply_filename, self.filename+postfix+'_C.bin')
        os.system('rm '+self.ply_filename)
        
        return 

    def decode(self, postfix=''):
        gpcc_decode(self.filename+postfix+'_C.bin', self.ply_filename)
        coords = read_ply_ascii_geo(self.ply_filename)
        os.system('rm '+self.ply_filename)
        
        return coords


class FeatureCoder():
    """encode/decode feature using learned entropy model
    """
    def __init__(self, filename, entropy_model):
        self.filename = filename
        self.entropy_model = entropy_model.cpu()

    def encode(self, feats, postfix=''):
        strings, min_v, max_v = self.entropy_model.compress(feats.cpu())
        shape = feats.shape
        with open(self.filename+postfix+'_F.bin', 'wb') as fout:
            fout.write(strings)
        with open(self.filename+postfix+'_H.bin', 'wb') as fout:
            fout.write(np.array(shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(min_v), dtype=np.int8).tobytes())
            fout.write(np.array(min_v, dtype=np.float32).tobytes())
            fout.write(np.array(max_v, dtype=np.float32).tobytes())
            
        return 

    def decode(self, postfix=''):
        with open(self.filename+postfix+'_F.bin', 'rb') as fin:
            strings = fin.read()
        with open(self.filename+postfix+'_H.bin', 'rb') as fin:
            shape = np.frombuffer(fin.read(4*2), dtype=np.int32)
            len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            min_v = np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0]
            max_v = np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0]
            
        feats = self.entropy_model.decompress(strings, min_v, max_v, shape, channels=shape[-1])
        
        return feats


class Coder():
    def __init__(self, model, filename):
        self.model = model 
        self.filename = filename
        self.coordinate_coder = CoordinateCoder(filename)
        self.feature_coder = FeatureCoder(self.filename, model.entropy_bottleneck)

    @torch.no_grad()
    def encode(self, current_frame, previous_frame, postfix=''):
        # Encoder
        y_list_1 = self.model.encoder(previous_frame)
        y_list_T = self.model.encoder(current_frame)
        # Predictor network
        pred_F = self.model.predictor(y_list_T[0].C, y_list_1, device)
        # Residual
        y_R = ME.SparseTensor(y_list_T[0].F-pred_F.F, coordinates=y_list_T[0].C, tensor_stride=y_list_T[0].tensor_stride, device=device)
        
        y = sort_spare_tensor(y_R)
        num_points = [len(ground_truth) for ground_truth in y_list_T[2:4] + [current_frame]]
        with open(self.filename+postfix+'_num_points.bin', 'wb') as f:
            f.write(np.array(num_points, dtype=np.int32).tobytes())
        self.feature_coder.encode(y.F, postfix=postfix)
        self.coordinate_coder.encode((y.C//y.tensor_stride[0]).detach().cpu()[:,1:], postfix=postfix)
        
        return y

    @torch.no_grad()
    def decode(self, previous_frame, postfix=''):
        # decode coords
        y_C = self.coordinate_coder.decode(postfix=postfix)
        y_C = torch.cat((torch.zeros((len(y_C),1)).int(), torch.tensor(y_C).int()), dim=-1)
        indices_sort = np.argsort(array2vector(y_C, y_C.max()+1))
        y_C = y_C[indices_sort]*8
        y_C = y_C.to(device)
        # decode feat
        y_F = self.feature_coder.decode(postfix=postfix).to(device)
        # Network outputs
        y_list_1 = self.model.encoder(previous_frame)
        pred_F = self.model.predictor(y_C, y_list_1, device)
        pred_F =  sort_spare_tensor(pred_F)
        # tensor
        y = ME.SparseTensor(features=y_F+pred_F.F, coordinates=y_C, tensor_stride=8, device=device)
        # decode label
        with open(self.filename+postfix+'_num_points.bin', 'rb') as fin:
            num_points = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
            num_points = [[num] for num in num_points]
        # decode
        _, out = self.model.decoder(y, nums_list=num_points, ground_truth_list=[None]*3, training=False)
        
        return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckptdir", default='ckpts/r7.pth')
    parser.add_argument("--current_file", default='/home/anique/Anique/Coding/Dataset_2/cat2_Dynamic_Objects/8isequences/soldier/Ply/soldier_vox10_0550.ply')
    parser.add_argument("--previous_file", default='/home/anique/Anique/Coding/Dataset_2/cat2_Dynamic_Objects/8isequences/soldier/Ply/soldier_vox10_0551.ply')
    parser.add_argument("--res", type=int, default=1024, help='resolution')
    args = parser.parse_args()
    
    print("Previous File: ", args.previous_file)
    print("Current File: ", args.current_file)
    # load data
    start_time = time.time()
    x_1 = load_sparse_tensor_downsample(args.previous_file, device, ds=1)
    x_T = load_sparse_tensor_downsample(args.current_file, device, ds=1)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')

    outdir = './output'
    if not os.path.exists(outdir): os.makedirs(outdir)
    filename = os.path.split(args.current_file)[-1].split('.')[0]
    filename = os.path.join(outdir, filename)
    print(filename)
    
    # model
    print('='*10, 'Test', '='*10)
    model = Network().to(device)
    assert os.path.exists(args.ckptdir)
    ckpt = torch.load(args.ckptdir)
    model.entropy_bottleneck.load_state_dict(ckpt['entropy_bottleneck'])
    model.predictor.load_state_dict(ckpt['predictor'])
    model.encoder.load_state_dict(ckpt['encoder'])
    model.decoder.load_state_dict(ckpt['decoder'])
    print('load checkpoint from \t', args.ckptdir)
    
    
    # coder
    coder = Coder(model=model, filename=filename)
    
    # encode
    start_time = time.time()
    _ = coder.encode(x_T, x_1)
    print('Enc Time:\t', round(time.time() - start_time, 3), 's')

    # decode
    start_time = time.time()
    x_dec = coder.decode()
    print('Dec Time:\t', round(time.time() - start_time, 3), 's')

    # bitrate
    bits = np.array([os.path.getsize(filename + postfix)*8 \
                            for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])
    bpps = (bits/len(x_T)).round(3)
    print('bits:\t', bits, '\nbpps:\t', bpps)
    print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))
    
    # distortion
    start_time = time.time()
    write_ply_ascii_geo(filename+'_dec.ply', x_dec.C.detach().cpu().numpy()[:,1:])
    print('Write PC Time:\t', round(time.time() - start_time, 3), 's')
    
    start_time = time.time()
    pc_error_metrics = pc_error(args.current_file, filename+'_dec.ply', res=args.res, show=False)
    print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
    # print('pc_error_metrics:', pc_error_metrics)
    print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
    
    
    ### MANUAL TEST # 1
    print("====== MANUAL TEST # 1 ======")
    y_list_1 = model.encoder(x_1)
    y_list_T = model.encoder(x_T)
    
    # Predictor Network
    pred_F = model.predictor(y_list_T[0].C, y_list_1, device)
    # Residual
    y = ME.SparseTensor(y_list_T[0].F-pred_F.F, coordinates=y_list_T[0].C, tensor_stride=y_list_T[0].tensor_stride, device=device)
    
    # Quantizer & Entropy Model
    y_q, likelihood = model.get_likelihood(y, quantize_mode="symbols")
    
    
    # Decoder
    y_rec = ME.SparseTensor(y_q.F+pred_F.F, coordinates=y_q.C, tensor_stride=y_q.tensor_stride, device=device)
    ground_truth_list = y_list_T[2:4] + [x_T]
    nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list]
    _, out = model.decoder(y_rec, nums_list, ground_truth_list, training=False)
    
    bits = -torch.sum(torch.log2(likelihood))
    bpp = bits/float(x_T.shape[0])
    
    print('bits:\t', bits.item(), '\nbpps:\t', bpp.item())
    
    # distortion
    start_time = time.time()
    write_ply_ascii_geo(filename+'_dec.ply', out.C.detach().cpu().numpy()[:,1:])
    print('Write PC Time:\t', round(time.time() - start_time, 3), 's')
    
    start_time = time.time()
    pc_error_metrics = pc_error(args.current_file, filename+'_dec.ply', res=args.res, show=False)
    print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
    # print('pc_error_metrics:', pc_error_metrics)
    print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
    
    
    ### MANUAL TEST # 2
    print("====== MANUAL TEST # 2 ======")
    y_list_1 = model.encoder(x_1)
    y_list_T = model.encoder(x_T)
    
    # Predictor Network
    pred_F = model.predictor(y_list_T[0].C, y_list_1, device)
    # Residual
    y_R = ME.SparseTensor(y_list_T[0].F-pred_F.F, coordinates=y_list_T[0].C, tensor_stride=y_list_T[0].tensor_stride, device=device)
    
    # Quantizer & Entropy Model
    feats = y_R.F
    coords = y_R.C
    strings, min_v, max_v = model.entropy_bottleneck.compress(feats.cpu())
    shape = feats.shape
    
    feats_2 = model.entropy_bottleneck.decompress(strings, int(min_v), int(max_v), shape, channels=shape[-1])
    
    y = ME.SparseTensor(features=feats_2, coordinates=coords, tensor_stride=8, device=device)
    
    # Decoder
    y_rec = ME.SparseTensor(y.F+pred_F.F, coordinates=y.C, tensor_stride=y.tensor_stride, device=device)
    ground_truth_list = y_list_T[2:4] + [x_T]
    nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list]
    _, out = model.decoder(y_rec, nums_list, ground_truth_list, training=False)
    
    # distortion
    start_time = time.time()
    write_ply_ascii_geo(filename+'_dec.ply', out.C.detach().cpu().numpy()[:,1:])
    print('Write PC Time:\t', round(time.time() - start_time, 3), 's')
    
    start_time = time.time()
    pc_error_metrics = pc_error(args.current_file, filename+'_dec.ply', res=args.res, show=False)
    print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
    # print('pc_error_metrics:', pc_error_metrics)
    print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
    
        
    ### MANUAL TEST # 3
    print("====== MANUAL TEST # 3 ======")
    y_list_1 = model.encoder(x_1)
    y_list_T = model.encoder(x_T)
    
    # Predictor Network
    pred_F = model.predictor(y_list_T[0].C, y_list_1, device)
    # Residual
    y_R1 = ME.SparseTensor(y_list_T[0].F-pred_F.F, coordinates=y_list_T[0].C, tensor_stride=y_list_T[0].tensor_stride, device=device)
    
    y_R = sort_spare_tensor(y_R1)
    
    # Quantizer & Entropy Model
    feats = y_R.F
    coords = y_R.C
    strings, min_v, max_v = model.entropy_bottleneck.compress(feats.cpu())
    shape = feats.shape
    
    feats_2 = model.entropy_bottleneck.decompress(strings, int(min_v), int(max_v), shape, channels=shape[-1])
    
    y = ME.SparseTensor(features=feats_2, coordinates=coords, tensor_stride=8, device=device)
      
    # Decoder
    pred_F1 =  sort_spare_tensor(pred_F)
    y_rec = ME.SparseTensor(y.F+pred_F1.F, coordinates=y.C, tensor_stride=y.tensor_stride, device=device)
    ground_truth_list = y_list_T[2:4] + [x_T]
    nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list]
    _, out = model.decoder(y_rec, nums_list, ground_truth_list, training=False)
    
    # distortion
    start_time = time.time()
    write_ply_ascii_geo(filename+'_dec.ply', out.C.detach().cpu().numpy()[:,1:])
    print('Write PC Time:\t', round(time.time() - start_time, 3), 's')
    
    start_time = time.time()
    pc_error_metrics = pc_error(args.current_file, filename+'_dec.ply', res=args.res, show=False)
    print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
    # print('pc_error_metrics:', pc_error_metrics)
    print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
    