import torch
import MinkowskiEngine as ME

from model.autoencoder import Encoder, Decoder
from model.entropy_model import EntropyBottleneck
from model.predictor import Predictor


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(channels=[1,16,32,64,32,8])
        self.decoder = Decoder(channels=[8,64,32,16])
        self.entropy_bottleneck = EntropyBottleneck(8)
        self.predictor = Predictor(channels=[1,16,32,64,32,8])
    
    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
            quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F, 
            coordinate_map_key=data.coordinate_map_key, 
            coordinate_manager=data.coordinate_manager, 
            device=data.device)
        
        return data_Q, likelihood
    
    def forward(self, coords_1, coords_T, device, training=True):
        feats_1 = torch.ones(coords_1.shape[0],1)*1
        feats_T = torch.ones(coords_T.shape[0],1)*1
        x_1 = ME.SparseTensor(feats_1, coordinates=coords_1, device=device)
        x_T = ME.SparseTensor(feats_T, coordinates=coords_T, device=device)
        
        y_list_1 = self.encoder(x_1)
        y_list_T = self.encoder(x_T)
        
        # Predictor Network
        pred_F = self.predictor(y_list_T[0].C, y_list_1, device)
        # Residual
        y = ME.SparseTensor(y_list_T[0].F-pred_F.F, coordinates=y_list_T[0].C, tensor_stride=y_list_T[0].tensor_stride, device=device)
        
        # Quantizer & Entropy Model
        y_q, likelihood = self.get_likelihood(y, quantize_mode="noise" if training else "symbols")
        
        
        # Measuring Variances.
        with torch.no_grad():
            var_F2 = torch.var(y_list_T[0].F)
            var_res = torch.var(y.F)
        
        # Decoder
        y_rec = ME.SparseTensor(y_q.F+pred_F.F, coordinates=y_q.C, tensor_stride=y_q.tensor_stride, device=device)
        ground_truth_list = y_list_T[2:4] + [x_T]
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] for ground_truth in ground_truth_list]
        out_cls_list, out = self.decoder(y_rec, nums_list, ground_truth_list, training)
        
        return {'pred_F':pred_F,
                'GT_F':y_list_T[0],
                'out':out,
                'out_cls_list':out_cls_list,
                'likelihood':likelihood, 
                'ground_truth_list':ground_truth_list,
                'var_F2':var_F2,
                'var_res':var_res}
    

if __name__ == '__main__':
    model = Network()
    print(model)

