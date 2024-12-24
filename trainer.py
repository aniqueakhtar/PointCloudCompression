import open3d as o3d
import os, time, logging
from tqdm import tqdm
import numpy as np
import torch

from utils.loss import get_bce, get_bits, get_metrics
from utils.pc_error_wrapper import pc_error
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tensorboardX import SummaryWriter


class Trainer():
    def __init__(self, args, model):
        self.args = args
        self.logger = self.getlogger(args.logdir)
        self.writer = SummaryWriter(log_dir=args.logdir)
        self.step = 0
        self.model = model.to(device)
        # self.logger.info(model)
        # Loss Functions
        self.crit = torch.nn.L1Loss()
        # optimizer
        self.optimizer = self.set_optimizer()
        self.load_state_dict()
        self.record_set = {'bce':[], 'bces':[], 'bpp':[],'sum_loss':[], 'metrics':[], 'L1loss':[],
                           'var_F2':[], 'var_res':[], 'PSNR':[]}
        self.record_set_avg = {'bce':[], 'bces':[], 'bpp':[],'sum_loss':[], 'metrics':[], 'L1loss':[],
                           'var_F2':[], 'var_res':[], 'PSNR':[]}
        self.time = time.time()

    def getlogger(self, logdir):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
        handler.setFormatter(formatter)
        # console = logging.StreamHandler()
        # console.setLevel(logging.INFO)
        # console.setFormatter(formatter)
        logger.addHandler(handler)
        # logger.addHandler(console)

        return logger

    def load_state_dict(self):
        """selectively load model
        """

        if self.args.init_ckpt=='':
            self.logger.info('Random initialization.')
        else:
            self.logger.info('Load checkpoint from ' + self.args.init_ckpt)
            ckpt = torch.load(self.args.init_ckpt)
            self.model.entropy_bottleneck.load_state_dict(ckpt['entropy_bottleneck'])
            self.model.predictor.load_state_dict(ckpt['predictor'])
            self.model.encoder.load_state_dict(ckpt['encoder'])
            self.model.decoder.load_state_dict(ckpt['decoder'])

            if self.args.reset:
                self.logger.info('Reset is True, Starting Step 0, Optimizer reset')
            else:
                self.step = ckpt['step']
                self.logger.info('Loading Optimizer state from checkpoint')
                self.optimizer.load_state_dict(ckpt['optimizer'])

            self.logger.info('Starting Step = ' + str(self.step))

        return

    def save_model(self):
        self.logger.info(f'save checkpoints: {self.args.ckptdir}/iter{str(self.step)}')
        torch.save({'step': self.step,
                    'encoder': self.model.encoder.state_dict(),
                    'decoder': self.model.decoder.state_dict(),
                    'entropy_bottleneck': self.model.entropy_bottleneck.state_dict(),
                    'predictor': self.model.predictor.state_dict(),
                    'optimizer':  self.optimizer.state_dict(),
                    }, os.path.join(self.args.ckptdir, 'iter' + str(self.step) + '.pth'))

        return

    def set_optimizer(self):
        optimizer = torch.optim.Adam([{"params":self.model.entropy_bottleneck.parameters(), 'lr':self.args.lr},
                                {"params":self.model.predictor.parameters(), 'lr':self.args.lr},
                                {"params":self.model.encoder.parameters(), 'lr':self.args.lr},
                                {"params":self.model.decoder.parameters(), 'lr':self.args.lr}],
                                betas=(0.9, 0.999), weight_decay=1e-4)

        return optimizer

    def calculate_PSNR(self, coords_T, coords_P):
        GT_pcd = o3d.geometry.PointCloud()
        GT_pcd.points = o3d.utility.Vector3dVector(coords_T.cpu())
        GTfile = 'tmp/'+'GT.ply'
        o3d.io.write_point_cloud(GTfile, GT_pcd, write_ascii=True)

        rec_pcd = o3d.geometry.PointCloud()
        rec_pcd.points = o3d.utility.Vector3dVector(coords_P.cpu())
        recfile = 'tmp/'+'rec.ply'
        o3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)

        pc_error_metrics = pc_error(infile1=GTfile, infile2=recfile, res=1024)

        return pc_error_metrics['mseF,PSNR (p2point)'][0]


    @torch.no_grad()
    def record_avg_iter(self):
        for k, v in self.record_set.items():
            self.record_set_avg[k].append(v)

        return

    @torch.no_grad()
    def record_avg(self):
        main_tag = 'Test_Avg'

        for k, v in self.record_set_avg.items():
            self.record_set_avg[k]=np.mean(v, axis=0)[0]

        # Writing values to Tensorboard.
        for k, v in self.record_set_avg.items():
            if np.isnan(v).any(): continue
            if isinstance(v, np.ndarray):
                d = {}
                for i,j in enumerate(v):
                    d[k+str(i)] = j
                self.writer.add_scalars(main_tag=main_tag+'/'+k, tag_scalar_dict=d, global_step=self.step)
            else:
                self.writer.add_scalar(tag=main_tag+'/'+k, scalar_value=v, global_step=self.step)

        for k in self.record_set_avg.keys():
            self.record_set_avg[k] = []

        return

    @torch.no_grad()
    def record(self, main_tag):
        # print record
        self.logger.info('='*10+'  '+main_tag + ' Step ' + str(self.step))
        for k, v in self.record_set.items():
            self.record_set[k]=np.mean(np.array(v), axis=0)
        for k, v in self.record_set.items():
            self.logger.info(k+': '+str(np.round(v, 4).tolist()))

        # Writing values to Tensorboard.
        for k, v in self.record_set.items():
            if np.isnan(v).any(): continue
            if isinstance(v, np.ndarray):
                d = {}
                for i,j in enumerate(v):
                    d[k+str(i)] = j
                self.writer.add_scalars(main_tag=main_tag+'/'+k, tag_scalar_dict=d, global_step=self.step)
            else:
                self.writer.add_scalar(tag=main_tag+'/'+k, scalar_value=v, global_step=self.step)

        # return zero
        for k in self.record_set.keys():
            self.record_set[k] = []

        return


    def change_training(self):
        self.logger.info('='*10 +' '+ '!'*10 +' '+ '='*10)
        self.logger.info(f'Changing Loss Function')
        self.reconstruction_loss = True

        self.logger.info(f'Training Encoder and Decoder')
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        self.logger.info(f'Reseting the Optimizer')
        self.optimizer = self.set_optimizer()
        self.logger.info('='*10 +' '+ '!'*10 +' '+ '='*10)

        return

    def loss_function(self, out_set, size):
        L1loss = self.crit(out_set['GT_F'].F, out_set['pred_F'].F)
        bpp = get_bits(out_set['likelihood'])/size
        bce, bce_list = 0, []
        for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
            curr_bce = get_bce(out_cls, ground_truth)/float(out_cls.__len__())
            bce += curr_bce
            bce_list.append(curr_bce.item())

        sum_loss = self.args.alpha * bce + self.args.beta * bpp

        return sum_loss, bce, bce_list, bpp, L1loss



    @torch.no_grad()
    def test(self, dataloader, main_tag='Test'):  # should be called eval rather than test.
        self.logger.info('Testing Files length:' + str(len(dataloader)))
        start_time = time.time()
        self.model.eval()
        for i, (coords_G2, coords_R1) in enumerate(dataloader):
            # forward
            out_set = self.model(coords_R1, coords_G2, device=device, training=False)
            # loss
            sum_loss, bce, bce_list, bpp, L1loss = self.loss_function(out_set, float(coords_G2.shape[0]))
            metrics = []
            for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                metrics.append(get_metrics(out_cls, ground_truth))
            # Calculate PSNR
            PSNR = self.calculate_PSNR(coords_G2[:,1:], out_set['out'].C[:,1:])
            # record
            self.record_set['L1loss'].append(L1loss.item())
            self.record_set['bpp'].append(bpp.item())
            self.record_set['bce'].append(bce.item())
            self.record_set['bces'].append(bce_list)
            self.record_set['sum_loss'].append(sum_loss.item())
            self.record_set['metrics'].append(metrics)
            self.record_set['var_F2'].append(out_set['var_F2'].cpu().numpy())
            self.record_set['var_res'].append(out_set['var_res'].cpu().numpy())
            self.record_set['PSNR'].append(PSNR)
            torch.cuda.empty_cache() # empty cache.
            self.record_avg_iter()
            self.record(main_tag=main_tag+'_'+str(i))

        self.record_avg()
        self.logger.info(f'Total Testing time: {((time.time()-start_time)/60):.2f} min')

        return


    def train(self, dataloader):
        self.logger.info('Training '+str(self.args.test_step)+' iterations')
        start_time = time.time()
        train_iter = iter(dataloader)

        self.model.train()
        i = 0
        while(True):
            if i>=self.args.test_step: break
            for _ in tqdm(range(self.args.base_step)):
                i+=1
                self.step+=1
                self.optimizer.zero_grad()
                # data
                coords_1, coords_2 = train_iter.next()
                # forward
                out_set = self.model(coords_1, coords_2,  device=device, training=True)
                # loss
                sum_loss, bce, bce_list, bpp, L1loss = self.loss_function(out_set, float(coords_2.shape[0]))
                # backward & optimize
                sum_loss.backward()
                self.optimizer.step()
                # metrics
                metrics = []
                for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                    metrics.append(get_metrics(out_cls, ground_truth))
                # record
                with torch.no_grad():
                    self.record_set['L1loss'].append(L1loss.item())
                    self.record_set['bpp'].append(bpp.item())
                    self.record_set['bce'].append(bce.item())
                    self.record_set['bces'].append(bce_list)
                    self.record_set['sum_loss'].append(sum_loss.item())
                    self.record_set['metrics'].append(metrics)
                    self.record_set['var_F2'].append(out_set['var_F2'].cpu().numpy())
                    self.record_set['var_res'].append(out_set['var_res'].cpu().numpy())
                torch.cuda.empty_cache()# empty cache.

                if self.step % int(self.args.lr_step) == 0:
                    if self.step>0:
                        self.args.lr =  max(self.args.lr/2, self.args.min_lr)
                        self.optimizer = self.set_optimizer()
                        self.logger.info('LR:' + str(np.round([params['lr'] for params in self.optimizer.param_groups], 6).tolist()))

            self.logger.info(f'Training time: {((time.time()-start_time)/60):.2f} min')
            self.logger.info(f'Total time: {((time.time()-self.time)/60):.2f} min')
            self.logger.info('LR:' + str(np.round([params['lr'] for params in self.optimizer.param_groups], 6).tolist()))
            with torch.no_grad(): self.record(main_tag='Train')

        self.save_model()

        return self.step
