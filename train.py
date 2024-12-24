import argparse
from data import make_data_loader
from model.Network import Network
from trainer import Trainer
import os, torch
import multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--prefix", type=str, default='bs4_a1_b1', help="prefix of checkpoints/logger, etc.")

    # Variables related to dataset.
    parser.add_argument("--dataset_train", default='../../Data/train/dataset_v1/')
    parser.add_argument("--train", default=['loot_kd2.npz',
                                            'longdress_kd2.npz',
                                            'queen_kd2.npz'])

    parser.add_argument("--dataset_test", default='../../Data/test/dataset_v5/')
    parser.add_argument("--test", default=['redandblack.npz', 'soldier.npz',
                                           'basketball.npz', 'dancer.npz', 'exercise.npz', 'model.npz'])   # Should be called Eval rather than test

    parser.add_argument("--alpha", type=float, default=1., help="weights for distoration.")
    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")

    parser.add_argument("--init_ckpt", default='')
    parser.add_argument("--reset", default=False, action='store_true', help='reset training')

    parser.add_argument("--global_step", type=int, default=int(20000))
    parser.add_argument("--base_step", type=int, default=int(5),  help='frequency for recording state.')
    parser.add_argument("--test_step", type=int, default=int(10),  help='frequency for test and save.')
    parser.add_argument("--logdir", type=str, default='logs', help="logger direction.")
    parser.add_argument("--ckptdir", type=str, default='ckpts', help="ckpts direction.")
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--lr_step", type=int, default=5000, help="step for adjusting lr_scheduler.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # log
    args = parse_args()
    # creating folders
    args.logdir = os.path.join(args.logdir, args.prefix)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    args.ckptdir = os.path.join(args.ckptdir, args.prefix)
    if not os.path.exists(args.ckptdir):
        os.makedirs(args.ckptdir)
    # model
    model = Network()
    # trainer
    trainer = Trainer(args, model=model)

    # dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Train Path: {args.dataset_train}')
    print(f'Train Files: {args.train}')
    print(f'Test Path: {args.dataset_test}')
    print(f'Test Files: {args.test}')

    # Load data.
    train_dataloader = make_data_loader(path=args.dataset_train,
                                        files=args.train,
                                        batch_size=args.batch_size,
                                        train=True,
                                        shuffle=True,
                                        num_workers=mp.cpu_count(),
                                        repeat=True)

    print(f'Length of Train Loader: {len(train_dataloader)}')

    test_dataloader = make_data_loader(path=args.dataset_test,
                                        files=args.test,
                                        batch_size=1,
                                        train=False,
                                        shuffle=False,
                                        num_workers=mp.cpu_count(),
                                        repeat=False)

    print(f'Length of Test Loader: {len(test_dataloader)}')

    print('alpha: ' + str(round(args.alpha,2)) + '\tbeta: ' + str(round(args.beta,2)))

    # training
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')

    # trainer.test(test_dataloader, 'Test')
    while True:
        step = trainer.train(train_dataloader)
        trainer.test(test_dataloader, 'Test')

        if step >= args.global_step:
            print("Finished Training !!")
            break
