import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

import argparse

from model import VAE
from celeba_dataset import *
from util import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    dataset = celeba_dataset(args.data_path, args.img_size)

    logger = SummaryWriter(args.logdir)

    dataloader = DataLoader(dataset, args.batch_size, True, pin_memory=True)
    model = VAE(3, 500, 128).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), args.lr)

    torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'model_start.ckpt'))

    global_step = 0
    for epoch in range(args.total_epoch):
        for i, imgs in enumerate(dataloader):
            global_step += 1

            imgs = imgs.to(device)

            recon_imgs, mu, logvar, train_z = model(imgs)
            debug1 = torch.max(mu)
            debug2 = torch.max(logvar)

            # recon_loss = torch.nn.BCELoss()(recon_imgs.reshape(imgs.shape[0], -1), imgs.reshape(imgs.shape[0], -1))
            recon_loss = torch.nn.MSELoss()(recon_imgs.reshape(imgs.shape[0], -1), imgs.reshape(imgs.shape[0], -1))
            if torch.isnan(mu).any():
                print('mu is problem')
            if torch.isnan(logvar).any():
                print('logvar is problem')
            kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1)).data.item()
            true_z = torch.rand_like(train_z)
            mmd_loss = compute_mmd(train_z, true_z)
            total_loss = recon_loss + mmd_loss

            # assert not torch.isnan(total_loss).any() print("What is the problem?")

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            orig_imgs = make_grid(imgs[:16], 4, normalize=True)
            recon_imgs = make_grid(recon_imgs[:16], 4, normalize=True)

            if (global_step + 1) % args.log_freq == 0:
                logger.add_scalars('loss', {'recon_loss': recon_loss.data.item(),
                                            'mmd_loss': mmd_loss.data.item(),
                                            'kld_loss': kld_loss,
                                            'total_loss': total_loss.data.item()}, global_step)
                logger.add_image('images', orig_imgs, global_step)
                logger.add_image('recon_imgs', recon_imgs, global_step)

            print("Epoch: %d/%d, Step: %d/%d, Recon: %.4f, MMD: %.4f, KLD: %.4f"%(epoch+1, args.total_epoch, i+1, len(dataloader), recon_loss.data.item(), mmd_loss.data.item(), kld_loss))

        if (epoch+1) % args.save_freq == 0 :
            torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'model_%d.ckpt'%(epoch+1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the vae with celeba or MNIST')

    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--total_epoch', default=600)

    parser.add_argument('--data_path', default='../dataset/celeba_dataset/img_align_celeba/')
    parser.add_argument('--ckpt_path', default='./ckpt')
    parser.add_argument('--logdir', default='./logs/')
    parser.add_argument('--save_freq', default=5)
    parser.add_argument('--log_freq', default=20)

    args = parser.parse_args()
    main(args)