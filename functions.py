
import os
import torch
import numpy as np
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
from scipy.io import savemat
from torch.utils.data import Dataset, DataLoader
import copy
from tools import EarlyStopping
import random
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]
        return data, index

    def __len__(self):
        return len(self.dataset)


def load_data(dir, fname, scale_opt=None, data_type=None):
    if data_type =='mat':
        hf = h5py.File(dir, 'r')
        geno = np.transpose(np.array(hf[fname][:])).astype(np.float32)  # original coding NaN=0, aa=1, Aa=2, AA=3
        if scale_opt == 'NaN_distinct':
            geno = geno/3
        elif scale_opt == 'NaN_random':
            geno[geno == 0] = random.randint(1,3)
            geno = geno/3
        elif scale_opt == 'Normalize':
            geno = (geno+1)/2  # original coding -1, 0, 1
        elif scale_opt == 'Scale':
            geno = (geno - 1) / 2  # original coding 1, 2, 3
        else:
            geno = geno
    elif data_type =='vcf':
        os.system("sed 's/^#CHROM/CHROM/' " + dir + fname + ".vcf" + " > " + dir + fname + ".temp.reformat.vcf")
        vcf = pd.read_csv(dir + fname + ".temp.reformat.vcf", sep="\t", comment='#')
        vcf = vcf.iloc[:, vcf.columns.get_loc('FORMAT') + 1:] # 0/0 is REF/REF
        # NaN = 0 = ./.
        # AA = 3 = 0/0
        # Aa= 2 = 0/1 or 1/0
        # aa = 1 = 1/1
        vcf = vcf.replace(to_replace=["./.",".|.","./0",".|0","0/.","0|.","./1",".|1","1/.","1|.",
                                    "0/0","0|0",
                                    "0/1","0|1","1/0","1|0",
                                    "1/1","1|1"], value=["0","0","0","0","0","0","0","0","0","0",
                                                        "3","3",
                                                        "2","2","2","2",
                                                        "1","1"])
        geno = np.transpose(np.array(vcf)).astype(np.float32)  # original coding NaN=0, aa=1, Aa=2, AA=3
        if scale_opt == 'NaN_distinct':
            geno = geno/3
        elif scale_opt == 'NaN_random':
            geno[geno == 0] = random.randint(1,3)
            geno = geno/3
        elif scale_opt == 'Normalize':
            geno = (geno+1)/2  # original coding -1, 0, 1
        elif scale_opt == 'Scale':
            geno = (geno - 1) / 2  # original coding 1, 2, 3
        else:
            geno = geno
    return geno


def load_ibs(dir_ibs, fname_ibs):
    hf = h5py.File(dir_ibs, 'r')
    ibs = np.array(hf[fname_ibs][:]).astype(np.float32)
    return ibs


def categorical_cross_entropy(y_pred, y_true):
    return -(y_true * torch.log(y_pred)).sum(dim=1).sum()


def recon_loss(recon_x, x, loss_opt=None):
    if loss_opt =='MAE':
        recon_loss = F.l1_loss(recon_x, x, reduction='sum')
    elif loss_opt == 'BCE':
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    return recon_loss


def vae_loss(recon_x, x, mu, logvar, variational_beta, loss_opt=None):
    # recon
    rcon_loss = recon_loss(recon_x, x, loss_opt)
    # KL-divergence
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return rcon_loss + variational_beta * kldivergence


def save_checkpoint(model, optimizer, scheduler, epoch, path, V=None):
    if V is None:
        torch.save(
            {'model': model,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'scheduler_state_dict': scheduler.state_dict()},
            os.path.join(path,'checkpoint_{:04d}.pt'.format(epoch)))
    else:
        torch.save(
            {'model': model,
             'model_state_dict': model.state_dict(),
             'V': V,
             'optimizer_state_dict': optimizer.state_dict(),
             'scheduler_state_dict': scheduler.state_dict()},
            os.path.join(path,'checkpoint_SAEIBS_{:04d}.pt'.format(epoch)))


def run_AE(model, inputdata, idx_val, batch_size, optimizer, scheduler, device, num_epochs, savepath, num_patience, loss_opt=None):
    # make validation set
    valdata = inputdata[idx_val]
    traindata = np.delete(inputdata, idx_val, axis=0)
    # load data in batch
    traindataloader = DataLoader(MyDataset(torch.from_numpy(traindata)), batch_size=batch_size, shuffle=False)
    valdataloader = DataLoader(MyDataset(torch.from_numpy(valdata)), batch_size=batch_size, shuffle=False)

    print('Training AE...')
    train_losses = []
    val_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(path=savepath + '/bestcheckpoint_AE.pt', patience=num_patience, verbose=True, delta=0.1)

    for epoch in range(num_epochs):
        model, train_loss = train_AE(model, traindataloader, optimizer, scheduler, device, loss_opt)
        val_loss = validate_AE(model, valdataloader, device, loss_opt)
        print('Epoch [%d / %d] training loss: %f validation loss: %f' % (epoch + 1, num_epochs, train_loss, val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        early_stopping(val_loss, model, epoch)
        if epoch > 50 and epoch % 100 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, savepath + '/checkpoints/')
        if early_stopping.early_stop:
            print('Early stopping at epoch: %d' % epoch)
            break
    checkpoint = torch.load(savepath + '/bestcheckpoint_AE.pt')  # reload the best checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    val_loss_min = early_stopping.val_loss_min
    return model, train_losses, val_losses, val_loss_min


def run_VAE(model, inputdata, idx_val, batch_size, optimizer, scheduler, device, num_epochs, savepath, num_patience, variational_beta, loss_opt=None):
    # make validation set
    valdata = inputdata[idx_val]
    traindata = np.delete(inputdata, idx_val, axis=0)
    # load data in batch
    traindataloader = DataLoader(MyDataset(torch.from_numpy(traindata)), batch_size=batch_size, shuffle=False)
    valdataloader = DataLoader(MyDataset(torch.from_numpy(valdata)), batch_size=batch_size, shuffle=False)

    print('Training AE...')
    train_losses = []
    val_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(path=savepath + '/checkpoint.pt', patience=num_patience, verbose=True, delta=0.1)

    for epoch in range(num_epochs):
        model, train_loss = train_VAE(model, traindataloader, optimizer, scheduler, device,variational_beta, loss_opt)
        val_loss = validate_VAE(model, valdataloader, device, variational_beta, loss_opt)
        print('Epoch [%d / %d] training loss: %f validation loss: %f' % (epoch + 1, num_epochs, train_loss, val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    checkpoint = torch.load(savepath + '/checkpoint.pt')  # reload the best checkpoint
    model.load_state_dict(checkpoint)
    val_loss_min = early_stopping.val_loss_min
    return model, train_losses, val_losses, val_loss_min


def run_DAE(model, inputdata, idx_val, datapath_noisyRef, fname, scale_opt, cond_DAE_loss, batch_size, optimizer, scheduler, device, num_epochs, savepath, num_patience, epoch_noisy, beta=None):
    # make validation set
    valdata = inputdata[idx_val]
    traindata = np.delete(inputdata, idx_val, axis=0)
    # load data in batch
    traindataloader = DataLoader(MyDataset(torch.from_numpy(traindata)), batch_size=batch_size, shuffle=False)
    valdataloader = DataLoader(MyDataset(torch.from_numpy(valdata)), batch_size=batch_size, shuffle=False)

    print('Training DAE...')
    train_losses = []
    val_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(path=savepath + '/checkpoint.pt', patience=num_patience, verbose=True, delta=0.1)

    for epoch in range(num_epochs):
        # feed noisy data every several epoch ==> need more early-stopping patience
        if (epoch % epoch_noisy == 0):
            i = int(epoch / epoch_noisy)
            tmp_dirty1kg_dir = datapath_noisyRef + fname + '_' + str(i + 1) + '.mat'
            geno1kg_dirty = load_data(tmp_dirty1kg_dir, fname, scale_opt)
            valdata_dirty = geno1kg_dirty[idx_val]
            traindata_dirty = np.delete(geno1kg_dirty, idx_val, axis=0)
            traindataloader_dirty = DataLoader(MyDataset(torch.from_numpy(traindata_dirty)), batch_size=batch_size, shuffle=False)
            valdataloader_dirty = DataLoader(MyDataset(torch.from_numpy(valdata_dirty)), batch_size=batch_size, shuffle=False)

        model, train_loss = train_DAE(model, traindataloader, traindataloader_dirty, cond_DAE_loss, optimizer, scheduler, device, beta)
        val_loss = validate_DAE(model, valdataloader, valdataloader_dirty, cond_DAE_loss, device, beta)
        print('Epoch [%d / %d] training loss: %f validation loss: %f' % (epoch + 1, num_epochs, train_loss, val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    checkpoint = torch.load(savepath + '/checkpoint.pt')
    model.load_state_dict(checkpoint)
    val_loss_min = early_stopping.val_loss_min
    return model, train_losses, val_losses, val_loss_min


def run_SAEIBS(model, inputdata, idx_val, batch_size, optimizer, scheduler, device, num_epochs, savepath, num_patience, ref_ibs=None, loss_opt=None):
    # make validation set
    valdata = inputdata[idx_val]
    ibs_val = np.diag(ref_ibs[idx_val, idx_val])
    traindata = np.delete(inputdata, idx_val, axis=0)
    ibs_train = np.delete(np.delete(ref_ibs, idx_val, axis=0), idx_val, axis=1)
    # load data in batch
    traindataloader = DataLoader(MyDataset(torch.from_numpy(traindata)), batch_size=batch_size, shuffle=False)
    valdataloader = DataLoader(MyDataset(torch.from_numpy(valdata)), batch_size=batch_size, shuffle=False)

    print('Training SAEIBS...')
    if 'SAE' in type(model).__name__:
        if model.emb is None:
            with torch.no_grad():
                for b, (x_data, _) in enumerate(traindataloader):
                    x = x_data.to(device)
                    emb = model.encoder(x)
                    if b == 0:
                        embedding = copy.deepcopy(emb)
                    else:
                        embedding = torch.cat([embedding, emb], 0)
                model.initialize_svd(embedding, torch.from_numpy(ibs_train).to(device))
            del x, embedding

    train_losses = []
    val_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(path=savepath + '/bestcheckpoint_SAEIBS.pt', patience=num_patience, verbose=True, delta=0.1)

    for epoch in range(num_epochs):
        model, train_loss, V, mean_emb = train_SAEIBS(model, traindataloader, optimizer, scheduler, device, ibs_train, loss_opt)
        val_loss = validate_SAEIBS(model, valdataloader, device, ibs_val, loss_opt)
        print('Epoch [%d / %d] training loss: %f validation loss: %f' % (epoch + 1, num_epochs, train_loss, val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        early_stopping(val_loss, model, epoch, V, mean_emb)
        if epoch % 100 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, savepath+'/checkpoints/', V)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    checkpoint = torch.load(savepath + '/bestcheckpoint_SAEIBS.pt')  # reload best checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    V = checkpoint['V']
    val_loss_min = early_stopping.val_loss_min
    return model, train_losses, val_losses, val_loss_min, V


def run_DSAEIBS(model, inputdata, idx_val, datapath_noisyRef, fname, scale_opt, cond_DAE_loss, batch_size, optimizer, scheduler, device, num_epochs, savepath, num_patience, epoch_noisy, ref_ibs=None, RefIBS_noisy=None,beta=None):
    # make validation set
    valdata = inputdata[idx_val]
    ibs_val = np.diag(ref_ibs[idx_val, idx_val])
    traindata = np.delete(inputdata, idx_val, axis=0)
    ibs_train = np.delete(np.delete(ref_ibs, idx_val, axis=0), idx_val, axis=1)
    # load data in batch
    traindataloader = DataLoader(MyDataset(torch.from_numpy(traindata)), batch_size=batch_size, shuffle=False)
    valdataloader = DataLoader(MyDataset(torch.from_numpy(valdata)), batch_size=batch_size, shuffle=False)

    print('Training DSAEIBS...')
    if 'SAE' in type(model).__name__:
        if model.emb is None:
            with torch.no_grad():
                for b, (x_data, _) in enumerate(traindataloader):
                    x = x_data.to(device)
                    emb = model.encoder(x)
                    if b == 0:
                        embedding = copy.deepcopy(emb)
                    else:
                        embedding = torch.cat([embedding, emb], 0)
                model.initialize_svd(embedding, torch.from_numpy(ibs_train).to(device))
            del x, embedding

    train_losses = []
    val_losses = []
    # initialize the early_stopping object
    early_stopping = EarlyStopping(path=savepath + '/checkpoint.pt', patience=num_patience, verbose=True, delta=0.1)

    for epoch in range(num_epochs):
        if (epoch % epoch_noisy == 0):
            i = int(epoch / epoch_noisy)
            tmp_dirty1kg_dir = datapath_noisyRef + fname + '_' + str(i + 1) + '.mat'
            tmp_dirty1kgIBS_dir = datapath_noisyRef + RefIBS_noisy + '_' + str(i + 1) + '.mat'
            geno1kg_dirty = load_data(tmp_dirty1kg_dir, fname, scale_opt)
            ref_ibs_dirty = load_ibs(tmp_dirty1kgIBS_dir, RefIBS_noisy)

            valdata_dirty = geno1kg_dirty[idx_val]
            ibs_val_dirty = np.diag(ref_ibs_dirty[idx_val, idx_val])
            traindata_dirty = np.delete(geno1kg_dirty, idx_val, axis=0)
            ibs_train_dirty = np.delete(np.delete(ref_ibs_dirty, idx_val, axis=0), idx_val, axis=1)
            traindataloader_dirty = DataLoader(MyDataset(torch.from_numpy(traindata_dirty)), batch_size=batch_size, shuffle=False)
            valdataloader_dirty = DataLoader(MyDataset(torch.from_numpy(valdata_dirty)), batch_size=batch_size, shuffle=False)

        model, train_loss, V, mean_emb = train_DSAEIBS(model, traindataloader, traindataloader_dirty, cond_DAE_loss, optimizer, scheduler, device, ibs_train,ibs_train_dirty, beta)
        val_loss = validate_DSAEIBS(model, valdataloader, valdataloader_dirty, cond_DAE_loss, device, ibs_val, ibs_val_dirty, beta)
        print('Epoch [%d / %d] training loss: %f validation loss: %f' % (epoch + 1, num_epochs, train_loss, val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        early_stopping(val_loss, model, epoch, V, mean_emb)
        if early_stopping.early_stop:
            print('Early stopping at epoch: %d' % epoch)
            break
    checkpoint = torch.load(savepath + '/checkpoint.pt')  # reload best checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    V = checkpoint['V']
    val_loss_min = early_stopping.val_loss_min
    return model, train_losses, val_losses, val_loss_min, V


def train_AE(model, dataloader, optimizer, scheduler, device, loss_opt=None):
    # set to training mode
    model.train()
    total_loss = 0
    for b, (x_batch, ind) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        x_batch_recon, h = model(x_batch)
        # reconstruction error
        loss = recon_loss(x_batch_recon, x_batch, loss_opt)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        scheduler.step()
        # loss
        total_loss += loss.item()
    return model, total_loss / len(dataloader.dataset)


def train_VAE(model, dataloader, optimizer, scheduler, device, variational_beta, loss_opt=None):
    # set to training mode
    model.train()
    total_loss = 0
    for b, (x_batch, ind) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        x_batch_recon, h, latent_mu, latent_logvar = model(x_batch)
        # vae loss
        loss = vae_loss(x_batch_recon, x_batch, latent_mu, latent_logvar, variational_beta, loss_opt)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        scheduler.step()
        # loss
        total_loss += loss.item()
    return model, total_loss/len(dataloader.dataset)


def train_DAE(model, dataloader, dataloader_dirty, cond_DAE_loss, optimizer, scheduler, device, beta=None):
    # set to training mode
    model.train()
    total_loss = 0
    for b, ((x_batch,_), (x_batch_dirty,_)) in enumerate(zip(dataloader, dataloader_dirty)):
        x_batch = x_batch.to(device)
        x_batch_dirty = x_batch_dirty.to(device)
        # reconstruction
        x_batch_recon_dirty, latent_dirty = model(x_batch_dirty)
        # reconstruction error
        if cond_DAE_loss == '+ project loss':
            x_batch_recon, latent = model(x_batch)
            loss = recon_loss(x_batch_recon_dirty, x_batch) + beta * recon_loss(latent_dirty, latent)
        else:
            loss = recon_loss(x_batch_recon_dirty, x_batch)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        scheduler.step()
        # loss
        total_loss += loss.item()
    return model, total_loss/len(dataloader.dataset)


def train_SAEIBS(model, dataloader, optimizer, scheduler, device, ibs=None, loss_opt=None):
    # set to training mode
    model.train()
    total_loss = 0
    for b, (x_batch, ind) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        if ibs is not None:
            ibs_batch = ibs[ind, ind]
            ibs_batch = torch.diag(torch.from_numpy(ibs_batch).to(device))
        # reconstruction
        x_batch_recon, _, V, mean_emb = model(x_batch, ibs_batch, ind)
        # reconstruction error
        loss = recon_loss(x_batch_recon, x_batch, loss_opt)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        scheduler.step()
        # loss
        total_loss += loss.item()
    return model, total_loss/len(dataloader.dataset), V, mean_emb


def train_DSAEIBS(model, dataloader,  dataloader_dirty, cond_DAE_loss, optimizer, scheduler, device, ibs=None, ibs_dirty=None, beta=None):
    # set to training mode
    model.train()
    total_loss = 0
    for b, ((x_batch,ind), (x_batch_dirty,ind_dirty)) in enumerate(zip(dataloader, dataloader_dirty)):
        x_batch = x_batch.to(device)
        x_batch_dirty = x_batch_dirty.to(device)
        if ibs is not None and ibs_dirty is not None:
            ibs_batch = ibs[ind, ind]
            ibs_batch = torch.diag(torch.from_numpy(ibs_batch).to(device))
            ibs_dirty_batch = ibs_dirty[ind_dirty, ind_dirty]
            ibs_dirty_batch = torch.diag(torch.from_numpy(ibs_dirty_batch).to(device))
        # reconstruction
        x_batch_recon_dirty, latent_dirty, V, mean_emb = model(x_batch_dirty, ibs_dirty_batch, ind_dirty)
        # reconstruction error
        if cond_DAE_loss == '+ project loss':
            x_batch_recon, latent, _, _ = model(x_batch, ibs_batch, ind)
            loss = recon_loss(x_batch_recon_dirty, x_batch) + beta * recon_loss(latent_dirty, latent)
        else:
            loss = recon_loss(x_batch_recon_dirty, x_batch)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        scheduler.step()
        # loss
        total_loss += loss.item()
    return model, total_loss/len(dataloader.dataset), V, mean_emb


def validate_AE(model, dataloader, device, loss_opt=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for b, (x_batch, ind) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            x_batch_recon, h = model(x_batch)
            loss = recon_loss(x_batch_recon, x_batch, loss_opt)
            total_loss += loss.item()
    return total_loss/len(dataloader.dataset)


def validate_VAE(model, dataloader, device, variational_beta, loss_opt=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for b, (x_batch, ind) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            x_batch_recon, h, latent_mu, latent_logvar = model(x_batch)
            loss = vae_loss(x_batch_recon, x_batch, latent_mu, latent_logvar, variational_beta, loss_opt)
            total_loss += loss.item()
    return total_loss/len(dataloader.dataset)


def validate_DAE(model, dataloader, dataloader_dirty, cond_DAE_loss, device, beta=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for b, ((x_batch, _), (x_batch_dirty, _)) in enumerate(zip(dataloader, dataloader_dirty)):
            x_batch = x_batch.to(device)
            x_batch_dirty = x_batch_dirty.to(device)
            # reconstruction
            x_batch_recon_dirty, latent_dirty = model(x_batch_dirty)
            # reconstruction error
            if cond_DAE_loss == 'project_loss':
                x_batch_recon, latent = model(x_batch)
                loss = recon_loss(x_batch_recon_dirty, x_batch) + beta * recon_loss(latent_dirty, latent)
            else:
                loss = recon_loss(x_batch_recon_dirty, x_batch)
            total_loss += loss.item()
    return total_loss/len(dataloader.dataset)


def validate_SAEIBS(model, dataloader, device, ibs=None, loss_opt=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for b, (x_batch, ind) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            if ibs is not None:
                ibs_batch = ibs[ind, ind]
                ibs_batch = torch.diag(torch.from_numpy(ibs_batch).to(device))
            x_batch_recon, _, _, _ = model(x_batch, ibs_batch, ind)
            loss = recon_loss(x_batch_recon, x_batch, loss_opt)
            total_loss += loss.item()
    return total_loss/len(dataloader.dataset)


def validate_DSAEIBS(model, dataloader, dataloader_dirty, cond_DAE_loss, device, ibs=None, ibs_dirty=None, beta=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for b, ((x_batch, ind), (x_batch_dirty, ind_dirty)) in enumerate(zip(dataloader, dataloader_dirty)):
            x_batch = x_batch.to(device)
            x_batch_dirty = x_batch_dirty.to(device)
            if ibs is not None and ibs_dirty is not None:
                ibs_batch = ibs[ind, ind]
                ibs_batch = torch.diag(torch.from_numpy(ibs_batch).to(device))
                ibs_dirty_batch = ibs_dirty[ind_dirty, ind_dirty]
                ibs_dirty_batch = torch.diag(torch.from_numpy(ibs_dirty_batch).to(device))
            # reconstruction
            x_batch_recon_dirty, latent_dirty, _, _ = model(x_batch_dirty, ibs_dirty_batch, ind_dirty)
            # reconstruction error
            if cond_DAE_loss == '+ project loss':
                x_batch_recon, latent, _, _ = model(x_batch, ibs_batch, ind)
                loss = recon_loss(x_batch_recon_dirty, x_batch) + beta * recon_loss(latent_dirty, latent)
            else:
                loss = recon_loss(x_batch_recon_dirty, x_batch)
            total_loss += loss.item()
    return total_loss/len(dataloader.dataset)


def project(model, data, batch_size, latent_dim, device):
    dataloader = DataLoader(MyDataset(torch.from_numpy(data)), batch_size=batch_size, shuffle=False)
    # set to evaluation mode
    model.eval()
    latent = torch.zeros(0, latent_dim)
    for b, (x_batch, _) in enumerate(dataloader):
        with torch.no_grad():
            x_batch = x_batch.to(device)
            # latent space
            if 'VariationalAutoencoder' in type(model).__name__:
                _, z, _, _ = model(x_batch)
            else:
                _, z = model(x_batch)
            latent = torch.cat((latent.to(device), z))
    latent = latent.to('cpu').detach().numpy()
    return latent


def projectSAEIBS_traindata(model, data, ibs, V, batch_size, device):
    dataloader = DataLoader(MyDataset(torch.from_numpy(data)), batch_size=batch_size, shuffle=False)
    # set to evaluation mode
    model.eval()
    with torch.no_grad():
        for b, (x_batch, _) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            emb = model.encoder(x_batch)
            if b == 0:
                embedding = copy.deepcopy(emb)
            else:
                embedding = torch.cat([embedding, emb], 0)

        embedding = torch.mm(torch.from_numpy(ibs).to(device), embedding)
        mean_emb = torch.mean(embedding, 0)
        latent = torch.matmul(embedding - mean_emb, V)
    return latent.to('cpu').detach().numpy(), mean_emb.to('cpu').detach().numpy()


def projectSAEIBS_newdata(model, data, ibs_connect, V, mean_emb, batch_size, device):
    dataloader = DataLoader(MyDataset(torch.from_numpy(data)), batch_size=batch_size, shuffle=False)
    # set to evaluation mode
    model.eval()
    with torch.no_grad():
        for b, (x_batch, _) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            emb = model.encoder(x_batch)
            if b == 0:
                embedding = copy.deepcopy(emb)
            else:
                embedding = torch.cat([embedding, emb], 0)

        embedding = torch.mm(torch.from_numpy(ibs_connect).to(device), embedding)
        latent = torch.matmul(embedding - torch.from_numpy(mean_emb).to(device), V)
    return latent.to('cpu').detach().numpy()


def load_project_save(dir, fname, model, savename, key, savepath, batch_size, scale_opt, latent_dim, device):
    data = load_data(dir, fname, scale_opt)
    latent = project(model, data, batch_size, latent_dim, device)
    savemat(savepath + '/' + savename + '.mat', mdict={key: latent})


def load_projectSAEIBS_save(dir, fname, dir_ibs, fname_ibs, model, V, mean_emb, savename, key, savepath, batch_size, scale_opt, device):
    data = load_data(dir, fname, scale_opt)
    ibs_connect = load_ibs(dir_ibs, fname_ibs)
    latent = projectSAEIBS_newdata(model, data, ibs_connect, V, mean_emb, batch_size, device)
    savemat(savepath + '/' + savename + '.mat', mdict={key: latent})


def orthogonality(latent, path, run_parameters, modelname):
    # covariance of embedding
    cov_emb = np.cov(np.transpose(latent))
    plt.imshow(cov_emb, cmap='seismic', interpolation='nearest')
    plt.xlabel("Dimension")
    plt.ylabel("Dimension")
    plt.suptitle("Covariance matrix\n (1 KG " + modelname + run_parameters + ")", fontsize=12)
    plt.savefig(path + '/Cov_' + run_parameters + '.jpg')
    plt.show()
    return cov_emb



