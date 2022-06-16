import os
import sys

sys.path.append("/usr/local/micapollo01/MIC/DATA/STAFF/myuan0/tmp/3_ANCESTRY/AncestrySAEIBS/")
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch.backends.cudnn as cudnn
import pandas
from functions import *
from model import *
from plot_functions import *

device_idx = 0
torch.cuda.get_device_name(device_idx)
device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
# set deterministic
torch.manual_seed(6)
np.random.seed(6)
cudnn.deterministic = True

# hyper-parameters
latent_dim = 4
hidden_dim = [512, 128, 64]
drop_rate = 0.25

# training condition
model_name = 'SAEIBS'
actFn = 'sigmoid'
scale_opt = 'Scale'
cond_dropout = False  # default is false
maxNum_epochs = 3000
batch_size = 256
num_epochs_pre = 100  # pre-train AE
num_patience_pre = 300
num_patience_sae = 100
loss_opt = 'MSE'
compute_wd_loss = 'TRUE'


# make path
run_parameters = str(len(hidden_dim)) + 'Layer_' + str(latent_dim) + 'Latent_' + "".join(str(x) for x in hidden_dim)
path = os.getcwd() + '/Results/main_1KG_HDGP/runs_' + model_name + '/' + model_name + '_' + run_parameters 
if cond_dropout:
    path = path + '_dropout' + str(drop_rate)
if not os.path.exists(path):
    os.makedirs(path)
path_checkpoint = path + '/checkpoints'
if not os.path.exists(path_checkpoint):
    os.makedirs(path_checkpoint)

# load data
datapath = '/usr/local/micapollo01/MIC/DATA/STAFF/myuan0/tmp/3_ANCESTRY/DATA/'
geno_1kg = load_data(datapath + '/SimExp_1KG_HDGP/Genotypes_1kg.mat', 'Genotypes_1kg', scale_opt)
geno_hdgp_sub = load_data(datapath + '/HDGP_sub/Genotypes_hdgp_sub.mat', 'Genotypes_hdgp_sub', scale_opt)

# Load population label
pop_dir = '/usr/local/micapollo01/MIC/DATA/STAFF/myuan0/tmp/3_ANCESTRY/DATA/1KG/SUPPOP.csv'
population = np.array(pandas.read_csv(pop_dir, header=None))

# load IBS
ibs_1kg = load_ibs(datapath + '/SimExp_1KG_HDGP/IBS_1kg.mat', 'IBS_1kg')
ibs_hdgp_sub = load_ibs(datapath + '/HDGP_sub/IBSconnect_hdgp_sub.mat', 'IBSconnect_hdgp_sub')


# make validation set
idx_val = range(0, geno_1kg.shape[0], 10)  # 10% data for validation
population_val = population[idx_val]
population_train = np.delete(population, idx_val, axis=0)

# %%

# pre-train an AE model
input_dim = geno_1kg.shape[1]
pre_model = Autoencoder(input_dim, hidden_dim[:-1], hidden_dim[-1], cond_dropout, drop_rate, actFn).to(device)

num_params = sum(p.numel() for p in pre_model.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

optimizer = torch.optim.Adam(params=pre_model.parameters(), lr=1e-3, weight_decay=1e-9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9999)

pre_model, train_loss1, val_loss1, _, _ = run_AE(pre_model, geno_1kg, idx_val, batch_size, optimizer, scheduler, device,
                                              num_epochs_pre, path, num_patience_pre, loss_opt, compute_wd_loss)

# check pre_train loss
plot_loss(train_loss1, val_loss1, path, '_premodel')
plot_loss_zoom(100, np.arange(100, len(train_loss1)), train_loss1[100:], val_loss1[100:], path, '_premodel_after100')


# Save logFile
log_file_pre = path + '/Log_pre_train' + model_name + '_' + run_parameters + '.txt'
loss_pre_train = path + '/Loss_pre_train_' + model_name + '_' + run_parameters + '.txt'
loss_pre_val = path + '/Loss_pre_val_' + model_name + '_' + run_parameters + '.txt'
with open(log_file_pre, 'w') as f:
    f.write('Model: ' + str(pre_model) + '\n\n')
    f.write('Number of parameters: ' + str(num_params) + '\n\n')
    f.write('Optimizer: ' + str(optimizer) + '\n\n')
    f.write('Missing value: ' + str(scale_opt) + '\n\n')
    f.write('Activation function of last layer in decoder: ' + str(actFn) + '\n\n')
    f.write('Dropout: ' + str(cond_dropout) + ' ' + str(drop_rate) + '\n\n')
    f.write('Batch size: ' + str(batch_size) + '\n\n')
    f.write('Pre-train epochs: ' + str(num_epochs_pre) + '\n\n')
    f.write('Pre-train Patience: ' + str(num_patience_pre) + '\n\n')
    f.write('Pre-train min val loss at epoch: ' + str(ind[0]) + '\n\n')
with open(loss_pre_train, 'w') as f:
    f.write(str(train_loss1))
with open(loss_pre_val, 'w') as f:
    f.write(str(val_loss1))


# %% train SAE
model = SAEIBS(input_dim, hidden_dim, latent_dim, cond_dropout, drop_rate, actFn).to(device)

update_dict = {}
pre_model_dict = pre_model.state_dict()
for k in pre_model_dict:
    update_dict[k] = pre_model_dict[k]
model_dict = model.state_dict()
for k in model_dict:
    if k not in update_dict:
        update_dict[k] = model_dict[k]

model_dict.update(update_dict)
model.load_state_dict(update_dict)

del pre_model, pre_model_dict, model_dict, update_dict

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9999)

model, train_loss2, val_loss2, val_loss_min, V = run_SAEIBS(model, geno_1kg, idx_val, batch_size, optimizer, scheduler, device,
                                                            maxNum_epochs, path, num_patience_sae, ibs_1kg)


# %%

# plot loss
train_loss = train_loss1 + train_loss2
val_loss = val_loss1 + val_loss2

plot_loss(train_loss, val_loss, path)
plot_loss(train_loss2, val_loss2, path, '_SAEIBS')

# Project
group = population
cdict = {'EUR': 'blue', 'EAS': 'red', 'AMR': 'yellow', 'SAS': 'purple', 'AFR': 'green'}
latent_1kg, mean_emb_1kg = projectSAEIBS_traindata(model, geno_1kg, ibs_1kg, V, batch_size, device)
plot_latent(latent_1kg, cdict, group, "1 KG " + model_name, "_1KG", path, run_parameters)

group2 = np.append(population, np.array([['HDGP'] for _ in range(geno_hdgp_sub.shape[0])]))
cdict2 = {'EUR': 'blue', 'EAS': 'red', 'AMR': 'yellow', 'SAS': 'purple', 'AFR': 'green', 'HDGP': 'grey'}
latent_hdgp_sub = projectSAEIBS_newdata(model, geno_hdgp_sub, ibs_hdgp_sub, V, mean_emb_1kg, batch_size, device)
latent = np.concatenate((latent_1kg, latent_hdgp_sub), axis=0)
plot_latent(latent, cdict2, group2, "1KG HDGP; " + model_name, "_1KG_HDGP", path, run_parameters)


latent_path = path + '/latent'
if not os.path.exists(latent_path):
    os.makedirs(latent_path)

# save 1KG latent
latent_1KG = latent_1kg
savemat(latent_path + '/latent_1KG.mat', mdict={'latent_1KG': latent_1KG})
# save HDGP-sub latent
savemat(latent_path + '/latent_hdgp_sub.mat', mdict={'latent_hdgp_sub': latent_hdgp_sub})

# Save Model
path_model = path + '/' + model_name + '_' + run_parameters + '.tar'
torch.save({'model': model,
            'model_state_dict': model.state_dict(),
            'V': V,
            'train_loss': train_loss,
            'val_loss': val_loss}, path_model)

# Save logFile
log_file = path + '/Log_' + model_name + '_' + run_parameters + '.txt'
loss_file1 = path + '/Loss_total_train_' + model_name + '_' + run_parameters + '.txt'
loss_file2 = path + '/Loss_total_val_' + model_name + '_' + run_parameters + '.txt'
loss_file3 = path + '/Loss_SAEIBS_train_' + model_name + '_' + run_parameters + '.txt'
loss_file4 = path + '/Loss_SAEIBS_val_' + model_name + '_' + run_parameters + '.txt'
with open(log_file, 'w') as f:
    f.write('Model: ' + str(model) + '\n\n')
    f.write('Number of parameters: ' + str(num_params) + '\n\n')
    f.write('Optimizer: ' + str(optimizer) + '\n\n')
    f.write('Missing value: ' + str(scale_opt) + '\n\n')
    f.write('Activation function of last layer in decoder: ' + str(actFn) + '\n\n')
    f.write('Dropout: ' + str(cond_dropout) + ' ' + str(drop_rate) + '\n\n')
    f.write('Batch size: ' + str(batch_size) + '\n\n')
    f.write('Pre-train Patience: ' + str(num_patience_pre) + '\n\n')
    f.write('SAEIBS Patience: ' + str(num_patience_sae) + '\n\n')
with open(loss_file1, 'w') as f:
    f.write(str(train_loss))
with open(loss_file2, 'w') as f:
    f.write(str(val_loss_min) + '\n\n')
    f.write(str(val_loss))
with open(loss_file3, 'w') as f:
    f.write(str(train_loss2))
with open(loss_file4, 'w') as f:
    f.write(str(val_loss2))


# Plot cov matrix
cov_emb = orthogonality(latent_1kg, path, run_parameters, model_name+' ')
