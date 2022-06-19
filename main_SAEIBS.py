import os
import sys

sys.path.append("/usr/local/micapollo01/MIC/DATA/STAFF/myuan0/tmp/3_ANCESTRY/AncestrySAEIBS/")
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch.backends.cudnn as cudnn
import pandas
from functions import *
from model import Autoencoder, SAEIBS
from plot_functions import plot_latent
from arguments import parse_args, save_args

device_idx = 0
torch.cuda.get_device_name(device_idx)
device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
# set deterministic
torch.manual_seed(6)
np.random.seed(6)
cudnn.deterministic = True


def main():
    args = parse_args()
    args.work_dir = os.path.dirname(os.path.realpath(__file__))
    run_parameters = str(len(args.hidden_dim)) + 'Layer_' + str(args.latent_dim) + 'Latent_' + "".join(str(x) for x in args.hidden_dim)
    args.out_dir = os.path.join(args.work_dir, 'Results/' + args.model_name + '/' + args.model_name + '_' + run_parameters)
    args.checkpoints_dir = os.path.join(args.out_dir, 'checkpoints')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    print(args)
    save_args(args, args.out_dir)

    # Load data
    geno_1kg = load_data(args.ref_data_dir, 'Genotypes_1kg', args.scale_opt)
    geno_hdgp_sub = load_data(args.target_data_dir, 'Genotypes_hdgp', args.scale_opt)
    # load IBS
    ibs_1kg = load_ibs(args.ref_ibs_dir, 'IBS_1kg')
    ibs_hdgp_sub = load_ibs(args.target_ibs_dir, 'IBSconnect_hdgp')
    # Load population label
    population = np.array(pandas.read_csv(args.pop_dir, header=None))

    # make validation set
    idx_val = range(0, geno_1kg.shape[0], 10)  # 10% data for validation
    population_val = population[idx_val]
    population_train = np.delete(population, idx_val, axis=0)

    # pre-train an AE model
    input_dim = geno_1kg.shape[1]
    pre_model = Autoencoder(input_dim, args.hidden_dim[:-1], args.hidden_dim[-1], args.cond_dropout, args.drop_rate, args.actFn).to(device)

    num_params = sum(p.numel() for p in pre_model.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)

    optimizer = torch.optim.Adam(params=pre_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.lr_decay)

    pre_model, train_loss1, val_loss1, _, = run_AE(pre_model, geno_1kg, idx_val, args.batch_size, optimizer, scheduler, device,
                                                     args.pretrain_epochs, args.out_dir, args.patience, args.loss_opt)

    # train SAE-IBS
    model = SAEIBS(input_dim, args.hidden_dim, args.latent_dim, args.cond_dropout, args.drop_rate, args.actFn).to(device)

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

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.lr_decay)

    model, train_loss2, val_loss2, val_loss_min, V = run_SAEIBS(model, geno_1kg, idx_val, args.batch_size, optimizer, scheduler, device,
                                                                args.maxNum_epochs, args.out_dir, args.patience, ibs_1kg)

    # Project
    group = population
    cdict = {'EUR': 'blue', 'EAS': 'red', 'AMR': 'yellow', 'SAS': 'purple', 'AFR': 'green'}
    latent_1kg, mean_emb_1kg = projectSAEIBS_traindata(model, geno_1kg, ibs_1kg, V, args.batch_size, device)
    plot_latent(latent_1kg, cdict, group, "1 KG " + args.model_name, "_1KG", args.out_dir, run_parameters)

    group2 = np.append(population, np.array([['HDGP'] for _ in range(geno_hdgp_sub.shape[0])]))
    cdict2 = {'EUR': 'blue', 'EAS': 'red', 'AMR': 'yellow', 'SAS': 'purple', 'AFR': 'green', 'HDGP': 'grey'}
    latent_hdgp_sub = projectSAEIBS_newdata(model, geno_hdgp_sub, ibs_hdgp_sub, V, mean_emb_1kg, args.batch_size, device)
    latent = np.concatenate((latent_1kg, latent_hdgp_sub), axis=0)
    plot_latent(latent, cdict2, group2, "1KG HDGP; " + args.model_name, "_1KG_HDGP", args.out_dir, run_parameters)

    latent_path = args.out_dir + '/latent'
    if not os.path.exists(latent_path):
        os.makedirs(latent_path)

    # save 1KG latent
    latent_1KG = latent_1kg
    savemat(latent_path + '/latent_1KG.mat', mdict={'latent_1KG': latent_1KG})
    # save HDGP-sub latent
    savemat(latent_path + '/latent_hdgp.mat', mdict={'latent_hdgp': latent_hdgp_sub})

    # Save Model
    path_model = args.out_dir + '/' + args.model_name + '_' + run_parameters + '.tar'
    torch.save({'model': model,
                'model_state_dict': model.state_dict(),
                'V': V}, path_model)

    # Plot cov matrix
    cov_emb = orthogonality(latent_1kg, args.out_dir, run_parameters, args.model_name + ' ')


if __name__ == '__main__':
    main()