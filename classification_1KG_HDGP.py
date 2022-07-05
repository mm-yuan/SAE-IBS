
import os
import sys

sys.path.append("/usr/local/micapollo01/MIC/DATA/STAFF/myuan0/tmp/3_ANCESTRY/AncestrySAEIBS/")
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch.backends.cudnn as cudnn
import scipy.io as sio
import pandas

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score

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


dir = os.getcwd() + '/Results/main_1KG_HDGP/'
savepath = dir + '/Evaluation_Classification/'
if not os.path.exists(savepath):
    os.makedirs(savepath)
figpath = savepath + '/Fig/'
if not os.path.exists(figpath):
    os.makedirs(figpath)


# LOAD PCA RESULTS
datapath = '/usr/local/micapollo01/MIC/DATA/STAFF/myuan0/tmp/3_ANCESTRY/'
Z_1kg_pca_dir = datapath + '/DATA/SimExp_1KG_HDGP/PCAscores_1kg.mat'
hf = h5py.File(Z_1kg_pca_dir, 'r')
Z_1kg_pca = np.transpose(np.array(hf['PCAscores_1kg']))
Z_hdgp_pca_dir = datapath + '/DATA/HDGP_sub/PCAscores_hdgp_sub.mat'
hf = h5py.File(Z_hdgp_pca_dir, 'r')
Z_hdgp_pca = np.transpose(np.array(hf['PCAscores_hdgp_sub']))


# LOAD AE RESULTS
dir = os.getcwd() + '/Results/main_1KG_HDGP/'

Z_1kg_ae2 = sio.loadmat(dir + 'runs_AE/AE_3Layer_2Latent_51212864/latent/latent_1KG.mat')['latent_1KG']
Z_1kg_ae4 = sio.loadmat(dir + 'runs_AE/AE_3Layer_4Latent_51212864/latent/latent_1KG.mat')['latent_1KG']
Z_1kg_ae8 = sio.loadmat(dir + 'runs_AE/AE_3Layer_8Latent_51212864/latent/latent_1KG.mat')['latent_1KG']
Z_1kg_ae12 = sio.loadmat(dir + 'runs_AE/AE_3Layer_12Latent_51212864/latent/latent_1KG.mat')['latent_1KG']

Z_hdgp_ae2 = sio.loadmat(dir + 'runs_AE/AE_3Layer_2Latent_51212864/latent/latent_hdgp_sub.mat')['latent_hdgp_sub']
Z_hdgp_ae4 = sio.loadmat(dir + 'runs_AE/AE_3Layer_4Latent_51212864/latent/latent_hdgp_sub.mat')['latent_hdgp_sub']
Z_hdgp_ae8 = sio.loadmat(dir + 'runs_AE/AE_3Layer_8Latent_51212864/latent/latent_hdgp_sub.mat')['latent_hdgp_sub']
Z_hdgp_ae12 = sio.loadmat(dir + 'runs_AE/AE_3Layer_12Latent_51212864/latent/latent_hdgp_sub.mat')['latent_hdgp_sub']


# Load SAEIBS RESULTS
Z_1kg_saeibs2 = sio.loadmat(dir + 'runs_SAEIBS/SAEIBS_3Layer_2Latent_51212864/latent/latent_1KG.mat')['latent_1KG']
Z_1kg_saeibs4 = sio.loadmat(dir + 'runs_SAEIBS/SAEIBS_3Layer_4Latent_51212864/latent/latent_1KG.mat')['latent_1KG']
Z_1kg_saeibs8 = sio.loadmat(dir + 'runs_SAEIBS/SAEIBS_3Layer_8Latent_51212864/latent/latent_1KG.mat')['latent_1KG']
Z_1kg_saeibs12 = sio.loadmat(dir + 'runs_SAEIBS/SAEIBS_3Layer_12Latent_51212864/latent/latent_1KG.mat')['latent_1KG']

Z_hdgp_saeibs2 = sio.loadmat(dir + 'runs_SAEIBS/SAEIBS_3Layer_2Latent_51212864/latent/latent_hdgp_sub.mat')['latent_hdgp_sub']
Z_hdgp_saeibs4 = sio.loadmat(dir + 'runs_SAEIBS/SAEIBS_3Layer_4Latent_51212864/latent/latent_hdgp_sub.mat')['latent_hdgp_sub']
Z_hdgp_saeibs8 = sio.loadmat(dir + 'runs_SAEIBS/SAEIBS_3Layer_8Latent_51212864/latent/latent_hdgp_sub.mat')['latent_hdgp_sub']
Z_hdgp_saeibs12 = sio.loadmat(dir + 'runs_SAEIBS/SAEIBS_3Layer_12Latent_51212864/latent/latent_hdgp_sub.mat')['latent_hdgp_sub']


# Load population label
pop_1kg_dir = '/usr/local/micapollo01/MIC/DATA/STAFF/myuan0/tmp/3_ANCESTRY/DATA/1KG/SUPPOP.csv'
suppopulation_1kg = pandas.read_csv(pop_1kg_dir, header=None)
suplabels_1kg_true = pandas.get_dummies(np.squeeze(np.array(suppopulation_1kg))).values.argmax(1)

pop_hdgp_dir = '/usr/local/micapollo01/MIC/DATA/STAFF/myuan0/tmp/3_ANCESTRY/DATA/HDGP_sub/label_hdgp_sub.csv'
suppopulation_hdgp = pandas.read_csv(pop_hdgp_dir, header=None)
suplabels_hdgp_true = pandas.get_dummies(np.squeeze(np.array(suppopulation_hdgp))).values.argmax(1)


# %%

def score_plot(X, labels_true, labels_pred, title, filename):
    acc = balanced_accuracy_score(y_true=labels_true, y_pred=labels_pred)

    fig = plt.figure(1, figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels_pred, s=10, cmap='viridis')
    plt.title(title)
    side_text = plt.figtext(0.92, 0.5, 'ACC: ' + str(acc)
                            , bbox=dict(facecolor='white'))
    fig.subplots_adjust(top=0.8)
    fig.savefig(figpath + filename, bbox_extra_artists=(side_text,), bbox_inches='tight')
    plt.close()
    return acc


def KNN(X, title, filename, num_neighbor, labels_true):
    model = KNeighborsClassifier(n_neighbors=num_neighbor)
    model.fit(X, labels_true)
    labels_pred = model.predict(X)
    f1_score = score_plot(X, labels_true, labels_pred, title, filename)
    return f1_score


def KNN_PredictTarget(X, Y, title, filename, num_neighbor, Xlabels_true, Ylabels_true):
    model = KNeighborsClassifier(n_neighbors=num_neighbor)
    model.fit(X, Xlabels_true)
    Ylabels_pred = model.predict(Y)
    score = score_plot(Y, Ylabels_true, Ylabels_pred, title, filename)
    return score


# %%


df = {}
df[1] = Z_1kg_pca[:, :2]; df[2] = Z_1kg_ae2; df[3] = Z_1kg_saeibs2
df[4] = Z_1kg_pca[:, :4]; df[5] = Z_1kg_ae4; df[6] = Z_1kg_saeibs4
df[7] = Z_1kg_pca[:, :8]; df[8] = Z_1kg_ae8; df[9] = Z_1kg_saeibs8
df[10] = Z_1kg_pca[:, :12]; df[11] = Z_1kg_ae12; df[12] = Z_1kg_saeibs12

df2 = {}
df2[1] = Z_hdgp_pca[:, :2]; df2[2] = Z_hdgp_ae2; df2[3] = Z_hdgp_saeibs2
df2[4] = Z_hdgp_pca[:, :4];  df2[5] = Z_hdgp_ae4; df2[6] = Z_hdgp_saeibs4
df2[7] = Z_hdgp_pca[:, :8]; df2[8] = Z_hdgp_ae8; df2[9] = Z_hdgp_saeibs8
df2[10] = Z_hdgp_pca[:, :12];  df2[11] = Z_hdgp_ae12; df2[12] = Z_hdgp_saeibs12

title = [' model for 2 PC', ' model for AE 2 latent axes', ' model for SAEIBS 2 latent axes',
         ' model for 4 PC', ' model for AE 4 latent axes', ' model for SAEIBS 4 latent axes',
         ' model for 8 PC', ' model for AE 8 latent axes', ' model for SAEIBS 8 latent axes',
         ' model for 12 PC', ' model for AE 12 latent axes', ' model for SAEIBS 12 latent axes']

filename = ['_2PC', '_2AE', '_2SAEIBS',
            '_4PC', '_4AE', '_4SAEIBS',
            '_8PC', '_8AE', '_8SAEIBS',
            '_12PC', '_12AE', '_12SAEIBS']


def plot_score(score, scorename):
    score_m = np.transpose(score.reshape((4, 3)))
    x = [2, 4, 8, 12]
    y1 = score_m[0, :]
    y2 = score_m[1, :]
    y3 = score_m[2, :]
    plt.figure(figsize=(6, 6))
    plt.plot(x, y1, 'tab:red',  label="PCA", linewidth=2)
    plt.plot(x, y2, 'tab:purple', label="AE", linewidth=2)
    plt.plot(x, y3, 'tab:orange', label="SAE-IBS", linewidth=2)
    plt.legend(fontsize=10, frameon=False,loc='center left', bbox_to_anchor=(0.75, 0.7))
    plt.xlabel('Dimensionality of Latent Space', fontsize=12)
    plt.ylabel(scorename, fontsize=12)
    plt.title('C) KNN Classification', x=0.1, y=1)
    plt.tight_layout()
    plt.savefig(savepath + 'KNN_' + scorename + '_SUPPOP.png', format='png', bbox_inches='tight', dpi=1200)
    plt.show()



score_ls = np.zeros([1, len(df)])
for (i, t, f) in zip(df, title, filename):
    score = KNN_PredictTarget(df[i], df2[i], 'KNN'+ title[i - 1], 'KNN' + filename[i - 1],
                                     3, suplabels_1kg_true, suplabels_hdgp_true)
    score_ls[0, i - 1] = score

# make table
score_tab = np.transpose(score_ls.reshape((4, 3)))
print('Classification Accuracy', score_tab)
plot_score(score_ls, 'Classification Accuracy')





