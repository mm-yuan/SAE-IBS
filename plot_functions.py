

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plot_loss(train_loss, val_loss, path, title=None):
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if title is not None:
        plt.savefig(path + '/loss' + title + '.jpg')
    else:
        plt.savefig(path + '/loss.jpg')
    plt.show()


def plot_latent(latent, cdict, group, figure_title, figure_name, path, run_parameters):
    ncol = latent.shape[1]
    if ncol == 2:
        fig, ax = plt.subplots()
        for g in np.unique(group):
            ix = np.where(group == g)[0].astype(int)
            ax.scatter(latent[ix, 0], latent[ix, 1], c=cdict[g], label=g, s=2)
        ax.legend()
    else:
        w, h = plt.figaspect(0.2)
        fig, ax = plt.subplots(1, int(ncol / 2), figsize=(w, h))
        for i, j in zip(range(0, ncol, 2), range(0, int(ncol / 2))):
            for g in np.unique(group):
                ix = np.where(group == g)
                ax[j].scatter(latent[ix, i], latent[ix, i + 1], c=cdict[g], label=g, s=2)
            #ax[j].legend()
            ax[j].set_xlabel('Axis' + str(i + 1), fontsize=10)
            ax[j].set_ylabel('Axis' + str(i + 2), fontsize=10)
    fig.suptitle("Latent Space \n (" + figure_title + "_" + run_parameters + ")", fontsize=12)
    plt.tight_layout()
    plt.savefig(path + '/Latent' + figure_name + "_" + run_parameters + '.jpg')
    plt.show()


def plot_rmsd(data, fname, path, title):
    plt.figure(figsize=(8, 6))
    c1 = "tab:red"
    box1 = plt.boxplot(data[:, (0, 1)], positions=[1, 6], patch_artist=True,
                       boxprops=dict(facecolor=c1, color=c1),
                       capprops=dict(color=c1),
                       whiskerprops=dict(color=c1),
                       flierprops=dict(color=c1, markeredgecolor=c1),
                       medianprops=dict(color=c1), showfliers=False, widths=0.3,
                       )
    c2 = "tab:green"
    box2 = plt.boxplot(data[:, (2, 3)], positions=[1.5, 6.5], patch_artist=True, showfliers=False, widths=0.3)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box2[item], color=c2)
    plt.setp(box2["boxes"], facecolor=c2)
    plt.setp(box2["fliers"], markeredgecolor=c2)
    c3 = "tab:blue"
    box3 = plt.boxplot(data[:, (4, 5)], positions=[2, 7], patch_artist=True, showfliers=False, widths=0.3)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box3[item], color=c3)
    plt.setp(box3["boxes"], facecolor=c3)
    plt.setp(box3["fliers"], markeredgecolor=c3)
    c4 = "tab:purple"
    box4 = plt.boxplot(data[:, (6, 7)], positions=[2.5, 7.5], patch_artist=True, showfliers=False, widths=0.3)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box4[item], color=c4)
    plt.setp(box4["boxes"], facecolor=c4)
    plt.setp(box4["fliers"], markeredgecolor=c4)
    c5 = "tab:pink"
    box5 = plt.boxplot(data[:, (8, 9)], positions=[3, 8], patch_artist=True, showfliers=False, widths=0.3)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box5[item], color=c5)
    plt.setp(box5["boxes"], facecolor=c5)
    plt.setp(box5["fliers"], markeredgecolor=c5)
    c6 = "tab:brown"
    box6 = plt.boxplot(data[:, (10, 11)], positions=[3.5, 8.5], patch_artist=True, showfliers=False, widths=0.3)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box6[item], color=c6)
    plt.setp(box6["boxes"], facecolor=c6)
    plt.setp(box6["fliers"], markeredgecolor=c6)
    c7 = "tab:orange"
    box7 = plt.boxplot(data[:, (12, 13)], positions=[4, 9], patch_artist=True, showfliers=False, widths=0.3)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box7[item], color=c7)
    plt.setp(box7["boxes"], facecolor=c7)
    plt.setp(box7["fliers"], markeredgecolor=c7)
    c8 = "tab:olive"
    box8 = plt.boxplot(data[:, (14, 15)], positions=[4.5, 9.5], patch_artist=True, showfliers=False, widths=0.3)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box8[item], color=c8)
    plt.setp(box8["boxes"], facecolor=c8)
    plt.setp(box8["fliers"], markeredgecolor=c8)
    c9 = "tab:grey"
    box9 = plt.boxplot(data[:, (16, 17)], positions=[5, 10], patch_artist=True, showfliers=False, widths=0.3)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box9[item], color=c9)
    plt.setp(box9["boxes"], facecolor=c9)
    plt.setp(box9["fliers"], markeredgecolor=c9)

    plt.legend([box1["boxes"][0], box2["boxes"][0], box3["boxes"][0], box4["boxes"][0],
                box5["boxes"][0], box6["boxes"][0], box7["boxes"][0], box8["boxes"][0], box9["boxes"][0]],
               ['PCA', 'UPCA', 'SUGIBS', 'AE','DAE','DAE-L','SAE-IBS','D-SAE-IBS','D-SAE-IBS-L'], loc='upper right', fontsize=10, frameon=False)
    plt.xticks([3, 8], ['Axis1', 'Axis2'], fontsize=12)
    plt.ylabel('NRMSD', fontsize=12)
    plt.title(title, x=0.01, y=1)
    plt.tight_layout()
    plt.savefig(path + fname + '_RMSD.png', format='png', bbox_inches='tight', dpi=1200)
    plt.show()


def legend_without_duplicate_labels(ax,location):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    order = [0, 2, 3,  1]
    tmp = [unique[i] for i in order]
    lgnd = ax.legend(*zip(*tmp),fontsize=14, frameon=False, loc=location)
    for handle in lgnd.legendHandles:
        handle.set_sizes([10])


def plot_z(latent, cdict, group, figure_name, title, path, location, legend=None, w=None, h=None, nrow=None, fig_col=None, outlier=None, leg_out=None):
    ncol = latent.shape[1]
    if ncol == 2:
        fig, ax = plt.subplots(figsize=(w, h), gridspec_kw={'width_ratios':[1],'height_ratios':[1]})
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        for g in np.unique(group):
            ix = np.where(group == g)[0].astype(int)
            ax.scatter(latent[ix, 0], latent[ix, 1], c=cdict[g], label=g, s=0.1) #s=1
        if legend == 'TRUE':
            if leg_out is not None:
                ax.legend(fontsize=10, frameon=False, loc=location, bbox_to_anchor=leg_out,markerscale=10)
            else:
                ax.legend(fontsize=14, frameon=False, loc=location)
        ax.set_xlabel('- Axis1', fontsize=12)
        ax.set_ylabel('Axis2', fontsize=12)
    else:
        fig, ax = plt.subplots(nrow, fig_col, figsize=(w, h), gridspec_kw={'width_ratios':[1]*int(fig_col),'height_ratios':[1]*int(nrow)})
        row_list = np.repeat(list(range(0, nrow, 1)), fig_col)
        col_list = list(range(fig_col)) * nrow
        for i, row, col in zip(range(0, ncol, 2), row_list, col_list):
            # for g in np.unique(group):
            for g in list(reversed(np.unique(group))): #--> plot PCA relative study
                ix = np.where(group == g)[0].astype(int)
                if nrow == 1:
                    ax[col].scatter(latent[ix, i], latent[ix, i + 1], c=cdict[g], label=g, s=0.1)
                    if outlier is not None:
                        ax[col].scatter(latent[outlier, i], latent[outlier, i + 1], c='magenta', label='REL', s=2)
                    ax[col].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax[col].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                else:
                    ax[row, col].scatter(latent[ix, i], latent[ix, i + 1], c=cdict[g], label=g, s=0.1)
                    if outlier is not None:
                        ax[row, col].scatter(latent[outlier, i], latent[outlier, i + 1], c='magenta', label='REL', s=2)
                    ax[row, col].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    ax[row, col].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if nrow != 1:
                ax[row, col].set_xlabel('Axis' + str(i + 1), fontsize=12)
                ax[row, col].set_ylabel('Axis' + str(i + 2), fontsize=12)
            else:
                ax[col].set_xlabel('Axis' + str(i + 1), fontsize=12)
                ax[col].set_ylabel('Axis' + str(i + 2), fontsize=12)
            if legend == 'TRUE':
                if (nrow != 1 and row == row_list[-1] and col == col_list[-1]):
                    if leg_out is not None:
                        ax[row, col].legend(fontsize=14, frameon=False, loc=location, bbox_to_anchor=leg_out,markerscale=10)
                    else:
                        #ax[row, col].legend(fontsize=14, frameon=False, loc=location,markerscale=10)
                        legend_without_duplicate_labels(ax[row, col], location)
                if (outlier is not None and nrow == 1 and row == row_list[-1] and col == col_list[-1]):
                    #ax[col].legend(fontsize=10, frameon=False, loc=location)
                    legend_without_duplicate_labels(ax[col],location)
                if (outlier is None and nrow == 1 and row == row_list[-1] and col == col_list[-1]):
                    ax[col].legend(fontsize=14, frameon=False, loc=location, bbox_to_anchor=leg_out,markerscale=10)
    plt.tight_layout()
    fig.suptitle(title, x=0, y=1, ha='left')
    plt.savefig(path + '/Latent' + figure_name + '.png', format='png', bbox_inches='tight', dpi=1200)
    plt.show()



