import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns

import sys
import os
home_directory_path = os.path.expanduser('~')
sys.path.append(home_directory_path +'/utils')

from prepare_utils import split_data


tm_batting = pd.read_csv('data/team_batting.csv')
tm_pitching = pd.read_csv('data/team_pitching.csv')


# Split data
tm_batting_train, _, _ = split_data(tm_batting,
                                    validate_size=.15,
                                    test_size=.15, 
                                    random_state=123)

tm_pitching_train = tm_pitching.loc[tm_batting_train.index]


def bin_wins(df=tm_batting_train):
    # Define the bin edges and labels
    bins = [0, 60, 70, 80, 90, 100, float('inf')]
    labels = ['60-', '60-69', '70-79', '80-89', '90-99', '100+']

    # Create the 'W_bins' column
    df['W_bins'] = pd.cut(df['W'], bins=bins, labels=labels, right=False)
    return df

tm_batting_train = bin_wins(tm_batting_train)
tm_pitching_train = bin_wins(tm_pitching_train)


def plot_key_corrs(df=tm_batting_train):
    fig, axes = plt.subplots(1, 2, figsize=(6, 6), gridspec_kw={'wspace': .4, 'width_ratios': [0.8, 1]})

    stats_pitching = ['WHIP', 'OPS', 'OBP', 'SLG', 'FIP', 'TB', 'H9', 'BA', 'H', 'SO/W',
       'BB9', 'BB', 'HardH%', '2B', 'BB%', 'SO%', 'HR9', 'HR', 'SO', 'PAge',
       'EV', 'SO9', 'GB%', 'GB/FB', 'SB', 'GO/AO', 'CS']

    sns.heatmap(tm_pitching_train.loc[:, stats_pitching+['W']].corr()['W'].to_frame().iloc[:-1,:],
                linewidths=.5, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, ax=axes[0], cbar=False)

    axes[0].set_title('Pitching Stat Correlations' + ' '*10, pad=10)
    axes[0].tick_params(axis='both', which='both', bottom=False, left=False)
        
    stats_batting = ['OPS+', 'TotA', 'rOBA', 'OPS', 'OBP', 'HardH%', 'SecA', 'SLG', 'HR',
     'HR%', 'BA', 'H', '2B', 'BatAge', 'GB/FB', 'Pull%',
     'GB%', 'Oppo%', 'SO%', 'SB%', 'FB%', 'Cent%', 'SO', 'CS', 'SB', 'LD%']

    sns.heatmap(tm_batting_train.loc[:, stats_batting+['W']].corr()['W'].to_frame().iloc[:-1,:],
                linewidths=.5, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, ax=axes[1])

    axes[1].set_title('Offensive Stat Correlations' + ' '*10, pad=10)
    axes[1].tick_params(axis='both', which='both', bottom=False, left=False)

    plt.show()


def plot_percent_medians(df=tm_batting_train):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 2.5))

    
    sns.barplot(data=df, x='W_bins', y='Pull%', errorbar=None,
                order= sorted(df['W_bins'].unique())[1:] + ['100+'],
                palette=['#0955AA'], ax=axes[0], estimator='median')
    axes[0].set_yticks(np.arange(0,31,10))

    
    sns.barplot(data=df, x='W_bins', y='Cent%', errorbar=None,
                order= sorted(df['W_bins'].unique())[1:] + ['100+'],
                palette=['#0955AA'], ax=axes[1], estimator='median')
    axes[1].set_yticks(np.arange(0,61,20))


    sns.barplot(data=df, x='W_bins', y='Oppo%', errorbar=None,
                order= sorted(df['W_bins'].unique())[1:] + ['100+'],
                palette=['#0955AA'], ax=axes[2], estimator='median')

    axes[2].set_yticks(np.arange(0, 21,5))

    axes[0].patches[-2].set_edgecolor('#C40000')
    axes[0].patches[-1].set_edgecolor('#C40000')
    axes[2].patches[-2].set_edgecolor('#C40000')
    axes[2].patches[-1].set_edgecolor('#C40000')

    plt.suptitle('Winning Teams Slightly Pull the Ball More')
    plt.tight_layout()
    sns.despine()
    
    for ax in axes:
        # make ticks smaller
        ax.tick_params(axis='both', labelsize=8)
        for p in ax.patches:
            # change x labels
            ax.set_xlabel('Wins')
            # annotate bars
            ax.annotate(f'{round(p.get_height(), 1)}', (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom',  fontsize=8)
    
    plt.show()
    
def plot_wins_vs_percentages(df=tm_batting_train):
    # Create the subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7.5, 2.5))

    # Scatterplot 1
    sns.scatterplot(data=tm_batting_train, x='Pull%', y='W', s=8, ax=axes[0])
    axes[0].set_xlabel('Pull%')
    axes[0].set_yticks(ticks=np.arange(50, 121, 10))
    axes[0].set_ylabel('Wins', rotation=0, labelpad=15)

    rectangle = patches.Rectangle((27, 97), 7, 15, linewidth=.5,
                                  edgecolor='#C40000', facecolor='none')
    axes[0].add_patch(rectangle)
    axes[0].add_patch(rectangle)


    # Scatterplot 2
    sns.scatterplot(data=tm_batting_train, x='Cent%', y='W', s=8, ax=axes[1])
    axes[1].set_xlabel('Cent%')
    axes[1].set_yticks(ticks=np.arange(50, 121, 10))
    axes[1].set_ylabel('')
    rectangle = patches.Rectangle((49, 97), 6.5, 14, linewidth=.5,
                                  edgecolor='#C40000', facecolor='none')
    axes[1].add_patch(rectangle)
    axes[1].add_patch(rectangle)


    # Scatterplot 3
    sns.scatterplot(data=tm_batting_train, x='Oppo%', y='W', s=8, ax=axes[2])
    axes[2].set_xlabel('Oppo%')
    axes[2].set_yticks(ticks=np.arange(50, 121, 10))
    axes[2].set_ylabel('')
    rectangle = patches.Rectangle((15.75, 97), 3.25, 15, linewidth=.5,
                                  edgecolor='#C40000', facecolor='none')
    axes[2].add_patch(rectangle)

    # Adjust the spacing between subplots
    plt.suptitle('Winning Teams\' Pull/Center/Oppo Percentages')
    plt.tight_layout()
    sns.despine()

    for ax in axes:
        # make ticks smaller
        ax.tick_params(axis='both', labelsize=8)

    # Show the plot
    plt.show()
    
     
def plot_W_by_OPS(df=tm_batting_train):
    sns.lmplot(data=df, x='OPS+', y='W', height=3,
               scatter_kws={'s': 2},
               line_kws={'linewidth': .5})

    plt.xticks(ticks=np.arange(70, 121, 10), fontsize=8)
    plt.xlabel('OPS+')
    plt.yticks(ticks=np.arange(50, 111, 10), fontsize=8)
    plt.ylabel('Wins', rotation=0, labelpad=15)


    plt.title('OPS+ leads to more wins', fontsize=10)

    plt.tick_params(axis='both', labelsize=7)

    sns.despine()
    plt.show()


def plot_median_OPS(df=tm_batting_train):
    plt.figure(figsize=(4, 2))

    sns.barplot(data=df, x='W_bins', y='OPS+', errorbar=None,
                order= sorted(df['W_bins'].unique())[1:] + ['100+'],
                palette=['#0955AA'], estimator='median')

    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{round(p.get_height(), 1)}', (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha='center', va='bottom')

    plt.ylabel('OPS+', rotation=0, labelpad=15)
    plt.xlabel('Wins')
    plt.tick_params(axis='both', labelsize=8)

    plt.title('Median OPS+', fontsize=10, pad=15)
    sns.despine()
    plt.show()
        
        
def plot_OPS_distributions(df=tm_batting_train):
    plt.figure(figsize=(4, 2.5))

    sns.swarmplot(data=df, x='W_bins', y='OPS+',
                  order= sorted(df['W_bins'].unique())[1:] + ['100+'],
                  size=2.5, color='#0955AA', legend=False)

    plt.xlabel('Wins')
    plt.ylabel('OPS+', rotation=0, labelpad=15)
    plt.tick_params(axis='both', labelsize=8)

    plt.title('Winning Teams Have Higher OPS\'s', fontsize=10)
    sns.despine()
    plt.show()







