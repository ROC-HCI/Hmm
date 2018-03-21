#!/usr/bin/env python

import matplotlib.pylab as plt
import pandas as pd

# Truth-Teller Training Score vs Number of Hidden States
def plot_truth_score(df):
    fig, ax = plt.subplots()
    labels = []
    for key, grp in df.groupby(['seed']):
        ax = grp.plot(ax=ax, kind='line', x='k', y='truthTrainScore')
        labels.append(key)
    lines, _ = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best',title='Random Seed')
    plt.ylabel('Log Sum Score on Truth-Teller Training Data')
    plt.xlabel('k (# Hidden States)')
    plt.title('Truth-Teller Training Score vs Number of Hidden States')
    plt.show()

# Bluffer Training Score vs Number of Hidden States
def plot_bluff_score(df):
    fig, ax = plt.subplots()
    labels = []
    for key, grp in df.groupby(['seed']):
        ax = grp.plot(ax=ax, kind='line', x='k', y='bluffTrainScore')
        labels.append(key)
    lines, _ = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best',title='Random Seed')
    plt.ylabel('Log Sum Score on Bluffer Training Data')
    plt.xlabel('k (# Hidden States)')
    plt.title('Bluffer Training Score vs Number of Hidden States')
    plt.show()

# Classification Accuracy vs Number of Hidden States
def plot_correct(df):
    fig, ax = plt.subplots()
    plt.ylim([0,30])
    labels = []
    for key, grp in df.groupby(['seed']):
        ax = grp.plot(ax=ax, kind='line', x='k', y='correct')
        labels.append(key)
    lines, _ = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc='best',title='Random Seed')
    plt.ylabel('Correctly Classified as Truther/Bluffer (Out of 30)')
    plt.xlabel('k (# Hidden States)')
    plt.title('Classification Accuracy vs Number of Hidden States')
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv('results.csv', skipinitialspace=True)
    df.sort_values('k', inplace=True)
    
    plot_truth_score(df)
    plot_bluff_score(df)
    plot_correct(df)
    

