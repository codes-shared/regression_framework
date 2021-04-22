#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: albaortega
"""
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random



def create_file (d, name_file):
    # crear clase ejecucion d
    timestamp =   (datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
    file = open(d['path_to_save']+name_file+'_'+d['num_ejecucion']+'.txt', "a+") 
    file.write('INIT: '+timestamp)
    return file

def open_file (path, file, mode="a+"):
    file = open(path+file, mode) 
    return file

def close_file (f):
    f.close()

def write_file (f, message):
    f.write(message)


def get_dummies (df, drop=False, nan_col = False, autom=False, ignored_cols=list(), columns=list()):
    """
    Can convert specified columns or search for object columns automatically.

    Parameters
    ----------
    df : Pandas.Dataframe()
        Dataset.
    drop : boolean
        To drop columns after transformation or not.
    nan_col : boolean
        To create a new columns with missing or not.
    autom : boolean
        Search objects columns automatically and transform.
    ignored_cols : list()
        Columns to ignore.
    columns : list()
        Defined columns list to transform in dummy.
    
    Returns
    -------
    Pandas.DataFrame()
        Pandas dataframe with with dummified columns.
    """
    l_drop = list()
    if autom:
        for i in df.columns:        
            if (i not in ignored_cols) & (df[i].dtype == 'object'):
                print('\nTransform <'+i+'> variable to dummy')
                
                l_drop.append(i)
                df = pd.concat([df, pd.get_dummies(df[i], prefix= i, prefix_sep='_', dummy_na=nan_col, drop_first=True)], axis=1)
        if drop:
            [print('\nRemoving <'+i+'> variable') for i in l_drop]
            df = df.drop(l_drop, axis=1)


    elif (columns == list()) & (autom == False):
        print('You must to introduce a list of columns to transform or activate autom param <autom = True>')
    else:
        for i in columns:        
            if (df[i].dtype == 'object'):
                print('\nTransform <'+i+'> variable to dummy')
                l_drop.append(i)
                df = pd.concat([df, pd.get_dummies(df[i], prefix= i, prefix_sep='_', dummy_na=True, drop_first=True)], axis=1)
        if drop:
            [print('\nRemoving <'+i+'> variable') for i in l_drop]
            df = df.drop(l_drop, axis=1)

    return df



def frequency_encoder (df, categorical_name, drop=False, new_col_name=None):
    """
    Pandas dataframe series frequency encoder in a new column.

    Parameters
    ----------
    df : Pandas.Dataframe()
        Dataset.
    categorical_name : string
        Name of the column to encoder.
    drop : boolean
        To drop column after encoding or not.
    new_col_name : string
        Name of the new column.

    Returns
    -------
    Pandas.DataFrame()
        Pandas dataframe with a frequency encoded column 

    """
    if new_col_name is None:
        new_col_name = 'freq_'+categorical_name
    g = df[categorical_name].value_counts()/df.shape[0]
    for i in g.index:
        df.loc[df[categorical_name] == i, new_col_name] = g.loc[g.index==i].values[0]
    if drop:
        df = df.drop(categorical_name, axis=1)
    return df


    
def missing_validation (df, ignored_cols=list()):

    for i in df.columns:
        if (i not in ignored_cols) & (df[i].isnull().sum() > 0):
            print('\nImpute values in <'+i+'> variable before continue')
            return -1
    return 1
    
def numeric_validation (df, ignored_cols=list()):

    for i in df.columns:
        if (i not in ignored_cols) & (df[i].dtype == 'object'):
            print('\nConvert values in <'+i+'> variable to numeric before continue')
            return -1
    return 1


def dataset_validation (df, ignored_cols=list()):
    """
    Missing and categorical values validation in a dataset.

    Parameters
    ----------
    df : Pandas.Dataframe()
        Dataset.
    ignored_cols : list()
        Columns to ignore.
    """
    if (missing_validation (df, ignored_cols) == 1) & (numeric_validation (df, ignored_cols) == 1):
        print('\nSuccessful validation')




def seed_optimization (df, target_name, verbose=True, prop=0.3, max_it=10, min_rand=0, max_rand=999, seed=''):
    """
    Seed optimization in dataset splitting with a continue target.

    Parameters
    ----------
    df : Pandas.Dataframe()
        Dataset.
    target_name : string
        Name of the target column.
    verbose : boolean
        To print traces.
    prop : float
        Train - test split ratio. 
    max_it : int
        Number of maximum iterations.
    min_rand : int
        Minimun value to obtain a random number seed.
    max_rand : int
        Maximum value to obtain a random number seed.
    seed : int
        Defined seed value

    Returns
    -------
    int
        Optimal seed

    """
    it=0
    while it < max_it:
        if it == 0:
            if '' == seed:
                seed = random.randint(min_rand, max_rand)
                x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != target_name], df[target_name], test_size=prop, random_state=seed)
                
            else:
                x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != target_name], df[target_name], test_size=prop, random_state=seed)
                max_it = 1

            
            m_train, s_train = np.mean(y_train), np.std(y_train) 
            m_test, s_test = np.mean(y_test), np.std(y_test)

            criterion_old = (0.6*abs(m_train - m_test))+(0.4*abs(s_train - s_test))
            old_seed = seed

            if verbose:
                print('Iter: '+str(it)+' | Seed: '+str(seed)+' | Criterion value: '+str(criterion_old))
                print('Train mean: '+str(m_train)+' -- Test mean: '+str(m_test))
                print('Train std: '+str(s_train)+' -- Test std: '+str(s_test)+'\n')

        else: 
            seed = random.randint(min_rand, max_rand)
            x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != target_name], df[target_name], test_size=prop, random_state=seed)
            
            m_train, s_train = np.mean(y_train), np.std(y_train) 
            m_test, s_test = np.mean(y_test), np.std(y_test)
            
            criterion = (0.6*abs(m_train - m_test))+(0.4*abs(s_train - s_test))

            if criterion < criterion_old: 
                criterion_old = criterion
                old_seed = seed

            if verbose:
                # cambiar salida
                print('Iter: '+str(it)+' | Seed: '+str(seed)+' | Criterion value: '+str(criterion)) 
                print('Train mean: '+str(m_train)+' -- Test mean: '+str(m_test))
                print('Train std: '+str(s_train)+' -- Test std: '+str(s_test)+'\n')
                
        it = it+1

    return old_seed


def split_dataset (df, target_name, prop=0.3, seed='', autom=False):
    """
    Split dataset by a seed.

    Parameters
    ----------
    df : Pandas.Dataframe()
        Dataset.
    target_name : string
        Name of the target column.
    prop : float
        Train - test split ratio. 
    seed : int
        Defined seed value.
    autom : boolean
        Dataset splitting with a random seed or not.

    Returns
    -------
    Pandas.Dataframe()
        train features dataset
    Pandas.Dataframe()
        test features dataset
    Pandas.Series()
        train target values
    Pandas.Series()
        test target values
    """
    if autom:
        seed = seed_optimization(df, target_name, verbose=False, prop=prop, max_it=1, min_rand=0, max_rand=999, seed=None)
        print('Automatic splitting - Seed: '+str(seed))
        x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != target_name], df[target_name], test_size=prop, random_state=seed)
    else:
        x_train, x_test, y_train, y_test = train_test_split(df.loc[:, df.columns != target_name], df[target_name], test_size=prop, random_state=seed)

    return x_train, x_test, y_train, y_test


