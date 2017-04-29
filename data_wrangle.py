# coding: utf-8
import pandas as pd
"""Titanic データの前処理をする

同じディレクトリに train.csv, test.csv を置いて使う。

Available functions:
    wrangle_data - データを加工する。返り値は train_df, test_df
                   (pandas のデータフレーム)
"""

def wrangle_data ():
    # Preprocess training data
    train_df = pd.read_csv ('train.csv')
    test_df = pd.read_csv ('test.csv')
    
    train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # train_df
    title_list = train_df['Title'].sort_values (inplace=False).unique ()
    pclass_list = train_df['Pclass'].sort_values (inplace=False).unique ()
    for title in title_list:
        for pclass in pclass_list:
            guess_df = train_df[(train_df['Title'] == title) & (train_df['Pclass'] == pclass)]['Age'].dropna ()
            train_df.loc[(train_df['Age'].isnull()) & (train_df['Title'] == title) & (train_df['Pclass'] == pclass), 'Age'] = guess_df.median ()
            
    # test_df
    title_list_t = test_df['Title'].sort_values (inplace=False).unique ()
    pclass_list_t = test_df['Pclass'].sort_values (inplace=False).unique ()
    for title in title_list_t:
        for pclass in pclass_list_t:
            guess_df = test_df[(test_df['Title'] == title) & (test_df['Pclass'] == pclass)]['Age'].dropna ()
            test_df.loc[(test_df['Age'].isnull()) & (test_df['Title'] == title) & (test_df['Pclass'] == pclass), 'Age'] = guess_df.median ()
            
    # test_df 'Title' == 'Ms' の補完
    guess_df = train_df[train_df['Title'] == 'Ms']['Age'].dropna ()
    test_df.loc[test_df['Age'].isnull(), 'Age'] = guess_df.median ()
       
    # train_df 'Embarked' の補完
    freq_port = train_df.Embarked.dropna().mode()[0]
    train_df['Embarked'] = train_df['Embarked'].fillna (freq_port)
    
    # test_df 'Fare' の補完
    guess_df = test_df[test_df['Pclass'] == 3].dropna()
    test_df.loc[test_df['Fare'].isnull(), 'Fare'] = guess_df['Fare'].median()
    
    df_list = [train_df, test_df]
    for df in df_list:
        df.loc[df['Age'] <= 8, 'Age'] = 0
        df.loc[(df['Age'] > 8) & (df['Age'] <= 16), 'Age'] = 1
        df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 2
        df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 3
        df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 4
        df.loc[(df['Age'] > 64) & (df['Age'] <= 80), 'Age'] = 5
        df['Fellow'] = df['SibSp'] + df['Parch']
        df['IsAlone'] = 0
        df.loc[df['Fellow'] == 0, 'IsAlone'] = 1
        df['Sex'] = df['Sex'].map ({'female': 0, 'male': 1}).astype (int)
        df['Embarked'] = df['Embarked'].map ({'C': 0, 'Q': 1, 'S': 2}).astype (int)
        df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
        df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
        df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
        df.loc[df['Fare'] > 31, 'Fare'] = 3
        df['Fare'] = df['Fare'].astype(int)
        
    train_df = train_df.drop (['PassengerId', 'Ticket', 'Cabin', 'Name', 'Title', 'SibSp', 'Parch', 'Fellow'], axis=1)
    test_df = test_df.drop (['Ticket', 'Cabin', 'Name', 'Title', 'SibSp', 'Parch', 'Fellow'], axis=1)
    
    return train_df, test_df

if __name__ == '__main__':
    
    train_df, test_df = wrangle_data ()
    
    print (train_df.describe())
    print (test_df.describe())
