import pandas as pd
import numpy as np
import math
import json
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
% matplotlib inline

def load_data():
    # read in the json files
    '''
    INPUT
    None
    
    OUTPUT
    portfolio, profile, transcript - 3 dataframe including portfolio, profile and transcript information
    '''

    portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
    profile = pd.read_json('data/profile.json', orient='records', lines=True)
    transcript = pd.read_json('data/transcript.json', orient='records', lines=True)
    
    portfolio.columns = ['channels', 'difficulty', 'duration', 'offer_id', 'offer_type', 'reward']
    profile.columns = ['cus_age', 'cus_became_member_on', 'cus_gender', 'cus_id', 'cus_income']
    transcript.columns = ['event', 'cus_id', 'hours_since_start', 'value']
    
    return portfolio, profile, transcript

portfolio, profile, transcript = load_data()

def check_NA():
    
    '''
    INPUT
    None
    
    OUTPUT
    3 String sentences showing if there's any missing value in 3 datasets
    '''
    print("There's {} Null Value in portfolio".format(portfolio.isna().sum().sum()))
    print("There's {} Null Value in profile".format(profile.isna().sum().sum()))
    print("There's {} Null Value in transcript".format(transcript.isna().sum().sum()))

def id_mapper(id_col):
    '''
    INPUT
    id_col - pandas series including id numbers
    
    OUTPUT
    id_encoded - pandas series including changed id numbers
    '''
    coded_dict = dict()
    cter = 1
    id_encoded = []
    
    for val in id_col:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        id_encoded.append(coded_dict[val])
    return id_encoded


def Combine_df(portfolio=portfolio, profile=profile, transcript=transcript):
    '''
    INPUT
    portfolio, profile, transcript - 3 datasets that need to be combined
    
    OUTPUT
    df - 1 result combined dataframe
    '''
    transcript['transaction_amount'] = transcript.value.apply(lambda x: (x['amount']) if 'amount' in x.keys() else np.nan)
    transcript['offer_id'] = transcript.value.apply(lambda x: x['offer id'] if 'offer id' in x.keys() else \
                                               x['offer_id'] if 'offer_id' in x.keys() else np.nan)
    transcript['reward_amount'] = transcript.value.apply(lambda x: (x['reward']) if 'reward' in x.keys() else np.nan)
    transcript = transcript.drop(['value'], axis=1)
    
    df = transcript.merge(profile, how='left', on='cus_id').merge(portfolio, how='left', on='offer_id')
    
    cus_id = id_mapper(df.cus_id)
    offer_id = ['off'+str(i) for i in id_mapper(df.offer_id[df.offer_id.isna()==False])]
    df.cus_id = cus_id
    df.offer_id[df.offer_id.isna()==False] = offer_id
    df = pd.concat([df, pd.get_dummies(df.event)], axis=1)
    
    return df


df = Combine_df()

def people_not_targeted(df):
    '''
    INPUT
    df - dataframe result from Combine_df()
    
    OUTPUT
    cus_list - A list including customer ids that are teased out based on criteria
    '''
    cus_lst = np.intersect1d(df[df['event']=='offer completed'].cus_id.unique(), df[df['event']=='transaction'].\
                         cus_id.unique(), assume_unique=True)
    df1 = df[df['cus_id'].isin(cus_lst)==True].groupby('cus_id').agg({'offer completed':'sum', 'offer received': \
                                                'sum', 'offer viewed': 'sum', 'transaction': 'sum'}).reset_index()
    df1['view:received'] = df1['offer viewed']/df1['offer received']
    df1['complete:trans'] = df1['offer completed']/df1['transaction']
    df1['offer_interact'] = df1['view:received']*df1['complete:trans']
    cus_list = df1[(df1['offer_interact']-df1['offer_interact'].mean())/(df1['offer_interact']).std()<-1].cus_id.tolist()
    
    return cus_list


def generate_training_data(df):
    '''
    INPUT
    df - Combined dataframe from Combine_df()
    
    OUTPUT
    df_result - training datasets for K-means Model
    '''

    cus_list = people_not_targeted(df)
    df_selected = df[df['cus_id'].isin(cus_list)==False].reset_index(drop=True)
    df_selected.loc[df_selected['offer_type'].isnull()==True, 'offer_type']=''
    df_selected['offer_event'] = df_selected['offer_type']+' '+df_selected['event']
    df_selected = pd.concat([df_selected, pd.get_dummies(df_selected['offer_event'])], axis=1)
    df_selected = pd.concat([df_selected.cus_id, df_selected[df_selected.columns[-9:]]], axis=1)
    df_selected = df_selected.groupby('cus_id').sum()
    
    df_bogo = df_selected[df_selected.columns[1:4]].div(df_selected[df_selected.columns[1:4]].sum(axis=1), axis=0).\
        reset_index().fillna(0)
    df_discount = df_selected[df_selected.columns[4:7]].div(df_selected[df_selected.columns[4:7]].sum(axis=1), axis=0).\
            fillna(0).reset_index(drop=True)
    df_info = df_selected[df_selected.columns[7:9]].div(df_selected[df_selected.columns[7:9]].sum(axis=1), axis=0).\
            fillna(0).reset_index(drop=True)
    df_result = pd.concat([df_bogo, df_discount, df_info], axis=1)
    
    #X = df_result[df_result.columns[1:]]
    
    return df_result


def get_best_k(k_max=15):
    '''
    INPUT
    k_max - the maximum k to be included in analysis
    
    OUTPUT
    plot - a graph generated by matplotlib package
    '''
    df_result = generate_training_data()
    X = df_result[df_result.columns[1:]]
    wcss = []
    for i in range(1, k_max):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, k_max), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


def get_best_group(k, offer_type):
    '''
    INPUT
    k - integer used for K-means 
    offer_type - a string representing offer types including 'informational', 'discount' or 'bogo'
    
    OUTPUT
    best_group - a dataframe including the best group profile information
    '''

    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    df_result = generate_training_data(df)
    X = df_result[df_result.columns[1:]]
    pred_y = kmeans.fit_predict(X)
    df_result['cluster'] = pred_y
    X['cluster'] = pred_y
    
    if offer_type == 'informational':
        cluster_num = X.groupby('cluster').mean().sort_values(by=offer_type+" offer viewed", ascending=False).index[0]
    elif offer_type in ['discount', 'bogo']:
        cluster_num = X.groupby('cluster').mean().sort_values(by=offer_type+" offer completed", ascending=False).index[0]
    else:
        print("The offer type is not accepted")
    
    pro_cus_id=id_mapper(profile.cus_id)
    profile.cus_id = pro_cus_id
    
    df_cluster = profile.merge(df_result[['cus_id', 'cluster']], how='left', on='cus_id')
    best_group = df_cluster[df_cluster['cluster']==cluster_num]
    
    return best_group