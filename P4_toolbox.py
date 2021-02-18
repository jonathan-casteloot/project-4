import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns

def loss_ratio(vector, dataset , main_dataframe, min_value=0, max_value=0):
    
    if max_value == 0:
        max_value = dataset[vector].max()
        
    if min_value == 0:
        min_value = dataset[vector].min()

    # ratio with adjustables limits
    intra = dataset[dataset[vector] < min_value].shape[0] + dataset[dataset[vector] > max_value].shape[0]
    inter = dataset[dataset[vector] < min_value].shape[0] + dataset[dataset[vector] > max_value].shape[0]

    print("% intra loss  ", round(intra / dataset[vector].shape[0],4))
    print("% global loss ", round(inter / main_dataframe[vector].shape[0],4))

    
def drop_values(vector, dataset, main_dataframe, min_value, max_value):
    
    row_indexer = dataset[dataset[vector] >= max_value].index
    main_dataframe.drop(index=row_indexer, inplace=True)
    
    row_indexer = dataset[dataset[vector] <= min_value].index
    main_dataframe.drop(index=row_indexer, inplace=True)

    
def isolation_forest_min_max_values(vector, dataset, contamination):
    # model selection
    from sklearn.ensemble import IsolationForest
    
    # feature set
    X = np.array(dataset[vector]).reshape(-1,1)
    
    # model train
    model = IsolationForest(contamination=contamination, random_state=0)
    model.fit(X)
    
    # model prediction
    prediction = model.predict(X)
    
    return [X[prediction == 1].min(),X[prediction == 1].max()]


def kmeans_graph(feature_vector, target_vector, dataset, n_clusters , zoom=0):
    # model selection
    from sklearn.cluster import KMeans

    # X = selected Feature, y = Target
    X = np.array(dataset[feature_vector]).reshape(-1,1)
    y = np.array(dataset[target_vector]).reshape(-1,1)

    # model train
    model = KMeans(n_clusters=n_clusters, random_state=0)
    model.fit(X,y)

    # model prediction
    prediction = model.predict(X)

    # visual result
    plt.figure(figsize=(20,10),dpi=200)
    
    # zoom feature
    if zoom != 0:
        plt.axis(xmax=zoom)
        
    plt.scatter(X, y, c=prediction);

    
def elbow_method_graph(feature_vector, target_vector, dataset, max_n_clusters):
    # model selection
    from sklearn.cluster import KMeans
    
    array = np.array(dataset[[feature_vector,target_vector]])
    
    # inertia = cost function
    inertia = []

    # benchmark
    k_range = range(1,max_n_clusters)
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=0).fit(array)
        inertia.append(model.inertia_)
    
    # visual result
    plt.figure(dpi=100)
    plt.plot(k_range, inertia)
    plt.xlabel('Clusters number')
    plt.ylabel('Inertia cost')

   
def shapiro_ratio_categ_global(vector_name, target_vector, dataset):
    from scipy import stats

    for category_name in dataset[vector_name].unique():
        categ = dataset[dataset[vector_name] == category_name].copy()
        print(category_name + ' ' + str(round(stats.shapiro(categ[target_vector])[0],2)))

    print('global ' + str(round(stats.shapiro(dataset[target_vector])[0],2)))
    
def eta2(vector_quant, vector_quali, data):
    
    # function body
    from sklearn.feature_selection import f_classif

    # features set
    dataset = data[[vector_quant, vector_quali]].copy()

    # to set up n_samples, n_features
    dataset = one_hot_matrix(vector_quali, dataset)

    n_samples_n_features = np.array(dataset)

    # reshape(-1,1) : from scalar (1d) to vector(2d)
    n_samples = np.array(dataset.iloc[:,0]).reshape(-1,1)

    # matrix columns
    results_matrix = pd.DataFrame((f_classif(n_samples_n_features, n_samples)))
    results_matrix.columns = dataset.columns
    results_matrix.drop(vector_quant, axis=1, inplace=True)
    results_matrix = results_matrix.T
    
    results_matrix.iloc[:,0] = round((results_matrix.iloc[:,0]/results_matrix.iloc[:,0].sum())*100)
    results_matrix.iloc[:,1] = round(results_matrix.iloc[:,1],3)
    
    results_matrix.columns = ['Fisher %','p-value']

    # heatmpap global settings
    plt.style.use('default')
    sns.set(font_scale=2.5)
    plt.figure(figsize=(5,5))
    
    # eta value title
    title = 'η2 : ' + str(eta_squared(data[vector_quali], data[vector_quant]))
    plt.suptitle(title, fontsize=30)
    
    sns.heatmap(results_matrix, annot=True, cbar=False)

    plt.show()


def one_hot_matrix(vector,data):
    
    dataset = data.copy()
    
    one_hot_matrix = pd.get_dummies(dataset[vector],dtype='bool')

    # to merge new features
    dataset = dataset.merge(one_hot_matrix,how ='left', left_index=True, right_index=True)

    # to drop vector
    dataset = dataset.drop(vector, axis=1)

    return dataset


def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
        
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    
    eta2 = round((SCE/SCT),2)
    
    if eta2==0:
        eta2=int(eta2)
    
    return eta2


def chi2(vector_x, vector_y, data):
    
    from sklearn.feature_selection import chi2
    
    contingency_table = data[[vector_x, vector_y]].copy()
    
    columns_names     = one_hot_matrix(vector_y, contingency_table).columns[1:]
    
    contingency_table = one_hot_matrix(vector_y, contingency_table)
    contingency_table = contingency_table.groupby(vector_x).sum()

    # matrix columns
    results_matrix = pd.DataFrame(chi2(contingency_table, contingency_table.index))
    results_matrix.columns = columns_names
    results_matrix = results_matrix.T
    
    results_matrix.iloc[:,0] = round((results_matrix.iloc[:,0]/results_matrix.iloc[:,0].sum())*100)
    results_matrix.iloc[:,1] = round(results_matrix.iloc[:,1],3)
    
    results_matrix.columns = ['Chi-2 %','p-value']

    # heatmpap global settings
    plt.style.use('default')
    sns.set(font_scale=2.5)
    plt.figure(figsize=(5,5))
    
    # xi_n value title
    title = 'ξn : ' + str(xi_n(vector_x, vector_y, data))
    plt.suptitle(title, fontsize=30)
    sns.heatmap(results_matrix, annot=True, cbar=False)

    plt.show()


def xi_n(X, y, data):

    # to create contingency table
    contingency_table = data[[X,y]].pivot_table(index=X,
                                                columns=y,
                                                aggfunc=len,
                                                margins=True,
                                                margins_name="Total",
                                                fill_value=0)


    # to create independence table
    tx = contingency_table.loc[:,["Total"]]
    ty = contingency_table.loc[["Total"],:]
    n = len(data)
    independence_table = tx.dot(ty) / n

    # to compute xi_n
    contingency_table = contingency_table.iloc[0:4,0:4]
    independence_table = independence_table.iloc[0:4,0:4]

    measure = (contingency_table-independence_table)**2/independence_table
    xi_n = int(measure.sum().sum().round())
    
    return xi_n


def lorenz_gini_graph_dataframe(vector_name, data):
    import quantecon as qe 

    # X = selected Feature
    X = np.array(data[vector_name])

    # to compute lorenz curve
    cum_frequences, cum_weights = qe.lorenz_curve(X)
    gini = qe.gini_coefficient(X)

    # visual result
    plt.style.use('default')
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(5,5),dpi=100)
    plt.axis(xmin=0, xmax=1 ,ymin=0, ymax=1)
    plt.title('Gini : ' + str(round(gini,2)))

    # to plot the equality line
    plt.plot([0, 1], lw=2, c='gray', ls='dotted', label='Equality')

    # to plot the Lorenz curve
    plt.plot(cum_frequences, cum_weights, lw=2, c='r', label='Lorenz')
    #plt.savefig('lorenz_gini.png')
    plt.legend(frameon=False);
    plt.show()

    # result into dataframe
    vectors = {vector_name     : np.sort(X),
               'cum_frequences': cum_frequences[1:],
               'cum_weights'   : cum_weights[1:]}

    dataset = pd.DataFrame(data=vectors)
    
    return dataset


def boxplot_stats_dataframe(vector_name, data):

    boxplot = pd.DataFrame(data=[])

    boxplot.loc[vector_name,'Q1'] = data[vector_name].quantile(0.25)
    
    boxplot.loc[vector_name,'Q3'] = data[vector_name].quantile(0.75)

    boxplot.loc[vector_name,'IQR'] = boxplot.loc[vector_name,'Q3'] - boxplot.loc[vector_name,'Q1']

    boxplot.loc[vector_name,'b_min'] = boxplot.loc[vector_name,'Q1'] - 1.5 * boxplot.loc[vector_name,'IQR']

    if boxplot.loc[vector_name,'b_min'] < 0:
        boxplot.loc[vector_name,'b_min'] = data[vector_name].min()
    
    boxplot.loc[vector_name,'b_max'] = boxplot.loc[vector_name,'Q3'] + 1.5 * boxplot.loc[vector_name,'IQR']

    return boxplot


def r2(target_vector, data):
    
    plt.figure(dpi=100)
    
    corr_vector = pd.DataFrame(((data.corr()**2).round(2)))[0:1]
    corr_vector = corr_vector.dropna(axis=1)
    one_level_column(corr_vector)

    sns.heatmap(corr_vector.iloc[:,1:], annot=True, cbar=False, square=True)

    plt.xlabel('')
    plt.ylabel('')
    plt.show();
    
    
def one_level_column(data):
    
    if data.columns.nlevels > 1 :
        names_col =[]
        for id_col in range(0,data.columns.shape[0]):
            name = (data.columns.get_level_values(0)[id_col] + '_' 
                  + data.columns.get_level_values(1)[id_col])
    
            names_col.append(name)

        data.columns = names_col
        

def pie_chart(lorenz_table, dataframe, color_list):

    data = dataframe.loc[lorenz_table.index]['categ'].value_counts(normalize=True).round(2).reset_index()
    data['index'] = data['index'].map({'c_1': 'catégorie 1', 'c_0':'catégorie 0', 'c_2':'catégorie 2'})
    
    plt.style.use('default')

    plt.pie(data['categ'], 
            labels=data['index'], 
            autopct=lambda x: str(int(round(x))) + ' %', 
            labeldistance=None, 
            wedgeprops={'edgecolor' : 'w', 'linewidth' : 2},
            textprops={'fontsize': 20, 'color':'w'},
            colors=color_list
           )

    plt.legend(loc='upper center', 
               fontsize=20, 
               bbox_to_anchor=(0.25, 0.75, 0.5, 0.5),
               frameon=False)

    plt.show()


def frequences_graph_stats(vector, data):

    frequences = pd.DataFrame(data[vector].value_counts(normalize=True).sort_values().reset_index())
    frequences = frequences.set_index(vector)
    sns.distplot(frequences);
    frequences.columns = [vector]

    print(frequences.describe().round(2))

    plt.figure()
    sns.boxplot(frequences[vector])
    plt.show()
    
def frequence_weight_ratios(lorenz_table, age):
    print((1 - lorenz_table[lorenz_table['age'] == age].iloc[0,1]).round(2))
    print((1 - lorenz_table[lorenz_table['age'] == age].iloc[0,2]).round(2))