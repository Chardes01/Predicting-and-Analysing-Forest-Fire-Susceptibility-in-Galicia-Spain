import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_curve,roc_auc_score,auc,precision_recall_curve, auc,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from matplotlib.colors import ListedColormap
import geopandas as gpd
from spatialkfold.clusters import spatial_kfold_clusters
from shapely.geometry import Point 
import ast
import geopandas as gpd
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
import numpy as np
from sklearn.utils import shuffle
from imblearn.ensemble import EasyEnsembleClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from imblearn.metrics import specificity_score
import warnings

'''
This file entails all utils for the result file to prepare the data and provide visualisation tools.

'''



def average_ndvi(ndvi_values): 
    if type(ndvi_values) == int:
        print(ndvi_values)
        return False
    average = sum(ndvi_values)/len(ndvi_values)
    slope,offset = 0.004,-0.08
    physical_value = average * slope + offset

    return physical_value


def prepare_data(df,label,conf=False):
    df['label'] = label
    df['ndvi'] = df.apply(lambda row: (average_ndvi(row['ndvi'])), axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df_resign_dir = df[df['dir'] == 99]
    df_resign_dir.loc[:,'dir'] = -1
    df.update(df_resign_dir)
    if label == 1:
        df_val_confidence = df[df['confidence'] == 'l']
        df = df[df['confidence'] != 'l']
        df= df[df['forest_type'].apply(lambda x: x in [20,21,22,23,24,25,26,27,28,29])]
        if conf:
            return df,df_val_confidence
    return df

def get_x_y(df_true,df_neg,sample_neg=True,sample_num=None):
    df_true['label'] = 1
    df_neg['label'] = 0
    if sample_num:
        df_true_prep = df_true.sample(sample_num)
    df_true_prep = prepare_data(df_true,1)
    df_neg = prepare_data(df_neg,0)
    if sample_neg:
        len_true = len(df_true_prep)
        df_neg = df_neg.sample(len_true)
    df_together = pd.concat([df_true_prep,df_neg],axis=0,ignore_index=True)
    #df_id = df_together['id']
    df_y = df_together['label']
    df_x = df_together.drop(columns='label')
    return df_x,df_y


def resign_id(df_true,df_neg):
    ids = range(len(df_true) + len(df_neg))
    df_true.loc[:,'id'] = ids[:len(df_true)]
    df_neg.loc[:,'id'] = ids[len(df_true):]
    return df_true,df_neg

def sample_len_df(df):
    df_true = df[df['label'] == 1]
    len_true = len(df_true)
    df_neg = df[df['label'] == 0]
    df_neg = df_neg.sample(len_true)
    df = pd.merge(df_true,df_neg,how='outer')
    return df


def get_spatial_k_folds(df,k):
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf['id_gdf'] = range(len(gdf))
    gdf_clusters = spatial_kfold_clusters(
        gdf=gdf,
        name='id_gdf',
        nfolds=k,
        algorithm='kmeans',
        n_init="auto",
        random_state=42
    )
    cols_tab = cm.get_cmap('tab20', 10)
    cols = [cols_tab(i) for i in range(10)]
    color_ramp = ListedColormap(cols)


    fig, ax = plt.subplots(1,1 , figsize=(9, 4)) 
    gdf_clusters.plot(column='folds', ax=ax, cmap= color_ramp, markersize = 2, legend=True)
    ax.set_title('Spatially Clustered Folds\nUsing KMeans')
    plt.show()
    return gdf_clusters

def add_folds(df_folds,df):
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    existing_coords = np.radians(df_folds[['lat', 'lon']].values)
    existing_folds = df_folds['folds'].values
    tree = BallTree(existing_coords, metric='haversine')
    def assign_to_fold(gdf):
        new_coords = np.radians(gdf[['lat', 'lon']].values)
        distances, indices = tree.query(new_coords, k=1)
        assigned_folds = existing_folds[indices.flatten()]
        return assigned_folds
    gdf['folds'] = assign_to_fold(gdf)
    return gdf

def visualise_folds(df):
    cols_tab = cm.get_cmap('tab20', 10)
    cols = [cols_tab(i) for i in range(10)]
    color_ramp = ListedColormap(cols)
    fig, ax = plt.subplots(1,1 , figsize=(9, 4)) 
    df.plot(column='folds', ax=ax, cmap= color_ramp, markersize = 2, legend=True)
    ax.set_title('Spatially Clustered Folds\nUsing KMeans')
    plt.show()



def evaluate_all_models_metrics(models,df_test,names,features):
    warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
    fig, ax = plt.subplots(3, 3,figsize=(18, 16)) 
    for i,model in enumerate(models):
        y_pred_proba = model.predict_proba(df_test[features])
        y_true = df_test['label']
        y_scores = y_pred_proba[:,1] 
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        specificity = 1 - fpr

        idx = np.argmin(np.abs(tpr - specificity))
        optimal_threshold = thresholds[idx]
        y_pred_optimal = (y_scores > optimal_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_optimal)

        ax[2][i].plot(thresholds, tpr, label="Sensitivity (Recall)", color="blue")
        ax[2][i].plot(thresholds, specificity, label="Specificity", color="green")
        ax[2][i].axvline(optimal_threshold, color='red', linestyle='--', label=f"Sensitivity==Specificity: {optimal_threshold:.2f}")
        ax[2][i].set_xlabel("Threshold")
        ax[2][i].set_ylabel("Rate")
        ax[2][i].legend(loc="upper right")
        
        inset_ax = inset_axes(ax[2][i], width="30%", height="30%", loc="lower right", bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax[2][i].transAxes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=inset_ax, colorbar=False)
        inset_ax.set_title("Confusion Matrix", fontsize=10)

        plt.tight_layout()
        
    
        roc_auc = auc(fpr, tpr)
        y_probs_neg = y_pred_proba[:,0]
        fnr, tnr, _ = roc_curve(1- y_true,y_probs_neg)

        ax[1][i].plot(fpr, tpr, label=f'ROC curve: Positive (AUC= {roc_auc:.2f}')
        ax[1][i].plot(fnr, tnr, label=f'ROC curve: Negative')
        ax[1][i].plot([0, 1], [0, 1], 'k--', label='Random Guessing')
        ax[1][i].set_xlabel('False Positive/Negative Rate')
        ax[1][i].set_ylabel('True Positive/Negative Rate')
        ax[1][i].legend()
        plt.tight_layout()
    

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        specificity_scores = []
        f1_scores = []
        thresholds = np.arange(0,1,0.01)
        for threshold in thresholds:
            y_prob_bin = (y_scores > threshold).astype(int)
            accuracy_scores.append(accuracy_score(y_true,y_prob_bin))
            precision_scores.append(precision_score(y_true,y_prob_bin,zero_division=1))
            recall_scores.append(recall_score(y_true,y_prob_bin))
            specificity_scores.append(specificity_score(y_true,y_prob_bin))
            f1_scores.append(f1_score(y_true,y_prob_bin))
        ax[0][i].plot(thresholds,accuracy_scores,label=f'Accuracy')
        ax[0][i].plot(thresholds,recall_scores,label='Recall')
        ax[0][i].plot(thresholds,precision_scores,label=f'Precision')
        ax[0][i].plot(thresholds,f1_scores,label=f'F1-Score: Max at {round(thresholds[np.argmax(f1_scores)],2)} with {round(max(f1_scores),2)}')            
        ax[0][i].set_xlabel('Threshold')
        ax[0][i].set_ylabel('Rate')
        ax[0][i].legend()
        ax[0][i].set_title(names[i])
        plt.tight_layout()

    plt.show()

def cluster_dataset():   
    df_true = pd.read_csv('Daten/Dataset/complete_true_samples/veg_season_updated_weather.csv', converters={
        'ndvi': ast.literal_eval,
    })

    df_neg = pd.read_csv('Daten/Dataset/neg_samples/all_Apr_Sep_second.csv', converters={
        'ndvi': ast.literal_eval,
    })

    cluster_approach = input("Cluster approach: 1 - all data points; 2 - true data points ")
    if cluster_approach == "1":
        print("K-Fold Clustering with all datapoints")
        df_true,df_neg = resign_id(df_true,df_neg)
        df_true = prepare_data(df_true,1)
        df_neg = prepare_data(df_neg,0)
        df = pd.concat([df_true,df_neg],axis=0,ignore_index=True)
        gdf_clusters = get_spatial_k_folds(df,7)
        gdf_clusters = shuffle(gdf_clusters)
    elif cluster_approach == "2":   
        print("K-Fold Clustering with first clustering positive samples and sorting negative samples") 
        df_true,df_neg = resign_id(df_true,df_neg)
        df_true = prepare_data(df_true,1)
        df_neg = prepare_data(df_neg,0)
        df_true = get_spatial_k_folds(df_true,7)
        df_neg = add_folds(df_true,df_neg)
        gdf_clusters = pd.concat([df_true,df_neg],axis=0,ignore_index=True)
        visualise_folds(gdf_clusters)
        gdf_clusters = shuffle(gdf_clusters)
    else: 
        print("Not valid")
        return None
    
    return gdf_clusters

def visualise_spatial(models,clusters,df_test_list,gdf_clusters,features):
    warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect.")
    fig, ax = plt.subplots(2, round(len(clusters)/2),figsize=(20, 10))  
    for cluster,model,df_test in zip(clusters,models,df_test_list):
        i = cluster - 1
        y_pred_proba = model.predict_proba(df_test[features])
        y_true = df_test['label']
        y_scores = y_pred_proba[:,1] 
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)      
        fig.suptitle('Spatial k-fold Validation')    
        roc_auc = auc(fpr, tpr)
        y_probs_neg = y_pred_proba[:,0]
        fnr, tnr, _ = roc_curve(1- y_true,y_probs_neg)
        if i < len(clusters)/2:
            j = 0
        else:
            j= 1
            i = i - round(len(clusters)/2)

        ax[j][i].plot(fpr, tpr, label=f'ROC curve: Positive (AUC= {roc_auc:.2f}')
        ax[j][i].plot(fnr, tnr, label=f'ROC curve: Negative')
        ax[j][i].plot([0, 1], [0, 1], 'k--', label='Random Guessing')
        ax[j][i].set_xlabel('False Positive/Negative Rate')
        ax[j][i].set_ylabel('True Positive/Negative Rate')
        ax[j][i].legend()
        plt.tight_layout()
        ax[j][i].set_title(f'Fold {cluster}:')

    
    folds = gdf_clusters[['folds','geometry','label']]
    boundaries = folds.dissolve(by='folds').geometry.convex_hull
    boundaries_df = pd.DataFrame(boundaries)
    boundaries_df['centroid'] = boundaries.centroid
    boundaries.boundary.plot(ax=ax[j][int(len(clusters)/2)], color='black',linewidth=2)
    folds[folds['label'] == 0].plot(ax=ax[j][int(len(clusters)/2)],color='green',markersize=10)
    folds[folds['label'] == 1].plot(ax=ax[j][int(len(clusters)/2)],color='red',markersize=10)
    for idx, row in boundaries_df.iterrows():
        centroid = row['centroid']
        ax[j][int(len(clusters)/2)].text(centroid.x, centroid.y, idx, 
                horizontalalignment='center', verticalalignment='center',
                fontsize=20, color='black')
        
    plt.tight_layout()
    plt.show()


def spatial_evaluation(model_names,gdf_clusters,features,sample_neg=True,model_parameters=None):
    group_cvs = LeaveOneGroupOut()
    spatial_folds = gdf_clusters['folds']
   
    for train_idx, test_idx in group_cvs.split(gdf_clusters, groups=spatial_folds):
        models_fitted = []
        X,test = gdf_clusters.iloc[train_idx], gdf_clusters.iloc[test_idx]
        print(f'Group left out: {test.iloc[0]['folds']}')
        for i,model_type in enumerate(model_names):
            if model_type == 'Easy Ensemble Classifier':
                model = EasyEnsembleClassifier(estimator=XGBClassifier(objective='binary:logistic', random_state=42,colsample_bytree= 0.6,learning_rate= 0.3,max_depth = 7,min_child_weight =1,n_estimators= 200,subsample=1.0),n_estimators=20,random_state=42) #with grid search
                sample_neg = False
            elif model_type == 'Random Forest Classifier':
                model = RandomForestClassifier(random_state=42,**model_parameters[i])
            elif model_type == 'XGB Classifier':
                model = XGBClassifier(objective='binary:logistic', random_state=42,eval_metric=['auc','logloss'],**model_parameters[i])
            else:
                print("No valid model type!")
                break
           
            if sample_neg:
                X_true = X[X['label'] == 1]
                X_neg = X[X['label'] == 0]
                try:
                    X_neg_subset = X_neg.sample(len(X_true))
                    train = pd.merge(X_true,X_neg_subset,how='outer')
                except ValueError:
                    X_true_subset = X_true.sample(len(X_neg))
                    train = pd.merge(X_true_subset,X_neg,how='outer')
                X_train = train[features]  
                y_train = train['label']  
            else:
                X_train = X[features] 
                y_train = X['label']  

            print(f'Training with {len(X_train)} samples')
            model.fit(X_train,y_train)   
            models_fitted.append(model)  
        evaluate_all_models_metrics(models_fitted,test,model_names,features=features)
 

def spatial_evaluation_one_model(model_type,gdf_clusters,features,sample_neg=True,model_parameters=None):
    group_cvs = LeaveOneGroupOut()
    spatial_folds = gdf_clusters['folds']
    models_fitted = []
    test_sets = []
    clusters = []
    for train_idx, test_idx in group_cvs.split(gdf_clusters, groups=spatial_folds):
        X,test = gdf_clusters.iloc[train_idx], gdf_clusters.iloc[test_idx]
        clusters.append(test.iloc[0]['folds'])
     
        if model_type == 'Easy Ensemble Classifier':
            model = EasyEnsembleClassifier(estimator=XGBClassifier(objective='binary:logistic', random_state=42,colsample_bytree= 0.6,learning_rate= 0.3,max_depth = 7,min_child_weight =1,n_estimators= 200,subsample=1.0),n_estimators=20,random_state=42) #with grid search
            sample_neg = False
        elif model_type == 'Random Forest Classifier':
            model = RandomForestClassifier(random_state=42,**model_parameters)
        elif model_type == 'XGB Classifier':
            model = XGBClassifier(objective='binary:logistic', random_state=42,eval_metric=['auc','logloss'],**model_parameters)
        else:
            print("No valid model type!")
            break

        
        if sample_neg:
            X_true = X[X['label'] == 1]
            X_neg = X[X['label'] == 0]
            try:
                X_neg_subset = X_neg.sample(len(X_true))
                train = pd.merge(X_true,X_neg_subset,how='outer')
            except ValueError:
                X_true_subset = X_true.sample(len(X_neg))
                train = pd.merge(X_true_subset,X_neg,how='outer')
            X_train = train[features] 
            y_train = train['label']  
        else:
            X_train = X[features]  
            y_train = X['label']  
        model.fit(X_train,y_train)   
        models_fitted.append(model) 
        test_sets.append(test)
    print(model_type)
    visualise_spatial(models_fitted,clusters,test_sets,gdf_clusters=gdf_clusters,features=features)  
 


 
def visualise_temporal(models,df_test,names,features):
    warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
    fig, ax = plt.subplots(3, 3,figsize=(18, 16))  
    for i,model in enumerate(models):
        y_pred_proba = model.predict_proba(df_test[features])
        y_true = df_test['label']
        y_scores = y_pred_proba[:,1] 
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        specificity = 1 - fpr
        idx = np.argmin(np.abs(tpr - specificity))
        optimal_threshold = thresholds[idx]

        y_pred_optimal = (y_scores > optimal_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_optimal)

        ax[2][i].plot(thresholds, tpr, label="Sensitivity (Recall)", color="blue")
        ax[2][i].plot(thresholds, specificity, label="Specificity", color="green")
        ax[2][i].axvline(optimal_threshold, color='red', linestyle='--', label=f"Sensitivity==Specificity: {optimal_threshold:.2f}")
        ax[2][i].set_xlabel("Threshold")
        ax[2][i].set_ylabel("Rate")
        ax[2][i].legend(loc="upper right")
    

        inset_ax = inset_axes(ax[2][i], width="30%", height="30%", loc="lower right", bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax[2][i].transAxes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=inset_ax, colorbar=False)
        inset_ax.set_title("Confusion Matrix", fontsize=10)

        plt.tight_layout()
     
    
        roc_auc = auc(fpr, tpr)
        y_probs_neg = y_pred_proba[:,0]
        fnr, tnr, _ = roc_curve(1- y_true,y_probs_neg)
  
        ax[1][i].plot(fpr, tpr, label=f'ROC curve: Positive (AUC= {roc_auc:.2f}')
        ax[1][i].plot(fnr, tnr, label=f'ROC curve: Negative')
        ax[1][i].plot([0, 1], [0, 1], 'k--', label='Random Guessing')
        ax[1][i].set_xlabel('False Positive/Negative Rate')
        ax[1][i].set_ylabel('True Positive/Negative Rate')
        ax[1][i].legend()
        plt.tight_layout()
    

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        specificity_scores = []
        f1_scores = []
        thresholds = np.arange(0,1,0.01)
        for threshold in thresholds:
            y_prob_bin = (y_scores > threshold).astype(int)
            accuracy_scores.append(accuracy_score(y_true,y_prob_bin))
            precision_scores.append(precision_score(y_true,y_prob_bin,zero_division=1))
            recall_scores.append(recall_score(y_true,y_prob_bin))
            specificity_scores.append(specificity_score(y_true,y_prob_bin))
            f1_scores.append(f1_score(y_true,y_prob_bin))
        ax[0][i].plot(thresholds,accuracy_scores,label=f'Accuracy')
        ax[0][i].plot(thresholds,recall_scores,label='Recall')
        ax[0][i].plot(thresholds,precision_scores,label=f'Precision')

        ax[0][i].plot(thresholds,f1_scores,label=f'F1-Score: Max at {round(thresholds[np.argmax(f1_scores)],2)} with {round(max(f1_scores),2)}')          
        ax[0][i].set_xlabel('Threshold')
        ax[0][i].set_ylabel('Rate')
        ax[0][i].legend()
        ax[0][i].set_title(names[i])
    plt.show()


def visualise_temporal_one_model(models,df_test_list,names,features,save=False,id=None):
    warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
    fig, ax = plt.subplots(3, 3,figsize=(18, 16)) 
    for i,(model,df_test) in enumerate(zip(models,df_test_list)):
        y_pred_proba = model.predict_proba(df_test[features])
        y_true = df_test['label']
        y_scores = y_pred_proba[:,1] 
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        specificity = 1 - fpr
        idx = np.argmin(np.abs(tpr - specificity))
        optimal_threshold = thresholds[idx]
        y_pred_optimal = (y_scores > optimal_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_optimal)
     

        ax[2][i].plot(thresholds, tpr, label="Sensitivity (Recall)", color="blue")
        ax[2][i].plot(thresholds, specificity, label="Specificity", color="green")
        ax[2][i].axvline(optimal_threshold, color='red', linestyle='--', label=f"Sensitivity==Specificity: {optimal_threshold:.2f}")
        ax[2][i].set_xlabel("Threshold")
        ax[2][i].set_ylabel("Rate")
        ax[2][i].legend(loc="upper right")
        

        inset_ax = inset_axes(ax[2][i], width="30%", height="30%", loc="lower right", bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax[2][i].transAxes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=inset_ax, colorbar=False)
        inset_ax.set_title("Confusion Matrix", fontsize=10)

        plt.tight_layout()
    
        roc_auc = auc(fpr, tpr)
        y_probs_neg = y_pred_proba[:,0]
        fnr, tnr, _ = roc_curve(1- y_true,y_probs_neg)

        ax[1][i].plot(fpr, tpr, label=f'ROC curve: Positive (AUC= {roc_auc:.2f}')
        ax[1][i].plot(fnr, tnr, label=f'ROC curve: Negative')
        ax[1][i].plot([0, 1], [0, 1], 'k--', label='Random Guessing')
        ax[1][i].set_xlabel('False Positive/Negative Rate')
        ax[1][i].set_ylabel('True Positive/Negative Rate')
        ax[1][i].legend()
        plt.tight_layout()
    

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        specificity_scores = []
        f1_scores = []
        thresholds = np.arange(0,1,0.01)
        for threshold in thresholds:
            y_prob_bin = (y_scores > threshold).astype(int)
            accuracy_scores.append(accuracy_score(y_true,y_prob_bin))
            precision_scores.append(precision_score(y_true,y_prob_bin,zero_division=1))
            recall_scores.append(recall_score(y_true,y_prob_bin))
            specificity_scores.append(specificity_score(y_true,y_prob_bin))
            f1_scores.append(f1_score(y_true,y_prob_bin))
        ax[0][i].plot(thresholds,accuracy_scores,label=f'Accuracy')
        ax[0][i].plot(thresholds,recall_scores,label='Recall')
        ax[0][i].plot(thresholds,precision_scores,label=f'Precision')
        ax[0][i].plot(thresholds,f1_scores,label=f'F1-Score: Max at {round(thresholds[np.argmax(f1_scores)],2)} with {round(max(f1_scores),2)}')            
        ax[0][i].set_xlabel('Threshold')
        ax[0][i].set_ylabel('Rate')
        ax[0][i].legend()
        ax[0][i].set_title(names[i])
    plt.show()


def evaluate_temporal(df,features): 
    best_parameters_rf = [{'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100},{'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100},{'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100}]
    best_parameters_xgb = [{'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 200, 'subsample': 1.0},{'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 100, 'subsample': 0.6},{'colsample_bytree': 0.6, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.8}]

    xgb_model = XGBClassifier(objective='binary:logistic', random_state=42,**best_parameters_xgb[0])
    rfc_model = RandomForestClassifier(random_state=42,**best_parameters_rf[0])
    ensemble_model = EasyEnsembleClassifier(estimator=XGBClassifier(objective='binary:logistic', random_state=42,colsample_bytree= 0.6,learning_rate= 0.3,max_depth = 7,min_child_weight =1,n_estimators= 200,subsample=1.0),n_estimators=20,random_state=42) #with grid search
    models = [rfc_model,xgb_model,ensemble_model]
    names = ['Random Forest Classifier','XGB Classifier','Easy Ensemble Classifier']
    for year in [2020,2021,2022]:
        for model in models:
            df_test = df[df['date'].dt.year == year]
            df_train = df[df['date'].dt.year != year]
            if model != ensemble_model:
                df_train = sample_len_df(df_train)
            df_train = shuffle(df_train)
            model.fit(df_train[features],df_train['label'])
        print(f"Evaluation for year {year}:")
        visualise_temporal(models,df_test,names,features)


def evaluate_temporal_one_model(df,model_type,features,sample_neg=True):  
    best_parameters_rf = [{'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100},{'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100},{'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100}]
    best_parameters_xgb = [{'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 200, 'subsample': 1.0},{'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 100, 'subsample': 0.6},{'colsample_bytree': 0.6, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.8}]
    models = []
    df_test_list = []
    names = ['2020','2021','2022']
    for year in [2020,2021,2022]:
        
        if model_type == 'Easy Ensemble Classifier':
            model = EasyEnsembleClassifier(estimator=XGBClassifier(objective='binary:logistic', random_state=42,colsample_bytree= 0.6,learning_rate= 0.3,max_depth = 7,min_child_weight =1,n_estimators= 200,subsample=1.0),n_estimators=20,random_state=42) #with grid search
            sample_neg = False
        elif model_type == 'Random Forest Classifier':
            model = RandomForestClassifier(random_state=42,**best_parameters_rf[0])
        elif model_type == 'XGB Classifier':
            model = XGBClassifier(objective='binary:logistic', random_state=42,eval_metric=['auc','logloss'],**best_parameters_xgb[0])
        else:
            print("No valid model type!")
            break
        model = RandomForestClassifier(random_state=42,**best_parameters_rf[0])
    
        df_test = df[df['date'].dt.year == year]
        df_test_list.append(df_test)
        df_train = df[df['date'].dt.year != year]
        if sample_neg:
            df_train = sample_len_df(df_train)
        df_train = shuffle(df_train)
        model.fit(df_train[features],df_train['label'])
        models.append(model)
    visualise_temporal_one_model(models,df_test_list,names,features)


def visualise_spatial_temporal(model,val_df,year,features):
    warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect.")
    spatial_folds = val_df['folds']
    group_cvs = LeaveOneGroupOut()
    df_test_list = []
    clusters = []
    for train_idx, test_idx in group_cvs.split(val_df, groups=spatial_folds):
        test = val_df.iloc[test_idx]
        df_test_list.append(test)
        clusters.append(test.iloc[0]['folds'])
    fig, ax = plt.subplots(2, round(len(clusters)/2),figsize=(20, 10)) 
    for cluster,df_test in zip(clusters,df_test_list):
        i = cluster - 1
        y_pred_proba = model.predict_proba(df_test[features])
        y_true = df_test['label']
        y_scores = y_pred_proba[:,1] 
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
      
        fig.suptitle(f'Spatial Validation for year {year}')    
        roc_auc = auc(fpr, tpr)
        y_probs_neg = y_pred_proba[:,0]
        fnr, tnr, _ = roc_curve(1- y_true,y_probs_neg)

        if i < len(clusters)/2:
            j = 0
        else:
            j= 1
            i = i - round(len(clusters)/2)

        ax[j][i].plot(fpr, tpr, label=f'ROC curve: Positive (AUC= {roc_auc:.2f}')
        ax[j][i].plot(fnr, tnr, label=f'ROC curve: Negative')
        ax[j][i].plot([0, 1], [0, 1], 'k--', label='Random Guessing')
        ax[j][i].set_xlabel('False Positive/Negative Rate')
        ax[j][i].set_ylabel('True Positive/Negative Rate')
        ax[j][i].legend()
        plt.tight_layout()

        ax[j][i].set_title(f'Fold {cluster}:')
      
    folds = val_df[['folds','geometry','label']]
    boundaries = folds.dissolve(by='folds').geometry.convex_hull
    boundaries_df = pd.DataFrame(boundaries)
    boundaries_df['centroid'] = boundaries.centroid
    boundaries.boundary.plot(ax=ax[j][int(len(clusters)/2)], color='black',linewidth=2)
    folds[folds['label'] == 0].plot(ax=ax[j][int(len(clusters)/2)],color='green',markersize=10)
    folds[folds['label'] == 1].plot(ax=ax[j][int(len(clusters)/2)],color='red',markersize=10)
    for idx, row in boundaries_df.iterrows():
        centroid = row['centroid']
        ax[j][int(len(clusters)/2)].text(centroid.x, centroid.y, idx, 
                horizontalalignment='center', verticalalignment='center',
                fontsize=20, color='black')
        
    plt.tight_layout()
    plt.show()
     

def evaluate_spatial_temporal(df,features): 
    best_parameters_rf = [{'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100},{'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100},{'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 100}]
    best_parameters_xgb = [{'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 200, 'subsample': 1.0},{'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 3, 'n_estimators': 100, 'subsample': 0.6},{'colsample_bytree': 0.6, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.8}]

    xgb_model = XGBClassifier(objective='binary:logistic', random_state=42,**best_parameters_xgb[0])
    rfc_model = RandomForestClassifier(random_state=42,**best_parameters_rf[0])
    ensemble_model = EasyEnsembleClassifier(estimator=XGBClassifier(objective='binary:logistic', random_state=42,colsample_bytree= 0.6,learning_rate= 0.3,max_depth = 7,min_child_weight =1,n_estimators= 200,subsample=1.0),n_estimators=20,random_state=42) #with grid search
  
    models = [rfc_model,xgb_model,ensemble_model]
    names = ['Random Forest Classifier','XGB Classifier','Easy Ensemble Classifier']
    for year in [2020,2021,2022]:
        for model,name in zip(models,names):
            df_test = df[df['date'].dt.year == year]
            df_train = df[df['date'].dt.year != year]
            if model != ensemble_model:
                df_train = sample_len_df(df_train)
            df_train = shuffle(df_train)
            model.fit(df_train[features],df_train['label'])
            print(name)
            visualise_spatial_temporal(model,df_test,year,features)
        