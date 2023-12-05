# In[1]:

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.chdir("/home/omkarc/omkar/pulearning")
import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score, roc_auc_score
import random
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import csv



parser = argparse.ArgumentParser()
parser.add_argument("--starti", type = int)
parser.add_argument("--endi", type = int)
parser.add_argument("--algotypei")  # "base" or "selecting_pos"
parser.add_argument("--selectiontypei") # "ranking" or "quantile":
parser.add_argument("--featuretypei") # "onto_binary":

args = parser.parse_args()
start_i = args.starti
end_i = args.endi
algo_type = args.algotypei
selection_type = args.selectiontypei
feature_type = args.featuretypei


# In[1]:

disease_df = pd.read_csv("dis_genet_train_final.csv", low_memory=False, index_col = 0)
disease_df_test = pd.read_csv("dis_genet_test_final.csv", low_memory=False, index_col = 0)


# In[1]:
unipath_results = pd.read_csv("scores_disease_adjpvalog_train.csv", index_col=0)
unipath_results_old = pd.read_csv("scores_disease_adjpvalog.csv", index_col=0)
peakscores = pd.read_csv("peakscore_mean_all_normal.csv", index_col =0)
gene_list = pd.read_csv("gene_list_unique.csv")

go_bp = pd.read_csv("go_biological_process_2021.csv", low_memory=False, index_col=0)
go_mf = pd.read_csv("go_molecular_function_2021.csv", low_memory=False, index_col=0)
go_cc = pd.read_csv("go_cellular_component_2021.csv", low_memory=False, index_col=0)
all_genesets = pd.concat([go_bp, go_mf, go_cc])
binary_gene_feature_onto_pre = pd.read_csv("gene_binary_cp_mf_cc_r.csv", low_memory = False, index_col = 0)
binary_gene_feature_onto = binary_gene_feature_onto_pre[~binary_gene_feature_onto_pre.index.isna()]

unipath_epigenome = pd.read_csv("scores_disease_adjpvalog_epigenome_train.csv", index_col=0)
binary_gene_feature_epigenome = pd.read_csv("peakscores_binary.csv", low_memory=False, index_col = 0)
gene_set_binary_epigenome = pd.read_csv("gene_set_binary_epigenome.csv", low_memory=False, index_col = 0)
binary_onto_epigene = pd.read_csv("binary_onto_epigene.csv", low_memory=False, index_col = 0)

gene_pairs = pd.read_csv("/home/omkarc/omkar/pulearning/benchmark/NIAPU/data/biogrid_gene_pairs_clean_Hs_4_4_206.txt", header = None, sep = " ")
adjacency_matrix_ppi = pd.read_csv("adjacency_matrix_ppi.csv", low_memory=False, index_col=0)
# In[1]:


def cross_val_score_defined(estimator, x, y, x_data_test, y_data_test, x_all, i):
    # scores = []
    #kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #kf = StratifiedGroupKFold(n_splits=5, shuffle=False)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    score_dict = {
        'train_accuracy': [],
        'train_balanced_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1_score': [],
        'val_acc': [],
        'val_bal_accu': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1_score': [],
        'val_auc': [],
        'test_accuracy': [],
        'test_balanced_accuracy': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1_score': [],
        'test_auc': [],
    }

    for train_index, val_index in kf.split(x, y):
        x_train, x_val = x.iloc[train_index], x.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        estimator.fit(x_train, y_train)
        val_pred_prbs = estimator.predict(x_val)
        y_val_binary = y_val
        y_val_binary[y_val_binary == 0.1] = 0; y_train[y_train == 0.1] = 0;
        y_val_binary[y_val_binary == 0.9] = 1; y_train[y_train == 0.9] = 1;
        y_val_binary[y_val_binary == 0.2] = 1; y_train[y_train == 0.2] = 0;

        thresholds = np.arange(0, 1, 0.001)
        best_threshold_val = None
        best_bal_acc_score = 0
        for threshold in thresholds:
            y_pred_binary = np.where(val_pred_prbs > threshold, 1, 0)
            bal_acc = metrics.f1_score(y_val, y_pred_binary)
            if bal_acc > best_bal_acc_score:
                best_bal_acc_score = bal_acc
                best_threshold_val = threshold

        # validation metrics
        val_pred = np.where(val_pred_prbs > best_threshold_val, 1, 0)
        val_accuracy = metrics.accuracy_score(y_val, val_pred)
        val_balanced_accuracy = best_bal_acc_score
        val_precision, val_recall, val_f1_score, _ = metrics.precision_recall_fscore_support(y_val, val_pred, average='binary')
        val_auc = roc_auc_score(y_val, val_pred_prbs)

        # train metrics
        pred_prbs_train = estimator.predict(x_train)
        train_pred = np.where(pred_prbs_train > best_threshold_val, 1, 0)
        accuracy_train = metrics.accuracy_score(y_train, train_pred)
        balanced_acc_train = metrics.balanced_accuracy_score(y_train, train_pred)
        precision_train, recall_train, f1_score_train, _ = metrics.precision_recall_fscore_support(y_train, train_pred, average='binary')

        # test metrics.
        pred_test_prbs = estimator.predict(x_data_test)
        test_auc = roc_auc_score(y_data_test, pred_test_prbs)
        pred_test = np.where(pred_test_prbs > best_bal_acc_score, 1, 0)
        test_accuracy = metrics.accuracy_score(y_data_test, pred_test)
        test_balanced_accuracy =  metrics.balanced_accuracy_score(y_data_test, pred_test)
        test_precision, test_recall, test_f1_score, _ = metrics.precision_recall_fscore_support(y_data_test, pred_test, average='binary')

        score_dict['train_accuracy'].append(accuracy_train)
        score_dict['train_balanced_accuracy'].append(balanced_acc_train)
        score_dict['train_precision'].append(precision_train)
        score_dict['train_recall'].append(recall_train)
        score_dict['train_f1_score'].append(f1_score_train)
        score_dict['val_acc'].append(val_accuracy)
        score_dict['val_bal_accu'].append(val_balanced_accuracy)
        score_dict['val_precision'].append(val_precision)
        score_dict['val_recall'].append(val_recall)
        score_dict['val_f1_score'].append(val_f1_score)
        score_dict['val_auc'].append(val_auc)
        score_dict['test_accuracy'].append(test_accuracy)
        score_dict['test_balanced_accuracy'].append(test_balanced_accuracy)
        score_dict['test_precision'].append(test_precision)
        score_dict['test_recall'].append(test_recall)
        score_dict['test_f1_score'].append(test_f1_score)
        score_dict['test_auc'].append(test_auc)

    plot_filename = '{}{}{}{}{}{}{}'.format('results/f1_score/', i, "_", algo_type, "_", feature_type, ".png")
    csv_filename = '{}{}{}{}{}{}{}'.format('results/f1_score/', i, "_", algo_type, "_", feature_type, ".csv")
    
    plot_save_f1_scores_and_thresholds(y_data_test, pred_test_prbs, plot_filename, csv_filename)

    check_test_bal = pd.DataFrame(score_dict['val_f1_score'])
    max_cv = check_test_bal.max().index[0]

    split_number = 0

    # Iterate through the folds
    for train_index, val_index in kf.split(x, y):
        split_number += 1
        if split_number == max_cv:  # Check if it's the third split
            x_train, x_val = x.iloc[train_index], x.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            estimator.fit(x_train, y_train)
            
            break  # Exit the loop after the third split
        pred_prbs_all = pd.DataFrame(estimator.predict(x_all))
        pred_prbs_all.index = x_all.index
        
    return score_dict, pred_prbs_all



# In[0]:

def plot_save_f1_scores_and_thresholds(y_labels, y_predicted_probs, plot_filename, csv_filename):
    thresholds = np.arange(0, 1.05, 0.05)
    f1_scores = []

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Threshold', 'F1 Score'])

        for thresh in thresholds:
            y_pred = (y_predicted_probs >= thresh).astype(int)
            f1 = f1_score(y_labels, y_pred)
            f1_scores.append(f1)
            writer.writerow([thresh, f1])

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, marker='o')
    plt.title('F1 Score vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.savefig(plot_filename)

# In[0]:

def predict_genes(pred_prbs_all, pos_genes):
    y_label = pred_prbs_all * [0] 
    y_label[y_label.index.isin(pos_genes)] = 1
    thresholds = np.arange(0, 1, 0.001)
    best_threshold_val = None
    best_f1_score = 0
    for threshold in thresholds:
        y_pred_binary = np.where(pred_prbs_all.iloc[:,0] > threshold, 1, 0)
        f1 = metrics.f1_score(y_label, y_pred_binary)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold_val = threshold
    final_pred = pd.DataFrame(np.where(pred_prbs_all.iloc[:,0] > best_threshold_val, 1, 0))
    final_pred.index = pred_prbs_all.index
    return final_pred
    
#In[0]
def build_x_y_data_base_validation(pos_genes, i, disease_df, disease_df_test):
    pos_genes_test = disease_df_test[disease_df_test.index.isin([disease_df.index[i]])].dropna(axis = 1)
    
    pos_all_genes =  pos_genes + pos_genes_test.iloc[0,:].tolist()
    gene_for_neg_train_val_sample = list(set(gene_list.iloc[:,0].tolist()) - set(pos_all_genes))
    random_generator = random.Random(1)
    neg_genes_train_val = random_generator.sample(gene_for_neg_train_val_sample, len(pos_genes) * 5)
    gene_for_neg_test_sample = set(gene_for_neg_train_val_sample) - set(neg_genes_train_val)

    random_generator = random.Random(1)
    neg_genes_test = random_generator.sample(gene_for_neg_test_sample, len(pos_genes_test.iloc[0,:]) * 5)

    if feature_type == "binary_epige":
        x_pos_train_val = binary_gene_feature_epigenome[binary_gene_feature_epigenome.index.isin(pos_genes)]
        x_neg_train_val = binary_gene_feature_epigenome[binary_gene_feature_epigenome.index.isin(neg_genes_train_val)]
        x_pos_test = binary_gene_feature_epigenome[binary_gene_feature_epigenome.index.isin(pos_genes_test.iloc[0,:].tolist())]
        x_neg_test = binary_gene_feature_epigenome[binary_gene_feature_epigenome.index.isin(neg_genes_test)]
        x_all = binary_gene_feature_epigenome
        
    if feature_type == "binary_onto_epige":
        x_pos_train_val = binary_onto_epigene[binary_onto_epigene.index.isin(pos_genes)]
        x_neg_train_val = binary_onto_epigene[binary_onto_epigene.index.isin(neg_genes_train_val)]
        x_pos_test = binary_onto_epigene[binary_onto_epigene.index.isin(pos_genes_test.iloc[0,:].tolist())]
        x_neg_test = binary_onto_epigene[binary_onto_epigene.index.isin(neg_genes_test)]
        x_all = binary_onto_epigene
        
    if feature_type == "onto_binary":
        x_pos_train_val = binary_gene_feature_onto[binary_gene_feature_onto.index.isin(pos_genes)]
        x_neg_train_val = binary_gene_feature_onto[binary_gene_feature_onto.index.isin(neg_genes_train_val)]
        x_pos_test = binary_gene_feature_onto[binary_gene_feature_onto.index.isin(pos_genes_test.iloc[0,:].tolist())]
        x_neg_test = binary_gene_feature_onto[binary_gene_feature_onto.index.isin(neg_genes_test)]
        x_all = binary_gene_feature_onto
        
    if feature_type == "epigenome":
        x_pos_train_val = peakscores[peakscores.index.isin(pos_genes)]
        x_neg_train_val = peakscores[peakscores.index.isin(neg_genes_train_val)]
        x_pos_test = peakscores[peakscores.index.isin(pos_genes_test.iloc[0,:].tolist())]
        x_neg_test = peakscores[peakscores.index.isin(neg_genes_test)]
        x_all = peakscores
        
    y_pos_train_val = pd.Series([1] * (x_pos_train_val.shape[0]))
    y_neg_train_val = pd.Series([0] * (x_neg_train_val.shape[0]))
    y_pos_test = pd.Series([1] * (x_pos_test.shape[0]))
    y_neg_test = pd.Series([0] * (x_neg_test.shape[0]))

    x_data_train_valid = pd.concat([x_pos_train_val, x_neg_train_val])
    y_data_train_valid = pd.concat([y_pos_train_val, y_neg_train_val])
    x_data_test = pd.concat([x_pos_test, x_neg_test])
    y_data_test = pd.concat([y_pos_test, y_neg_test])

    return x_data_train_valid, y_data_train_valid, x_data_test, y_data_test, x_all

# In[1]:

def make_genes_ranking(i, enriched_gene_sets, gene_rank_quantile):
    gene_set_scores = unipath_results.loc[unipath_results.index.isin(enriched_gene_sets.index.tolist()),disease_df.index[i]] 
    binary_gene_feature_onto_pos = binary_gene_feature_onto[enriched_gene_sets.index.tolist()]
    gene_score_pre = binary_gene_feature_onto_pos.dot(gene_set_scores).sort_values(ascending=False)
    gene_score = gene_score_pre[gene_score_pre > 0 ]
    gene_score_selected = gene_score[gene_score >= gene_score.quantile(gene_rank_quantile)]
    binary_gene_feature_onto_selected = binary_gene_feature_onto_pos[binary_gene_feature_onto_pos.index.isin(gene_score_selected.index)]
    binary_gene_feature_onto_selected.loc[gene_score_selected.index].loc["score"] = gene_score_selected
    
    return binary_gene_feature_onto_selected.index.tolist(), binary_gene_feature_onto_selected

def make_genes_ranking_likely_pos(i, enriched_gene_sets, pos_genes, likely_pos_quantile):
    gene_set_scores = unipath_results.loc[unipath_results.index.isin(enriched_gene_sets.index.tolist()),disease_df.index[i]] 
    binary_gene_feature_onto_pos = binary_gene_feature_onto[enriched_gene_sets.index.tolist()]
    gene_score_pre = binary_gene_feature_onto_pos.dot(gene_set_scores).sort_values(ascending=False)
    gene_score = gene_score_pre[gene_score_pre > 0 ]
    gene_score = gene_score[~gene_score.index.isin(pos_genes)]
    likely_pos_gene_score = gene_score[gene_score >= gene_score.quantile(likely_pos_quantile)]
    gene_score_weakly_pos_likely_neg = gene_score[~gene_score.index.isin(likely_pos_gene_score.index)]
    weakly_pos_gene_score = gene_score_weakly_pos_likely_neg[gene_score_weakly_pos_likely_neg > np.quantile(gene_score_weakly_pos_likely_neg, 0.99)]
    likely_neg_gene_score = gene_score_weakly_pos_likely_neg[gene_score_weakly_pos_likely_neg <= np.quantile(gene_score_weakly_pos_likely_neg, 0.04)]

    reliable_neg_genes_score = gene_score_pre[gene_score_pre <= 0]
    binary_features_train_test = pd.concat([binary_gene_feature_onto_pos, binary_gene_feature_onto_pos.dot(gene_set_scores)], axis = 1)
    binary_features_train_test.columns = binary_gene_feature_onto_pos.columns.tolist() + ["score"]
    
    return likely_pos_gene_score, reliable_neg_genes_score, binary_features_train_test, weakly_pos_gene_score, likely_neg_gene_score

def make_genes_ranking_likely_pos1(i, enriched_gene_sets, pos_genes, likely_pos_quantile):
    gene_set_scores = unipath_results.loc[unipath_results.index.isin(enriched_gene_sets.index.tolist()),disease_df.index[i]] 
    binary_gene_feature_onto_pos = binary_gene_feature_onto[enriched_gene_sets.index.tolist()]
    gene_score_pre = pd.DataFrame(binary_gene_feature_onto_pos.dot(gene_set_scores).sort_values(ascending=False))
    gene_pairs_pos = gene_pairs[gene_pairs.iloc[:,0].isin(pos_genes)]
    gene_pairs_pos_first_genes = gene_pairs_pos[~gene_pairs_pos.iloc[:,1].isin(pos_genes)]
    mask = gene_score_pre.index.isin(gene_pairs_pos_first_genes.iloc[:, 1])
    gene_score_pre.iloc[mask, gene_score_pre.columns[0]] = +10
    gene_score = gene_score_pre[gene_score_pre.iloc[:,0] > 0 ]
    gene_score = gene_score[~gene_score.index.isin(pos_genes)]
    likely_pos_gene_score = gene_score[gene_score.iloc[:,0] >= gene_score.iloc[:,0].quantile(likely_pos_quantile)]

    reliable_neg_genes_score = gene_score_pre[gene_score_pre.iloc[:,0] <= 0]
    binary_features_train_test = pd.concat([binary_gene_feature_onto_pos, binary_gene_feature_onto_pos.dot(gene_set_scores)], axis = 1)
    binary_features_train_test.columns = binary_gene_feature_onto_pos.columns.tolist() + ["score"]
    
    return likely_pos_gene_score, reliable_neg_genes_score, binary_features_train_test

# In[1]:
    
def make_genes_ranking_likely_pos_epigenome_onto(i, enriched_gene_sets, pos_genes, enriched_gene_sets_epi, disease_df_selected, likely_pos_quantile):
    gene_set_scores = unipath_results.loc[unipath_results.index.isin(enriched_gene_sets.index.tolist()),disease_df_selected.index[i]] 
    binary_gene_feature_onto_pos = binary_gene_feature_onto[enriched_gene_sets.index.tolist()]
    gene_score_pre = binary_gene_feature_onto_pos.dot(gene_set_scores).sort_values(ascending=False)
    gene_score = gene_score_pre[gene_score_pre > 0 ]
    gene_score = gene_score[~gene_score.index.isin(pos_genes)]
    likely_pos_gene_score = gene_score[gene_score >= gene_score.quantile(likely_pos_quantile)]
    gene_score[~gene_score.index.isin(likely_pos_gene_score.index)]
    reliable_neg_genes_score = gene_score_pre[gene_score_pre <= 0]    
    binary_features_train_test_onto = pd.concat([binary_gene_feature_onto_pos, binary_gene_feature_onto_pos.dot(gene_set_scores)], axis = 1)
    binary_features_train_test_onto.columns = binary_gene_feature_onto_pos.columns.tolist() + ["score_onto"]

    gene_set_scores_epi = unipath_epigenome.loc[unipath_epigenome.index.isin(enriched_gene_sets_epi.index.tolist()), disease_df_selected.index[i]] 
    binary_gene_feature_epigenome_pos = binary_gene_feature_epigenome[enriched_gene_sets_epi.index.tolist()]
    gene_score_pre_epi = binary_gene_feature_epigenome_pos.dot(gene_set_scores_epi).sort_values(ascending=False)
    gene_score_epi = gene_score_pre_epi[gene_score_pre_epi > 0 ]
    gene_score_epi = gene_score_epi[~gene_score_epi.index.isin(pos_genes)]
    likely_pos_gene_score_epi = gene_score_epi[gene_score_epi >= gene_score_epi.quantile(likely_pos_quantile)]
    reliable_neg_genes_score_epi = gene_score_pre_epi[gene_score_pre_epi <= 0]
    binary_features_train_test_epi = pd.concat([binary_gene_feature_epigenome_pos, binary_gene_feature_epigenome_pos.dot(gene_set_scores_epi)], axis = 1)
    binary_features_train_test_epi.columns = binary_gene_feature_epigenome_pos.columns.tolist() + ["score_epi"]

    reliable_neg_genes_score = set(reliable_neg_genes_score.index.tolist()) & set(reliable_neg_genes_score_epi.index.tolist())
    binary_features_train_test = pd.concat([binary_features_train_test_onto, binary_features_train_test_epi], axis = 1)
    binary_features_train_test = binary_features_train_test.fillna(0)
    likely_pos_gene = likely_pos_gene_score.index.tolist() + likely_pos_gene_score_epi.index.tolist()
    return likely_pos_gene, reliable_neg_genes_score, binary_features_train_test


# In[1]:
def mutually_exclusive_lists(pos_genes, likely_pos_gene_score, weakly_pos_gene_score, reliable_neg_genes_score, likely_neg_gene_score):
    likely_pos_genes = list(set(likely_pos_gene_score.index.tolist()))
    weakly_pos_genes = list(set(weakly_pos_gene_score.index.tolist()))
    reliable_neg_genes = list(set(reliable_neg_genes_score.index.tolist()))
    likely_neg_genes = list(set(likely_neg_gene_score.index.tolist()))

    reliable_neg_genes = list(set(reliable_neg_genes) - set(pos_genes + likely_pos_genes + weakly_pos_genes))
    likely_neg_genes = list(set(likely_neg_genes) - set(pos_genes + likely_pos_genes + weakly_pos_genes))

    random_generator = random.Random(1)
    # weakly_pos_genes = random_generator.sample(weakly_pos_genes, len(likely_pos_genes))
    reliable_neg_genes = random_generator.sample(reliable_neg_genes, len(pos_genes) * 15)
    likely_neg_genes = random_generator.sample(likely_neg_genes, len(pos_genes) * 2)
    
    return pos_genes, likely_pos_genes, weakly_pos_genes, reliable_neg_genes, likely_neg_genes

def divide_train_test(pos_genes, pos_genes_test, likely_pos_genes, weakly_pos_genes, reliable_neg_genes, likely_neg_genes):
    random_generator = random.Random(1)
    neg_genes_for_sample = list(set(gene_list.iloc[:,0]) - set(pos_genes_test.iloc[0,:].tolist() + pos_genes + likely_pos_genes + weakly_pos_genes + reliable_neg_genes + likely_neg_genes))
    neg_genes_test = random_generator.sample(neg_genes_for_sample, len(pos_genes) * 5)
    return neg_genes_test
    
# In[1]
def build_x_y_data_base_pos_selection_validation_set(pos_genes, i, feature_type, likely_pos_quantile):
    pos_genes_test = disease_df_test[disease_df_test.index.isin([disease_df.index[i]])].dropna(axis = 1)
    quntile_threshold = 0.01
    scores_unipath_pre = unipath_results[disease_df.index[i]]
    scores_unipath = scores_unipath_pre[scores_unipath_pre > 0]
    scores_unipath_quant = scores_unipath[scores_unipath > scores_unipath.quantile(quntile_threshold)]
    enriched_gene_sets_list = scores_unipath_quant.index.tolist()
    
    enriched_gene_sets = all_genesets[all_genesets.index.isin(enriched_gene_sets_list)]

    likely_pos_gene_score, reliable_neg_genes_score, binary_features_train_test, weakly_pos_gene_score, likely_neg_gene_score = make_genes_ranking_likely_pos(i, enriched_gene_sets, pos_genes, likely_pos_quantile)
    pos_genes, likely_pos_genes, weakly_pos_genes, reliable_neg_genes, likely_neg_genes = mutually_exclusive_lists(pos_genes, likely_pos_gene_score, weakly_pos_gene_score, reliable_neg_genes_score, likely_neg_gene_score)
    neg_genes_test = divide_train_test(pos_genes, pos_genes_test, likely_pos_genes, weakly_pos_genes, reliable_neg_genes, likely_neg_genes)

    print("No. of likely positive genes:", len(likely_pos_genes), "  No. of positive genes:", len(pos_genes), "  No. of weakly positive genes:", len(weakly_pos_genes), "  No. of reliable negative:", len(reliable_neg_genes),
          "No. of likely negative genes:", len(likely_neg_genes))

    #binary_features_train_test = pd.concat([binary_gene_feature_onto, binary_features_train_test.score], axis = 1)        
    binary_features_train_test = binary_gene_feature_onto        
    if feature_type == "onto_binary":
        x_pos_train = binary_features_train_test[binary_features_train_test.index.isin(pos_genes)]
        x_likely_pos_train = binary_features_train_test[binary_features_train_test.index.isin(likely_pos_genes)]
        x_weakly_pos_train = binary_features_train_test[binary_features_train_test.index.isin(weakly_pos_genes)]
        x_reliable_neg_train = binary_features_train_test[binary_features_train_test.index.isin(reliable_neg_genes)]
        x_likely_neg_train = binary_features_train_test[binary_features_train_test.index.isin(likely_neg_genes)]

        x_pos_test = binary_features_train_test[binary_features_train_test.index.isin(pos_genes_test.iloc[0,:].tolist())]
        x_neg_test = binary_features_train_test[binary_features_train_test.index.isin(neg_genes_test)]
        
        x_all = binary_features_train_test
        
    # if feature_type == "epigenome":
    #     x_pos_train = peakscores[peakscores.index.isin(pos_genes)]
    #     x_reliable_neg_train = peakscores[peakscores.index.isin(neg_genes_train_)]
        
    #     x_pos_test = peakscores[peakscores.index.isin(pos_genes_test.iloc[0,:].tolist())]
    #     x_neg_test = peakscores[peakscores.index.isin(neg_genes_test)]
    #     x_all = peakscores
        
    y_pos_train = pd.Series([1] * (x_pos_train.shape[0]))
    y_likely_pos_train = pd.Series([1] * (x_likely_pos_train.shape[0]))
    y_weakly_pos_train = pd.Series([0.9] * (x_weakly_pos_train.shape[0]))
    y_reliable_neg_train = pd.Series([0] * (x_reliable_neg_train.shape[0]))
    y_likely_neg_train = pd.Series([0.2] * (x_likely_neg_train.shape[0]))

    y_pos_test = pd.Series([1] * (x_pos_test.shape[0]))
    y_neg_test = pd.Series([0] * (x_neg_test.shape[0]))

    x_data_train = pd.concat([x_pos_train, x_likely_pos_train, x_weakly_pos_train, x_reliable_neg_train, x_likely_neg_train])
    y_data_train = pd.concat([y_pos_train, y_likely_pos_train, y_weakly_pos_train, y_reliable_neg_train, y_likely_neg_train])
    x_data_test = pd.concat([x_pos_test, x_neg_test])
    y_data_test = pd.concat([y_pos_test, y_neg_test])
    
    return x_data_train, y_data_train, x_data_test, y_data_test, x_all



#In[0]
def analyze_dbscan_result(gene_cluster, new_adj_mat):
    cluster_id_list = gene_cluster.cluster_id.value_counts().index.tolist()
    cluster_purity_list = []
    likely_neg_genes = []
    for i in range(len(cluster_id_list)):
        group_i = gene_cluster[gene_cluster.cluster_id.isin([cluster_id_list[i]])]
        rg_genes = group_i[group_i.label == -1]
        if rg_genes.shape[0] == 0:
            continue 
        uk_genes = group_i[group_i.label == 0]
        p_genes = group_i[group_i.label == 1]
        neg_purity = rg_genes.shape[0]/group_i.shape[0] * 100
        pos_purity = p_genes.shape[0]/group_i.shape[0] * 100
        total_members = group_i.shape[0]
        cluster_purity_list.append([neg_purity, pos_purity, total_members]) 

        for j in range(uk_genes.shape[0]):
            point_index = uk_genes.gene.iloc[j]
            distances = new_adj_mat[point_index]
            nearest_neighbors_top = np.argsort(distances)[:5]
            nearest_neighbors = nearest_neighbors_top[nearest_neighbors_top.index.isin(rg_genes.gene)]
            likely_neg_genes.append([point_index, nearest_neighbors.shape[0]])
    cluster_purity_df = pd.DataFrame(cluster_purity_list) 
    return sum(100 - cluster_purity_df.iloc[:,0]), cluster_purity_df.shape[0]

#In[0]:

import numpy as np

# Define the labels
labels = ['A', 'B', 'C']

# Create an adjacency matrix initialized with zeros
adj_matrix_binary = np.zeros((len(labels), len(labels)))

# Set the specific connections
adj_matrix_binary[labels.index('A'), labels.index('B')] = 1
adj_matrix_binary[labels.index('B'), labels.index('A')] = 1  # Assuming undirected graph
adj_matrix_binary[labels.index('C'), labels.index('A')] = 1
adj_matrix_binary[labels.index('A'), labels.index('C')] = 1  # Assuming undirected graph

print(adj_matrix_binary)
adj_matrix_binary = pd.DataFrame(adj_matrix_binary)
adj_matrix_binary.index = labels
adj_matrix_binary.columns = labels

adj_b = adj_matrix_binary 
# Create an adjacency matrix initialized with zeros
adj_matrix_values_df = pd.DataFrame(0, index=labels, columns=labels)

# Set the specific connections
adj_matrix_values_df.at['A', 'B'] = 2.5
adj_matrix_values_df.at['B', 'A'] = 2.5  # Assuming undirected graph
adj_matrix_values_df.at['C', 'A'] = 1.5
adj_matrix_values_df.at['A', 'C'] = 1.5  # Assuming undirected graph
adj_matrix_values_df.at['C', 'B'] = 4
adj_matrix_values_df.at['B', 'C'] = 4  # Assuming undirected graph

print(adj_matrix_values_df)
adj_v = adj_matrix_values_df

max_val = adj_v.max().max()
adj_v_normal = adj_v / max_val 

adj_b + adj_v_normal



# In[0]:
def identify_likely_negative(pos_genes, reliable_neg_genes_score):
    
    reliable_neg_sample_nodes = reliable_neg_genes_score.index.tolist()
    D = pairwise_distances(X=binary_gene_feature_onto, metric='euclidean', n_jobs=15)

    adj_matrix = 10 - pd.DataFrame(D)
    adj_matrix.index = binary_gene_feature_onto.index
    adj_matrix.columns = binary_gene_feature_onto.index
    max_val = adj_matrix.max().max()
    adj_matrix_normal = adj_matrix / max_val
    adj_mat_ppi_1 = adjacency_matrix_ppi[adjacency_matrix_ppi.index.isin(adj_matrix_normal.index)]
    adj_mat_ppi_2 = adj_mat_ppi_1.loc[:, adj_mat_ppi_1.columns.isin(adj_matrix_normal.index)]

    new_adj_mat = adj_mat_ppi_2 + adj_matrix_normal 
    new_adj_mat = new_adj_mat.fillna(0)
    features = adj_matrix.values    
    features = pd.DataFrame(features)

    def dbscan_grid_search(new_adj_mat, reliable_neg_sample_nodes, pos_genes):
        eps_values=np.arange(0.5, 3, 0.5)
        min_samples_values=range(2, 5)
        cluster_info = []
        # Grid search
        for eps in eps_values:
            for min_samples in min_samples_values:
                print(eps, min_samples)
                # Run DBSCAN
                clusters = DBSCAN(eps=eps, min_samples=min_samples, n_jobs= 200).fit(new_adj_mat)
                gene_cluster = pd.concat([pd.DataFrame(new_adj_mat.columns), pd.DataFrame(clusters.labels_)], axis = 1)
                gene_cluster.columns = ["gene", "cluster_id"]
                # gene_cluster['cluster_id'].value_counts()
                gene_cluster["label"] = 0
                gene_cluster.loc[gene_cluster.iloc[:,0].isin(reliable_neg_sample_nodes), "label" ] = -1
                gene_cluster.loc[gene_cluster.iloc[:,0].isin(pos_genes), "label" ] = 1

                cluster_purity, total_number_of_clusters = analyze_dbscan_result(gene_cluster, new_adj_mat)
                cluster_info.append([cluster_purity, total_number_of_clusters, eps, min_samples])
                # Only score clusters with more than 1 cluster present (excluding noise)
                # if len(set(db.labels_)) > 1:
                #     # score = silhouette_score(X, db.labels_)
                #     if score > best_score:
                #         best_score = score
                #         best_eps = eps
                #         best_min_samples = min_samples
        cluster_info_df = pd.DataFrame(cluster_info)
        cluster_info_df.columns = ["impurity", "no_of_clusters", "eps", "min_samples"]
        return cluster_info_df

    cluster_info_df = dbscan_grid_search(new_adj_mat, reliable_neg_sample_nodes, pos_genes)

    def parameter_selection_likely_neg_selection(cluster_info_df):
        min_impurity = cluster_info_df.impurity / cluster_info_df.no_of_clusters 
        eps_final = cluster_info_df.loc[min_impurity.idxmin(), "eps"]
        min_samples = cluster_info_df.loc[min_impurity.idxmin(), "min_samples"]
        clusters = DBSCAN(eps=eps_final, min_samples=min_samples, n_jobs= 200).fit(new_adj_mat)
        gene_cluster = pd.concat([pd.DataFrame(new_adj_mat.columns), pd.DataFrame(clusters.labels_)], axis = 1)
        gene_cluster.columns = ["gene", "cluster_id"]
        gene_cluster["label"] = 0
        gene_cluster.loc[gene_cluster.iloc[:,0].isin(reliable_neg_sample_nodes), "label" ] = -1
        gene_cluster.loc[gene_cluster.iloc[:,0].isin(pos_genes), "label" ] = 1

        cluster_id_list = gene_cluster.cluster_id.value_counts().index.tolist()
        likely_neg_genes = []
        cluster_purity_list = []
        for i in range(len(cluster_id_list)):
            group_i = gene_cluster[gene_cluster.cluster_id.isin([cluster_id_list[i]])]
            rg_genes = group_i[group_i.label == -1]
            uk_genes = group_i[group_i.label == 0]
            p_genes = group_i[group_i.label == 1]
            neg_purity = rg_genes.shape[0]/group_i.shape[0] 
            pos_purity = p_genes.shape[0]/group_i.shape[0] 
            total_members = group_i.shape[0]
            cluster_purity_list.append([neg_purity, pos_purity, total_members, cluster_id_list[i]]) 
        cluster_purity_df = pd.DataFrame(cluster_purity_list) 

        not_one_clust_df = cluster_purity_df[cluster_purity_df.iloc[:,0] != 1]
        cluster_ids_to_select = not_one_clust_df[not_one_clust_df.iloc[:,0] > 0.9].iloc[:,3]

        for id in cluster_ids_to_select:
            group_i = gene_cluster[gene_cluster.cluster_id.isin([id])]
            rg_genes = group_i[group_i.label == -1]
            uk_genes = group_i[group_i.label == 0]
            p_genes = group_i[group_i.label == 1]
            for j in range(uk_genes.shape[0]):
                point_index = uk_genes.gene.iloc[j]
                distances = new_adj_mat[point_index]
                nearest_neighbors_top = np.argsort(distances)[:5]
                nearest_neighbors = nearest_neighbors_top[nearest_neighbors_top.index.isin(rg_genes.gene)]
                likely_neg_genes.append([point_index, nearest_neighbors.shape[0]])
        likely_neg_genes = pd.DataFrame(likely_neg_genes)
        return likely_neg_genes.iloc[:, 0].tolist()
    likely_neg_genes = parameter_selection_likely_neg_selection(cluster_info_df)
    return likely_neg_genes


# In[1]

def mutually_exclusive_lists1(pos_genes, likely_pos_gene_score, reliable_neg_genes_score, likely_neg_genes_list):
    likely_pos_genes = list(set(likely_pos_gene_score.index.tolist()))
    reliable_neg_genes = list(set(reliable_neg_genes_score.index.tolist()))

    reliable_neg_genes = list(set(reliable_neg_genes) - set(pos_genes + likely_pos_genes ))
    likely_neg_genes = list(set(likely_neg_genes_list) - set(pos_genes + likely_pos_genes ))

    random_generator = random.Random(1)
    reliable_neg_genes = random_generator.sample(reliable_neg_genes, len(pos_genes) * 20)
    if len(likely_neg_genes) > len(pos_genes) * 5:
        likely_neg_genes = random_generator.sample(likely_neg_genes, len(pos_genes) * 5)
    else:
        likely_neg_genes = likely_neg_genes
    return pos_genes, likely_pos_genes, reliable_neg_genes, likely_neg_genes


def divide_train_test1(pos_genes, pos_genes_test, likely_pos_genes, reliable_neg_genes, likely_neg_genes):
    random_generator = random.Random(1)
    neg_genes_for_sample = list(set(gene_list.iloc[:,0]) - set(pos_genes_test.iloc[0,:].tolist() + pos_genes + likely_pos_genes + reliable_neg_genes + likely_neg_genes))
    neg_genes_test = random_generator.sample(neg_genes_for_sample, len(pos_genes) * 5)
    return neg_genes_test


def build_x_y_data_base_pos_selection_validation_set1(pos_genes, i, likely_pos_quantile):
    pos_genes_test = disease_df_test[disease_df_test.index.isin([disease_df.index[i]])].dropna(axis = 1)
    quntile_threshold = 0.01
    scores_unipath_pre = unipath_results[disease_df.index[i]]
    scores_unipath = scores_unipath_pre[scores_unipath_pre > 0]
    scores_unipath_quant = scores_unipath[scores_unipath > scores_unipath.quantile(quntile_threshold)]
    enriched_gene_sets_list = scores_unipath_quant.index.tolist()
    
    enriched_gene_sets = all_genesets[all_genesets.index.isin(enriched_gene_sets_list)]

    likely_pos_gene_score, reliable_neg_genes_score, binary_features_train_test = make_genes_ranking_likely_pos1(i, enriched_gene_sets, pos_genes, likely_pos_quantile)
    likely_neg_genes_list = identify_likely_negative(pos_genes, reliable_neg_genes_score)
    print("No. of likely positive genes:", len(likely_pos_gene_score), "No. of positive genes", len(pos_genes), "No. of likely negative genes", len(likely_neg_genes_list))

    pos_genes, likely_pos_genes, reliable_neg_genes, likely_neg_genes = mutually_exclusive_lists1(pos_genes, likely_pos_gene_score, reliable_neg_genes_score, likely_neg_genes_list)
    neg_genes_test = divide_train_test1(pos_genes, pos_genes_test, likely_pos_genes, reliable_neg_genes, likely_neg_genes)

    binary_features_train_test = binary_gene_feature_onto        
    x_pos_train = binary_features_train_test[binary_features_train_test.index.isin(pos_genes)]
    x_likely_pos_train = binary_features_train_test[binary_features_train_test.index.isin(likely_pos_genes)]
    x_reliable_neg_train = binary_features_train_test[binary_features_train_test.index.isin(reliable_neg_genes)]
    x_likely_neg_train = binary_features_train_test[binary_features_train_test.index.isin(likely_neg_genes)]

    x_pos_test = binary_features_train_test[binary_features_train_test.index.isin(pos_genes_test.iloc[0,:].tolist())]
    x_neg_test = binary_features_train_test[binary_features_train_test.index.isin(neg_genes_test)]
    x_all = binary_features_train_test

    y_pos_train = pd.Series([1] * (x_pos_train.shape[0]))
    y_likely_pos_train = pd.Series([0.9] * (x_likely_pos_train.shape[0]))
    y_reliable_neg_train = pd.Series([0] * (x_reliable_neg_train.shape[0]))
    y_likely_neg_train = pd.Series([0.1] * (x_likely_neg_train.shape[0]))

    y_pos_test = pd.Series([1] * (x_pos_test.shape[0]))
    y_neg_test = pd.Series([0] * (x_neg_test.shape[0]))

    x_data_train = pd.concat([x_pos_train, x_likely_pos_train, x_reliable_neg_train, x_likely_neg_train])
    y_data_train = pd.concat([y_pos_train, y_likely_pos_train, y_reliable_neg_train, y_likely_neg_train])
    x_data_test = pd.concat([x_pos_test, x_neg_test])
    y_data_test = pd.concat([y_pos_test, y_neg_test])

    return x_data_train, y_data_train, x_data_test, y_data_test, x_all

#In[0]:
def select_diseases(algo_type, feature_type):
    disease_df_selected = None; disease_df_test_selected = None
    if algo_type == "base" and feature_type == "onto_binary":
        disease_df_selected = disease_df
        disease_df_test_selected = disease_df_test
    if algo_type == "base" and feature_type == "binary_epige" or feature_type == "binary_onto_epige":
        sum_score_more =  unipath_epigenome.sum(0) > 10
        select_index = sum_score_more[sum_score_more == True].index
        disease_df_selected = disease_df[disease_df.index.isin(select_index)]
        disease_df_test_selected = disease_df_test[disease_df_test.index.isin(select_index)]
    if algo_type == "selecting_pos" and feature_type == "onto_binary":
        disease_df_selected = disease_df
        disease_df_test_selected = disease_df_test
    if algo_type == "selecting_pos_onto_epigene" or algo_type == "selecting_pos_epigene":
        sum_score_more =  unipath_epigenome.sum(0) > 10
        select_index = sum_score_more[sum_score_more == True].index
        disease_df_selected = disease_df[disease_df.index.isin(select_index)]
        disease_df_test_selected = disease_df_test[disease_df_test.index.isin(select_index)]

    return disease_df_selected, disease_df_test_selected
#In[0]:
def build_x_y_data_combined_feature(disease_df_selected, disease_df_test_selected, pos_genes, i):
    pos_genes_test = disease_df_test_selected[disease_df_test_selected.index.isin([disease_df_test_selected.index[i]])].dropna(axis = 1)
    quntile_threshold = 0.02
    scores_unipath_pre = unipath_results[disease_df_selected.index[i]]
    scores_unipath = scores_unipath_pre[scores_unipath_pre > 0]
    scores_unipath_quant = scores_unipath[scores_unipath > scores_unipath.quantile(quntile_threshold)]
    enriched_gene_sets_list = scores_unipath_quant.index.tolist()
    enriched_gene_sets = all_genesets[all_genesets.index.isin(enriched_gene_sets_list)]

    scores_unipath_epi_pre = unipath_epigenome[disease_df_selected.index[i]]
    scores_unipath_epi = scores_unipath_epi_pre[scores_unipath_epi_pre > 0]
    scores_unipath_epi_quant = scores_unipath_epi[scores_unipath_epi > scores_unipath_epi.quantile(quntile_threshold)]
    enriched_gene_sets_epi_list = scores_unipath_epi_quant.index.tolist()
    enriched_gene_sets_epi = gene_set_binary_epigenome[gene_set_binary_epigenome.index.isin(enriched_gene_sets_epi_list)]

    likely_pos_gene, reliable_neg_genes_score, binary_features_train_test = make_genes_ranking_likely_pos_epigenome_onto(i, enriched_gene_sets, pos_genes, enriched_gene_sets_epi, disease_df_selected, likely_pos_quantile)

    print("No. of likely positive genes:", len(likely_pos_gene), "  No. of positive genes", len(pos_genes))

    pos_all_genes = pos_genes + likely_pos_gene + pos_genes_test.iloc[0,:].tolist()
    selected_pos_genes = pos_genes + likely_pos_gene
    neg_genes_train_validation_pre =  list(set(reliable_neg_genes_score) - set(pos_all_genes))
    random_generator = random.Random(1)
    neg_genes_train_val = random_generator.sample(neg_genes_train_validation_pre, len(selected_pos_genes) * 5)

    neg_genes_test_for_sample = set(neg_genes_train_validation_pre) - set(neg_genes_train_val) 
    random_generator = random.Random(1)
    neg_genes_test = random_generator.sample(neg_genes_test_for_sample, pos_genes_test.shape[1] * 5)

    x_pos_train_val = binary_features_train_test[binary_features_train_test.index.isin(pos_genes)]
    x_neg_train_val = binary_features_train_test[binary_features_train_test.index.isin(neg_genes_train_val)]
    x_pos_test = binary_features_train_test[binary_features_train_test.index.isin(pos_genes_test.iloc[0,:].tolist())]
    x_neg_test = binary_features_train_test[binary_features_train_test.index.isin(neg_genes_test)]
    x_all = binary_features_train_test
    
    y_pos_train_val = pd.Series([1] * (x_pos_train_val.shape[0]))
    y_neg_train_val = pd.Series([0] * (x_neg_train_val.shape[0]))
    y_pos_test = pd.Series([1] * (x_pos_test.shape[0]))
    y_neg_test = pd.Series([0] * (x_neg_test.shape[0]))

    x_data_train_valid = pd.concat([x_pos_train_val, x_neg_train_val])
    y_data_train_valid = pd.concat([y_pos_train_val, y_neg_train_val])
    x_data_test = pd.concat([x_pos_test, x_neg_test])
    y_data_test = pd.concat([y_pos_test, y_neg_test])
    
    return x_data_train_valid, y_data_train_valid, x_data_test, y_data_test, x_all

# In[0]:
def make_genes_ranking_likely_pos_epigenome_only(i, pos_genes, enriched_gene_sets_epi, disease_df_selected, likely_pos_quantile):
    gene_set_scores_epi = unipath_epigenome.loc[unipath_epigenome.index.isin(enriched_gene_sets_epi.index.tolist()), disease_df_selected.index[i]] 
    binary_gene_feature_epigenome_pos = binary_gene_feature_epigenome[enriched_gene_sets_epi.index.tolist()]
    gene_score_pre_epi = binary_gene_feature_epigenome_pos.dot(gene_set_scores_epi).sort_values(ascending=False)
    gene_score_epi = gene_score_pre_epi[gene_score_pre_epi > 0]
    gene_score_epi = gene_score_epi[~gene_score_epi.index.isin(pos_genes)]
    likely_pos_gene_score_epi = gene_score_epi[gene_score_epi >= gene_score_epi.quantile(likely_pos_quantile)]
    reliable_neg_genes_score_epi = gene_score_pre_epi[gene_score_pre_epi <= 0]
    binary_features_train_test_epi = pd.concat([binary_gene_feature_epigenome_pos, binary_gene_feature_epigenome_pos.dot(gene_set_scores_epi)], axis = 1)
    binary_features_train_test_epi.columns = binary_gene_feature_epigenome_pos.columns.tolist() + ["score_epi"]
    
    return likely_pos_gene_score_epi, reliable_neg_genes_score_epi, binary_features_train_test_epi

# In[0]:

def build_x_y_data_epi_feature(disease_df_selected, disease_df_test_selected, pos_genes, i):
    pos_genes_test = disease_df_test_selected[disease_df_test_selected.index.isin([disease_df_test_selected.index[i]])].dropna(axis = 1)
    quntile_threshold = 0.02
    scores_unipath_epi_pre = unipath_epigenome[disease_df_selected.index[i]]
    scores_unipath_epi = scores_unipath_epi_pre[scores_unipath_epi_pre > 0]
    scores_unipath_epi_quant = scores_unipath_epi[scores_unipath_epi > scores_unipath_epi.quantile(quntile_threshold)]
    enriched_gene_sets_epi_list = scores_unipath_epi_quant.index.tolist()
    enriched_gene_sets_epi = gene_set_binary_epigenome[gene_set_binary_epigenome.index.isin(enriched_gene_sets_epi_list)]

    likely_pos_gene, reliable_neg_genes_score, binary_features_train_test = make_genes_ranking_likely_pos_epigenome_only(i, pos_genes, enriched_gene_sets_epi, disease_df_selected, likely_pos_quantile)

    pos_all_genes = pos_genes + likely_pos_gene.index.tolist() + pos_genes_test.iloc[0,:].tolist()
    selected_pos_genes = pos_genes + likely_pos_gene.index.tolist()
    neg_genes_train_validation_pre =  list(set(reliable_neg_genes_score.index.tolist()) - set(pos_all_genes))
    random_generator = random.Random(1)
    neg_genes_train_val = random_generator.sample(neg_genes_train_validation_pre, len(selected_pos_genes) * 5)

    neg_genes_test_for_sample = set(neg_genes_train_validation_pre) - set(neg_genes_train_val) 
    random_generator = random.Random(1)
    neg_genes_test = random_generator.sample(neg_genes_test_for_sample, pos_genes_test.shape[1] * 5)

    x_pos_train_val = binary_features_train_test[binary_features_train_test.index.isin(pos_genes)]
    x_neg_train_val = binary_features_train_test[binary_features_train_test.index.isin(neg_genes_train_val)]
    x_pos_test = binary_features_train_test[binary_features_train_test.index.isin(pos_genes_test.iloc[0,:].tolist())]
    x_neg_test = binary_features_train_test[binary_features_train_test.index.isin(neg_genes_test)]
    x_all = binary_features_train_test
    y_pos_train_val = pd.Series([1] * (x_pos_train_val.shape[0]))
    y_neg_train_val = pd.Series([0] * (x_neg_train_val.shape[0]))
    y_pos_test = pd.Series([1] * (x_pos_test.shape[0]))
    y_neg_test = pd.Series([0] * (x_neg_test.shape[0]))

    x_data_train_valid = pd.concat([x_pos_train_val, x_neg_train_val])
    y_data_train_valid = pd.concat([y_pos_train_val, y_neg_train_val])
    x_data_test = pd.concat([x_pos_test, x_neg_test])
    y_data_test = pd.concat([y_pos_test, y_neg_test])
    
    return x_data_train_valid, y_data_train_valid, x_data_test, y_data_test, x_all

# In[1]: 
def call_ml_model(start_i, end_i, algo_type, feature_type, likely_pos_quantile):
    model_metrics = list()
    predicted_genes = pd.DataFrame()
    
    for i in range(start_i, end_i):
        print(i)
        try:
            if algo_type == "base":
                disease_df, disease_df_test = select_diseases(algo_type, feature_type)
                print(disease_df.index[i])
                pos_genes = disease_df.iloc[i, :].dropna().tolist()
                disease_name = disease_df.index[i]
                x_data_train_valid, y_data_train_valid, x_data_test, y_data_test, x_all = build_x_y_data_base_validation(pos_genes, i, disease_df, disease_df_test)
                
            elif algo_type == "selecting_pos":
                disease_df, disease_df_test = select_diseases(algo_type, feature_type)
                pos_genes = disease_df.iloc[i, :].dropna().tolist()
                #x_data_train_valid, y_data_train_valid, x_data_test, y_data_test, x_all = build_x_y_data_base_pos_selection_validation_set(pos_genes, i, feature_type, likely_pos_quantile)
                x_data_train_valid, y_data_train_valid, x_data_test, y_data_test, x_all = build_x_y_data_base_pos_selection_validation_set1(pos_genes, i, likely_pos_quantile)
                disease_name = disease_df.index[i]
                
            elif algo_type == "selecting_pos_onto_epigene":
                disease_df_selected, disease_df_test_selected = select_diseases(algo_type, feature_type)
                pos_genes = disease_df_selected.iloc[i, :].dropna().tolist()
                print(disease_df_selected.index[i])
                x_data_train_valid, y_data_train_valid, x_data_test, y_data_test, x_all = build_x_y_data_combined_feature(disease_df_selected, disease_df_test_selected, pos_genes, i)
                disease_name = disease_df_selected.index[i]

            elif algo_type == "selecting_pos_epigene":
                disease_df_selected, disease_df_test_selected = select_diseases(algo_type, feature_type)
                pos_genes = disease_df_selected.iloc[i, :].dropna().tolist()
                print(disease_df_selected.index[i])
                x_data_train_valid, y_data_train_valid, x_data_test, y_data_test, x_all = build_x_y_data_epi_feature(disease_df_selected, disease_df_test_selected, pos_genes, i)
                disease_name = disease_df_selected.index[i]

            estimator = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=50)
            score_dict, pred_prbs_all = cross_val_score_defined(estimator, x_data_train_valid, y_data_train_valid, x_data_test, y_data_test, x_all, i)
            #final_pred = predict_genes(pred_prbs_all, pos_genes); final_pred.columns = [disease_name]
            pred_prbs_all.columns = [disease_name]
            predicted_genes = pd.concat([predicted_genes, pred_prbs_all], axis = 1)
            
            accuracy = sum(score_dict['train_accuracy'])/len(score_dict['train_accuracy'])
            balanced_accuracy = sum(score_dict['train_balanced_accuracy'])/len(score_dict['train_balanced_accuracy'])
            precision = sum(score_dict['train_precision'])/len(score_dict['train_precision'])
            recall = sum(score_dict['train_recall'])/len(score_dict['train_recall'])
            train_f1_score = sum(score_dict['train_f1_score'])/len(score_dict['train_f1_score'])
            test_accuracy = sum(score_dict['test_accuracy'])/len(score_dict['test_accuracy'])
            test_balanced_accuracy = sum(score_dict['test_balanced_accuracy'])/len(score_dict['test_balanced_accuracy'])
            test_precision = sum(score_dict['test_precision'])/len(score_dict['test_precision'])
            test_recall = sum(score_dict['test_recall'])/len(score_dict['test_recall'])
            test_f1_score = sum(score_dict['test_f1_score'])/len(score_dict['test_f1_score'])
            test_auc = sum(score_dict['test_auc'])/len(score_dict['test_auc'])

            val_acc = sum(score_dict['val_acc'])/len(score_dict['val_acc'])
            val_bal_accu = sum(score_dict['val_bal_accu'])/len(score_dict['val_bal_accu'])
            val_precision = sum(score_dict['val_precision'])/len(score_dict['val_precision'])
            val_recall = sum(score_dict['val_recall'])/len(score_dict['val_recall'])
            val_f1_score = sum(score_dict['val_f1_score'])/len(score_dict['val_f1_score'])
            val_auc = sum(score_dict['val_auc'])/len(score_dict['val_auc'])
            
            model_metric_i = [disease_name, accuracy, balanced_accuracy, precision, recall, train_f1_score,
                            val_acc, val_bal_accu, val_precision, val_recall, val_f1_score, val_auc,
                            test_accuracy, test_balanced_accuracy, test_precision, test_recall, test_f1_score, test_auc]
            model_metrics.append(model_metric_i)
        except:
            pass
    model_metrics_df = pd.DataFrame(model_metrics)
    model_metrics_df.columns = ["disease_name", "train_accu", "train_bal_accu", "train_precision", "train_recall", "train_f1_score",
                                "val_acc", "val_bal_accu", "val_precision", "val_recall", "val_f1_score", "val_auc", 
                                 "test_accu", "test_bal_accu", "test_precision", "test_recall", "test_f1", "test_auc"]
    return model_metrics_df.round(4), predicted_genes

# In[1]: 
# algo_type = "base" or "selecting_pos" or "selecting_pos_onto_epigene" or "selecting_pos_epigene"
# feature_type = "binary_epige" or "binary_onto_epige" or "onto_binary" or "epigenome"
# start_i = 0
# end_i = 1
# algo_type = "selecting_pos" or "selecting_pos_onto_epigene"
# start_i = 0
# end_i = 1
# algo_type = "selecting_pos"
# feature_type = "onto_binary"
likely_pos_quantile = 0.99
# feature_type = "onto_binary" 
model_metrics_out, predicted_genes = call_ml_model(start_i, end_i, algo_type, feature_type, likely_pos_quantile)
model_metrics_out.to_csv('{}{}{}{}{}{}{}{}{}{}{}'.format('results/', algo_type, "_", feature_type, "_", start_i, "_", end_i, "_", likely_pos_quantile, '_likely_neg1.csv'))
predicted_genes.to_csv('{}{}{}{}{}{}{}{}{}{}{}'.format('results/dir_predicted_genes/predicted_genes_', algo_type, "_", feature_type, "_", start_i, "_", end_i, "_", likely_pos_quantile, '_likely_neg1.csv'))
