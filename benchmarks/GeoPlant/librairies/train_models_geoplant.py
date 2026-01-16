import os
# import wandb

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from librairies.losses import bce_loss, ce_loss, deepmaxent_loss, poisson_loss
from librairies.model import TimeSpectralViT, make_predictions, ModifiedResNet18, ModifiedVisionTransformer,HierarchicalTransformer
import numpy as np
import rasterio
from rasterio.mask import mask
import torch
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from scipy.stats import pearsonr
from scipy import stats
from librairies.utils import set_seed
import copy
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import KFold
import verde as vd
from librairies.model import deepmaxent_model, make_predictions


def resize_array(array_3d, new_shape, categorical_indices,sum_option=False):
    resized_array = np.zeros(new_shape)
    block_shape = tuple(old_dim // new_dim for old_dim, new_dim in zip(array_3d.shape, new_shape))
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                block = array_3d[i*block_shape[0]:(i+1)*block_shape[0], j*block_shape[1]:(j+1)*block_shape[1], k*block_shape[2]:(k+1)*block_shape[2]]
                if k in categorical_indices:
                    resized_array[i, j, k] = stats.mode(block, axis=None).mode
                else:
                    if(sum_option):
                        resized_array[i, j, k] = np.sum(block)
                    else:
                        resized_array[i, j, k] = np.mean(block)

    return resized_array

def cv_eval_deepmodel(args):

    results_df = pd.DataFrame(columns=["lr","weight_decay","batch_size","hidden_nbr", "Seed","block_i", "loss_cal", "loss_val","AUC_val"])        
    n = 0
    
    # results_df = pd.DataFrame(columns=["Seed", "AUC_macro", "AUC_micro", "Pearson"])

    PO = pd.read_csv(args.trainPOfile)
    PA = pd.read_csv(args.trainPAfile)
    PA_test = pd.read_csv(args.testPAfile)
    
    # Keep only 2020 data
    PO = PO[PO["year"] == 2020]
    PA = PA[PA["year"] == 2020]
    
    
    
    # Filter PO and PA to have common species in both datasets
    PO_sub = PO[PO["speciesId"].isin(PA["speciesId"])]
    PA_sub = PA[PA["speciesId"].isin(PO["speciesId"])]
    
    unique_spid = PO_sub["speciesId"].unique()
    # unique_spid = unique_spid[:500]
    unique_spid = np.random.choice(unique_spid, size=args.species_nbr, replace=False)
    selected_species = unique_spid

    PO_sub = PO_sub[PO_sub["speciesId"].isin(selected_species)]
    PA_sub = PA_sub[PA_sub["speciesId"].isin(selected_species)]
    # PA_test_sub = PA_test[PA_test["speciesId"].isin(PO_sub["speciesId"])]
    # Regrouper PO_sub et PA_sub par surveyId, speciesId, lat et lon
    PO_sub = PO_sub.groupby("surveyId").agg({"speciesId": list, "lat": "first", "lon": "first"}).reset_index()
    PA_sub = PA_sub.groupby("surveyId").agg({"speciesId": list, "lat": "first", "lon": "first"}).reset_index()
    
    # # Compute PO_sub with surveyId that are in PO_data_cleaned
    # PO_sub = PO_sub[PO_sub["surveyId"].isin(PO_data_cleaned["red"]["surveyId"])]
    # # recheck if PA have same speciesId
    # PA_sub = PA_sub[PA_sub["speciesId"].isin(PO_sub["speciesId"])]
    
    # PO_sub['speciesId']=PO_sub['speciesId'].astype(int)
    # PA_sub['speciesId']=PA_sub['speciesId'].astype(int)
    
    # Now, let's compute y_target as a matrix of 0 and 1 when surveyId as row and colonn corresponding to surveyId unique and similair with PO and PA 
    # unique_spid = PO_sub["speciesId"].unique()
    
    # Filter PO_sub, PA_sub, PA_data, and PO_data_cleaned based on the selected species



    # Utilisation de la fonction pour PO
    po_files = {
        "blue": args.PO_blue_file,
        "green": args.PO_green_file,
        "red": args.PO_red_file,
        "nir": args.PO_nir_file,
        "swir1": args.PO_swir1_file,
        "swir2": args.PO_swir2_file
    }



    # Utilisation de la fonction pour PA (si nécessaire)
    pa_files = {
        "blue": args.PA_blue_file,
        "green": args.PA_green_file,
        "red": args.PA_red_file,
        "nir": args.PA_nir_file,
        "swir1": args.PA_swir1_file,
        "swir2": args.PA_swir2_file
    }
    
    
    PO_data = load_and_filter_data(po_files, PO_sub)
    PA_data = load_and_filter_data(pa_files, PA_sub)
    
    num_lines = len(next(iter(PO_data.values())))

    # Initialiser un tensor vide
    X_train = np.zeros((num_lines, 6, 4, 21))
    # Initialiser un tensor vide
    # tensor = np.zeros((6, 4, 21))

    # Dictionnaire pour mapper les bandes aux indices
    band_indices = {'blue': 0, 'green': 1, 'red': 2, 'nir': 3, 'swir1': 4, 'swir2': 5}

    # Remplir le tensor avec les données
    for band, data in PO_data.items():
        band_index = band_indices[band]
        for year in range(2000, 2021):
            for quarter in range(1, 5):
                column_name = f"{year}_{quarter}"
                if column_name in data.columns:
                    X_train[:,band_index, quarter-1, year-2000] = data[column_name].values
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_train = torch.nan_to_num(X_train)
    
    num_lines = len(next(iter(PA_data.values())))

    # Initialiser un tensor vide
    X_Validation = np.zeros((num_lines, 6, 4, 21))
        # Remplir le tensor avec les données
    for band, data in PA_data.items():
        band_index = band_indices[band]
        for year in range(2000, 2021):
            for quarter in range(1, 5):
                column_name = f"{year}_{quarter}"
                if column_name in data.columns:
                    X_Validation[:,band_index, quarter-1, year-2000] = data[column_name].values
    X_Validation = torch.tensor(X_Validation, dtype=torch.float32)
    X_Validation = torch.nan_to_num(X_Validation)
    
    
    # Now let's make a matrix of 0 and 1 with surveyId as row and speciesId as colonn
    unique_siteid_po = PO_sub["surveyId"].unique()
    unique_siteid_pa = PA_sub["surveyId"].unique()
    y_train_po = torch.zeros(len(unique_siteid_po), len(unique_spid))
    for idx, siteid in enumerate(unique_siteid_po):
        speciesIdinthissite = PO_sub[PO_sub["surveyId"] == siteid]
        for species_list in speciesIdinthissite["speciesId"]:
            for spid in species_list:
                idx_spid = np.where(unique_spid == spid)[0][0]
                y_train_po[idx, idx_spid] += 1
        # unique_speciesIdinthissite = speciesIdinthissite["speciesId"].unique()
        # for spid in unique_speciesIdinthissite:
        #     idx_spid = np.where(unique_spid == spid)[0][0]
        #     y_tensor_po[idx, idx_spid] = 1
        
    # Now let's make a matrix of 0 and 1 with surveyId as row and speciesId as colonn
    y_validation_pa = torch.zeros(len(unique_siteid_pa), len(unique_spid))
    for idx, siteid in enumerate(unique_siteid_pa):
        speciesIdinthissite = PA_sub[PA_sub["surveyId"] == siteid]
        for species_list in speciesIdinthissite["speciesId"]:
            for spid in species_list:
                idx_spid = np.where(unique_spid == spid)[0][0]
                y_validation_pa[idx, idx_spid] += 1
        
        
    y_validation_pa[y_validation_pa>1] = 1
    
    
    # occurrence_grid_species = np.zeros(
    #     (nrows, ncols, len(PO_sub["spid"].unique())),dtype=np.int16
    # )

    # idx = 0
    # for spid in tqdm(PO_sub["spid"].unique()):
    #     tab_df_sub = PO_sub[PO_sub["spid"] == spid]
    #     grouped = (
    #         tab_df_sub.groupby(["row", "col", "spid"])
    #         .size()
    #         .reset_index(name="count")
    #     )
    #     occurrence_grid_species[grouped["row"], grouped["col"], idx] = grouped[
    #         "count"
    #     ]
    #     idx += 1
        

    # X_occurence_reshape = reshape_arr(occurrence_grid_species)
    # X_occurence_reshape = mask_arr(X_occurence_reshape, mask_real)
    # y_tensor_plant = torch.tensor(X_occurence_reshape, dtype=torch.float32)
    
    # idx_to_keep = np.where(X_occurence_reshape.sum(1) != 0)[0]
    # zero_indices = np.where(X_occurence_reshape.sum(1) == 0)[0]
    # random_zero_indices = np.random.choice(
    #     zero_indices, size=min(len(idx_to_keep), len(zero_indices)), replace=False
    # )
    # random_zero_indices = np.random.choice(
    #     zero_indices, size=min(0, len(zero_indices)), replace=False
    # )


    # idx_to_keep = np.concatenate([idx_to_keep, random_zero_indices])
    # y_tensor_plant = y_tensor_plant[idx_to_keep, :]
    # X_tensor = X_tensor[idx_to_keep, :]
    # X_lat_long = X_lat_long[idx_to_keep, :]

    if args.cross_validation == "plain":
        kfold = KFold(n_splits=int(1/args.validation_size), shuffle=True,random_state=args.global_seed).split(X_train)
    elif args.cross_validation == "blocked":
        kfold = vd.BlockKFold(shape=args.num_cv_blocks, n_splits=int(1/args.validation_size), shuffle=True, balance=True, random_state=args.global_seed).split(PO_sub[['lat', 'lon']].to_numpy())
    x_trains = []
    y_trains = []
    x_vals = []
    y_vals = []
    for ind_train, ind_val in kfold:
        x_trains.append(X_train[ind_train])
        y_trains.append(y_train_po[ind_train])
        x_vals.append(X_train[ind_val])
        y_vals.append(y_train_po[ind_val])
    
    for hidden_nbr in args.hidden_nbr_tested:
        args.hidden_nbr = hidden_nbr
        for lr in args.lr_tested:
            args.learning_rate = lr
            for weight_decay in args.w_tested:
                args.weight_decay = weight_decay
                for batch_size in args.batch_tested:
                    args.batch_size = batch_size
                    for idx_seed in range(args.repeat_seed):
                        for iblock_i in range(len(x_trains)):
                            X_cal = x_trains[iblock_i]
                            y_cal = y_trains[iblock_i]
                            X_val = x_vals[iblock_i]
                            y_val = y_vals[iblock_i]
                            set_seed(args.list_of_seed[idx_seed])
                            results = cv_deepmodel(
                                X_cal,
                                y_cal,
                                args,
                                X_val, 
                                y_val
                            )

                            loss_val_array = np.array([t.item() for t in results['loss_val_by_batch']])
                            loss_cal_array = np.array(results['loss_by_batch'])
                            
                            AUC_val = np.array(results['AUC_val'])
                            

                            # Find val loss min index 
                            min_val_loss_idx = np.argmin(loss_val_array)
                            min_val_loss = loss_val_array[min_val_loss_idx]
                            min_val_loss_cal = loss_cal_array[min_val_loss_idx]
                            
                            
                            new_row = pd.DataFrame(
                                {
                                    "lr": [lr],
                                    "weight_decay": [weight_decay],
                                    "batch_size": [batch_size],
                                    "hidden_nbr": [hidden_nbr],
                                    "Seed": [args.list_of_seed[idx_seed]],
                                    "block_i": [iblock_i],
                                    "loss_cal": [min_val_loss_cal],
                                    "loss_val": [min_val_loss],
                                    "AUC_val": [AUC_val]
                                }
                            )

                            print(new_row)
                            results_df = pd.concat([results_df, new_row], ignore_index=True)
                            new_row.to_csv(
                                f"{args.outputdir}/results_backup_cv.csv",
                                mode="a",
                                header=not os.path.exists(
                                    f"{args.outputdir}/results_backup_cv.csv"
                                ),
                                index=False,
                            )
    results_df.to_csv(f"{args.outputdir}/results_cv.csv", index=False) 



def load_and_filter_data(file_dict, subset):
    data_dict = {}
    for band, file_path in file_dict.items():
        data = pd.read_csv(file_path)
        data = data[data["surveyId"].isin(subset["surveyId"])]
        data_dict[band] = data
    return data_dict   
def find_nan_columns(data_dict):
    nan_columns_dict = {}
    for band, data in data_dict.items():
        nan_columns_dict[band] = data.columns[data.isna().any()].tolist()
    
    # Find columns that are NaN for at least one band
    common_nan_columns = list(set().union(*nan_columns_dict.values()))
    
    return nan_columns_dict, common_nan_columns   

def compute_mean_std(data_dict, common_columns):
    mean_dict = {}
    std_dict = {}
    for band, data in data_dict.items():
        filtered_data = data.drop(columns=common_columns)
        mean_dict[band] = filtered_data.mean(0)
        std_dict[band] = filtered_data.std(0)
    return mean_dict, std_dict

def count_nan_per_column(data_dict):
    nan_count_dict = {}
    for band, data in data_dict.items():
        nan_count_dict[band] = data.isna().sum()
    return nan_count_dict

def remove_rows_with_nan(data_dict):
    # Find all surveyIds with NaNs in any DataFrame
    survey_ids_with_nan = set()
    for data in data_dict.values():
        survey_ids_with_nan.update(data.loc[data.isna().any(axis=1), 'surveyId'])
    
    # Remove rows with these surveyIds from all DataFrames
    for band in data_dict:
        data_dict[band] = data_dict[band][~data_dict[band]['surveyId'].isin(survey_ids_with_nan)]
    
    return data_dict


def process_and_concatenate(data_dict, bands):
    scalers = {}
    X_train_list = []
    X_validation_list = []
    column_indices = {}

    start_idx = 0
    for band in bands:
        scaler = StandardScaler(with_std=False)
        X_train_band = torch.tensor(scaler.fit_transform(data_dict[band]["train"].values[:, 1:]), dtype=torch.float32)
        X_validation_band = torch.tensor(scaler.transform(data_dict[band]["validation"].values[:, 1:]), dtype=torch.float32)
        
        X_train_list.append(X_train_band)
        X_validation_list.append(X_validation_band)
        
        end_idx = start_idx + X_train_band.shape[1]
        column_indices[band] = (start_idx, end_idx)
        start_idx = end_idx
        
        scalers[band] = scaler

    X_train = torch.cat(X_train_list, dim=1)
    X_validation = torch.cat(X_validation_list, dim=1)

    return X_train, X_validation, column_indices, scalers


def eval_deepmodel(args):

    results_df = pd.DataFrame(columns=["Seed", "AUC_macro", "AUC_micro", "Pearson"])

    PO = pd.read_csv(args.trainPOfile)
    PA = pd.read_csv(args.trainPAfile)
    
    # # Keep only 2020 data
    # PO = PO[PO["year"] == 2020]
    # PA = PA[PA["year"] == 2020]
    
    # C'est pas 2017 le problème
    PO = PO[PO["year"] != args.year_to_remove]
    PA = PA[PA["year"] != args.year_to_remove]
    
    
    
    
    # Filter PO and PA to have common species in both datasets
    PO_sub = PO[PO["speciesId"].isin(PA["speciesId"])]
    PA_sub = PA[PA["speciesId"].isin(PO["speciesId"])]
    
    
    unique_spid = PO_sub["speciesId"].unique()
    # unique_spid = unique_spid[:500]
    unique_spid = np.random.choice(unique_spid, size=args.species_nbr, replace=False)
    selected_species = unique_spid

    PO_sub = PO_sub[PO_sub["speciesId"].isin(selected_species)]
    PA_sub = PA_sub[PA_sub["speciesId"].isin(selected_species)]
    # PA_test_sub = PA_test[PA_test["speciesId"].isin(PO_sub["speciesId"])]

    # Regrouper PO_sub et PA_sub par surveyId
    PO_sub = PO_sub.groupby("surveyId").agg({"speciesId": list}).reset_index()
    PA_sub = PA_sub.groupby("surveyId").agg({"speciesId": list}).reset_index()
    
    
    # Select some surveyIds to create a PA_test
    selected_survey_ids = np.random.choice(PA_sub["surveyId"].unique(), size=int(len(PA_sub) * 0.2), replace=False)
    PA_test = PA_sub[PA_sub["surveyId"].isin(selected_survey_ids)]
    PA_sub = PA_sub[~PA_sub["surveyId"].isin(selected_survey_ids)]
    
    
    # # Compute PO_sub with surveyId that are in PO_data_cleaned
    # PO_sub = PO_sub[PO_sub["surveyId"].isin(PO_data_cleaned["red"]["surveyId"])]
    # # recheck if PA have same speciesId
    # PA_sub = PA_sub[PA_sub["speciesId"].isin(PO_sub["speciesId"])]
    
    # PO_sub['speciesId']=PO_sub['speciesId'].astype(int)
    # PA_sub['speciesId']=PA_sub['speciesId'].astype(int)
    
    # Now, let's compute y_target as a matrix of 0 and 1 when surveyId as row and colonn corresponding to surveyId unique and similair with PO and PA 
    # unique_spid = PO_sub["speciesId"].unique()
    
    # Filter PO_sub, PA_sub, PA_data, and PO_data_cleaned based on the selected species



    # Utilisation de la fonction pour PO
    po_files = {
        "blue": args.PO_blue_file,
        "green": args.PO_green_file,
        "red": args.PO_red_file,
        "nir": args.PO_nir_file,
        "swir1": args.PO_swir1_file,
        "swir2": args.PO_swir2_file
    }



    # Utilisation de la fonction pour PA (si nécessaire)
    pa_files = {
        "blue": args.PA_blue_file,
        "green": args.PA_green_file,
        "red": args.PA_red_file,
        "nir": args.PA_nir_file,
        "swir1": args.PA_swir1_file,
        "swir2": args.PA_swir2_file
    }
    
    
    
    PO_data = load_and_filter_data(po_files, PO_sub)
    PA_data = load_and_filter_data(pa_files, PA_sub)
    PA_test_data = load_and_filter_data(pa_files, PA_test)



    if args.architecture == "transformer":
        num_lines = len(next(iter(PO_data.values())))

        # Initialiser un tensor vide
        X_train = np.zeros((num_lines, 6, 4* 21))
        # Initialiser un tensor vide
        # tensor = np.zeros((6, 4, 21))

        # Dictionnaire pour mapper les bandes aux indices
        band_indices = {'blue': 0, 'green': 1, 'red': 2, 'nir': 3, 'swir1': 4, 'swir2': 5}

        # Remplir le tensor avec les données

        for band, data in PO_data.items():
            band_index = band_indices[band]
            for year in range(2000, 2021):
                for quarter in range(1, 5):
                    column_name = f"{year}_{quarter}"
                    if column_name in data.columns:
                        X_train[:, band_index, (year-2000)*4 + (quarter-1)] = data[column_name].values
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_train = torch.nan_to_num(X_train)
        
        num_lines = len(next(iter(PA_data.values())))

        # Initialiser un tensor vide
        X_Validation = np.zeros((num_lines, 6, 4* 21))
            # Remplir le tensor avec les données
        for band, data in PA_data.items():
            band_index = band_indices[band]
            for year in range(2000, 2021):
                for quarter in range(1, 5):
                    column_name = f"{year}_{quarter}"
                    if column_name in data.columns:
                        X_Validation[:, band_index, (year-2000)*4 + (quarter-1)] = data[column_name].values
        X_Validation = torch.tensor(X_Validation, dtype=torch.float32)
        X_Validation = torch.nan_to_num(X_Validation)
        
        
        num_lines = len(next(iter(PA_test_data.values())))

        # Initialiser un tensor vide
        X_test = np.zeros((num_lines, 6, 4* 21))
            # Remplir le tensor avec les données
        for band, data in PA_test_data.items():
            band_index = band_indices[band]
            for year in range(2000, 2021):
                for quarter in range(1, 5):
                    column_name = f"{year}_{quarter}"
                    if column_name in data.columns:
                        X_test[:, band_index, (year-2000)*4 + (quarter-1)] = data[column_name].values
        X_test = torch.tensor(X_test, dtype=torch.float32)
        X_test = torch.nan_to_num(X_test)



    else:

    
        num_lines = len(next(iter(PO_data.values())))

        # Initialiser un tensor vide
        X_train = np.zeros((num_lines, 6, 4, 21))
        # Initialiser un tensor vide
        # tensor = np.zeros((6, 4, 21))

        # Dictionnaire pour mapper les bandes aux indices
        band_indices = {'blue': 0, 'green': 1, 'red': 2, 'nir': 3, 'swir1': 4, 'swir2': 5}





        # Remplir le tensor avec les données
        for band, data in PO_data.items():
            band_index = band_indices[band]
            for year in range(2000, 2021):
                for quarter in range(1, 5):
                    column_name = f"{year}_{quarter}"
                    if column_name in data.columns:
                        X_train[:,band_index, quarter-1, year-2000] = data[column_name].values
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_train = torch.nan_to_num(X_train)
        
        num_lines = len(next(iter(PA_data.values())))

        # Initialiser un tensor vide
        X_Validation = np.zeros((num_lines, 6, 4, 21))
            # Remplir le tensor avec les données
        for band, data in PA_data.items():
            band_index = band_indices[band]
            for year in range(2000, 2021):
                for quarter in range(1, 5):
                    column_name = f"{year}_{quarter}"
                    if column_name in data.columns:
                        X_Validation[:,band_index, quarter-1, year-2000] = data[column_name].values
        X_Validation = torch.tensor(X_Validation, dtype=torch.float32)
        X_Validation = torch.nan_to_num(X_Validation)
        
        
        num_lines = len(next(iter(PA_test_data.values())))

        # Initialiser un tensor vide
        X_test = np.zeros((num_lines, 6, 4, 21))
            # Remplir le tensor avec les données
        for band, data in PA_test_data.items():
            band_index = band_indices[band]
            for year in range(2000, 2021):
                for quarter in range(1, 5):
                    column_name = f"{year}_{quarter}"
                    if column_name in data.columns:
                        X_test[:,band_index, quarter-1, year-2000] = data[column_name].values
        X_test = torch.tensor(X_test, dtype=torch.float32)
        X_test = torch.nan_to_num(X_test)
    

    # PO_data_cleaned = remove_rows_with_nan(PO_data)
    
    
    # for band in PA_data:
    #     PA_data[band] = PA_data[band][PA_data[band]["speciesId"].isin(selected_species)]

    # for band in PO_data_cleaned:
    #     PO_data_cleaned[band] = PO_data_cleaned[band][PO_data_cleaned[band]["speciesId"].isin(selected_species)]
    
    
    # Now let's make a matrix of 0 and 1 with surveyId as row and speciesId as colonn
    unique_siteid_po = PO_sub["surveyId"].unique()
    unique_siteid_pa = PA_sub["surveyId"].unique()
    y_train_po = torch.zeros(len(unique_siteid_po), len(unique_spid))
    for idx, siteid in enumerate(unique_siteid_po):
        speciesIdinthissite = PO_sub[PO_sub["surveyId"] == siteid]
        for species_list in speciesIdinthissite["speciesId"]:
            for spid in species_list:
                idx_spid = np.where(unique_spid == spid)[0][0]
                y_train_po[idx, idx_spid] += 1
        # unique_speciesIdinthissite = speciesIdinthissite["speciesId"].unique()
        # for spid in unique_speciesIdinthissite:
        #     idx_spid = np.where(unique_spid == spid)[0][0]
        #     y_tensor_po[idx, idx_spid] = 1
        
    # Now let's make a matrix of 0 and 1 with surveyId as row and speciesId as colonn
    y_validation_pa = torch.zeros(len(unique_siteid_pa), len(unique_spid))
    for idx, siteid in enumerate(unique_siteid_pa):
        speciesIdinthissite = PA_sub[PA_sub["surveyId"] == siteid]
        for species_list in speciesIdinthissite["speciesId"]:
            for spid in species_list:
                idx_spid = np.where(unique_spid == spid)[0][0]
                y_validation_pa[idx, idx_spid] += 1

    y_validation_pa[y_validation_pa>1] = 1

    unique_siteid_pa_test = PA_test["surveyId"].unique()
    y_test = torch.zeros(len(unique_siteid_pa_test), len(unique_spid))
    for idx, siteid in enumerate(unique_siteid_pa_test):
        speciesIdinthissite = PA_test[PA_test["surveyId"] == siteid]
        for species_list in speciesIdinthissite["speciesId"]:
            for spid in species_list:
                idx_spid = np.where(unique_spid == spid)[0][0]
                y_test[idx, idx_spid] += 1
        
        
    y_test[y_test>1] = 1
    
    # empty_columns = np.where(~y_validation_pa.any(axis=0))[0]
    # print(f'Colonnes vides dans y_validation_pa: {empty_columns}')

    # nan_count_PO = count_nan_per_column(PO_data)

    # nan_columns_PA, common_columns_PA = find_nan_columns(PA_data)
    # nan_columns_PO, common_columns_PO = find_nan_columns(PO_data)
    # PO_mean, PO_std = compute_mean_std(PO_data, common_columns_PO)
    # PA_mean, PA_std = compute_mean_std(PA_data, common_columns_PA)


    # data_dict = {
    #     "blue": {"train": PO_data["blue"], "validation": PA_data["blue"]}, 
    #     "green": {"train": PO_data["green"], "validation": PA_data["green"]},
    #     "red": {"train": PO_data["red"], "validation": PA_data["red"]},
    #     "nir": {"train": PO_data["nir"], "validation": PA_data["nir"]},
    #     "swir1": {"train": PO_data["swir1"], "validation": PA_data["swir1"]},
    #     "swir2": {"train": PO_data["swir2"], "validation": PA_data["swir2"]}
    # }

    # bands = ["blue", "green", "red", "nir","swir1","swir2"]
    # X_train, X_validation,

    # Now you can use X_train and X_validation, and you have column_indices to know where each band starts and ends
    # print(column_indices)

    # o,n,m,p = X_train.shape
    # X_train = X_train.view(o,n*m*p)
    
    
    # scaler = StandardScaler(with_std=False)
    # X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    # X_train = X_train.view(o,n,m,p)

    # o,n,m,p = X_Validation.shape
    # X_Validation = X_Validation.view(o,n*m*p)
    # X_Validation = torch.tensor(scaler.transform(X_Validation), dtype=torch.float32)
    # X_Validation = X_Validation.view(o,n,m,p)

    # X_validation_band = torch.tensor(scaler.transform(data_dict[band]["validation"].values[:, 1:]), dtype=torch.float32)
    # X_Validation = X_Validation.to(args.device)
    del PO, PA, PA_test
    del PO_sub, PA_sub, po_files, pa_files, PO_data, PA_data

    print('Training')
    for loss_option in args.loss_options:
        for idx_seed in range(args.repeat_seed):
            # wandbname = "BIOSPACE" + str(args.list_of_seed[idx_seed])
            # wandb.init(
            #     name = wandbname,
            #     mode=args.mode_wandb,
            #     project="Biospace",
            #     config={
            #         "seed": args.list_of_seed[idx_seed],
            #         "loss": loss_option
            #     }
            # )
            
            
            # Set the random seed for reproducibility
            set_seed(args.list_of_seed[idx_seed])
            # results = train_deepmodel_with_eval(
            #     X_train,
            #     y_train_po, 
            #     args,
            #     X_Validation,
            #     y_validation_pa,
            #     hidden_size=250,
            #     loss_option=loss_option
            # )
            
            results = train_deepmodel_with_eval_test(
                X_train,
                y_train_po, 
                args,
                X_Validation,
                y_validation_pa,
                hidden_size=250,
                loss_option=loss_option
            )

            # Créer une nouvelle instance du modèle
            model = results["model"]
            model.eval()
            model = model.to("cpu")
            best_epoch = results["best_epoch"]
            print(f'best_epoch: {best_epoch}')
            # model = model.to("cpu")
            # X_tensor = X_tensor.to("cpu")
            filename_model = f"biospace_{args.list_of_seed[idx_seed]}"
            full_path = f"{args.outputdir}/models/{filename_model}.pth"



            # torch.save({
            #     'model_state_dict': model.state_dict(),
            #     'scaler': scaler,
            # }, full_path)   
            
            # TEST 
            loss_by_epoch = results["loss_by_epoch"]
            auc_by_epoch = results["auc_by_epoch"]
            loss_df = pd.DataFrame({
                "loss_by_epoch": loss_by_epoch,
                "auc_by_epoch": auc_by_epoch
            })

            loss_df.to_csv(f"{args.outputdir}/loss_curve.csv", index=False)
            
            
            if(args.architecture == "transformer"):
                X_test = X_test.permute([0,2,1])
                X_test = X_test.unsqueeze(3)
                X_test = X_test.unsqueeze(4)
            with torch.no_grad():
                predictions = model(X_test)

            y_test = y_test.numpy()
            predictions = torch.clamp(predictions,
                                        min=-np.inf,
                                        max=88.7)
            
            predictions = predictions.exp().detach().cpu().numpy()
            
            predictions[np.isinf(predictions)] = np.finfo(np.float32).max
            
            columns_with_multiple_classes = []
            for i in range(y_test.shape[1]):
                if len(np.unique(y_test[:, i])) > 1:
                    columns_with_multiple_classes.append(i)

            # Extract only the columns with more than 1 unique class
            y_test = y_test[:, columns_with_multiple_classes]
            predictions = predictions[:, columns_with_multiple_classes]
            
            auc_rel_macro = 0.5
            auc_rel_micro=0.5
            
            if len(np.unique(y_test)) > 1:
                auc_rel_macro = roc_auc_score(y_test, predictions)
                auc_rel_micro = roc_auc_score(y_test, predictions,average='micro')
            # auc_rel_macro = roc_auc_score(y_test, predictions)
            
            # auc_f1 = 2 * auc_rel_macro * auc_rel_micro / (auc_rel_macro + auc_rel_micro)
            print(f'AUC:{auc_rel_macro}')
            lbd_corr_list = [] 
            auc_by_species = []   
            for i in range(y_test.shape[1]):
                correlation, _ = pearsonr(predictions[:, i], y_test[:, i])       
                lbd_corr_list.append(correlation)
                if len(np.unique(y_test[:,i])) > 1:
                    auc_rel_macro_by_species = roc_auc_score(y_test[:,i], predictions[:,i ])
                    auc_by_species.append(auc_rel_macro_by_species)
                else:
                    auc_rel_macro_by_species = 0.5
                    auc_by_species.append(auc_rel_macro_by_species)
            auc_by_species = np.array(auc_by_species)
            lbd_corr_list = np.array(lbd_corr_list)
            new_row = pd.DataFrame(
                {
                    "loss": [loss_option],
                    "Seed": [args.list_of_seed[idx_seed]],
                    "AUC_macro": [auc_rel_macro],
                    "AUC_micro": [auc_rel_micro],
                    "Pearson": [lbd_corr_list.mean()],
                }
            )

            print(new_row)
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            new_row.to_csv(
                f"{args.outputdir}/results_backup.csv",
                mode="a",
                header=not os.path.exists(
                    f"{args.outputdir}/results_backup.csv"
                ),
                index=False,
            )
            # wandb.finish()
            
            new_row = pd.DataFrame(
                {
                    "AUC": auc_by_species
                }
            )
            new_row.to_csv(f"{args.outputdir}/AUC_by_species/auc_{args.list_of_seed[idx_seed]}.csv", index=False) 

            
            
            
            # OLD VAL 

            # with torch.no_grad():
            #     predictions = model(X_Validation)

            # y_validation_pa = y_validation_pa.numpy()
            # predictions = torch.clamp(predictions,
            #                             min=-np.inf,
            #                             max=88.7)
            
            # predictions = predictions.exp().detach().cpu().numpy()
            
            # predictions[np.isinf(predictions)] = np.finfo(np.float32).max
            # auc_rel_macro = roc_auc_score(y_validation_pa, predictions)
            # auc_rel_micro = roc_auc_score(y_validation_pa, predictions,average='micro')
            # # auc_f1 = 2 * auc_rel_macro * auc_rel_micro / (auc_rel_macro + auc_rel_micro)
            # print(f'AUC:{auc_rel_macro}')
            # lbd_corr_list = [] 
            # auc_by_species = []   
            # for i in range(y_validation_pa.shape[1]):
            #     correlation, _ = pearsonr(predictions[:, i], y_validation_pa[:, i])       
            #     lbd_corr_list.append(correlation)
            #     auc_rel_macro_by_species = roc_auc_score(y_validation_pa[:,i], predictions[:,i ])
            #     auc_by_species.append(auc_rel_macro_by_species)
            # auc_by_species = np.array(auc_by_species)
            # lbd_corr_list = np.array(lbd_corr_list)
            # new_row = pd.DataFrame(
            #     {
            #         "loss": [loss_option],
            #         "Seed": [args.list_of_seed[idx_seed]],
            #         "AUC_macro": [auc_rel_macro],
            #         "AUC_micro": [auc_rel_micro],
            #         "Pearson": [lbd_corr_list.mean()],
            #     }
            # )

            # print(new_row)
            # results_df = pd.concat([results_df, new_row], ignore_index=True)
            # new_row.to_csv(
            #     f"{args.outputdir}/results_backup.csv",
            #     mode="a",
            #     header=not os.path.exists(
            #         f"{args.outputdir}/results_backup.csv"
            #     ),
            #     index=False,
            # )
            # # wandb.finish()
            
            # new_row = pd.DataFrame(
            #     {
            #         "spid": unique_spid,
            #         "AUC": auc_by_species
            #     }
            # )
            # new_row.to_csv(f"{args.outputdir}/AUC_by_species/auc_{args.list_of_seed[idx_seed]}.csv", index=False) 
            # loss_by_batch = results["loss_by_batch"]
            # loss_df = pd.DataFrame(loss_by_batch)
            # loss_df.to_csv(f"{args.outputdir}/loss_curve.csv", index=False)
              
    results_df.to_csv(f"{args.outputdir}/results.csv", index=False) 


def cv_deepmodel(
    X_tensor,
    y_tensor, 
    args,
    X_val, 
    y_true,
    loss_option="deepmaxent"
):
    device = args.device
    X_tens_data = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(X_tens_data, batch_size=args.batch_size, shuffle=True)
    X_val = X_val.to(device)
    
    y_true[y_true>1] = 1
    
    if loss_option == "deepmaxent":
        criterion = deepmaxent_loss().to(device)
    elif loss_option == "poisson":
        criterion = poisson_loss().to(device)
    elif loss_option == "ce":
        criterion = ce_loss().to(device)
    elif loss_option == "bce":
        criterion = bce_loss().to(device)
    else:
        raise ValueError("Loss option not recognized")

        
    input_size = X_tensor.shape[1]
    output_size = y_tensor.shape[1]
    
    if (args.architecture == "resnet"):
        model = ModifiedResNet18(args.batch_size, output_size)
    else :
        model = deepmaxent_model(input_size, args.hidden_size, output_size, args.hidden_nbr)       
    model = model.to(device)
    
    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        optimizer = optim.Adam(
            [
                {"params": model.parameters(), "lr": args.learning_rate, "weight_decay":args.weight_decay},
            ],
        )
    else:
        optimizer = optim.Adam(
            [
                {"params": model.parameters(), "lr": args.learning_rate},
            ],
        )
    num_epochs = args.epoch_cv
    loss_by_batch = []
    loss_val_by_batch = []
    auc_epoch = []
    
    best_loss = float('inf')
    best_model = None

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss_train = 0.0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
        
            loss = criterion(outputs, batch_y)
                
            loss.backward()

            optimizer.step()
            total_loss_train += loss.item()
            

        model.eval()
        with torch.no_grad():
            outputs_PA = model(X_val)
            loss_val = criterion(outputs_PA.cpu(), y_true)
            loss_val_by_batch.append(loss_val)

            predictions = torch.clamp(outputs_PA,
                                        min=-np.inf,
                                        max=88.7)

            predictions = predictions.exp().detach().cpu().numpy()
            predictions[np.isinf(predictions)] = np.finfo(np.float32).max
            auc_rel_macro = 0
            idx=0
            for i in range(predictions.shape[1]):
                if(len(np.unique(y_true[:,i])) != 1):
                    auc_rel_macro += roc_auc_score(y_true[:,i], predictions[:,i])
                    idx+=1
            if (idx!=0):
                auc_value = auc_rel_macro/idx
                auc_epoch.append(auc_value)
                
        loss_by_batch.append(total_loss_train / len(train_loader))
        
        if (args.validation_criteria == "AUC" and auc_value < best_loss) or (args.validation_criteria != "AUC" and loss_val > best_loss):
            best_loss = auc_value if args.validation_criteria == "AUC" else loss_val
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        
    model.load_state_dict(best_model)
    predictions = make_predictions(model, X_tensor)
    
    if len(auc_epoch) > 0:
        auc_rel_value = auc_epoch[-1]
    else:
        auc_rel_value = 0
    result = {
        "predictions": predictions,
        "model": model,
        "loss_val_by_batch": loss_val_by_batch,
        "loss_by_batch": loss_by_batch,
        "AUC_val": auc_rel_value,
        "best_epoch": best_epoch
    }
    return result
    

def train_deepmodel(
    X_tensor,
    y_tensor, 
    args,
    hidden_size=250,
    loss_option="deepmaxent"
):
    device = args.device
    X_tens_data = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(X_tens_data, batch_size=250, shuffle=True)
    
    
    if loss_option == "deepmaxent":
        criterion = deepmaxent_loss().to(device)
    elif loss_option == "poisson":
        criterion = poisson_loss().to(device)
    elif loss_option == "ce":
        criterion = ce_loss().to(device)
    elif loss_option == "bce":
        criterion = bce_loss().to(device)
    else:
        raise ValueError("Loss option not recognized")

        
    input_size = X_tensor.shape[1]
    output_size = y_tensor.shape[1]
    
    if(args.architecture == "resnet"):
        model = ModifiedResNet18(args.batch_size, output_size)
    else : 
        model = deepmaxent_model(input_size, hidden_size, output_size, args.hidden_nbr)      
    model = model.to(device)
    # if hasattr(args, 'weight_decay') and args.weight_decay is not None:
    #     optimizer = optim.Adam(
    #         [
    #             {"params": model.parameters(), "lr": args.learning_rate, "weight_decay":args.weight_decay},
    #         ],
    #     )
    # else:
    #     optimizer = optim.Adam(
    #         [
    #             {"params": model.parameters(), "lr": args.learning_rate},
    #         ],
    #     )

    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    scheduler = CosineAnnealingLR(optimizer, T_max=25, verbose=True)
    
    num_epochs = args.epoch
    loss_by_batch = []
    best_loss = float('inf')
    best_model = None

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss_train = 0.0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) 
            loss.backward()

            optimizer.step()
            total_loss_train += loss.item()
            
        loss_by_batch.append(total_loss_train / len(train_loader))
        
        # wandb.log({"epoch": epoch,
        #             "total_loss": total_loss_train / len(train_loader)})
        
        if total_loss_train / len(train_loader) < best_loss:
            best_loss = total_loss_train / len(train_loader)
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f'loss: {best_loss}/ epoch: {best_epoch}')
        scheduler.step()   
        print("Scheduler:",scheduler.state_dict())
    model.load_state_dict(best_model)
    # predictions = make_predictions(model, X_tensor)
    result = {
        # "predictions": predictions,
        "best_epoch":best_epoch,
        "model": model,
        "loss_by_batch": loss_by_batch
    }
    return result
    
    

def train_deepmodel_with_eval(
    X_tensor,
    y_tensor, 
    args,
    X_val,
    y_true,
    hidden_size=250,
    loss_option="deepmaxent"
):
    device = args.device
    X_tens_data = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(X_tens_data, batch_size=args.batch_size, shuffle=True)
    X_val = X_val.to(device)
    
    if loss_option == "deepmaxent":
        criterion = deepmaxent_loss().to(device)
    elif loss_option == "poisson":
        criterion = poisson_loss().to(device)
    elif loss_option == "ce":
        criterion = ce_loss().to(device)
    elif loss_option == "bce":
        criterion = bce_loss().to(device)
    else:
        raise ValueError("Loss option not recognized")

        
    input_size = X_tensor.shape[1]
    output_size = y_tensor.shape[1]
    
    if(args.architecture == "resnet"):
        model = ModifiedResNet18(args.batch_size, output_size)
        model = ModifiedVisionTransformer(args.batch_size, output_size)
        
    else : 
        model = deepmaxent_model(input_size, hidden_size, output_size, args.hidden_nbr)      
    model = model.to(device)
    # if hasattr(args, 'weight_decay') and args.weight_decay is not None:
    #     optimizer = optim.Adam(
    #         [
    #             {"params": model.parameters(), "lr": args.learning_rate, "weight_decay":args.weight_decay},
    #         ],
    #     )
    # else:
    #     optimizer = optim.Adam(
    #         [
    #             {"params": model.parameters(), "lr": args.learning_rate},
    #         ],
    #     )

    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # scheduler = CosineAnnealingLR(optimizer, T_max=20, verbose=True)
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)  # Warm-up
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=15)  # CosineAnnealing sur les 15 dernières epochs
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
    
    
    num_epochs = args.epoch
    loss_by_batch = []
    auc_epoch = []
    best_loss = float('inf')
    best_model = None

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss_train = 0.0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) 
            loss.backward()

            optimizer.step()
            total_loss_train += loss.item()
            
        
        model.eval()
        with torch.no_grad():
            outputs_PA = model(X_val)
            # loss_val = criterion(outputs_PA.cpu(), y_true)
            # loss_val_by_batch.append(loss_val)

            predictions = torch.clamp(outputs_PA,
                                        min=-np.inf,
                                        max=88.7)

            predictions = predictions.exp().detach().cpu().numpy()
            predictions[np.isinf(predictions)] = np.finfo(np.float32).max
            auc_rel_macro = 0
            idx=0
            for i in range(predictions.shape[1]):
                if(len(np.unique(y_true[:,i])) != 1):
                    auc_rel_macro += roc_auc_score(y_true[:,i], predictions[:,i])
                    idx+=1
            if (idx!=0):
                auc_value = auc_rel_macro/idx
                print(f'AUC_val={auc_value}')
                auc_epoch.append(auc_value)
            
        loss_by_batch.append(total_loss_train / len(train_loader))
        
        # wandb.log({"epoch": epoch,
        #             "total_loss": total_loss_train / len(train_loader)})
        
        if total_loss_train / len(train_loader) < best_loss:
            best_loss = total_loss_train / len(train_loader)
            best_model = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            print(f'loss: {best_loss}/ epoch: {best_epoch}')
        scheduler.step()   
        # print("Scheduler:",scheduler.state_dict())
    model.load_state_dict(best_model)
    # predictions = make_predictions(model, X_tensor)
    result = {
        # "predictions": predictions,
        "best_epoch":best_epoch,
        "model": model,
        "loss_by_batch": loss_by_batch,
        "auc_val":auc_epoch
    }
    return result



def train_deepmodel_with_eval_test(
    X_tensor,
    y_tensor, 
    args,
    X_val,
    y_val,

    hidden_size=250,
    loss_option="deepmaxent"
):
    device = args.device
    if(args.architecture == "transformer"):
        X_tensor = X_tensor.permute([0,2,1])
        X_tensor = X_tensor.unsqueeze(3)
        X_tensor = X_tensor.unsqueeze(4)
        X_val = X_val.permute([0,2,1])
        X_val = X_val.unsqueeze(3)
        X_val = X_val.unsqueeze(4)
    X_tens_data = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(X_tens_data, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    X_val = X_val.to(device)
    
    if loss_option == "deepmaxent":
        criterion = deepmaxent_loss().to(device)
    elif loss_option == "poisson":
        criterion = poisson_loss().to(device)
    elif loss_option == "ce":
        criterion = ce_loss().to(device)
    elif loss_option == "bce":
        criterion = bce_loss().to(device)
    else:
        raise ValueError("Loss option not recognized")

        
    input_size = X_tensor.shape[1]
    output_size = y_tensor.shape[1]
    
    if(args.architecture =='transformer'):
        model = ModifiedVisionTransformer(args.batch_size, output_size)        
        image_size =1
        time_frames = 12  # 24 patches
        dim = 512
        depth = 3
        heads = 4
        dim_head = 64
        dropout = 0.2
        emb_dropout = 0.2
        channels = 10
        model = TimeSpectralViT(image_size=1, 
                time_frames=84, 
                dim=output_size, 
                depth=6, 
                heads=8, 
                spectral_bands=6, 
            )
    elif(args.architecture == "htransformer"):
        num_bands = 6
        num_seasons = 4
        num_years = 21
        embed_size = 64
        num_heads = 4
        hidden_size = 128
        num_layers = 2
        model = HierarchicalTransformer(num_bands, embed_size, num_heads, hidden_size, num_layers, num_seasons, num_years)
    elif(args.architecture == "resnet"):
        model = ModifiedResNet18(args.batch_size, output_size)
        # model = ModifiedVisionTransformer(args.batch_size, output_size)
        
    else : 
        model = deepmaxent_model(input_size, hidden_size, output_size, args.hidden_nbr)      
    model = model.to(device)
   

    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # scheduler = CosineAnnealingLR(optimizer, T_max=20, verbose=True)
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)  # Warm-up
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=15)  # CosineAnnealing sur les 15 dernières epochs
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
    
    
    num_epochs = args.epoch
    loss_by_batch = []
    auc_epoch = []
    best_loss = float('inf')
    best_model = None
    best_auc_val = 0
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss_train = 0.0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) 
            loss.backward()

            optimizer.step()
            total_loss_train += loss.item()
            
        
        model.eval()
        with torch.no_grad():
            auc_rel_macro = 0
            idx = 0
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.cpu().numpy() 
                outputs_PA = model(X_batch)
                
                predictions = torch.clamp(outputs_PA, min=-np.inf, max=88.7)
                predictions = predictions.exp().detach().cpu().numpy()
                predictions[np.isinf(predictions)] = np.finfo(np.float32).max
                
                for i in range(predictions.shape[1]):
                    if len(np.unique(y_batch[:, i])) != 1:
                        auc_rel_macro += roc_auc_score(y_batch[:, i], predictions[:, i])
                        idx += 1
            
            if idx != 0:
                auc_value = auc_rel_macro / idx
                print(f'AUC_val={auc_value}')
                auc_epoch.append(auc_value)
                
                if auc_value > best_auc_val:
                    best_auc_val = auc_value
                    best_model = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    print(f'New best model found at epoch {epoch} with AUC_val={auc_value}')
        
        #### OLD// 
        
        
        
        # model.eval()
        # with torch.no_grad():

        #     outputs_PA = model(X_val)
        #     # loss_val = criterion(outputs_PA.cpu(), y_true)
        #     # loss_val_by_batch.append(loss_val)

        #     predictions = torch.clamp(outputs_PA,
        #                                 min=-np.inf,
        #                                 max=88.7)

        #     predictions = predictions.exp().detach().cpu().numpy()
        #     predictions[np.isinf(predictions)] = np.finfo(np.float32).max
        #     auc_rel_macro = 0
        #     idx=0
        #     for i in range(predictions.shape[1]):
        #         if(len(np.unique(y_val[:,i])) != 1):
        #             auc_rel_macro += roc_auc_score(y_val[:,i], predictions[:,i])
        #             idx+=1
        #     if (idx!=0):
        #         auc_value = auc_rel_macro/idx
        #         print(f'AUC_val={auc_value}')
        #         auc_epoch.append(auc_value)
                
        #         if auc_value > best_auc_val:
        #             best_auc_val = auc_value
        #             best_model = copy.deepcopy(model.state_dict())
        #             best_epoch = epoch
        #             print(f'New best model found at epoch {epoch} with AUC_val={auc_value}')
            
        loss_by_batch.append(total_loss_train / len(train_loader))
        
        # wandb.log({"epoch": epoch,
        #             "total_loss": total_loss_train / len(train_loader)})
        
        # if total_loss_train / len(train_loader) < best_loss:
        #     best_loss = total_loss_train / len(train_loader)
        #     best_model = copy.deepcopy(model.state_dict())
        #     best_epoch = epoch
        #     print(f'loss: {best_loss}/ epoch: {best_epoch}')
            
            
            
        scheduler.step()   
        # print("Scheduler:",scheduler.state_dict())
    model.load_state_dict(best_model)
    # predictions = make_predictions(model, X_tensor)
    result = {
        # "predictions": predictions,
        "best_epoch":best_epoch,
        "model": model,
        "loss_by_epoch": loss_by_batch,
        "auc_by_epoch":auc_epoch
    }
    return result

def mask_arr(arrays, array_mask):
    reshaped_mask = array_mask.reshape(array_mask.shape[0] * array_mask.shape[1])

    mask_array = arrays[reshaped_mask == 1]
    return mask_array


def reshape_arr(arrays):
    reshaped_arrays = arrays.reshape(arrays.shape[0] * arrays.shape[1], arrays.shape[2])
    return reshaped_arrays


def make_results_directory(args):

    if not os.path.exists(f"{args.outputdir}"):
        os.mkdir(f"{args.outputdir}")
    if not os.path.exists(f"{args.outputdir}/models"):
        os.mkdir(f"{args.outputdir}/models")
