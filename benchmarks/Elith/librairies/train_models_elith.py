import os
# import wandb
from librairies.losses import bce_loss, ce_loss, deepmaxent_loss, poisson_loss
from librairies.model import make_predictions
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

    results_df = pd.DataFrame(columns=["Region", "Group", "lr","weight_decay","batch_size","hidden_nbr", "Seed","block_i", "loss_cal", "loss_val","AUC_val"])        
    n = 0
    for r in args.regions:
        n += 1

        dossier_path = args.dirdata + "/Environment/" + r
        borders = gpd.read_file(args.dirdata + "/Borders/" + r.lower() + ".gpkg")
        trainPOfile = args.dirdata + "/Records/train_po/" + r + "train_po.csv"

        PO = pd.read_csv(trainPOfile)
        
        if (r == 'NSW') :
            group_mapping = {
                'ba': 'bates',
                'db': 'birds',
                'nb': 'birds',
                'sr': 'reptile',
                'rt': 'plants',
                'ou': 'plants',
                'ot': 'plants',
                'ru': 'plants'
            }
            PO['group'] = PO['group'].replace(group_mapping)

        group_list = PO["group"].unique()

            

        for group in group_list:
            PO_sub = PO[PO["group"] == group]
                
           
            fichiers = os.listdir(dossier_path)
            idx = 0
            variables = []  #
            for fichier in tqdm(fichiers, desc="Load environment data"):
                nb_fichiers_tiff = len(
                    [
                        fichier
                        for fichier in fichiers
                        if fichier.endswith(".tif") or fichier.endswith(".tiff")
                    ]
                )

                if fichier.endswith(".tif") or fichier.endswith(".tiff"):
                    tiff_path = os.path.join(dossier_path, fichier)
                    filename_without_extension = os.path.splitext(fichier)[0]
                    variables.append(filename_without_extension)
                    with rasterio.open(tiff_path) as src:
                        out_image, out_transform = mask(
                            src,
                            [borders["geometry"].unary_union.__geo_interface__],
                            crop=False,
                            filled=True,
                            nodata=255,
                        )
                        V1 = out_image

                        if idx == 0:
                            ndim, nrows, ncols = V1.shape
                            array_3d = np.ones((nrows, ncols, nb_fichiers_tiff))
                            idx += 1
                            PO_sub["row"], PO_sub["col"] = zip(
                                *PO_sub.apply(
                                    lambda row: src.index(row["x"], row["y"]), axis=1
                                )
                            )
                        array_3d[:, :, idx - 1] = V1
                        idx += 1
            mask_real = np.ones_like(array_3d[:, :, 0])
            ime = array_3d[:, :, 0]
            value_to_exclude = 255.0
            mask_real[ime == value_to_exclude] = 0
            
            X_lat_long = torch.meshgrid(
                torch.linspace(0, nrows, nrows), torch.linspace(0, ncols, ncols)
            )
            X_lat_long = torch.stack(X_lat_long, dim=2)
            X_lat_long = reshape_arr(X_lat_long)
            X_lat_long = mask_arr(X_lat_long, mask_real)
            
            
            categorical_indices = [
                variables.index(var) for var in args.categoricalvars if var in variables
            ]
            continuous_indices = [
                i for i in range(len(variables)) if i not in categorical_indices
            ]
            
            
            array_3d = reshape_arr(array_3d)
            array_3d = mask_arr(array_3d, mask_real)

            X_temp = torch.tensor(array_3d, dtype=torch.float32)

            if categorical_indices:
                X_categorical = F.one_hot(X_temp[:, categorical_indices].long())
                X_categorical = X_categorical.view(X_categorical.shape[0], -1)
            else:
                X_categorical = torch.empty(0)

            scaler = StandardScaler()
            X_continuous = torch.tensor(
                scaler.fit_transform(X_temp[:, continuous_indices]), dtype=torch.float32
            )


            X_tensor = torch.cat((X_categorical, X_continuous), dim=1)
            variables = [variables[i] for i in categorical_indices + continuous_indices]
            occurrence_grid_species = np.zeros(
                (nrows, ncols, len(PO_sub["spid"].unique())),dtype=np.int16
            )

            idx = 0
            for spid in tqdm(PO_sub["spid"].unique()):
                tab_df_sub = PO_sub[PO_sub["spid"] == spid]
                grouped = (
                    tab_df_sub.groupby(["row", "col", "spid"])
                    .size()
                    .reset_index(name="count")
                )
                occurrence_grid_species[grouped["row"], grouped["col"], idx] = grouped[
                    "count"
                ]
                idx += 1
                

            X_occurence_reshape = reshape_arr(occurrence_grid_species)
            X_occurence_reshape = mask_arr(X_occurence_reshape, mask_real)
            y_tensor_plant = torch.tensor(X_occurence_reshape, dtype=torch.float32)
            
            idx_to_keep = np.where(X_occurence_reshape.sum(1) != 0)[0]
            zero_indices = np.where(X_occurence_reshape.sum(1) == 0)[0]
            random_zero_indices = np.random.choice(
                zero_indices, size=min(len(idx_to_keep), len(zero_indices)), replace=False
            )
            random_zero_indices = np.random.choice(
                zero_indices, size=min(0, len(zero_indices)), replace=False
            )


            idx_to_keep = np.concatenate([idx_to_keep, random_zero_indices])
            y_tensor_plant = y_tensor_plant[idx_to_keep, :]
            X_tensor = X_tensor[idx_to_keep, :]
            X_lat_long = X_lat_long[idx_to_keep, :]

            if args.cross_validation == "plain":
                kfold = KFold(n_splits=int(1/args.validation_size), shuffle=True,random_state=args.global_seed).split(X_tensor)
            elif args.cross_validation == "blocked":
                kfold = vd.BlockKFold(shape=args.num_cv_blocks, n_splits=int(1/args.validation_size), shuffle=True, balance=True,random_state=args.global_seed).split(X_lat_long.numpy())

            x_trains = []
            y_trains = []
            x_vals = []
            y_vals = []
            for ind_train, ind_val in kfold:
                x_trains.append(X_tensor[ind_train])
                y_trains.append(y_tensor_plant[ind_train])
                x_vals.append(X_tensor[ind_val])
                y_vals.append(y_tensor_plant[ind_val])
            
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
                                            "Region": [r],
                                            "Group": [group],
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
    
def eval_deepmodel(args):

    results_df = pd.DataFrame(columns=["Region", "Group", "Loss", "Seed", "AUC_macro", "AUC_micro", "Pearson"])

    n = 0
    for r in args.regions:
        n += 1

        dossier_path = args.dirdata + "/Environment/" + r
        borders = gpd.read_file(args.dirdata + "/Borders/" + r.lower() + ".gpkg")
        trainPOfile = args.dirdata + "/Records/train_po/" + r + "train_po.csv"

        PO = pd.read_csv(trainPOfile)

        if (r == 'NSW') :
            group_mapping = {
                'ba': 'bates',
                'db': 'birds',
                'nb': 'birds',
                'sr': 'reptile',
                'rt': 'plants',
                'ou': 'plants',
                'ot': 'plants',
                'ru': 'plants'
            }
            PO['group'] = PO['group'].replace(group_mapping)

        group_list = PO["group"].unique()

            

        for group in group_list:
            PO_sub = PO[PO["group"] == group]
            if len(group_list) > 1:
                suffix = "_" + group
            else:
                suffix = ""
                
            if (r=='NSW'):
                group_keys = [key for key, value in group_mapping.items() if value == group]
                dfs = []
                for key in group_keys:
                    test_PAfile = (
                        args.dirdata
                        + "/Records/test_pa/"
                        + r
                        + "test_pa_"
                        + key
                        + ".csv"
                    )
                    df = pd.read_csv(test_PAfile)
                    dfs.append(df)
                PA = pd.concat(dfs)  
                PA = PA.fillna(0)
            else:    
                test_PAfile = (
                    args.dirdata
                    + "/Records/test_pa/"
                    + r
                    + "test_pa"
                    + suffix
                    + ".csv"
                )
                PA = pd.read_csv(test_PAfile)

            fichiers = os.listdir(dossier_path)
            idx = 0
            variables = []  #
            for fichier in tqdm(fichiers, desc="Load environment data"):
                nb_fichiers_tiff = len(
                    [
                        fichier
                        for fichier in fichiers
                        if fichier.endswith(".tif") or fichier.endswith(".tiff")
                    ]
                )

                if fichier.endswith(".tif") or fichier.endswith(".tiff"):
                    tiff_path = os.path.join(dossier_path, fichier)
                    filename_without_extension = os.path.splitext(fichier)[0]
                    variables.append(filename_without_extension)
                    with rasterio.open(tiff_path) as src:
                        out_image, _ = mask(
                            src,
                            [borders["geometry"].unary_union.__geo_interface__],
                            crop=False,
                            filled=True,
                            nodata=255,
                        )
                        V1 = out_image

                        if idx == 0:
                            _, nrows, ncols = V1.shape
                            array_3d = np.ones((nrows, ncols, nb_fichiers_tiff))
                            idx += 1
                            PO_sub["row"], PO_sub["col"] = zip(
                                *PO_sub.apply(
                                    lambda row: src.index(row["x"], row["y"]), axis=1
                                )
                            )
                            PA["row"], PA["col"] = zip(
                                *PA.apply(
                                    lambda row: src.index(row["x"], row["y"]), axis=1
                                )
                            )
                        array_3d[:, :, idx - 1] = V1
                        idx += 1
            mask_real = np.ones_like(array_3d[:, :, 0])
            ime = array_3d[:, :, 0]
            value_to_exclude = 255.0
            mask_real[ime == value_to_exclude] = 0

            PA_env = torch.tensor(
                array_3d[PA["row"].values, PA["col"].values, :], dtype=torch.float32
            )
            
            categorical_indices = [
                variables.index(var) for var in args.categoricalvars if var in variables
            ]
            continuous_indices = [
                i for i in range(len(variables)) if i not in categorical_indices
            ]
            
            
            array_3d = reshape_arr(array_3d)
            array_3d = mask_arr(array_3d, mask_real)

            X_temp = torch.tensor(array_3d, dtype=torch.float32)

            if categorical_indices:
                X_categorical = F.one_hot(X_temp[:, categorical_indices].long())
                X_categorical = X_categorical.view(X_categorical.shape[0], -1)
                X_PA_categorical = F.one_hot(PA_env[:, categorical_indices].long())
                X_PA_categorical = X_PA_categorical.view(X_PA_categorical.shape[0], -1)
            else:
                X_categorical = torch.empty(0)
                X_PA_categorical = torch.empty(0)

            scaler = StandardScaler()
            X_continuous = torch.tensor(
                scaler.fit_transform(X_temp[:, continuous_indices]), dtype=torch.float32
            )

            X_PA_continuous = torch.tensor(
                scaler.transform(PA_env[:, continuous_indices]), dtype=torch.float32
            )
            X_PA_tensor = torch.cat((X_PA_categorical, X_PA_continuous), dim=1)
            X_tensor = torch.cat((X_categorical, X_continuous), dim=1)
            variables = [variables[i] for i in categorical_indices + continuous_indices]
            occurrence_grid_species = np.zeros(
                (nrows, ncols, len(PO_sub["spid"].unique())),dtype=np.int16
            )

            idx = 0
            for spid in tqdm(PO_sub["spid"].unique()):
                tab_df_sub = PO_sub[PO_sub["spid"] == spid]
                grouped = (
                    tab_df_sub.groupby(["row", "col", "spid"])
                    .size()
                    .reset_index(name="count")
                )
                occurrence_grid_species[grouped["row"], grouped["col"], idx] = grouped[
                    "count"
                ]
                idx += 1
                

            X_occurence_reshape = reshape_arr(occurrence_grid_species)
            X_occurence_reshape = mask_arr(X_occurence_reshape, mask_real)
            y_tensor_plant = torch.tensor(X_occurence_reshape, dtype=torch.float32)
            
            idx_to_keep = np.where(X_occurence_reshape.sum(1) != 0)[0]
            zero_indices = np.where(X_occurence_reshape.sum(1) == 0)[0]

            
            if(args.TGB):
                random_zero_indices = np.random.choice(
                    zero_indices, size=min(0, len(zero_indices)), replace=False
                )
            else: 
                random_zero_indices = np.random.choice(
                zero_indices, size=min(len(idx_to_keep)*10, len(zero_indices)), replace=False
            )

            idx_to_keep = np.concatenate([idx_to_keep, random_zero_indices])
            y_tensor_plant = y_tensor_plant[idx_to_keep, :]
            X_tensor = X_tensor[idx_to_keep, :]

            for loss_option in args.loss_options:
                for idx_seed in range(args.repeat_seed):
                    # wandbname = r + "_" + group + "_" + loss_option + "_" + str(args.list_of_seed[idx_seed])
                    # wandb.init(
                    #     name = wandbname,
                    #     mode=args.mode_wandb,
                    #     project="disentangling-method",
                    #     config={
                    #         "region": r,
                    #         "group": group,
                    #         "loss": loss_option,
                    #         "seed": args.list_of_seed[idx_seed]
                    #     }
                    # )
     
                    unique_spid = PO_sub["spid"].unique()
                    y_true = PA[unique_spid]

                    set_seed(args.list_of_seed[idx_seed])
                    results = train_deepmodel(
                        X_tensor,
                        y_tensor_plant,
                        args,
                        hidden_size=args.hidden_size,
                        device="cuda",
                        loss_option=loss_option
                    )

                    model = results["model"]
                    filename_model = f"{r}_{group}_{loss_option}_{args.list_of_seed[idx_seed]}"
                    full_path = f"{args.outputdir}/models/{filename_model}.pth"



                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'scaler': scaler,
                        'columns': variables
                    }, full_path)   
                    
                    
                    loss = results['loss_by_epoch']
                    
                    loss_df = pd.DataFrame({"loss": loss})
                    loss_df.to_csv(f"{args.outputdir}/curves/loss_{r}_{group}_{loss_option}_{args.list_of_seed[idx_seed]}.csv", index=False)
                    
                    with torch.no_grad():
                        predictions = model(X_PA_tensor)
                    predictions = torch.clamp(predictions,
                                              min=-np.inf,
                                              max=88.7)
                    predictions = predictions.exp().detach().cpu().numpy()
                    
                    predictions[np.isinf(predictions)] = np.finfo(np.float32).max
                    auc_rel_macro = roc_auc_score(y_true, predictions)
                    auc_rel_micro = roc_auc_score(y_true, predictions,average='micro')
                    auc_f1 = 2 * auc_rel_macro * auc_rel_micro / (auc_rel_macro + auc_rel_micro)
                    
                    lbd_corr_list = [] 
                    auc_by_species = []               
                    for i in range(y_true.shape[1]):
                        correlation, _ = pearsonr(predictions[:, i], y_true.iloc[:, i])       
                        lbd_corr_list.append(correlation)
                        auc_rel_macro_by_species = roc_auc_score(y_true.iloc[:,i], predictions[:,i ])
                        auc_by_species.append(auc_rel_macro_by_species)
                    auc_by_species = np.array(auc_by_species)
                    lbd_corr_list = np.array(lbd_corr_list)
                    new_row = pd.DataFrame(
                        {
                            "Region": [r],
                            "Group": [group],
                            "Loss": [loss_option],
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
                            "spid": unique_spid,
                            "AUC": auc_by_species
                        }
                    )
                    new_row.to_csv(f"{args.outputdir}/AUC_by_species/auc_{r}_{group}_{loss_option}_{args.list_of_seed[idx_seed]}.csv", index=False) 
              
              
    results_df.to_csv(f"{args.outputdir}/results.csv", index=False) 

def cv_deepmodel(
    X_tensor,
    y_tensor, 
    args,
    X_val, 
    y_true,
    loss_option="deepmaxent",
    device="cuda",
):

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
    device="cuda",
    loss_option="deepmaxent"
):

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
    
    model = deepmaxent_model(input_size, hidden_size, output_size, args.hidden_nbr)      
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
        # loss_by_batch.append(total_loss_train / len(train_loader))
        
        if total_loss_train / len(train_loader) < best_loss:
            best_loss = total_loss_train / len(train_loader)
            best_model = copy.deepcopy(model.state_dict())
                
    model.load_state_dict(best_model)
    predictions = make_predictions(model, X_tensor)
    result = {
        "predictions": predictions,
        "model": model,
        "loss_by_epoch": loss_by_batch
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
