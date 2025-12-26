#---- general packages
import numpy as np
import os
import argparse
import re
import sys
import json

from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

#----- models
sys.path.append('./02_models/')
from model_cnn import *

#---- generator and parameter
from hypara import *

#--- torch
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


project_dir = str(Path(__file__).resolve().parent.parent.parent) # Add project dir
sys.path.append(project_dir)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

#----- main
def main():
    print("\n ==================================================================== SETUP PARAMETERS...")
    print("-------- PARSER:")
    parser = init_argparse()
    args   = parser.parse_args()

    THRES_SCORE = 0.609
    IN_DIR = './01_feature/11_lfcc_01/'
    # OUT_DIR          = project_dir +  "/output/deepfake"
    OUT_DIR = args.output_dir
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        print("New output dir for DEEPFAKE_DETECTION created.")

    print("-------- Hyper parameters:")
    NF_AUD           = hypara().nF_aud  
    NT_AUD           = hypara().nT_aud
    NC_AUD           = hypara().nC_aud

    BATCH_SIZE       = hypara().batch_size
    START_BATCH      = hypara().start_batch
    LEARNING_RATE    = hypara().learning_rate
    IS_ARG           = hypara().is_arg
    CHECKPOINT_EVERY = hypara().check_every
    N_CLASS          = hypara().class_num
    NUM_EPOCHS       = hypara().epoch_num
    
    #Setting directory
    print("\n =============== Directory Setting...")
    stored_dir = os.path.abspath(os.path.join(os.path.curdir, OUT_DIR))
    print("+ Writing to {}\n".format(stored_dir))

    best_model_dir = os.path.join(os.path.curdir, "model")
    print("+ Best model Dir: {}\n".format(best_model_dir))
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    best_model_file = os.path.join(best_model_dir, "best_model")

    #--- model setup
    model = model_cnn()
    # print(model)
    device = get_default_device()
    if os.path.isfile(best_model_file):
        model.load_state_dict(torch.load(best_model_file, map_location=device))
        with open(os.path.join(os.path.curdir,"train_acc_log.txt"), "a") as text_file:
            text_file.write("Latest model is loaded from: {} ...\n".format(best_model_dir))
    else:        
        with open(os.path.join(os.path.curdir,"train_acc_log.txt"), "a") as text_file:
            text_file.write("New model instance is created...\n")
    model = to_device(model, device)

    #--- Optimization & Loss setup
    loss_fn = nn.BCELoss()  # binary cross entropy
    #loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) #weight_decay is for l2 norm

    #---read data
    file_list = os.listdir(IN_DIR)
    for file_name in file_list:
        prediction = None
        audio_name = file_name[:-4]
        file_dir  = os.path.join(IN_DIR, file_name)
        x_eval_batch = np.load(file_dir)

        #--- np to torch tensor
        x_eval_batch = torch.tensor(x_eval_batch, dtype=torch.float32)
        #--- push to gpu
        x_eval_batch = to_device(x_eval_batch, device)
        
        #--- eva real acc
        with torch.no_grad():
            #--- inference process on gpu
            eval_pred = model(x_eval_batch)
            # push to cpu
            eval_pred    = eval_pred.detach().cpu().numpy()
            eval_pred = np.sum(eval_pred, 0)/np.shape(eval_pred)[0]
            if eval_pred[1] >= THRES_SCORE:
                prediction = ['real']
                print('XXXXXXXXXXXXXXXXXXXX: FILE {}: REAL'.format(file_name))
            else:
                prediction = ['fake']    
                print('XXXXXXXXXXXXXXXXXXXX: FILE {}:FAKE'.format(file_name))

        # Write result
        save_dir = os.path.join(OUT_DIR, audio_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'deepfake.json'), "w") as outfile:
            json.dump(prediction, outfile)


def init_argparse():
    parser = argparse.ArgumentParser(
        usage="%(prog)s TODO",
        description="TODO" 
    )
    parser.add_argument(
        "--output_dir", required=True,
        help='Choose out directory'
    )
    return parser


if __name__ == "__main__":
    main()
