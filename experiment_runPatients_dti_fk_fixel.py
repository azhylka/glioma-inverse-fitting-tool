#!/usr/bin/python
#%%
import numpy as np
import os
import os.path as osp
import nibabel as nib
import time
from scipy import ndimage
import nibabel as nib
import matplotlib.pyplot as plt
import cmaesFixel
import tools
from scipy.ndimage import binary_dilation
import TumorGrowthToolkit.FK_DTI.tools as toolsDTI
import gc
import importlib
import subprocess
import sys
from multiprocessing import Pool, cpu_count
import wandb

wandb.login(key="b00045d336b878b4010142ee5780b67d7aadd383")

registered_HCP_data = "/mnt/Drive2/andrey/HCPTemplate2BraTS"
BraTS_data_dir = "/mnt/Drive4/jonas/datasets/brats_2021_train2/"    
output_dir = "/mnt/Drive2/andrey/cma_es_result"

doLog = True
experimentName ="Fixel"#"23_testDTI_new_STD_and_exp_butterfly"#"26_testDTI_fix_std"#"17_testDTIexponent" #"15_testDTI"# evolutionary_sampling26_testDTI_fix_std
run_tag = 'STD1_fixed'
debug = True # TODOCheck
if debug:
    experimentName += "debugGen50"
    doLog = False

#%%
def run(edema, necrotic, enhancing, affine, diffusionTensors, brainmask, fixel_dir, resultpath, gm, wm, cache_dir=None, runName = "run"):
    
    settings = {}

    if "DTI" in experimentName:
        settings["solverType"] = 'DTI' 
    elif 'FixelnOrig' in experimentName:
        settings['solverType'] = 'FixelnOrig'
    elif 'Fixel' in experimentName:
        settings["solverType"] = 'Fixel'
    else:
        settings["solverType"] = 'FK'

    print("Attention -----------------")
    print(f"running normal {settings['solverType']}")
    print("Attention -----------------")

    # fixed parameters that are not varied
    # only optimize origin, rho and final volume for now
    settings["fixedParameters"] = ["stopping_time",  "thresholdT1c", "thresholdFlair", "rho", # "desiredSTD", 
                                   "diffusionTensorExponent","diffusionEllipsoidScaling","RatioDw_Dg"] # TODO
    #",, , 
    ##"stopping_volume", "RatioDw_Dg","desiredSTD" "diffusionTensorExponent""Dw", #"diffusionTensorExponent",,"Dw",,"NxT1_pct", "NyT1_pct", "NzT1_pct"], , "thresholdFlair", 

    # init parameter
    settings["rho"] = 0.5 #0.5#0.1 # TODO
    settings["Dw"] = 15 # 5.0 #TODO
    settings["RatioDw_Dg"] = 10.0 #TODO
    settings["diffusionEllipsoidScaling"] = 1
    settings["diffusionTensorExponent"] = 1
    settings["desiredSTD"] = 1 #0 #  0.3 # 0 = Fisher kolmogorov # TODO
    settings["viewLossAsProbability"] = False#True # TODO multiplies the losses for flair and T1c
    settings['fixel_dir'] = fixel_dir
    settings['cache_dir'] = cache_dir

    settings["thresholdT1c"] = 0.66
    settings["thresholdFlair"] = 0.33
    settings["stopping_volume"] = 0.7*(np.sum(edema) + np.sum(necrotic) + np.sum(enhancing)) #10 *(np.sum(edema) + np.sum(necrotic) + np.sum(enhancing))  #  #TODO
    settings["stopping_time"] = 10000000000 # 100 days 

    print("*** Prior stopping volume:", settings["stopping_volume"], "\n*** Regular stopping volume:", 0.7*(np.sum(edema) + np.sum(necrotic) + np.sum(enhancing)))

    # center of mass
    com = ndimage.center_of_mass(necrotic + enhancing)
    settings["NxT1_pct"] = float(com[0] / np.shape(edema)[0])
    settings["NyT1_pct"] = float(com[1] / np.shape(edema)[1])
    settings["NzT1_pct"] = float(com[2] / np.shape(edema)[2])


    # set parameter ranges
    settings["rho_range"] = [0.01, 5.0]
    settings["Dw_range"] = [0.001, 120.0] #TODO
    settings["RatioDw_Dg_range"] = [0.1, 100.0] # TODO
    settings["desiredSTD_range"] = [0.0, 3.0]
    settings["thresholdT1c_range"] = [0.5, 0.9]
    settings["thresholdFlair_range"] = [0.01, 0.5]
    settings["NxT1_pct_range"] = [0,1]
    settings["NyT1_pct_range"] = [0,1]
    settings["NzT1_pct_range"] = [0,1]
    settings["diffusionEllipsoidScaling_range"] = [0.1, 100.0]
    settings["diffusionTensorExponent_range"] = [0.0, 3.0] # TODO
    settings["stopping_volume_range"] = [0.1 * (np.sum(edema) + np.sum(necrotic) + np.sum(enhancing)), np.sum(brainmask) /2]
    settings["stopping_time_range"] = [0, 1000000000]

    # algorithm settings
    settings["workers"] = 0#9# 9#9#0#9#0 #9# 1#9 #9#4 # 9 TODO
    settings["sigma0"] = 0.02 # 0.02 # TODO
    weighLossByVolume = False
    settings["weighLossByVolume"] = weighLossByVolume
    settings["use_homogen_gm"] =  True # TODO

    if weighLossByVolume:
        volumeCore = np.sum(necrotic) + np.sum(enhancing)
        volumeEdema = np.sum(edema) + np.sum(necrotic) + np.sum(enhancing)
        totalVolume = volumeCore + volumeEdema
        relVolumeCore = volumeCore/ totalVolume
        relVolumeEdema = 1 - relVolumeCore
        print("relVolumeCore", relVolumeCore)
        settings["lossLambdaT1"] = relVolumeCore
        settings["lossLambdaFlair"] = relVolumeEdema
        print("rel core volume:", relVolumeCore, "rel edema volume:", relVolumeEdema)
    else: #TODO
        settings["lossLambdaT1"] = 0.5 #0.2
        settings["lossLambdaFlair"] = 0.5 # 0.8

    # if dir it changes with generations: key = from relative generations, value = resolution factor
    
    settings["resolution_factor"] = 2/3 #{ 0: 0.5, 0.7: 0.6, 0.8:0.7, 0.85:0.8, 0.9: 0.9, 0.95: 1.0} # 0.5 #{ 0: 0.5, 0.75: 0.6, 0.85:0.8, 0.9: 0.8, 0.95: 1.0}
    settings["generations"] = 101 #101 #125#TODO int(1000 /9) +1 # there are 9 samples in each step
    if debug:
        settings["generations"] = 50
        # resolution_factor = 0.5

    print(' *********', settings['cache_dir'])
    solver = cmaesFixel.CmaesSolver(settings, diffusionTensors, edema, enhancing, necrotic, gm, wm, logNameProject = "evolutionary_sampling" + experimentName, logNameRun = runName)
    resultTumor, resultDict = solver.run()

    # save results
    resPathFolder = resultpath# os.path.join(resultpath.split("/")[0:-1])
    os.makedirs(resPathFolder, exist_ok=True)

    np.save(resultpath + "gen_"+ str(settings["generations"]) + "_settings.npy", settings)
    np.save(resultpath + "gen_"+ str(settings["generations"]) + "_results.npy", resultDict)
    nibImg = nib.Nifti1Image(resultTumor, affine)
    nib.save(nibImg, resultpath+"gen_"+ str(settings["generations"]) +"_result.nii.gz")

    tools.writeNii(resultTumor, path = resultpath+"gen_"+ str(settings["generations"]) +"_result.nii.gz", affine = affine)
    
    del solver, resultTumor, resultDict, nibImg
    print("DTI done for this Patient")
    gc.collect()

#%%
def process_patient(patientID):

    # try:
        
    segmPath = osp.join(BraTS_data_dir, patientID, 'preop', f'sub-{patientID}_ses-preop_space-sri_seg.nii.gz')
    dtiPath = osp.join(registered_HCP_data, patientID, 'trans_data_rig_dt.nii.gz')
    tissuePath = osp.join(registered_HCP_data, patientID, 'transformed_tissue.nii.gz')
    fixel_dir = osp.join(registered_HCP_data, patientID, 'fixels_15')
    cache_dir = fixel_dir #osp.join(output_dir, patientID)
    os.makedirs(cache_dir, exist_ok=True)


    segm = nib.load(segmPath)
    segmentation = segm.get_fdata()
    affine = segm.affine

    brainTissue = nib.load(tissuePath).get_fdata()
    diffusionTensorsLower = nib.load(dtiPath).get_fdata()
    diffusionTensors = toolsDTI.get_tensor_from_lower6_mrtrix(diffusionTensorsLower)

    print("found data for patient", patientID)
    # except Exception as e:
    #     print(f"patient {patientID} not found: {e}")
    #     return

    wm = brainTissue == 3
    gm = brainTissue == 2

    #exclude CSF
    CSFMask = brainTissue == 1
    # include tumor segmentation region,
    mask = CSFMask.copy()
    mask[segmentation > 0] = 0
    diffusionTensors[mask] = 0

    #exclude CSF 
    wm[mask] = 0
    gm[mask] = 0
    gm[np.logical_and(CSFMask, segmentation>0)] = True

    brainmask = brainTissue > 0

    edema = np.logical_or(segmentation == 3, segmentation == 2)
    necrotic = segmentation == 1
    enhancing = segmentation == 4

    if np.sum(edema) + np.sum(necrotic) + np.sum(enhancing) < 25 or np.sum(edema) < 25:
        print("Too small tumor for patient", patientID)
        return

    resultpath = osp.join(output_dir, experimentName, patientID, f'sub-{patientID}_{run_tag}') 
    print('Result path is', resultpath)
    run(edema, necrotic, enhancing, affine, diffusionTensors, brainmask, fixel_dir, resultpath, gm, wm, cache_dir=cache_dir, runName = f"/{patientID}")

    # Explicit cleanup
    del segm, segmentation, brainTissue, diffusionTensorsLower, diffusionTensors
    gc.collect()


#%%
def try_process_patient(patientID):
    # try:
    process_patient(patientID)
    # except Exception as e:
    #     print(f"Error processing patient {patientID}: {e}")

if True:
    patient_list = ['BraTS2021_00014'] #, 'BraTS2021_00212', 'BraTS2021_00016', 'BraTS2021_00070']
    # patient_list = os.listdir(registered_HCP_data)
    patient_list.sort()
    print(patient_list)
    try_process_patient(patient_list[0])
    # with Pool(3) as p:
    #     p.map(try_process_patient, patient_list)
        