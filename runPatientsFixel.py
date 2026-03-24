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
        
#%%
def run(edema, necrotic, enhancing, affine, diffusionTensors, brainmask, fixel_dir, wm, gm, resultpath):
    
    settings = {}
    # fixed parameters that are not varied
    #TODO
    settings["fixedParameters"] = ["thresholdT1c",
                                    "thresholdFlair", "diffusionTensorExponent"]#,"diffusionEllipsoidScaling"]#,  "Dw","NxT1_pct", "NyT1_pct", "NzT1_pct"]

    # init parameter
    settings["rho"] = 3.0
    settings["Dw"] = 0.15
    settings["diffusionEllipsoidScaling"] = 10.0 #TODO
    settings["diffusionTensorExponent"] = 1.0
    settings["thresholdT1c"] = 0.9
    settings["thresholdFlair"] = 0.25
    settings["fixel_dir"] = fixel_dir

    # center of mass
    com = ndimage.measurements.center_of_mass(edema)
    settings["NxT1_pct"] = float(com[0] / np.shape(edema)[0])
    settings["NyT1_pct"] = float(com[1] / np.shape(edema)[1])
    settings["NzT1_pct"] = float(com[2] / np.shape(edema)[2])

    # set parameter ranges
    settings["rho_range"] = [0.001, 5.0]
    settings["Dw_range"] = [0.001, 5.0]
    settings["thresholdT1c_range"] = [0.5, 0.9]
    settings["thresholdFlair_range"] = [0.001, 0.5]
    settings["NxT1_pct_range"] = [0,1]
    settings["NyT1_pct_range"] = [0,1]
    settings["NzT1_pct_range"] = [0,1]
    settings["diffusionEllipsoidScaling_range"] = [0.1, 100.0]
    settings["diffusionTensorExponent_range"] = [0.1, 10.0]

    # algorithm settings
    settings["workers"] =0 #9# 1#9 #9#4 # 9
    settings["sigma0"] = 0.06
    settings["lossLambdaT1"] = 0.2
    settings["lossLambdaFlair"] = 0.8

    # if dir it changes with generations: key = from relative generations, value = resolution factor
    #TODO
    settings["resolution_factor"] = 1.0 # { 0: 0.5, 0.3:0.6, 0.8: 0.8, 0.9: 1.0}
    settings["generations"] =  int(1000 /9) +1 # there are 9 samples in each step

    solver = cmaesFixel.CmaesSolver(settings, diffusionTensors, edema, enhancing, necrotic, wm=wm, gm=gm)
    resultTumor, resultDict = solver.run()

    # save results
    os.makedirs(resultpath, exist_ok=True)
    np.save(resultpath + "gen_"+ str(settings["generations"]) + "_settings.npy", settings)
    np.save(resultpath + "gen_"+ str(settings["generations"]) + "_results.npy", resultDict)
    tools.writeNii(resultTumor, path = resultpath+"gen_"+ str(settings["generations"]) +"_result.nii.gz", affine = affine)
    
    print("Done For This Patient")

# 18 patients old!!
#%%
#tgm
if  __name__ == '__main__':

    registered_HCP_data = "/mnt/Drive2/andrey/HCPTemplate2BraTS"
    BraTS_data_dir = "/mnt/Drive4/jonas/datasets/brats_2021_train2/"
    output_dir = "/mnt/Drive2/andrey/cma_es_result"

    patients = osp.listdir(registered_HCP_data)
    print(patients)
    for patient_id in patients:
        try:

            tumor_segmentation = nib.load(osp.join(BraTS_data_dir, patient_id, 'preop', 'sub-{patient_id}_ses-preop_space-sri_seg.nii.gz'))

            segmentation = tumor_segmentation.get_fdata()
            affine = tumor_segmentation.affine

            brainTissue = nib.load(osp.join(registered_HCP_data, patient_id, 'transformed_tissue.nii.gz')).get_fdata()

            brainmask = brainTissue > 0

            diffusionTensors = nib.load(osp.join(registered_HCP_data, patient_id, 'transfomed_dt.nii.gz')).get_fdata()

        except:
            print("patient not found ", patient_id)
            continue

        CSFMask = binary_dilation(brainTissue == 1, iterations = 1)

        diffusionTensors[CSFMask] = 0

        # different labels then other datasets
        edema = np.logical_or(segmentation == 3, segmentation == 2)
        necrotic = segmentation == 1
        enhancing = segmentation == 4

        datetime = time.strftime("%Y_%m_%d-%H_%M_%S")
        patient_results = osp.join(output_dir, patient_id + '_' + datetime)
        #TODO
        #resultpath = "/mnt/8tb_slot8/jonas/workingDirDatasets/tgm/cma-es_DTI_results_testing/" + ("0000" + str(patientID))[-3:] + "/"

        run(edema, necrotic, enhancing, affine, diffusionTensors, brainmask, patient_results)

#respond old!!
if False: # __name__ == '__main__':
    for patient_id in range(120,130):
        try:
            # save the parameters and the tumor
            patientNumber = ("000000" + str(patient_id))[-3:]
            print("patient number: ", patientNumber)

            patientPath = "/mnt/8tb_slot8/jonas/datasets/ReSPOND/respond/respond_tum_"+ patientNumber+"/d0/"
            
            segmentationNiiPath = patientPath + "sub-respond_tum_"+ patientNumber+"_ses-d0_space-sri_seg.nii.gz"
            tumor_segmentation = nib.load(segmentationNiiPath)

            segmentation = tumor_segmentation.get_fdata()
            affine = tumor_segmentation.affine

            tissuePath = patientPath + "sub-respond_tum_"+ patientNumber+"_ses-d0_space-sri_tissuemask.nii.gz"
            tissue = nib.load(tissuePath).get_fdata()

        except:
            print("patient not found ", patient_id)
            continue

        try:
            petPath = patientPath + "sub-respond_tum_"+ patientNumber+"_ses-d0_space-sri_fet.nii.gz"
            pet = nib.load(petPath).get_fdata()
            if pet.ndim == 4:
                pet = pet[:,:,:,0]
            #pet = pet * brainmask
            pet = pet / np.max(pet)
        except: # if no pet just set it to 0
            pet = np.zeros_like(tissue)

        # different labels then other datasets
        edema = np.logical_or(segmentation == 3, segmentation == 2)
        necrotic = segmentation == 1
        enhancing = segmentation == 4

        WM, GM = segmentation *0.0, segmentation *0.0
        WM[tissue == 3] = 1.0
        GM[tissue == 2] = 1.0

        WM[segmentation >0] = 1.0

        assert WM.shape == GM.shape == pet.shape == segmentation.shape

        datetime = time.strftime("%Y_%m_%d-%H_%M_%S")
        patient_results = '/mnt/8tb_slot8/jonas/workingDirDatasets/ReSPOND/cma-es_results/' + str(patientNumber) + 'newSettings/'
        run(edema, necrotic, enhancing, affine, pet, WM, GM, patient_results)
# %%
