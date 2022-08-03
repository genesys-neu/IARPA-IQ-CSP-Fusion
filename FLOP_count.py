import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from numpy import linalg as LA
import os
import argparse
import ModelHandler_Variant, ModelHandler
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--network_type', type=str, default='c',choices = ['iq', 'c', 'nc', 'fusion'],
help='Which data to use as input. Select from: raw IQ data, non-conjugate features, conjugate features.')
parser.add_argument('--dataset', type=str, default='nwra',choices = ['neu', 'powder', 'nwra'],
help='Which dataset to use.')
parser.add_argument('--anom_detection', type=bool, default=False)
parser.add_argument('--iq_slice_len',default=131072, type=int, choices = [131072, 262144, 524288], help='Slice length for processing IQ files')

args = parser.parse_args()
print('Argument parser inputs', args)

# Calculating the L2 Norm of One Layer
def calculate_one_layer_l2_norm(layer):
  l2_weight = 0
  if isinstance(layer, torch.nn.Conv1d) or isinstance(layer, torch.nn.Linear):
    l2_weight = LA.norm(layer.weight.cpu().detach().numpy())
  return l2_weight

def calculate_one_classifier_l2_norm(one_classifier):
  total_l2_norm = 0
  for index in range(len(one_classifier)):
    one_layer_l2_norm = calculate_one_layer_l2_norm(one_classifier[index])
    total_l2_norm += one_layer_l2_norm
    # print("The l2 Norm for layer", one_classifier[index], " is: ", one_layer_l2_norm)
  return total_l2_norm


def calculate_one_neural_network_l2_norm(neural_net):
  l2_weight = 0
  # print("======================================================")
  # print(neural_net)
  for out_layer in list(neural_net.children()):
      if isinstance(out_layer, torch.nn.Sequential):
        # print("Inside AlexNet")
        one_classifier_l2_norm = calculate_one_classifier_l2_norm(out_layer)
        l2_weight += one_classifier_l2_norm
  return l2_weight

with torch.cuda.device(0):
  # Model files for Real data for Anom detection
  if args.dataset =='nwra' and args.anom_detection == True:
    if args.network_type == 'nc': model_file = 'D:/IARPA_DATA/Saved_Models/NWRA_dataset_models/non_conjugate_'+str(args.iq_slice_len)+'.pt' # CSP features
    if args.network_type == 'c': model_file = 'D:/IARPA_DATA/Saved_Models/NWRA_dataset_models/conjugate_'+str(args.iq_slice_len)+'.pt'
    if args.network_type == 'iq':
        model_file = 'D:/IARPA_DATA/Saved_Models/NWRA_dataset_models/IQ_'+str(args.iq_slice_len)+'.pt'
        # model_file = 'D:\IARPA_DATA\Saved_Models\POWDER_dataset_models\IQ_131072_penultimate.pt'
    if args.network_type == 'fusion': model_file = 'D:/IARPA_DATA/Saved_Models/NWRA_dataset_models/non_conjugate_iq_'+str(args.iq_slice_len)+'.pt'

  # Model files for Real data for Mod recognition
  if args.dataset =='nwra' and args.anom_detection == False:
    # if args.network_type == 'nc': model_file = 'D:/IARPA_DATA/Saved_Models_Variant/NWRA_dataset_models/non_conjugate_'+str(args.iq_slice_len)+'.pt' # CSP features
    if args.network_type == 'c': model_file = 'D:/IARPA_DATA/Saved_Models_Variant/NWRA_dataset_models/Variant_model_c_'+str(args.iq_slice_len)+'_penultimate.pt'
    if args.network_type == 'iq':
        model_file = 'D:/IARPA_DATA/Saved_Models_Variant/NWRA_dataset_models/Variant_model_iq_'+str(args.iq_slice_len)+'_penultimate.pt'
        # model_file = 'D:\IARPA_DATA\Saved_Models\POWDER_dataset_models\IQ_131072_penultimate.pt'
    if args.network_type == 'fusion': model_file = 'D:/IARPA_DATA/Saved_Models_Variant/NWRA_dataset_models/Variant_model_conjugate_iq_'+str(args.iq_slice_len)+'_penultimate.pt'
  # model_file = 'D:/IARPA_DATA/Saved_Models/block_wise_trained_model_on_sythetic_dataset/non_conjugate_262144.pt'



  net = torch.load(model_file)
  print("The original model is: ", net)
  total_l2_norm = 0
  #print("Model Keys are: ", net.hidden1.weight)
  # if args.anom_detection == False and args.network_type == 'iq':
  #     net = torch.nn.Sequential(*list(net.children())[11:14])
  # if args.anom_detection == False and args.network_type == 'c':
  #     net = torch.nn.Sequential(*list(net.children())[11:14])
  # print(net)
  fusion_only_weights = 0
  for out_layer in list(net.children()):
      if args.network_type =='iq' and args.anom_detection==True:
          # print("The outer layer is: ", out_layer)
          if isinstance(out_layer, torch.nn.Sequential):
            one_classifier_l2_norm = calculate_one_classifier_l2_norm(out_layer)
          total_l2_norm += one_classifier_l2_norm
          # print("The l2 Norm for classifier", out_layer, " is: ", one_classifier_l2_norm)
      elif args.network_type == 'iq' and args.anom_detection == False:
          one_layer_l2_norm = calculate_one_layer_l2_norm(out_layer)
          total_l2_norm += one_layer_l2_norm
      elif args.network_type == 'fusion' and args.anom_detection==True:
          # print("The outer layer is: ", out_layer)
          # if isinstance(out_layer, torch.nn.Sequential):
          if isinstance(out_layer, ModelHandler.AlexNet1D):
            one_l2_norm = calculate_one_neural_network_l2_norm(out_layer)
            # print("*****after AlexNet: ", one_l2_norm)
          elif isinstance(out_layer, ModelHandler.FeatureNet):
            one_l2_norm = 0
            for in_layer in list(out_layer.children()):
                one_l2_norm += calculate_one_layer_l2_norm(in_layer)
                # total_l2_norm += one_l2_norm
            # print("*****after CSPNet: ", one_l2_norm)
          elif isinstance(out_layer, torch.nn.Sequential):
            one_l2_norm = calculate_one_classifier_l2_norm(out_layer)
          else:
            one_l2_norm = calculate_one_layer_l2_norm(out_layer)
          total_l2_norm += one_l2_norm
          # print("The l2 Norm for classifier", out_layer, " is: ", one_l2_norm)
      elif args.network_type == 'fusion' and args.anom_detection==False:
          # print("The outer layer is: ", out_layer)
          # if isinstance(out_layer, torch.nn.Sequential):
          # if isinstance(out_layer, ModelHandler.RFNet) :
          #   one_l2_norm = calculate_one_neural_network_l2_norm(out_layer)
            # print("*****after AlexNet: ", one_l2_norm)
          if isinstance(out_layer, ModelHandler_Variant.FeatureNet) or isinstance(out_layer, ModelHandler_Variant.RFNet):
            one_l2_norm = 0
            for in_layer in list(out_layer.children()):
                one_l2_norm += calculate_one_layer_l2_norm(in_layer)
                # total_l2_norm += one_l2_norm
                # print("Individual weights: ", one_l2_norm)
            # print("*****after CSPNet: ", one_l2_norm)
          elif isinstance(out_layer, torch.nn.Sequential):
            one_l2_norm = calculate_one_classifier_l2_norm(out_layer)
          else:
            one_l2_norm = calculate_one_layer_l2_norm(out_layer)
            fusion_only_weights += one_l2_norm
          total_l2_norm += one_l2_norm
          # print("The l2 Norm for classifier", out_layer, " is: ", one_l2_norm)
      else:
          one_layer_l2_norm = calculate_one_layer_l2_norm(out_layer)
          total_l2_norm += one_layer_l2_norm

          #print("The l2 Norm for layer", out_layer, " is: ", one_layer_l2_norm)

  print("***********Total l2 Norm weight is (working):***********", total_l2_norm)
  print("Fusion only weights: ", fusion_only_weights)
    # GET INFORMATION ABOUT THE NEURAL NETWORK
  # for name, param in net.named_parameters():
  #     print('name: ', name)
  #     print(type(param))
  #     print('param.shape: ', param.shape)
  #     print('param.requires_grad: ', param.requires_grad)
  #     print('=====')

  input_shape_nc = (1, 4)# (1, 8, 4)
  input_shape_iq = (args.iq_slice_len, 2, 1)
  if args.anom_detection == True: input_shape_fusion = (1, 256)
  else: input_shape_fusion = (1, 384)
  if args.network_type == 'nc' or args.network_type == 'c':
      macs, params = get_model_complexity_info(net, input_shape_nc, as_strings=True,
                                               print_per_layer_stat=True, verbose=True)

  if args.network_type == 'iq':
      macs, params = get_model_complexity_info(net, input_shape_iq, as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
  if args.network_type == 'fusion':
      if args.anom_detection == True: net_fusion = torch.nn.Sequential(*list(net.children())[2:7])
      else: net_fusion = torch.nn.Sequential(*list(net.children())[2:8])
      # print("Printing net fusion: ....", net_fusion)
      macs_iq, params_iq = get_model_complexity_info(net.modelA, input_shape_iq, as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      macs_csp, params_csp = get_model_complexity_info(net.modelB, input_shape_nc, as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      macs_fusion, params_fusion = get_model_complexity_info(net_fusion, input_shape_fusion, as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      macs = macs_iq+ macs_csp +macs_fusion
      params = params_iq + params_csp + params_fusion
      # macs = macs_csp + macs_fusion
      # params = params_csp + params_fusion

  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))

