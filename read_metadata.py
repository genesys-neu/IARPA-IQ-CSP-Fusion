import  pandas as pd
import os
import itertools

metadata_path = 'D:/IARPA_DATA/IQDataSet_LTE_DSSS_v2/Metadata/'
feature_path = 'D:/IARPA_DATA/' + 'NEU_LTE_DSSS_Dataset_2_CSP/'

snr_list = [0, 5, 10]
sir_list = [0, 5, 10]
# dsss_snr_sir_list = tuple(zip(snr_list, sir_list))
dsss_snr_sir_list = list(itertools.product(snr_list, sir_list))
dsss_dict= dict.fromkeys(dsss_snr_sir_list, 0)
lte_dict = dict.fromkeys(snr_list, 0)

print(lte_dict)
print(dsss_dict)

total_lte_count = 0
total_dsss_count = 0
for filename in os.listdir(feature_path):
    meta_filename = filename[::-1].split('_', 2)[-1][::-1]
    print("the meta file name: ", meta_filename)
    one_meta_file_df = pd.read_csv(metadata_path + meta_filename + '.csv', sep=",", header=0).dropna(axis=1, how='all')

    # try:
    #     one_file_df = pd.read_csv(feature_path + filename, sep=" ", header=None).dropna(axis=1, how='all')
    #
    # except pd.errors.EmptyDataError:
    #     print("Note:", filename, "was empty. Skipping.")
    #     # count = count - 1 # decreasing the total file count
    #     continue

    if 'LTE_DSSS' in meta_filename:
        #(SNR, SIR)
        sir = one_meta_file_df['LTE_DSSS_SIR_dB'].to_numpy().item()
        snr = one_meta_file_df['LTE_SNR_dB'].to_numpy().item()
        # print('DSSS SIR and LTE SNR: ', sir, snr)
        dsss_dict[(snr, sir)]=dsss_dict[(snr, sir)]+ 1
        total_dsss_count =total_dsss_count+ 1
    else:
        # print('LTE SNR: ', sir, snr)
        snr = one_meta_file_df['LTE_SNR_dB'].to_numpy().item()
        lte_dict[snr] = lte_dict[snr]+1
        total_lte_count = total_lte_count+1

print(lte_dict)
print(dsss_dict)
print("total LTE count: ", total_lte_count)
print("total DSSS count: ", total_dsss_count)
