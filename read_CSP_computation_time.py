import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams["font.family"] = "Times New Roman"


parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--iq_slice_len',default=131072, type=int, choices = [32768, 65536, 131072, 262144, 524288], help='Slice length for processing IQ files')
args = parser.parse_args()
print('Argument parser inputs', args)

filename = 'D:/IARPA_DATA/POWDER_data/Proc.log'
# filename = '/Users/debashri/Dropbox/Proc.log'
#filename = '/Users/debashri/Dropbox/test-proc.txt'

#Processing files for Pipeline 1
# filename = 'D:/IARPA_DATA/Pipeline1_timing_for_SSCA/POWDER/P1_proc_BW5MHz.log'
# filename = 'D:/IARPA_DATA/Pipeline1_timing_for_SSCA/POWDER/P1_proc_BW10MHz.log'
# filename = 'D:/IARPA_DATA/Pipeline1_timing_for_SSCA/NWRA/P1_proc_BW5MHz_V.log'
# filename = 'D:/IARPA_DATA/Pipeline1_timing_for_SSCA/NWRA/P1_proc_BW10MHz_V.log'
# filename = 'D:/IARPA_DATA/Pipeline1_timing_for_SSCA/NEU/P1_proc_Vini_2.log'


# data = pd.read_csv(filename, sep=' ', index_col=False, keep_default_na = False)
# data.fillna(0, inplace = True)
# print('The fisr BL:', data['BL'][0])
#
# args = parser.parse_args()
# print('Argument parser inputs', args)
#
#
# print("Size of block lenght", len(data['BL']))
# # for i, row in enumerate(data.values):
# #     # print(i, row)
# #     print(data['BL'][i])
# #     if data['BL'][i] == args.iq_slice_len:
# #         print (row)

calculate = False
total_time_per_file = 0
time_list_13 = []
time_list_26 = []
time_list_52 = []

# BLOCK LENGTH: 131072
with open(filename, encoding='utf8') as f:
    total_count = 0
    total_time = 0
    #total_time_per_file = 0
    for line in f:
        one_line =  line.strip()
        if one_line[0] =='B':
            oneline_list= one_line.split('   ')[2:3]
            #oneline_np = np.array(oneline_list)
            # print("The numpy array: ", oneline_np)
            # oneline_np = np.expand_dims(oneline_np, axis=0)
            print(oneline_list[0])
            if total_time_per_file > 0:
                time_list_13.append(total_time_per_file)
            calculate = False
            total_time_per_file = 0
            # for x, elem in enumerate(oneline_list):
            if str(args.iq_slice_len) == oneline_list[0].strip():
                print("Hereeeeeeeeeeeeeeeeeeee:", oneline_list[0].strip())
                #total_time_per_file = 0
                calculate = True

        # else:
        #     # print("Total time per file: ", total_time_per_file)
        #     calculate = False


        else:
            # print(one_line, calculate, total_time_per_file)
            if calculate == True:
                #print(one_line)
                total_time_per_file += float(one_line)
            # else:

            # total_time += float(one_line)
            # total_count += 1
            # time_list.append(float(one_line))

# print("The required average time: ", total_time/total_count)
# print("The mean and standard daviation:",total_time, np.mean(np.array(time_list), axis =0), np.std(np.array(time_list), axis =0) )
print("The list: ", time_list_13)
# print("The mean and std: ", np.mean(np.array(time_list_13), axis =0), np.std(np.array(time_list_13), axis =0))



#####################################################
# BLOCK LENGTH: 262144
with open(filename, encoding='utf8') as f:
    total_count = 0
    total_time = 0
    #total_time_per_file = 0
    for line in f:
        one_line =  line.strip()
        if one_line[0] =='B':
            oneline_list= one_line.split('   ')[2:3]
            #oneline_np = np.array(oneline_list)
            # print("The numpy array: ", oneline_np)
            # oneline_np = np.expand_dims(oneline_np, axis=0)
            print(oneline_list[0])
            if total_time_per_file > 0:
                time_list_26.append(total_time_per_file)
            calculate = False
            total_time_per_file = 0
            # for x, elem in enumerate(oneline_list):
            if str(262144) == oneline_list[0].strip():
                print("Hereeeeeeeeeeeeeeeeeeee:", oneline_list[0].strip())
                #total_time_per_file = 0
                calculate = True

        # else:
        #     # print("Total time per file: ", total_time_per_file)
        #     calculate = False


        else:
            # print(one_line, calculate, total_time_per_file)
            if calculate == True:
                #print(one_line)
                total_time_per_file += float(one_line)
            # else:

            # total_time += float(one_line)
            # total_count += 1
            # time_list.append(float(one_line))

# print("The required average time: ", total_time/total_count)
# print("The mean and standard daviation:",total_time, np.mean(np.array(time_list), axis =0), np.std(np.array(time_list), axis =0) )
print("The list: ", time_list_26)
# print("The mean and std: ", np.mean(np.array(time_list_26), axis =0), np.std(np.array(time_list_26), axis =0))

# x = np.linspace(0.1, 2 * np.pi, 41)
# y = np.exp(np.sin(x))
#####################################################
# BLOCK LENGTH: 524288
with open(filename, encoding='utf8') as f:
    total_count = 0
    total_time = 0
    #total_time_per_file = 0
    for line in f:
        one_line =  line.strip()
        if one_line[0] =='B':
            oneline_list= one_line.split('   ')[2:3]
            #oneline_np = np.array(oneline_list)
            # print("The numpy array: ", oneline_np)
            # oneline_np = np.expand_dims(oneline_np, axis=0)
            print(oneline_list[0])
            if total_time_per_file > 0:
                time_list_52.append(total_time_per_file)
            calculate = False
            total_time_per_file = 0
            # for x, elem in enumerate(oneline_list):
            if str(524288) == oneline_list[0].strip():
                print("Hereeeeeeeeeeeeeeeeeeee:", oneline_list[0].strip())
                #total_time_per_file = 0
                calculate = True

        # else:
        #     # print("Total time per file: ", total_time_per_file)
        #     calculate = False


        else:
            # print(one_line, calculate, total_time_per_file)
            if calculate == True:
                #print(one_line)
                total_time_per_file += float(one_line)
            # else:

            # total_time += float(one_line)
            # total_count += 1
            # time_list.append(float(one_line))

# print("The required average time: ", total_time/total_count)
# print("The mean and standard daviation:",total_time, np.mean(np.array(time_list), axis =0), np.std(np.array(time_list), axis =0) )
print("The list: ", time_list_52)
# print("The mean and std: ", np.mean(np.array(time_list_52), axis =0), np.std(np.array(time_list_52), axis =0))
print("The mean and std for 131072 block length: ", np.mean(np.array(time_list_13), axis =0), np.std(np.array(time_list_13), axis =0))
print("The mean and std for 262144 block length:  ", np.mean(np.array(time_list_26), axis =0), np.std(np.array(time_list_26), axis =0))
print("The mean and std for 524288 block length: ", np.mean(np.array(time_list_52), axis =0), np.std(np.array(time_list_52), axis =0))

samples_to_plot = len(time_list_13)

# PLOTTING THE
fig, ax = plt.subplots()
mean_13 = sum(time_list_13[0:samples_to_plot])/len(time_list_13[0:samples_to_plot])
mean_26 = sum(time_list_26[0:samples_to_plot])/len(time_list_26[0:samples_to_plot])
mean_52 = sum(time_list_52[0:samples_to_plot])/len(time_list_52[0:samples_to_plot])
x_13, y_13, text_13 = samples_to_plot//2 -500, mean_13, "Mean (131072): "+"{:.2f}".format(mean_13) + " s"
x_26, y_26, text_26 = samples_to_plot//2 -500, mean_26+1, "Mean (262144): "+"{:.2f}".format(mean_26) + " s"
x_52, y_52, text_52 = samples_to_plot//2 -500, mean_52, "Mean (524288): "+"{:.2f}".format(mean_52) + " s"
plt.xticks(fontsize=15,rotation=0)
plt.yticks(fontsize=15)
plt.ylabel('Processing Time (s)', fontsize=15)
plt.xlabel('Frame Index', fontsize=15)


plt.plot(time_list_13[0:samples_to_plot], color ='g', label="131072")
# plt.plot(mean_list_13[0:250],'b', '-.')
plt.axhline(y = mean_13, color = 'g', linestyle = '-.',linewidth=3)
ax.text(x_13, y_13, text_13,fontsize=15)
plt.plot(time_list_26[0:samples_to_plot], color ='y', label="262144")
plt.axhline(y = mean_26, color = 'y', linestyle = '-.',linewidth=3)
ax.text(x_26, y_26, text_26,fontsize=15)
plt.plot(time_list_52[0:samples_to_plot], color ='m', label="524288")
ax.text(x_52, y_52, text_52,fontsize=15)
plt.axhline(y = mean_52, color = 'm', linestyle = '-.',linewidth=3)
plt.legend(fontsize=15)
plt.show()

