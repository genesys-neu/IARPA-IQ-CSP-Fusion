# Instructions for running the inference code from CSP features (`main_inference.py`)

## About the Local Directory Path
Please give the proper path of the local directory where the CSP features are stored, to run `python main_inference.py`

Sample test folder `test/CSP/` is given for directly downloading and testing the inference file. 

## About the Trained Model
The model file `non_conjugate_all_SNR_all_SIR.pt` is already set in the `main_inference.py`.

The model is trained on the 4th column of the non conjugate CSP features, it skips the conjugate features. 

## Example of Running an Instance of Inference:
~~~
python main_inference.py --data_folder 'path\to\directory\where\the\csp\features\are\stored\'
~~~

## Example Output:
~~~
The prediction from CSP features in  OnlyLTE_frame_120_131072_3.NC  is:  OnlyLTE
Total time of execution for OnlyLTE_frame_120_131072_3.NC  is :  0.002008676528930664  seconds.

The prediction from CSP features in  Combined_LTE_DSSS_frame_127_262144_4.NC  is:  Combined_LTE_DSSS
Total time of execution for Combined_LTE_DSSS_frame_127_262144_4.NC  is :  0.002009153366088867  seconds.
~~~


# Instructions for running the inference code from either CSP/IQ/both features (`main_inference_all.py`)

## About the Local Directory Path for Data and Model Files 
Please give the proper path of the local directory where the files are stored and where the model files are stored.  The model files will be shared seperately, as they are total ~14GB.

The model is trained on the 4th column of the non conjugate CSP features, it skips the conjugate features. 

Sample test folder `Data_for_Inference_NWRA/` is given for directly downloading and testing the inference file. 

## Example of Running an Instance of Inference:
* To run IQ inference: 
~~~
`python main_inference_all.py --input iq --data_folder 'path\to\directory\where\the\data\is\stored\' --model_folder 'path\to\directory\where\the\models\are\stored\' 
~~~

* To run CSP feature inference: 
~~~
`python main_inference_all.py --input nc --data_folder 'path\to\directory\where\the\data\is\stored\' --model_folder 'path\to\directory\where\the\models\are\stored\' 
~~~

* To run fusion based IQ and CSP feature inference: 
~~~
`python main_inference_all.py --input iq nc --data_folder 'path\to\directory\where\the\data\is\stored\' --model_folder 'path\to\directory\where\the\models\are\stored\' 
~~~

## Example Output:
~~~
The prediction from IQ Samples in  lte_5MHz_1_1_fs_7.68MHz_9_51.tim  is:  LTE
Total time of execution for lte_5MHz_1_1_fs_7.68MHz_9_51.tim  is :  0.04089045524597168  seconds.

The prediction from CSP features in  lte_5MHz_1_1_fs_7.68MHz_10_51_131072_1.NC  is:  LTE
Total time of execution for lte_5MHz_1_1_fs_7.68MHz_10_51_131072_1.NC  is :  0.0009970664978027344  seconds.

 
The prediction from IQ Samples and CSP feature fusion in  lte_5MHz_1_1_fs_7.68MHz_9_51  is:  LTE
Total time of execution for lte_5MHz_1_1_fs_7.68MHz_9_51  is :  0.051860809326171875  seconds.
~~~
