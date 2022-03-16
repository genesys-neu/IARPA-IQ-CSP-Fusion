# Instructions for running the inference code from CSP features

Please give the proper path of the local directory where the CSP features are stored, to run `python main_inference.py`

Sample test folder `test/SCP/` is given for directly downloading and testing the inference file. 

The model file `non_conjugate_all_SNR_all_SIR.pt` is already set in the `main_inference.py`.

The model is trained on the 4th column of the non conjugate CSP features, it skips the conjugate features. 

Example:
~~~
python main_inference.py --data_folder 'path\to\direction\where\the\csp\features\are\stored\
~~~

Example Output:
~~~
The prediction from CSP features in  OnlyLTE_frame_120_131072_3.NC  is:  OnlyLTE
Total time of execution for OnlyLTE_frame_120_131072_3.NC  is :  0.002008676528930664  seconds.

The prediction from CSP features in  Combined_LTE_DSSS_frame_127_262144_4.NC  is:  Combined_LTE_DSSS
Total time of execution for Combined_LTE_DSSS_frame_127_262144_4.NC  is :  0.002009153366088867  seconds.
~~~


