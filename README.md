# Text-driven Head Motion Synthesis 
### 2019 MSc project

This is a document for the program of this project. Most files have been well commented, described and orgainized.

Dependencies:
- Python 3.7.3
- PyTorch 1.1.0
- matplotlib 3.1.0
- numpy 1.16.4
- scipy 1.2.1
- scikit-learn 0.21.2

### Run program
The file `train_dct.py` and `train_seq2seq.py` can be run directly. They are the main programs for training and you can modify hyperparameters inside to test different configurations. I tried to add GPU support but failed, so only add some basic GPU-related code and uncomment them.

For other files like `data_process.py`, `token_process.py`, `eval_dct.py`, `eval_seq2seq.py`,it is recommended to run them using ipython/jupyter kernel in PyCharm or VS Code. So you can do more operations on data. The files `dataset.py` and `models.py` contains the core code for the two models. And `utils.py` contains some useful tools. 

The other necessary data files are also included in this folder. With these, you can reproduce most of the experiments in the thesis. Because the time was limited when writing thesis, some visualization was done in jupyter notebook and that code was a messey, hard to organized. So I can only restore some code by memory. Also, I write the code in the end of `eval_xxx.py` file that saves all outputs of test set, so you can make use of that.

Besides, there are two folders `glove` and `Recordings_October_2014` is not included because they are too large and is public online and on afs, you can make a link to them.

### Subjective evaluation
The synthesised animation and survey webpage are in folder `subj_eval_case`. The webpage is made by very simple HTML. By swap \<video> labels, you can shuffle the order of test cases, which must not been seen by participants. 

---
You can contact me by email: mrcjava@gmail.com
Good luck!

Jiahua Chu, Aug 2019