# CONTaiNER
Source code and relevant scripts for our ACL 2022 paper: "[CONTaiNER: Few-Shot Named Entity Recognition via Contrastive Learning](https://arxiv.org/pdf/2109.07589.pdf)".

## Requirements

- Python 3.8.5
- PyTorch (tested version 1.8.1)
- Transformers (tested version 4.6.0)
- seqeval

You can install all required Python packages with `pip install -r requirements.txt`

## Dataset 
- To run on [Few-NERD dataset](https://arxiv.org/abs/2105.07464), download Few-NERD data:
```
   wget -O episode_data.zip https://cloud.tsinghua.edu.cn/f/8483dc1a34da4a34ab58/?dl=1
   wget -O data/few-nerd/inter/train.txt https://cloud.tsinghua.edu.cn/f/45d55face2a14c098a13/?dl=1
   wget -O data/few-nerd/intra/train.txt https://cloud.tsinghua.edu.cn/f/b169cfbeb90a48c1bf23/?dl=1
   ```
Update (6/13/2022): Looks like the previous links to Few-NERD dataset is expired. Thanks to [jiayuemoon](https://github.com/jiayuemoon) for pointing this out. Please follow this [issue](https://github.com/psunlpgroup/CONTaiNER/issues/5#issuecomment-1153709315) to get the updated link.
- Execute [process_fewnerd.sh](process_fewnerd.sh) for preprocessing and organization of support sets and test sets
- To run the other tests, please obtain permission (publicly not available) and download OntoNotes and other datasets (Update: the corresponding support sets have been added for the target datasets). Then preprocess them in a similar manner. The datasets are required to be in the OntoNotes like NER format as organized in the case of Few-NERD.

## Running CONTaiNER
- Install all dependencies in requirements.txt 
- If you want to use the evaluate the few-shot performance, download the pre-trained models (inter and intra) [here](https://pennstateoffice365-my.sharepoint.com/:u:/g/personal/sfd5525_psu_edu/Ed49crJl--dIrJHfgl6ctIQBStzGGt-47GjUThDdyINkmQ?download=1), and decompress in the main directory
- Run [exec_container.sh](exec_container.sh) with the following parameters: ```[task-group] [gpu to use] [way (5/10)] [shots (1/5)]```
  - Example: To run evaluation on Few-NERD (intra) on 5 way 5~10 shot task with GPU 0, run ```exec_container.sh intra 0 5 5```
- Since support set - test set pairs are evaluated separately, this test is lengthy. Thus by default the script only checks the performance on first 50 pairs. To do the full test, please increase the iteratrions in [exec_container.sh](exec_container.sh) to 5000.
- Finally, calculate the micro-averaged F1 as done in the [Few-NERD paper](https://arxiv.org/abs/2105.07464) by running [calc-micro-avg.py](src/calc-micro-avg.py) with the argument  ```--target_dir [directory to results.txt] --range [iterations in the script ran]```. Use the directory that contains all the results of the test.
- To train the model from scratch with ```inter``` and ```intra``` training data, remove any remaining model files in [saved_models](saved_models) and run [container.py](src/container.py) similar to [exec_container.sh](exec_container.sh) and use the argument ```--do_train```. To learn further about all the arguments, please see [container.py](src/container.py) and [exec_container.sh](exec_container.sh)

## Citation
If you use our work, please cite:
```bibtex
@inproceedings{das2022container,
  title={CONTaiNER: Few-Shot Named Entity Recognition via Contrastive Learning},
  author={Das, Sarkar Snigdha Sarathi and Katiyar, Arzoo and Passonneau, Rebecca J and Zhang, Rui},
  booktitle={ACL},
  year={2022}
}
```
