<font size="4">



## Environment
Anaconda3 (highly recommendation)
python3.6/3.7/3.8
vscode (IDE)
pytorch 1.10 (pip package)
torchvision 0.11.1 (pip package)




## Multi-label Classification

**MMID** stands for Multi-frame Medical Images Distillation

### MMID Stage1

- Train with `train_multi_avg_fusion.py` and `fusion_model.py`.
- Use the resulting checkpoint to predict the train studies with `batch_predict50_csv_buildmynewtrain.py` and `fusion_model_buildmynewtrain.py`.
- `fusion_model_buildmynewtrain.py` is a modified `fusionmodel` that returns the weights of each image contribution.

### MMID Stage2

- Train with `train_multi_org.py` using the distillation result training set.


MMID demonstrates a simple yet significantly performance-improving distillation strategy, used for medical imaging, especially OCT image classification, multi-label, or dataset organization.

## Medical Report Generation

- Train with `main_oct.py` and `r2genUPD_version2.py`, which is an improvement scheme based on OCT features on the R2GEN model.

**Three main improvements:**

1. For OCT multi-frame images, 10-frame sampling is used.
2. The multi-frame feature sampling fusion position is set behind the encoder. In other words, a mean feature fusion layer is added between the encoder and decoder.
3. The model simultaneously performs inference on six types of labels and then fuses them through cross-task feature fusion between the encoder and decoder.


This is the first time we have performed the task of report generation in OCT medical images, and the model improvements we have made demonstrate a more reliable performance.


</font>
