# comprehensive-OCT-evaluation


multi-label classification：

MMID Stage1：

train with train_multi_avg_fusion.py and  fusion_model.py.

using the result checkpoint to predict the train studies with batch_predict50_csv_buildmynewtrain.py and fusion_model_buildmynewtrain.py.

fusion_model_buildmynewtrain.py is a modified fusionmodel that return the weights of each image contribution.


MMID Stage2:

train with train_multi_org.py using the distillation result training set.








medical report generation：

train with main_oct.py and r2genUPD_version2.py which is a improvement scheme based on OCT features on the R2GEN model


There are three main improvements:

1,For OCT multi-frame images, 10-frame sampling is used.
2,The position of multi-frame feature sampling fusion is set behind the encoder. In other words, a mean feature fusion layer is added between the encoder and decoder.
3,The model simultaneously performs inference on six types of labels, and then fuses them through cross-task feature fusion between the encoder and decoder.
