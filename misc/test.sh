dataset=/scratch/nka77/shapenet/
batchSize=32
num_points=2500
nepoch=100
lr=0.001
# feature_transform=True
# python train_classification.py --dataset $dataset --batchSize $batchSize --num_points $num_points --lr $lr --nepoch $nepoch --feature_transform $feature_transform

lr=0.0001
# feature_transform=False
# class_choice=Table
# python train_segmentation.py --dataset $dataset --batchSize $batchSize --lr $lr --nepoch $nepoch --feature_transform $feature_transform --class_choice $class_choice


## CLASSIFICATION
feature_transform=True
python codes/show_cls.py --model cls/weights_with_transform.pt --feature_transform $feature_transform --dataset $dataset

feature_transform=False
python codes/show_cls.py --model cls/weights_without_transform.pt --feature_transform $feature_transform --dataset $dataset

## SEGMENTATION
class_choice=Chair
feature_transform=True
python codes/show_seg.py --model seg/weights_with_transform.pt --feature_transform $feature_transform --dataset $dataset --class_choice $class_choice

feature_transform=False
python codes/show_seg.py --model seg/weights_without_transform.pt --feature_transform $feature_transform --dataset $dataset --class_choice $class_choice

class_choice=Table
feature_transform=True
python codes/show_seg.py --model seg/weights_with_transform_table.pt --feature_transform $feature_transform --dataset $dataset --class_choice $class_choice

feature_transform=False
python codes/show_seg.py --model seg/weights_without_transform_table.pt --feature_transform $feature_transform --dataset $dataset --class_choice $class_choice

