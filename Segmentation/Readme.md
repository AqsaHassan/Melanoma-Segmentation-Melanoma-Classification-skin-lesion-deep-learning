This is segmentation code based on Double U-Net that was trained on ISIC 2018 Datasets. 

### Installations

To install the required libraries run the command
'pip install -r requirements'

It will install all the required libraries.

## Dataset

Put the ISIC2018 Dataset in the two folders,

- training_folder with images named 'ISIC2018_task1_training'

- ground truth folder with images named 'ISIC2018_task1_GroundTruth'

## Training

To train the model run the command 
'python train.py'

## Testing 

To test the trained model run the file in prediction model with the help of readme in that folder. You can also find the trained model in this prediciton folder as well.