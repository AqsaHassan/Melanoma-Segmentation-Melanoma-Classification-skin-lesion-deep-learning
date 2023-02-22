# Melanoma-Segmentation-Melanoma-Classification-skin-lesion-deep-learning

Skin lesion is one of the common contributors to deaths globally. However, early detection of this disease can lead to a very high chance of survival. Deep learning neural networks have been successful in demonstrating promising progress on medical imaging for skin lesion classification. Therefore, in this project, we have evaluated the performance comparison of the recent state-of-the-art deep convolutional neural networks, that have already achieved significant performance on lesion image segmentation and classification, for the task of skin lesion detection using HAM10000 through transfer learning. In order to address the limitation of training data, several data augmentation techniques applied to the training dataset to avoid overfitting for skin lesion classification. Moreover, different optimization techniques are used while training these models, such as learning rate decay and dropout, which made the model generalized better for the skin lesion classification. In addition, we investigated ensemble learning to build a single strong model that has a higher performance accuracy than any of the base models. Intensive experimentation has also been performed to identify the role of segmentation in the skin lesion classification where ISIC 2018 dataset has been used for training the double-UNet architecture. This achieved 0.76 in Dice coefficient, 0.72 in specificity and 0.80 sensitivity. The overall results of experimentation revealed that the ensemble of InceptionV3 and DenseNet201 has outperformed by achieving 0.76 accuracy without segmentation and 0.75 with segmentation.
