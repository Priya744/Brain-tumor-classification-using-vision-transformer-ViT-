# Brain-tumor-classification-using-vision-transformer-ViT-
A brain tumor is a growth of cells in or near the brain that multiplies abnormally and uncontrollably. Brain tumors are complex, with significant variations in size and location, making it challenging to fully understand their nature.
A Vision Transformer (ViT) can be used for brain tumor classification. This model, designed for image classification, uses a Transformer-like architecture on image patches. The image is divided into fixed-size patches, which are linearly embedded and combined with positional embeddings. This sequence is then fed into a Transformer encoder, and an additional learnable "classification token" is included for classification tasks.To increase the model's accuracy, two fully connected layers are also implemented. This architecture effectively classifies brain tumors into four categories: glioma tumor, meningioma tumor, pituitary tumor, and no tumor. 

Vision Transformer model:
![image](https://github.com/Priya744/Brain-tumor-classification-using-vision-transformer-ViT-/assets/98945781/a9302af1-e66f-439b-ae96-1ac7f2283d43)

Dataset:
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri 
Brain tumor classification dataset contains 2,870 mri images for training and 394 images for testing the model.
