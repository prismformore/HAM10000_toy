# A Toy Project for Skin Cancer Classification
A toy project for the HAM10000 dataset (https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/home).

I call it toy because the dataset contains stong unbalanced classes. The setting of test set should be reconsidered. This is also a open question as far as I am conerned.

## Dataset Statistics
|class|number|
-------- | :-----------:  | 
| nv | 6705  | 
| mel | 1113| 
| bkl | 1099|
| bcc  | 514 |
|akiec | 327|
|vasc  |142|
|df |115|
       
Here,

|class|diagnostic categories|
-------- | :-----------:  | 
|akiec| Actinic keratoses and intraepithelial carcinoma / Bowen's disease |
|bcc| basal cell carcinoma |
|bkl| benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)|
|df| dermatofibroma |
|mel| melanoma  |
|nv| melanocytic nevi |
|vasc| vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)|

# Analysis
Detailed in the jupyter notebook.

## Performance
Test for model ckp_step_30000, Test R1: 0.8878535628318787, R5: 0.9840266108512878

## Project Structure
in `/config` you should find:

- settings.py:          For parameters used in the project
- backbone.py:       Backbone network
- main.py:              Main program, run the training by `python main.py -a train', run the testing by 'python main.py -a test'
- models.py:          Network models definition
- solver.py:            Code of warm up learning rate
- transforms.py:    Some transformation of the input images
- clean.sh:             Clean the checkpoints and log



## Run the Code
Training: ```python main.py -a train```
Testing:  ```python main.py -a test -m checkpoint_name```

