# TexGANPrivate
Source code of the paper "Learning Texture Generators for 3D Shape Collections from Internet Photo Sets" by Rui Yu, [Yue Dong](http://yuedong.shading.me/), [Pieter Peers](https://www.cs.wm.edu/~ppeers/) and [Xin Tong](http://www.xtong.info/) from BMVC 2021.

# Dependencies
tensorflow-gpu==1.14  
opencv-python  
tqdm  

# Inference
* Copy ```exp_config\config_{car,ffhq,shoe}.py``` to ```exp_config\config.py```
* The entry of inference process is ```test_process/run_inference.py```.  
Run this command:  
```python test_process/run_inference.py --output_dir Path/To/Output/Folder --data_root Path/To/DemoInputData --model_path Path/To/PretrainedModelFile --dataset {car,ffhq,shoe}```  
The results will be found in ```Path/To/Output/Folder/test_result```  

# Training
* Copy ```exp_config\config_{car,ffhq}.py``` to ```exp_config\config.py```
* Run this command:  
```python train_process/train_chart6view.py --output_dir Path/To/Output/Folder --data_root Path/To/TrainingData --phase_kimg 200 --gpu_id 0,1,2,3 --batch_group 8 --syn_batch 1 --real_batch 1 --exp_type {car,ffhq} --grad_repeat 4```

# Cuhk Dataset #  
Pretrained model: [PretrainedModel](http://share.msraig.info/TexGAN/data/Car/PretrainedModel.zip)  
DemoInputData: [CarDemo](http://share.msraig.info/TexGAN/data/Car/CarDemo.zip)   
Training Data: [CarCuhk_RealExp](http://share.msraig.info/TexGAN/data/Car/CarCuhk_RealExp.zip)  
Source shapes of DemoInputData: [CarDemoSourceShape](http://share.msraig.info/TexGAN/data/Car/CarDemoSourceShape.zip)  

# FFHQ Dataset #
Pretrained model: [PretrainedModel](http://share.msraig.info/TexGAN/data/FFHQ/PretrainedModel.zip)  
DemoInputData: [FFHQDemo](http://share.msraig.info/TexGAN/data/FFHQ/FFHQDemo.zip)   
Training Data: [FFHQ_RealExp](http://share.msraig.info/TexGAN/data/FFHQ/FFHQ3d_RealExp.zip)  

# Shoe Dataset #
Pretrained model: [PretrainedModel](http://share.msraig.info/TexGAN/data/Shoe/PretrainedModel.zip)  
DemoInputData: [ShoeDemo](http://share.msraig.info/TexGAN/data/Shoe/ShoeDemo.zip)   
Due to copyright issue, we cannot release the training data of the Shoe dataset.  

