# Fitastic System
## Description
Fitastic is a ML & DL based system that ultilizes the personal workout records with typical feature patterns to recommend a customized workout and a diet plan. Instead of collecting users' height, weight, and other private information, Fitastic regards users’ heart rate changes as an important indicator to establish model. Calorie is an important input number for Fitastic to understand how much calorie users want to consume and take in, and recommend plans on that basis.

## Functions
* To provide personal workout recommendation based on input calorie
* To provide heart rate prediction during workout, and suggestions on speeding up or slowing down for reaching the target calorie consumption
* To make actual calorie consumption prediction after workout
* To provide personal diet recommendation based on input calorie and selected meal types

## Data
* Over 50 thousand workout records of over 900 users collected from [https://www.endomondo.com/](https://www.endomondo.com/)
* Almost 20 thousand recipes scraped from [https://www.allrecipes.com/](https://www.allrecipes.com/)

## Running Instruction
### Using docker (Strongly Recommended)
**Prerequisite**

```
OS: Linux
GPU: Nvidia graphics card 
Drive: Nvidia-driver-410+
```
**Steps**

1. install Nvidia-driver-410, referring to [https://discuss.luxonis.com/d/17-installing-nvidia-410-cuda-and-cudnn-on-ubuntu-16-04](https://discuss.luxonis.com/d/17-installing-nvidia-410-cuda-and-cudnn-on-ubuntu-16-04)
1. install NVIDIA docker toolkit, referring to [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
1. install docker compose, referring to [https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)
1. pull docker-compose.yml (link) and run

### Using Local IDE
**Prerequisite**

`GPU: Nvidia graphics card `

**Steps**

1. install CUDA Toolkit 11.3.0 (*suggested*), referring to [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)
2. install cuDNN 8.2.0 (*suggested*), referring to [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
3. install pytorch cuda 11.1 (*suggested*), referring to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
4. clone codes (link) from Github
5. download files from the following links: [https://fitastichr.s3.amazonaws.com/model_epoch_04.pt](),
[https://fitastichr.s3.amazonaws.com/mock_dataset.csv](), [https://fitastichr.s3.amazonaws.com/processed_endomondoHR_proper_interpolate_1k.csv](), [https://fitastichr.s3.amazonaws.com/processed_endomondoHR_proper_interpolate_5k.csv](), [https://fitastichr.s3.amazonaws.com/recipes.csv]() and put all files in the CS25-2 folder
6. run the following codes 

	```
	pip install flask 
	pip install xgboost==0.90
	pip install jinja2
	pip install numpy
	pip install pandas
	pip install pyecharts
	pip install scipy
	pip install sklearn
	pip install pymysql
	pip install matplotlib
	pip install hashlib
	```
7. run `python app.py` to launch Fitastic

## Test Account (Ideal users)
```
username: Eason
password: 123456
email: Eason@gmail.com
```
```
username: Peter
password: 123456
email: Peter@gmail.com
```

