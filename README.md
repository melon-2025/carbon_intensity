
To have all other information fo to the wiki home page [Home](Home)
# start

1. At the root of the project create a python virtual environment

Windows
python -m venv myenv
cd myenv\Scripts\activate

En mac
python3 -m venv myenv
source myenv/bin/activate

# 2nd step part one

Install Dependencies
Go to the root of the project and install requirements.txt. cou do this by writing the command below.
`pip install -r requirements.txt`

# 2nd step part two

- Download all the necessary datasets in their respective folders [DataSet Table](DataSet Table)
- From electricity Maps download the data set for carbon intensity dataset hourly   for 2021,2022,2023
- From swiss Grid download consumption and production dataset 15 min frequency dataset for 2021,2022,2023
- Place the datasets in the folder data/raw

# 3rd step you go to the preprocessing folder

- make sure all the datasets are placed in the correct folder
- then open the jupiter notebook called preprocess_data.ipynb
- then you select kernel to the enviroment name then you execute it

# 4th step you go to the folder model

- open the sarima_model_hourly.ipynb
- select kernel with the python environment
- you execute it, make sure all the datasets are in the correct folder

# 5th step you go to the app folder

- make sure you have the virtual environment like the previous steps
- select the kernel
- make sure in the folder data/models/sarimax_model_hourly.pkl exists
- make sure in the folder data/models/naive_data exists
- make sure in the folder data/test_data/test_data.csv exists
- make sure in the folder data/test_data/exog_test_hourly.csv exists
- then
- go to the line command CD app
- the execute the app `python app.py`

# to see the exploration

- go to exploration folder
- then open carbon_Intensity_hourly.ipynb then select kernel to the python environment you created and run it.
- open consumptionProduction.ipynb and run it after selecting the kernel

Select the environment when you open jupiter notebook
