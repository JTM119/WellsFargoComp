# WellsFargoComp
File Descriptions

WellsFargoComp.ipynb : This is a python notebook where I did all of my data exploration and model training. The models that are stored in this repo were created there

WellsFargoComp.py : This is a python file that loads in the file to predict from, modifies the data, and then runs it through the models.

mlp_feeder : This is the random forest model that feeds into the MLP

best-model.pt : This was the MLP model that gave the best F1 score. It encorporates the output from the mlp_feeder random forest model

tut3-model.pt : This was a plain MLP model that was saved

trainset.csv/xlsx : These were the files containing the training data in csv and excel formatting

testset_for_participants.csv/xlsx : These were the test files

output.csv : These are the predictions for each of the datapoints in the test set

environment_packages.txt : A list of the packages that I used and what versions for the python file. Note there were more used in the notebook

Rules : This described the rules for the competition

ModelVisualizationAndRationalization.pdf : This is a control flow diagram that shows how the data was manipulated and pushed through the model. It also gives a brief summary of all that was done

To run the WellsFargoComp.py please use Python WellsFargoComp.py


