# SCAM
same-radical characters information, Chinese text classification

* Using special Chinese radical information to enhance the model's text classification ability
* Using same-radical characters information

* our model in the *[model](./models)* file
* The radical dictionary and the conversion dictionary of the same-radical character position are in *[radical_vector](./radical_vector)* file

When you want to train, first clone it locally, then configure the location of the input data set in the *run.py* file, and finally run the *run.py* file.    
`python run.py --model SCAM`
