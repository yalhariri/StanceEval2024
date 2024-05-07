# StanceEval2024

The solution is based on Ollama and uses the model command-r:35b-v0.1-q8_0

You need to the Blind test file inside the Data folder and named as ```Mawqif_AllTargets_Blind Test.csv```
The output will be written to the CSV in the required format with the name output_testset_model.csv.


to run the python script, which have the exact same content for the notebook, you can run it by the following command:

```
python StancEval_Test.py ./Data/Mawqif_AllTargets_Blind\ Test.csv ./output_final_test.csv
```