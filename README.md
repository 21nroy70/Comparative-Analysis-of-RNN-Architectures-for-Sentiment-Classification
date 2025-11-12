Before running any of the .py files, make sure to run on the terminal (once you're in the directory ofc) as the **VERY FIRST THINGS TO DO**: 


1. conda create -n imdb-rnn python=3.10 -y

2. conda activate imdb-rnn

3. pip install -r requirements.txt

4. python -m src.preprocess --csv data/IMDB_Dataset.csv --out_dir data/processed --vocab_size 10000

5. Look at the 30 different commmands in train.py to see the commands (they have already been ran and saved in "metrics.csv" with the plots in the results folder

6. python -m src.evaluate

7. python -m src.evaluate --also_clipped true

8. python -m src.evaluate --sort_by f1_desc

9. python -m src.evaluate --sort_by accuracy_desc

10. python -m src.evaluate --sort_by time_asc

11. python -m src.evaluate --make_optimal_plots true

Include:

- Setup instructions (Python version, dependencies)
- How to run training and evaluation scripts
- Expected runtime and output files





