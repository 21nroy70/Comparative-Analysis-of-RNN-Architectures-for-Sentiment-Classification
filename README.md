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

- Python 3.10 and dependeices are in requirements.txt which is supposed to be ran in step 3 as you can see above
- Running the train.py on all 30 variations (which is ~ 20% of all possible combos) took me 5 hours because of 6 epochs, and especially with bilstm running the longest, followed by lstm. I have already included and updated the data with the finished runs which is stored in the "metrics.csv" or "summary_table_sorted.csv" where the final combos are sorted in a clean order to view. Likewise,you can see every plot in the results/plots folder, including the optimal plot with the 3 optimal models with their specific combo listed as well in the "optimal_summary.md" and "optimal_summary.csv" files. 





