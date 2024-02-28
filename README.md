I went to the website and saw that I first had to get the data the paper used before I could perform any analysis on it
– Data.txt, Setup.sas, to_csv.py –
Downloaded ICPSR 7757 which is the data source from the paper
It was in a SAS file so I wrote the python script `to_csv.py` to convert it to a csv file that can be processed by python more easily
– another_again.py –
Next is data preprocessing: dropping rows with invalid data, creating unique identifiers for each candidate across all rows they appear in
Then we create the new columns like ‘Vote Share t+1’ by comparing each column to its next with the same candidate identifier and ‘Normalized Margin of Victory’ by dividing the margin of victory by the magnitude of number of votes
Then we drop outliers, then initialize and fit a logistic regression
Then we plot, with a vertical line marking the discontinuity at x = 0
Oh no, it doesn’t work the graph doesn’t look like the paper
– file not even here –
Lets try having chat gpt do the process of the paper but it makes up any data it needs
That doesn’t work either, looks like chat gpt doesn’t understand what the target graphs are really supposed to look like
– simulate, simulate2_better_real.py –
What if i give it the graphs to start with and then make it work backwards to the simulated data so it’s guaranteed to understand what it needs to do with the graph
Yay that works
The end
