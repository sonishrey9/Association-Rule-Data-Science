# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# %%
transaction = []
with open("transactions_retail1.csv") as f:
    transaction = f.read()


# %%
# splitting the data into separate transactions using separator as "\n"
transaction = transaction.split("\n")


# %%
transaction


# %%
transaction_list = []
for i in transaction:
    transaction_list.append(i.split(","))


# %%
transaction_list


# %%
all_transaction_list = [i for item in transaction_list for i in item]


# %%
from collections import Counter # ,OrderedDict
item_frequencies = Counter(all_transaction_list)


# %%
# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])


# %%
# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


# %%
# barplot of top 10 
import matplotlib.pyplot as plt


# %%
plt.bar(height = frequencies[0:11], x = list(range(0, 11)))
plt.xticks(list(range(0, 11), ), items[0:11], rotation = 90)
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# %%
# Creating Data Frame for the transactions data
transaction_series = pd.DataFrame(pd.Series(transaction_list))
transaction_series = transaction_series.iloc[:9835, :] # removing the last empty transaction


# %%
transaction_series.columns = ["transactions"]
transaction_series


# %%
# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = transaction_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')
X


# %%
frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)


# %%
# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)
frequent_itemsets


# %%
plt.figure(figsize =(12,6))
plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11])
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()


# %%
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)


# %%
def to_list(i):
    return (sorted(list(i)))


# %%
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)


# %%
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
unique_rules_sets


# %%
index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# %%
# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]
rules_no_redudancy


# %%
# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(30)


# %%



