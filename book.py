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
from mlxtend.frequent_patterns import apriori, association_rules


# %%
books = pd.read_csv("book.csv")
books

# %% [markdown]
# # Dataframe is already sorted into dummy variable 

# %%
# here we use apiori function 

frequent_itemsets = apriori(books, min_support = 0.0075, max_len = 4, use_colnames = True)  #how we decided min support ?
frequent_itemsets


# %%
# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)


# %%
plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11])
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()


# %%
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

# %% [markdown]
# #  Extra part

# %%
def to_list(i):
    return (sorted(list(i)))


# %%
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)


# %%
ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)


# %%
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]


# %%
index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# %%
# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :] 


# %%
rules_no_redudancy


# %%
# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)


# %%



