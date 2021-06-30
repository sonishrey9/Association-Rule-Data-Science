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
myphone = pd.read_csv("myphonedata.csv")
myphone


# %%
myphone.info()


# %%
categorical_features= [feature for feature in myphone.columns if  myphone[feature].dtypes=='O']
categorical_features


# %%
numerical_features = [feature for feature in myphone.columns if myphone[feature].dtypes != 'O'] # list comprehension feature that are not equal to object type

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
myphone[numerical_features].head()


# %%
Y = myphone[numerical_features]
X = myphone[categorical_features]


# %%
categorical_features


# %%
V1 = pd.get_dummies(myphone["V1"])
V2 = pd.get_dummies(myphone["V2"])
V3 = pd.get_dummies(myphone["V3"])


# %%
new_myphone = pd.concat([V1,V2,V3, Y], axis =1)
new_myphone


# %%
frequent_itemsets = apriori(new_myphone , min_support = 0.0075, max_len = 4, use_colnames = True)  #how we decided min support ?
frequent_itemsets


# %%
# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)
frequent_itemsets


# %%
plt.figure(figsize =(15,8))

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

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
ma_X


# %%
ma_X = ma_X.apply(sorted)
ma_X 


# %%
rules_sets = list(ma_X) # it gives list out of groceries dataset

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)] #nested list


# %%
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
    
## it sets index but in random 


# %%
# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]
rules_no_redudancy


# %%
# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)


# %%



