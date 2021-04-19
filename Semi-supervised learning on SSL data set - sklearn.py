#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install sslbookdata 


# In[2]:


import sslbookdata as sbd
import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import confusion_matrix, classification_report


# # Dataset  Digit1

# In[3]:


digit1 = sbd.load_digit1(10, labels=100, return_X_y=False)


# In[4]:


digit1['data']


# In[5]:


digit1['target']


# In[6]:


len(digit1['target'])


# In[7]:


#shuffle data
rng = np.random.RandomState(2)
indices = np.arange(len(digit1['data']))
rng.shuffle(indices)


# In[8]:


indices


# In[9]:


# Take the first 500 data points (after shuffle) to train
train_data = digit1['data'][indices][:500]
train_target  = digit1['target'][indices][:500]


# In[10]:


train_target1 = np.copy(train_target)


# In[11]:


# Keep the label of the first 50 data points and remove the label of the remaining 450 data points
n_labeled_points =50
indices = np.arange(500)
unlabeled_set = indices[n_labeled_points:]


# In[12]:


# Assign 0 to the label of 450 "unlabeled" data points
train_target1[unlabeled_set] = 0


# In[13]:


#Label Propagation model train
lp_model = LabelPropagation()
lp_model.fit(train_data, train_target1)


# In[15]:


# Transductive learning: predict labels for "unlabeled input"
predicted_labels = lp_model.transduction_[unlabeled_set]
true_labels = train_target[unlabeled_set]


# In[16]:


predicted_labels


# In[17]:


cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)


# In[18]:


cm


# In[140]:


print(classification_report(true_labels, predicted_labels))


# In[ ]:


# load_coil2(split, labels=10, return_X_y=False)
# load_bci(split, labels=10, return_X_y=False)
# load_g241c(split, labels=10, return_X_y=False)
# load_coil(split, labels=10, return_X_y=False)
# load_g241n(split, labels=10, return_X_y=False)
# load_secstr(split, labels=100, extra_unlabeled=False, return_X_y=False)
# load_text(split, labels=10, return_X_y=False)


# # Dataset COIL ( 6 classes)

# In[32]:


coil = sbd.load_coil(split = 1, labels=10, return_X_y=False)


# In[33]:


coil['data']


# In[34]:


coil['target']


# In[35]:


#shuffle data
rng = np.random.RandomState(2)
indices = np.arange(len(coil['data']))
rng.shuffle(indices)


# In[36]:


# Take the first 1000 data points (after shuffle) to train
train_data = coil['data'][indices][:1000]
train_target  = coil['target'][indices][:1000]


# In[40]:


# Keep the label of the first 200 data points (20%) and remove the label of the remaining 950 data points
n_labeled_points = 200
indices = np.arange(1000)
unlabeled_set = indices[n_labeled_points:]


# In[41]:


# Assign 6 to the label of 950 "unlabeled" data points
train_target1 = np.copy(train_target)
train_target1[unlabeled_set] = 6


# In[49]:


#Label Propagation model train
lp_model = LabelPropagation(gamma= 20 , max_iter= 1000, n_neighbors = 2)
lp_model.fit(train_data, train_target1)


# In[50]:


# Transductive learning: predict labels for "unlabeled input"
predicted_labels = lp_model.transduction_[unlabeled_set]
true_labels = train_target[unlabeled_set]


# In[51]:


predicted_labels


# In[ ]:




