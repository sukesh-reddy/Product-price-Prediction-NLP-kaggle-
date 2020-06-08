####################################################################################
# Mercari Price Suggestion Challenge - Kaggle
# https://www.kaggle.com/c/mercari-price-suggestion-challenge (Hold Shift+click)
#####################################################################################



###############################
# Steps taken/performed
# 1) Data initial preprocessing (Target Variable Analysis)
# 2) Feature Engineering
# 3) Tokenization
# 4) TF/IDF of all tokena
# 5) Sparse matrix creation
# 5) Xgboost model training
# 6) Prediction
################################


###################################
# ---------- Initial Step----------

# Setting the working envoirnment
###################################

rm(list = ls())

setwd('C:\\Users\\sukes\\Desktop\\Projects\\product_price_prediction_NLP')

# Loading and installing the required libraries
library(data.table)
library(xgboost)
library(tm)
library(dplyr)
library(plyr)
library(stringi)
library(stringr)
library(quanteda)
library(Matrix)
library(ggplot2)

################################
# ------- First Step-----------
# Data initial preprocessing
###############################

# Read the data

train = fread('train.tsv',na.strings = c("",NA, 'NULL'))
test = fread('test.tsv',na.strings = c("",NA, 'NULL'))

# Looking into the data
View(head(train))

View(head(test))


###############################
# Target Variable Analysis
###############################

# Initial prepocessing - Remove entries with price 0 as they give littile help predicting the price 
train <- train %>%
          filter(price !=0)

ggplot(data=train,aes(x=price)) + 
  geom_histogram(fill='red') + 
  labs(title='Histogram of Prices')

ggplot(data=train,aes(x=log(price)+1)) + 
  geom_histogram(fill='red') + 
  labs(title='Histogram of Prices')

# Set our price to price log as our metrics are rmsLe and data is skewed

train$price_log <- log(train$price+1)

#########################################
# ----------------- Second Step----------
# Feature Engineering

# Word Counts
# letter Counts
# Splitting the categories into levels
# Filling the missing values
#########################################


# In natural language processing , it is always good to add additional text attributes as new features.
# Like - characterlength, word count, etc..

# Letter count/character count for the attributes - item description, item name
train$name_length <- nchar(train$name)
train$description_length <- nchar(train$item_description)

# Word Count for the attributes - item desciption, item name
train$name_words <- str_count(train$name,'\\S+')
train$description_words <- str_count(train$item_description,'\\S+')

View(head(train))


# Splitting the categories into attribute levels as we can see that category column combines the ...
# ... whole tree of categories in one line . Most items have three categories (1-top category,....)
# ... ( 2-first sub category, 3-second subcategory). Some items have additional forth subcategory.
temp <- str_split(train$category_name, "/",n=3, simplify = TRUE)

temp <- as_tibble(temp)

names(temp) <- paste0('category',1:3)

train <- bind_cols(train,temp)

train$category_name <- NULL

rm(temp)


# Filling the MIssing Values
colSums(is.na(train))

train$brand_name[is.na(train$brand_name)] <- 'undefined'
train$category1[is.na(train$category1)] <- 'undefined'
train$category2[is.na(train$category2)] <- 'undefined'
train$category3[is.na(train$category3)] <- 'undefined'

# Doing this same for test data SET
test$name_length = nchar(test$name)
test$description_length = nchar(test$item_description)
test$name_words = str_count(test$name, '\\S+')
test$description_words = str_count(test$item_description, '\\S+')

temp = as_tibble(str_split(test$category_name, "/", n = 3, simplify = TRUE))
names(temp) = paste0("category", 1:3)
test = bind_cols(test, temp)
test$category_name = NULL
rm(temp)

test$brand_name[is.na(test$brand_name)] = "undefined"
test$category1[is.na(test$category1)] = "undefined" 
test$category2[is.na(test$category2)] = "undefined" 
test$category3[is.na(test$category3)] = "undefined"

gc(TRUE)

#################################
# COmbine TRAIN and TEST for later use , like NLP tasks
##########################################################

nrow_train <- nrow(train)
nrow_test <- nrow(test)
price <- train$price_log
train$price <- NULL
all <- bind_rows(train,test)


####################################
# ------------Third Step------------

# Tokenization
###################################

# Tokenization/Feature Matrix :- What it does, is it creates a huge matrix making a separate variable 
# out of each word, that we have in our dataset. 
# So the new object that we create will have an inormous amount of columns indicating 1 or 0 
# in each of the word's rows showing if the word is present or not present in this item description 
# or name

##################################
# Tokenization of item description

# Initial PreProcessing - Removing numbers, punctuations, symbols, hypens
# Convrting to lowers
# REmoval of Stop words
# Applying stemming
# Careated bag of words/doumented frequrncy matrix
# Preprocess the document frequency matrix
# Apply TF_IDF
####################################

#Initial preprocessing as we will remove all the numbers, punctuation, symbol and hypens
all.items.tokens <- tokens(all$item_description,
                           what="word",
                           remove_numbers = T,
                           remove_punct = T,
                           remove_symbols = T,
                           remove_separators = T)

# For final further preprocessing we will choose a random row and 
# see how the item description is going to change

all.items.tokens[149]

# To Lower
all.items.tokens <- tokens_tolower(all.items.tokens)
all.items.tokens[149]

# Stopwords
all.items.tokens <- tokens_select(all.items.tokens,stopwords(),selection = 'remove')
all.items.tokens[149]

# Stem words
all.items.tokens <- tokens_wordstem(all.items.tokens, language = 'english')
all.items.tokens[149]

# Creating a BAG OF WORDS
all.items.dfm <- dfm(all.items.tokens)

dim(all.items.dfm) # As you can see below, we have created a matrix with 171K columns, each for every word

# We can preprocess such a matri/trim our matrix and include only most common variables 
all.items.dfm.trim <- dfm_trim(all.items.dfm,
                               min_termfreq = 300)

# Applying the TF-IDF
all.items.tfidf <- dfm_tfidf(all.items.dfm.trim)

####################################
# Tokenization of attribute items_name

# same steps like item-description
########################################

all.names.tokens = tokens(all$name, what = "word",
                          remove_numbers = TRUE, remove_punct = TRUE,
                          remove_symbols = TRUE, remove_separators = TRUE)

all.names.tokens = tokens_tolower(all.names.tokens)
all.names.tokens = tokens_select(all.names.tokens, stopwords(),
                                 selection = "remove")
all.names.tokens = tokens_wordstem(all.names.tokens, language = "english")
gc()

# bag of words
all.names.dfm = dfm(all.names.tokens)

# trim
all.names.dfm.trim = dfm_trim(all.names.dfm, min_termfreq = 100) 
gc()

# apply the TF IDF
all.names.tfidf = dfm_tfidf(all.names.dfm.trim)
all.names.tfidf
topfeatures((all.names.tfidf))


#####################################################
# ------------ FIFTH STEP-------------

# Creating a sparse matrix
#####################################################


# In the next steps when I had to create a sparse_matrix, 
# I have run into a NA problem, which had been found in our initial dataframe, 
# so in order to overpass it quickly, I have applied an NA PASS global rule. 
# But remember to opt out of this setting when you are done

previous_na_action = options('na.action')
options(na.action='na.pass')

# Spare matrix creation
sparse_matrix = sparse.model.matrix(~item_condition_id + brand_name + shipping + name_length + 
                                      description_length + name_words + description_words + 
                                      category1 + category2 + category3,
                                    data = all)         # here we have included all of our initial variables, except the text variables (name and description),
                                                        # which we have preprocessed separately

class(all.items.tfidf) = class(sparse_matrix)
class(all.names.tfidf) = class(sparse_matrix)

#COMBINE OUR NEWLY CREATED SPARSE_MATRIX WITH OUR PREPROCESSED TF/IDF TEXT DATAFRAMES
data = cbind(sparse_matrix, all.items.tfidf, all.names.tfidf)

# Turn off the na.action setting
options(na.action=previous_na_action$na.action)

# Splitting back the train and test sets
sparse_train <- data[seq_len(nrow(train)),]
  
sparse_test <- data[seq(from=nrow(train)+1,to=nrow(data)),]

#######################################
#------------- SIXTH STEP---------

# Modelling (XG BOOST)
######################################

# We will train our model using xgboostwith cross-validation

# Set the target variable
Label <- price

# Next we will create DMatrix from test and train sets in order to feed them properly to our xgboost

dtest1 <- xgb.DMatrix(sparse_test)
dtrain1 <- xgb.DMatrix(sparse_train,
                       label=data.matrix(Label))

# Set up xg boost paramters

xgb_params <- list(booster = 'gbtree',
                   colsample_bytree=0.7,
                   subsample=0.7,
                   eta=0.05,
                   objective='reg:linear',
                   max_depth = 5,
                   min_child_weight= 1,
                   eval_metric= "rmse")


##################################
# Modelling Steps
# Set timer to see hoe long will it take 
