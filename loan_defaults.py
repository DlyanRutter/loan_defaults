import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
from collections import Counter
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import make_scorer, accuracy_score, log_loss, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("/Applications/python_files/DR_Demo_Lending_Club.csv")

def deal_with_NaN(df):
    """
    Fills NaNs for mths_since_last_record, mths_since_last_delinq, revol_util, and
    collections_12_mths_ex_med. Removes rows if there are 5 or less missing values.
    Results in drop of 9 data points.
    Uncomment print missing line to check how many NaNs there were for each feature.
    """
    # 'emp_title' has many NaNs and only serves to complicate classification
    df.drop(['emp_title'], axis=1, inplace=True)

    # Impute NaNs with a placeholder text
    df.Notes.replace(np.nan, '[no_text]', regex=True, inplace=True)
    df.purpose.replace(np.nan, '[no_text]', regex=True, inplace=True)
    df.earliest_cr_line.replace(np.nan, '[no_text]', regex=True, inplace=True)

    # Numbers were input as strings here so convert to numerical, 10 is mode
    df['emp_length'] = df.emp_length.apply(lambda x: int(str(x).split()[0])
                                       if x != 'na' else 10)

    # These had thousands of missing values, but missing values were relevant    
    df.mths_since_last_record.fillna(-1, inplace=True) 
    df.mths_since_last_delinq.fillna(-1, inplace=True) 

    # Fill missing values in numerical rows with row averages
    df.collections_12_mths_ex_med.fillna(df['collections_12_mths_ex_med'].
                                         mean(), inplace=True)
    df.revol_util.fillna(df['revol_util'].mean(), inplace=True)
    df.annual_inc.fillna(df['annual_inc'].mean(), inplace=True)
    df.inq_last_6mths.fillna(df['inq_last_6mths'].mean(), inplace=True)
    df.pub_rec.fillna(df['pub_rec'].mean(), inplace=True)
    df.open_acc.fillna(df['open_acc'].mean(), inplace=True)
    df.total_acc.fillna(df['total_acc'].mean(), inplace=True)
    df.delinq_2yrs.fillna(df['delinq_2yrs'].mean(), inplace=True)
    return df

def simplify_dates(df):    
    """
    Addresses earliest_cr_line feature. First, extracts years from given dates.
    Second, groups years into decades and creates new column 'decade'. Third,
    extracts months from given dates. Fourth, groups months into quarters and
    creates new column 'quarter'. Returns df with new categorical columns.
    """
    # Get year integers from date string
    df['decade'] = df.earliest_cr_line.apply(lambda x: str(x).split('/')[-1])
    df['decade'] = pd.to_numeric(df['decade'], errors='coerce')

    # Separate years into decades 
    decades = ['unknown', 'fifties', 'sixties', 'seventies', 'eighties',\
               'nineties', 'mill']
    bins = (-1, 1949, 1959, 1969, 1979, 1989, 1999, 2010)
    categories = pd.cut(df.decade, bins, labels=decades)
    df.decade = categories

    # Get month integers from date string
    df['quarter'] = df.earliest_cr_line.apply(lambda x: str(x).split('/')[0])
    df['quarter'] = pd.to_numeric(df['quarter'], errors='coerce')

    # Separate months into quarters
    quarters = ['unknown', 'Q1', 'Q2', 'Q3', 'Q4']
    bins = (-1, 0, 3, 6, 9, 12)
    categories = pd.cut(df.quarter, bins, labels=quarters)
    df.quarter = categories
    return df.drop(['earliest_cr_line', 'Id'], axis=1)

def simplify_text(df, plot=False):
    """
    Converts Notes and purpose to numericals for text analysis.
    If plot is set to true, show a plot of most common words
    """
    # Remove stopwords from 'Notes' and 'purpose'
    stopwords_list = stopwords.words('english')
    df['Notes'] = df['Notes'].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in (stopwords_list)]))
    df['purpose'] = df['purpose'].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in (stopwords_list)]))

    # Establish count vectorizers
    cv = CountVectorizer()
    cv2 = CountVectorizer()

    # Transform 'Notes' and 'purpose', then update dataframe
    bag = cv.fit_transform(df['Notes']).toarray()
    bag2 = cv2.fit_transform(df['purpose']).toarray()
    df['Notes'] = bag
    df['purpose'] = bag2

    # Shows most frequently used words
    if plot == True:
        word_freq = dict(zip(cv.get_feature_names(),
                             np.asarray(bag.sum(axis=0)).ravel()))
        word_counter = Counter(word_freq)
        word_counter_df = pd.DataFrame(word_counter.most_common(20),
                                       columns = ['word', 'freq'])        
        fig, ax = plt.subplots(figsize=(12, 10))
        bar_freq_word = sns.barplot(x="word", y="freq",
                                    data=word_counter_df, palette="PuBuGn_d",
                                    ax=ax)
        plt.show()
    return df

def simplify_categories(df, encode=False):
    """
    Converts categorical features
    If encode is set to False, create dummies
    If encode is set to True, encode features using BinaryEncoder
    BinaryEncoder lowers num features; helps with runtime
    """
    # List of categorical features to be encoded 
    cat_features = ['addr_state', 'home_ownership', 'initial_list_status',\
                'pymnt_plan', 'policy_code', 'purpose_cat', 
                'verification_status', 'decade', 'quarter', 'zip_code']   
    if encode == True:
        # Create new df of encoded features
        encoder = ce.BinaryEncoder(cols=cat_features, drop_invariant=False)
        df_binary = encoder.fit_transform(df[cat_features])
    
        # Drop non-encoded categories from df
        df.drop(cat_features, axis=1, inplace=True)
    
        # Merge remaining df with encoded features
        df = pd.concat([df, df_binary], axis=1)
    else:
        df = pd.get_dummies(df, columns=cat_features, prefix=cat_features)
    return df

def outliers(train_df):
    """
    Finds rows in training set with more than two outliers, then drops those
    rows. Outliers are considered data points in the top and bottom 10%.
    """
    # Numeric categories
    numeric = ['annual_inc', 'collections_12_mths_ex_med', 'debt_to_income',
               'delinq_2yrs', 'emp_length', 'inq_last_6mths', 'open_acc',
               'mths_since_last_delinq', 'mths_since_last_major_derog',
               'mths_since_last_record', 'pub_rec', 'revol_bal', 'revol_util',
               'total_acc']
    outlier_indices = []
    for col in numeric:
        Q1 = np.percentile(train_df[col], 10) # 1st ten (10%)
        Q3 = np.percentile(train_df[col], 90) # last ten (90%)
        Range = Q3 - Q1 

        # Determine a list of indices of outliers for feature col
        outlier_step = 1.5 * Range
        "Outliers utliers for the feature '{}':".format(numeric)
        outlier_list_col = train_df[(train_df[col] < Q1 - outlier_step) |\
                              (train_df[col] > Q3 + outlier_step )].index
        
        # Append outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
                
    # Select, then drop indices containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = [k for k, v in outlier_indices.items() if v > 2]
    train_df = train_df.drop(train_df.index[multiple_outliers]).reset_index(
        drop=True)
    return train_df

def preprocess_df(df):
    """
    Applies NaN, dates, categories, and text preprocessing functions. Splits
    off a holdout subset. Removes outliers from train set. Splits sets into
    features and labels. Splits non-holdout set into train and test sets.
    Returns updated df, a holdout set, holdout set features, holdout set
    labels, a non-holdout set, non-holdout features, non-holdout labels,
    X_train, X_test, y_train, and y_test from train-test-split.
    """
    # Apply preprocessing functions
    df = deal_with_NaN(df)
    df = simplify_dates(df)
    df = simplify_categories(df)
    df = simplify_text(df)
    
    # Split off a holdout set 
    holdout, train = df.loc[8000:,:], df.loc[:8000,:]

    # Remove outliers from non-holdout set
    train = outliers(train)

    # Split non-holdout set and holdout set into features and labels
    train_labels = train['is_bad']
    holdout_labels = holdout['is_bad']
    train_features = train.drop(['is_bad'], axis=1)
    holdout_features = holdout.drop(['is_bad'], axis=1)

    # Split non-holdout set into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        train_features, train_labels, test_size=.2, random_state=23)
    
    return df, holdout, holdout_features, holdout_labels, train,\
           train_features, train_labels, X_train, X_test, y_train, y_test

def booster(grid_search=False):
    """
    Performs gradient boosting on the data frame
    If grid_search is set to True, performs a grid search
    Note: It takes a long time, so I have input the best parameters from the
    grid search I ran. 
    """
    def grid_search():
        # Parameters to check        
        parameters = {'n_estimators': [100, 150, 200], 
                      'max_features': ['log2', 'sqrt','auto'], 
                      'max_depth': [2, 3, 5, 7, 10], 
                      'min_samples_split': [2, 3, 5],
                      'min_samples_leaf': [1, 5, 8]}
                     
        # Create classifier and run grid
        clf = GradientBoostingClassifier()
        grid = GridSearchCV(clf,
                            parameters,
                            cv=5,
                            scoring=acc_scorer)
        grid = grid.fit(X_train, y_train)
        
        print ("Best Grid Score: {}".format(grid.best_score_))
        print ("Best Grid Params: {}".format(grid.best_params_))
        print ("Best Grid Estimator: {}".format(grid.best_estimator_))
        return grid
    
    # Set classifier as grid if grid_search=True, else run preconstructed clf
    if grid_search == True:        
        clf = grid_search()
    else:
        clf = GradientBoostingClassifier(n_estimators=150,
                                         max_features='auto',
                                         max_depth=3,
                                         min_samples_split=2,
                                         min_samples_leaf=1)
    # Fit classifier and make predictions
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Print performance metrics
    print ("Boost Accuracy Score: {}".format(accuracy_score(y_test,
                                                            predictions)))
    print ("Boost Log_loss: {}".format(log_loss(y_test, predictions)))
    print ("Boost F1 score: {}".format(f1_score(y_test, predictions)))
    return clf

def log_reg(grid_search=False):
    """
    Runs a Logistic Regression and returns a classifier.
    First runs a grid search if grid_search is set to True
    Note: I have already run a grid search and input the best parameters
    """
    def grid_search():
        # Parameters to check    
        alpha = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,13,14,
                 15,16,16.5,17,17.5,18]
        penalty = ['l1', 'l2']
        max_iter = [100, 200, 400]
        params = {'max_iter':max_iter, 'C':alpha, 'penalty':penalty}

        # Make scorer and set up grid
        acc_scorer = make_scorer(accuracy_score)    
        logreg = LogisticRegression(solver='liblinear')
        grid = GridSearchCV(logreg,
                            param_grid=params,
                            scoring=acc_scorer,
                            cv=5)
        grid.fit(X_train, y_train)

        # Print best grid paramters
        print ("Best Grid Score: {}".format(grid.best_score_))
        print ("Best Grid Params: {}".format(grid.best_params_))
        print ("Best Grid Estimator: {}".format(grid.best_estimator_))
        return grid

    # Run grid search if True or use preconstructed classifier
    if grid_search == True:        
        clf = grid_search()
    else:
        clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.8,
                                 max_iter=100)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Print performance metrics
    print ("Logistic Regression Accuracy Score: {}".format(
        accuracy_score(y_test, predictions)))
    print ("Logistic Regression Log loss: {}".format(
        log_loss(y_test, predictions)))
    print ("Logistic Regression F1 score: {}".format(
        f1_score(y_test, predictions)))
    return clf

def kfold_and_test_on_holdout(clf):
    """
    Computes and prints out-of-sample LogLoss, Accuracy, and F1 scores on
    StratifiedKfold validation and holdout set.  
    """
    # Set up features and labels
    X = train_df.drop(['is_bad'], axis=1)
    y = train_df['is_bad']
    kf = StratifiedKFold(n_splits=5)

    # Iterate through validation folds
    acc_outcomes = []
    log_loss_outcomes = []
    f1_outcomes = []
    fold = 0
    for train_index, test_index in kf.split(X, y):
        fold += 1
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        # Fit classifier and make predictions
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        # Get scores of predictions 
        f1 = f1_score(y_test, predictions)
        loss = log_loss(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)

        log_loss_outcomes.append(loss)
        f1_outcomes.append(f1)
        acc_outcomes.append(accuracy)

        # Print scores for each fold
        print ("fold {0} Log loss: {1}".format(fold, loss)) 
        print ("Fold {0} F1 score: {1}".format(fold, f1))
        print ("Fold {0} accuracy: {1}".format(fold, accuracy))

    # Get mean accuracy, log loss, and F1 from folds    
    mean_acc = np.mean(acc_outcomes)
    mean_log_loss = np.mean(log_loss_outcomes)
    mean_f1 = np.mean(f1_outcomes)
    
    # Print means
    print("Mean Validation Accuracy: {0}".format(mean_acc))
    print("Mean Validation Log Loss: {0}".format(mean_log_loss))
    print("Mean Validation f1: {0}".format(mean_f1))

    # Get prediction scores on holdout set 
    holdout_predictions = clf.predict(holdout_features)
    holdout_f1 = f1_score(holdout_labels, holdout_predictions)
    holdout_loss = log_loss(holdout_labels, holdout_predictions)
    holdout_acc = accuracy_score(holdout_labels, holdout_predictions)

    # Print prediction scores on holdout
    print("Holdout Accuracy: {0}".format(holdout_acc))
    print("Holdout Log Loss: {0}".format(holdout_loss))
    print("Holdout f1: {0}".format(holdout_f1))

df, holdout_df, holdout_features, holdout_labels, train_df, train_features,\
           train_labels, X_train, X_test, y_train, y_test = preprocess_df(df)

######### Uncomment, then run these to get models ###########

#logreg = log_reg(grid_search=True)
boost = booster(grid_search=True)
#kfold_and_test_on_holdout(logreg)
kfold_and_test_on_holdout(boost)
