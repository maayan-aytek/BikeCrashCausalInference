import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from constants import T_BINARY, Y
from figures import plot_evaluation_metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import cohen_kappa_score


def estimate_naive_ate(df):
    Y1 = np.array(df[df[T_BINARY] == 1][Y]).mean()
    Y0 = np.array(df[df[T_BINARY] == 0][Y]).mean()
    delta_all = Y1 - Y0
    return delta_all

def estimate_ate_ipw(T, Y, e, epsilon=1e-6):
    e_copy = e.copy() 
    e_copy[e_copy == 0] += epsilon
    e_copy[e_copy == 1] -= epsilon
    n = len(T)
    left_argument = sum(T * Y / e_copy)
    right_argument = sum(((1 - T) * Y) / (1 - e_copy))
    return (left_argument - right_argument) / n


def calculate_propensity_score(df, model, features, scale=False):
  df_copy = df.copy()
  X_df = df_copy[features]
  t_true = df_copy[T_BINARY]

  if scale:
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_df)
  else:
    X_scaled = X_df

  model.fit(X_scaled, t_true)
  propensity_scores = model.predict_proba(X_scaled)[:, 1]
  df_copy['propensity_score'] = propensity_scores
  return df_copy


def matching(label, propensity, calipher=0.05, replace=True):
    """
    Performs nearest-neighbour matching for a sample of test and control
    observations, based on the propensity scores for each observation.

    :param label: Series that contains the label for each observation.
    :param propensity: Series that contains the propensity score for each observation.
    :param calipher: Bound on distance between observations in terms of propensity score.
    :param replace: Boolean that indicates whether sampling is with (True) or without replacement (False).
    :return: matches
    """
    treated = propensity[label == 1]
    control = propensity[label == 0]

    # Randomly permute in case of sampling without replacement to remove any bias arising from the
    # ordering of the data set
    matching_order = np.random.permutation(label[label == 1].index)
    matches = {}

    for obs in matching_order:
        # Compute the distance between the treatment observation and all candidate controls in terms of
        # propensity score
        distance = abs(treated[obs] - control)

        # Take the closest match
        if distance.min() <= calipher or not calipher:
            matches[obs] = [distance.argmin()]
            # Remove the matched control from the set of candidate controls in case of sampling without replacement
            if not replace:
                control = control.drop(matches[obs])

    return matches


def matching_to_dataframe(match, covariates, remove_duplicates=False):
    """
    Converts a list of matches obtained from matching() to a DataFrame.
    Duplicate rows are controls that where matched multiple times.

    :param match: Dictionary with a list of matched control observations.
    :param covariates: DataFrame that contains the covariates for the observations.
    :param remove_duplicates: Boolean that indicates whether or not to remove duplicate rows from the result.
    If matching with replacement was used you should set this to False
    :return: matching as data frame
    """
    treated = list(match.keys())
    control = [ctrl for matched_list in match.values() for ctrl in matched_list]
    result = pd.concat([covariates.loc[treated], covariates.loc[control]])
    if remove_duplicates:
        return result.groupby(result.index).first()
    else:
        return result
    

def calc_matching_ate(propensity_df):
    propensity_df = propensity_df.reset_index(drop=True)
    matches = matching(label=propensity_df[T_BINARY],
                      propensity=propensity_df['propensity_score'],
                      calipher=0.1,
                      replace=True)

    matches_df = matching_to_dataframe(match=matches,
                                        covariates=propensity_df,
                                        remove_duplicates=False)
    Y1 = np.array(matches_df[matches_df[T_BINARY] == 1][Y]).mean()
    Y0 = np.array(matches_df[matches_df[T_BINARY] == 0][Y]).mean()
    delta_all = Y1 - Y0
    return delta_all


def calculate_s_learner_score(X, T, Y, model, scale=False):
    if scale:
      scaler = MinMaxScaler()
      X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
      X_scaled = X
    X_with_T = pd.concat([X_scaled, T], axis=1)
    Y_str = Y.astype(str)
    model.fit(X_with_T, Y_str)

    X_t1 = X_with_T.copy()
    X_t1[T_BINARY] = 1

    X_t0 = X_with_T.copy()
    X_t0[T_BINARY] = 0

    t1_y_preds = np.array(model.predict(X_t1).astype(float))
    t0_y_preds = np.array(model.predict(X_t0).astype(float))

    return (t1_y_preds - t0_y_preds).sum() / len(t1_y_preds)


def calculate_t_learner_score(X, T, Y, model0, model1, scale=False):
    if scale:
      scaler = MinMaxScaler()
      X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
      X_scaled = X

    Y_str = Y.astype(str)
    X_t1 = X_scaled[T == 1]
    Y_t1 = Y_str[T == 1]

    X_t0 = X_scaled[T == 0]
    Y_t0 = Y_str[T == 0]

    model1.fit(X_t1, Y_t1)
    model0.fit(X_t0, Y_t0)

    t1_y_preds = np.array(model1.predict(X_scaled).astype(float))
    t0_y_preds = np.array(model0.predict(X_scaled).astype(float))

    return (t1_y_preds - t0_y_preds).sum() / len(t1_y_preds)


def DR_ATE(X, Y, T, model0, model1, propensity_score, scale=False):
    if scale:
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        X_scaled = X

    Y_str = Y.astype(str)
    X_t1 = X_scaled[T == 1]
    Y_t1 = Y_str[T == 1]

    X_t0 = X_scaled[T == 0]
    Y_t0 = Y_str[T == 0]

    model1.fit(X_t1, Y_t1)
    model0.fit(X_t0, Y_t0)

    t1_y_preds = np.array(model1.predict(X_scaled).astype(float))
    t0_y_preds = np.array(model0.predict(X_scaled).astype(float))

    g1 = t1_y_preds + (T / propensity_score)*(Y - t1_y_preds)
    g0 = t0_y_preds + ((1 - T) / (1 - propensity_score))*(Y - t0_y_preds)
    return (g1 - g0).mean()


def eval_learner_models(X, T, Y, model, scale=False):
    if scale:
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        X_scaled = X

    X_with_T = pd.concat([X_scaled, T.reset_index(drop=True)], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X_with_T, Y, test_size=0.2, random_state=42)

    Y_train_str = Y_train.astype(str)
    Y_test_str = Y_test.astype(str)
    model.fit(X_train, Y_train_str)
    Y_pred = model.predict(X_test)

    return {'accuracy': accuracy_score(Y_test_str, Y_pred), 'f1': f1_score(Y_test_str, Y_pred, average='weighted')}


def estimate_learners_methods(df, features, model_name, learners_model_class, scale=False):
    df = df.reset_index(drop=True)
    s_learner = calculate_s_learner_score(df[features], df[T_BINARY], df[Y], model=learners_model_class(), scale=scale)
    t_learner = calculate_t_learner_score(df[features], df[T_BINARY], df[Y], model0=learners_model_class(), model1=learners_model_class(), scale=scale)
    estimation_df = pd.DataFrame({"learners_model": [model_name],
                                    "S Learner": [s_learner],
                                    "T learner": [t_learner]
                                    })
    return estimation_df


def estimate_propensity_methods(df, propensity_model_name, scale=False):
    ipw = estimate_ate_ipw(df[T_BINARY], df[Y], df['propensity_score'])
    matching = calc_matching_ate(df)
    estimation_df = pd.DataFrame({"propensity_model": [propensity_model_name],
                                   "IPW": [ipw],
                                    "Matching": [matching]
                                    })
    return estimation_df


def calc_bootstap_CI(df, method, features=None, model_class=None):
    deltas = []
    alpha = 0.95
    for _ in range(400):
        sample_df = df.sample(n=len(df), replace=True) 
        if method == "naive":
           result = estimate_naive_ate(sample_df)
        elif method == "IPW":
            result = estimate_ate_ipw(sample_df[T_BINARY], sample_df[Y], sample_df['propensity_score'])
        elif method == "s_learner":
           result = calculate_s_learner_score(sample_df[features], sample_df[T_BINARY], sample_df[Y], model_class(), scale=False)
        elif method == "t_learner":
           result = calculate_t_learner_score(sample_df[features], sample_df[T_BINARY], sample_df[Y], model_class(), model_class(), scale=False)
        elif method == "matching":
           result = calc_matching_ate(sample_df)
        else:
           raise ValueError("Unknown method:", method)
        deltas.append(result)

    lower_bound = np.percentile(deltas, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(deltas, (1 + alpha) / 2 * 100)
    return [lower_bound, upper_bound]


def eval_target_model(df, features, model_class, scale=False):
    X = df[features]
    y = df[Y] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if scale:
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)


    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [5, 10, 20, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    model = model_class(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred, weights="quadratic")

    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Quadratic Weighted Kappa: {kappa}")
    print(f"Accuracy: {accuracy}")
    print(f"F1: {f1}")

    return f1, grid_search.best_params_


def eval_propensity_models(df, models_dict, features, scale=False):
    df_copy = df.copy()
    X_df = df_copy[features]
    t_true = df_copy[T_BINARY]

    if scale:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_df)
    else:
        X_scaled = X_df
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, t_true, test_size=0.2, random_state=42)
    propensities_dict = {}
    for model_name, model in models_dict.items():
        df_copy = X_test.copy()
        model.fit(X_train.to_numpy(), Y_train)
        propensity_scores = model.predict_proba(X_test)[:, 1]
        df_copy['propensity_score'] = propensity_scores
        propensities_dict[model_name] = df_copy
    plot_evaluation_metrics(propensities_dict, Y_test)


