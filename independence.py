import pandas as pd
from scipy.stats import chi2_contingency
from constants import CATEGORICAL_COLUMNS, T_CATEGORY, Y


def column_distribution(df, column, seperator_col=T_CATEGORY):
    counts = df[[seperator_col, column]].groupby([seperator_col, column]).size()
    percentages = counts.groupby(level=0).apply(lambda x: round(100 * x / float(x.sum()), 2)).copy()
    counts.rename('No.', inplace=True)
    percentages.rename("%", inplace=True)
    dist = pd.concat([counts, percentages], axis=1).reset_index()
    dist = pd.melt(dist, id_vars=[seperator_col, column], value_vars=['No.', '%'])
    dist = pd.pivot_table(dist, index=column, columns=[seperator_col, 'variable'])
    return dist


def chi_square_test(df, column, seperator_col=T_CATEGORY):
    contingency_table = pd.crosstab(df[column], df[seperator_col])
    chi2, p_value = chi2_contingency(contingency_table)[:2]
    res = pd.DataFrame(data=[[round(chi2, 2), p_value]], columns=['Chi Square', 'p value'], index=[column])
    res['p value'] = res['p value'].apply(lambda x: round(x, 3) if x >= 0.05 else '<0.05')
    return res


def print_independence_table(df, columns=CATEGORICAL_COLUMNS, seperator_col=T_CATEGORY):
    df_list = []
    chi_list = []
    for column in columns:
        df_list.append(column_distribution(df, column, seperator_col=seperator_col))
        chi_list.append(chi_square_test(df, column, seperator_col=seperator_col))
    dist = pd.concat(df_list, keys=CATEGORICAL_COLUMNS)
    chi = pd.concat(chi_list, keys=CATEGORICAL_COLUMNS)
    return dist, chi


# def column_balance_prop(df, column):
#     percentages = df.groupby(['T', column]).agg({'est': 'sum'}).copy()
#     balance = percentages.reset_index()
#     balance = pd.pivot_table(balance, index=column, columns=['T'], values='est')
#     balance[0] = balance[0] / sum(balance[0])
#     balance[1] = balance[1] / sum(balance[1])
#     return balance


# def chi_square_test_prop(pivoted, column):
#     chi2, p_value = chi2_contingency(pivoted)[:2]
#     res = pd.DataFrame(data=[[round(chi2, 2), p_value]], columns=['Chi Square', 'p value'], index=[column])
#     res['p value'] = res['p value'].apply(lambda x: round(x, 3) if x >= 0.05 else '<0.05')
#     return res


# def print_balance_table_est(df, est, columns=CATEGORICAL_COLUMNS):
#     df_list = []
#     chi_list = []
#     df['est'] = est
#     for column in columns:
#         pivoted = column_balance_prop(df, column)
#         df_list.append(pivoted)
#         chi_list.append(chi_square_test_prop(pivoted.T, column))
#     balance = pd.concat(df_list, keys=CATEGORICAL_COLUMNS)
#     chi = pd.concat(chi_list, keys=CATEGORICAL_COLUMNS)
#     return balance, chi
