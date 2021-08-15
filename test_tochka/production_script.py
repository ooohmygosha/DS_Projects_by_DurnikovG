''' "Это коненчно не продакшен код и возвращает он совсем не веростности, а значения от нуля до единицы.'''

import pandas as pd
import numpy as np
import os
import pickle
from pandas.io.json import json_normalize
import joblib
from sklearn.linear_model import LogisticRegression
import lightgbm as lgbm
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

def PredictionPipepline(pth_model='lgb_final.pkl',
                        pth_feature='selected_features.pkl',
                        pth_account='df_accounts_sample.csv',
                        pth_bank='df_bankruptcies_sample.csv',
                        pth_cases='court_cases_sample/'
                        ):
    # get model and featurelist
    FINAL_MODEL = joblib.load(pth_model)
    feature_columns = joblib.load(pth_feature)

    # prepocesing data account
    df_account = pd.read_csv(pth_account, index_col=0)
    df_account = df_account.fillna(0)
    df_account.okei.replace({383: 1, 384: 1000, 385: 1_000_000}, inplace=True)
    df_account[
        ['long_term_liabilities_fiscal_year',
         'short_term_liabilities_fiscal_year',
         'balance_assets_fiscal_year']
    ] = df_account[
            ['long_term_liabilities_fiscal_year',
             'short_term_liabilities_fiscal_year',
             'balance_assets_fiscal_year']
        ].mul(df_account.okei, axis=0) / 1_000_000
    df_account.drop('okei', axis=1, inplace=True)
    df_account.drop_duplicates(inplace=True)

    list_inn = df_account.inn.drop_duplicates().to_list()
    df_acc_lags = df_account.copy().iloc[0:0]

    for inn in list_inn:
        inn_df = df_account[df_account.inn == inn].copy().sort_values('year')
        inn_df['lt_liabilities_diff'] = inn_df['long_term_liabilities_fiscal_year'] - inn_df['long_term_liabilities_fiscal_year'].shift()
        inn_df['st_liabilities_diff'] = inn_df['short_term_liabilities_fiscal_year'] - inn_df['short_term_liabilities_fiscal_year'].shift()
        inn_df['balance_diff'] = inn_df['balance_assets_fiscal_year'] - inn_df['balance_assets_fiscal_year'].shift()
        df_acc_lags = pd.concat([df_acc_lags, inn_df])

    df_acc_lags.fillna(0, inplace=True)
    df_acc_lags.head(15)

    df_account = df_acc_lags.copy()
    df_account['current_ratio'] = (
                                          df_account.long_term_liabilities_fiscal_year + df_account.short_term_liabilities_fiscal_year
                                  ) / (df_account.balance_assets_fiscal_year + 0.01)

    df_account['current_ratio'][df_account['current_ratio'] > 20] = 20

    # prepocesing data bank
    df_bank = pd.read_csv(pth_bank, index_col=0)
    df_bank.drop_duplicates()
    df_bank = df_bank.drop('bankrupt_id', axis=1)
    df_bank.rename(columns={'bancrupt_year' :'year'}, inplace=True)
    df_bank.year -= 1

    bankrupts = pd.merge(df_bank, df_account, on=['inn', 'year'])
    bankrupts.drop_duplicates('inn' ,inplace=True)
    bankrupts['is_bankrupt'] = 1

    solvents = df_account[~df_account.inn.isin(bankrupts.inn)]
    solvents = solvents.loc[solvents.groupby("inn")["year"].idxmax()]
    solvents['is_bankrupt'] = 0

    new_df = pd.concat([solvents, bankrupts]).reset_index(drop=True)
    new_df.rename(columns={'inn' :'company_inn'}, inplace=True)

    final_observation_year = new_df[['company_inn', 'year']].set_index('company_inn').to_dict()['year']

    path = pth_cases
    files = os.listdir(path)
    list_of_dfs = []

    for n, file in enumerate(files):
        # create dataframe for final data
        final_row = pd.DataFrame()

        # basic values
        company_name = 'Неизвестно'
        company_INN  = 0

        # read file
        with open(os.path.join(path, file), 'rb') as f:
            d = pickle.load(f)

            # first normalize - basic
        basic_df = json_normalize(d)

        company_INN = basic_df.inn.values[0]
        final_row = final_row.append({'company_inn': company_INN}, ignore_index=True)

        # check if no cases
        if len(basic_df.cases_list.values[0]) == 0:
            list_of_dfs.append(final_row)
            continue

            # second normalize - cases_list
        cases_list = pd.concat([pd.DataFrame(json_normalize(x)) for x in basic_df['cases_list']] ,ignore_index=True)
        cases_list.instanceDate = pd.to_datetime(cases_list.instanceDate, errors='coerce')
        cases_list.instanceDate = pd.DatetimeIndex(cases_list['instanceDate']).year

        # check only relevant data
        if company_INN in final_observation_year:
            cases_list = cases_list[cases_list.instanceDate <= final_observation_year[company_INN ] +1]

        # preparing dataframes
        client_cases = pd.DataFrame(columns=cases_list.columns)
        dfndt_cases = pd.DataFrame(columns=cases_list.columns)
        witness_cases = pd.DataFrame(columns=cases_list.columns)

        for i, x in enumerate(cases_list['case_sides']):
            # third normalize for every row - case_sides
            current_case = pd.DataFrame(json_normalize(x))
            if len(current_case) == 0:
                continue

            current_case.INN =  pd.to_numeric(current_case.INN, downcast='integer', errors='coerce')

            # get company name
            if i == 0:
                company_name = current_case[current_case.INN == company_INN].name.values[0]
            # check if don't have inn or name
            if company_INN in current_case.INN.values or company_name in current_case.name.values:

                type_ = current_case[(current_case.INN == company_INN) | (current_case.name == company_name)].type.values[0]

                if type_ == 0:
                    client_cases = pd.concat([client_cases, pd.DataFrame(cases_list.iloc[i]).T])

                elif type_ == 1:
                    dfndt_cases = pd.concat([dfndt_cases, pd.DataFrame(cases_list.iloc[i]).T])

                elif type_ == 2:
                    witness_cases = pd.concat([witness_cases, pd.DataFrame(cases_list.iloc[i]).T])
            else:
                continue

        full_cases = [client_cases, dfndt_cases, witness_cases]
        full_cases_names = ['client', 'defendant', 'witness']
        final_row['company_name'] = company_name

        # counting basic data
        for case_type, name in zip(full_cases ,full_cases_names):
            number_of_cases = len(case_type)
            number_active_cases = len(case_type[case_type.isActive == True])
            if name == 'client':
                win = case_type[(case_type.resultType == 'Выиграно') |
                            (case_type.resultType == 'Иск полностью удовлетворен')].copy()
                lose = case_type[(case_type.resultType == 'Проиграно') | (
                            case_type.resultType == 'В иске отказано полностью')].copy()
                number_win_cases = len(win)
                number_lose_cases = len(lose)
                sum_of_compensation_win = win['sum'].sum() / 1_000_000
                sum_of_compensation_lose = lose['sum'].sum() / 1_000_00

            elif name == 'defendant':
                lose = case_type[(case_type.resultType == 'Выиграно') | (
                            case_type.resultType == 'Иск полностью удовлетворен')].copy()
                win = case_type[(case_type.resultType == 'Проиграно') | (
                            case_type.resultType == 'В иске отказано полностью')].copy()
                number_win_cases = len(win)
                number_lose_cases = len(lose)
                sum_of_compensation_win = win['sum'].sum() / 1_000_000
                sum_of_compensation_lose = lose['sum'].sum() / 1_000_000
            else:
                number_win_cases = 0
                number_lose_cases = 0
                sum_of_compensation_win = 0
                sum_of_compensation_lose = 0

            types = case_type['caseType.code'].unique()
            count_types = {}
            for c in types:
                count_types[c] = len(case_type[case_type['caseType.code'] == c])

            final_row['_'.join(['number_of_cases', name])] = number_of_cases

            if number_of_cases > 0:
                final_row['_'.join(['prec_number_active_cases', name])] = number_active_cases / number_of_cases
                final_row['_'.join(['perc_win_cases', name])] = number_win_cases / number_of_cases
                final_row['_'.join(['per_lose_cases', name])] = number_lose_cases / number_of_cases
                final_row[
                    '_'.join(['sum_of_compensation_win_per_case', name])] = sum_of_compensation_win / number_of_cases
                final_row[
                    '_'.join(['sum_of_compensation_lose_per_case', name])] = sum_of_compensation_lose / number_of_cases

                for k in count_types:
                    final_row['_'.join(['perc_case_type', k, 'on', name])] = count_types[k] / number_of_cases

        list_of_dfs.append(final_row)

    pkl_df = pd.concat(list_of_dfs, axis=0, ignore_index=True)

    pkl_df.company_name.fillna('Неизвестно', inplace=True)
    pkl_df.fillna(0, inplace=True)

    pkl_df.company_inn = pkl_df.company_inn.astype('int64')

    df = pd.merge(pkl_df, new_df, on='company_inn')
    df.company_inn = df.company_inn.astype('str')

    x = df[list(feature_columns)]
    scalar = StandardScaler()
    x = scalar.fit_transform(x)

    return FINAL_MODEL.predict_proba(x)[:, 1]


porbabilities = PredictionPipepline(pth_model='lgb_final.pkl',
                    pth_feature='selected_features.pkl',
                    pth_account='df_accounts_sample.csv',
                    pth_bank='df_bankruptcies_sample.csv',
                    pth_cases='court_cases_sample/'
                    )

print(porbabilities)
