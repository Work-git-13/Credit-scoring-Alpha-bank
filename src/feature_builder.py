import numpy as np
import pandas as pd
code_columns = ['pre_since_opened', 'pre_since_confirmed', 'pre_pterm', 'pre_fterm', 'pre_till_pclose', 'pre_till_fclose', 'pre_loans_credit_limit', 'pre_loans_next_pay_summ', 'pre_loans_outstanding', 'pre_loans_total_overdue', 'pre_loans_max_overdue_sum', 'pre_loans_credit_cost_rate', 'pre_loans5', 'pre_loans530', 'pre_loans3060', 'pre_loans6090', 'pre_loans90', 'pre_util', 'pre_over2limit', 'pre_maxover2limit', 'enc_paym_0', 'enc_paym_1', 'enc_paym_2', 'enc_paym_3', 'enc_paym_4', 'enc_paym_5', 'enc_paym_6', 'enc_paym_7', 'enc_paym_8', 'enc_paym_9', 'enc_paym_10', 'enc_paym_11', 'enc_paym_12', 'enc_paym_13', 'enc_paym_14', 'enc_paym_15', 'enc_paym_16', 'enc_paym_17', 'enc_paym_18', 'enc_paym_19', 'enc_paym_20', 'enc_paym_21', 'enc_paym_22', 'enc_paym_23', 'enc_paym_24', 'enc_loans_account_holder_type', 'enc_loans_credit_status', 'enc_loans_credit_type', 'enc_loans_account_cur']
flag_columns = ['is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90', 'is_zero_util', 'is_zero_over2limit', 'is_zero_maxover2limit', 'pclose_flag', 'fclose_flag']
payment_columns = ['enc_paym_0', 'enc_paym_1', 'enc_paym_2', 'enc_paym_3', 'enc_paym_4', 'enc_paym_5', 'enc_paym_6', 'enc_paym_7', 'enc_paym_8', 'enc_paym_9', 'enc_paym_10', 'enc_paym_11', 'enc_paym_12', 'enc_paym_13', 'enc_paym_14', 'enc_paym_15', 'enc_paym_16', 'enc_paym_17', 'enc_paym_18', 'enc_paym_19', 'enc_paym_20', 'enc_paym_21', 'enc_paym_22', 'enc_paym_23', 'enc_paym_24']
def make_base_features(raw, application_ids, vocabulary):
    groups = raw.groupby('id', sort=False)
    index = pd.Index(application_ids, name='id')
    features = pd.DataFrame(index=index)
    product_counts = groups.size()
    features['n_credit_products'] = product_counts.reindex(index, fill_value=0)
    features['has_history'] = (features.n_credit_products > 0).astype('int8')

    # Для кодов допустимо число различных категорий, но не среднее значение кода.
    diversity = groups[code_columns].nunique().add_suffix('__nunique')
    missing_share = raw[code_columns].isna().groupby(raw.id).mean().add_suffix('__missing_share')
    # Среднее бинарного флага = доля единиц; отдельно учитываем пропуски флагов.
    flag_share = groups[flag_columns].mean().add_suffix('__share')
    flag_missing = raw[flag_columns].isna().groupby(raw.id).mean().add_suffix('__missing_share')
    features = features.join([diversity, missing_share, flag_share, flag_missing])

    frequency_blocks = []
    for col, categories in vocabulary.items():
        counts = raw.groupby(['id', col]).size().unstack(col, fill_value=0)
        counts = counts.reindex(columns=categories, fill_value=0)
        shares = counts.div(product_counts, axis=0).fillna(0)
        shares.columns = [f'{col}__code_{value}_share' for value in categories]
        frequency_blocks.append(shares)
        unknown = raw[col].notna() & ~raw[col].isin(categories)
        unknown_share = unknown.groupby(raw.id).mean().rename(f'{col}__unknown_share')
        frequency_blocks.append(unknown_share.to_frame())
    features = features.join(frequency_blocks)

    # rn используется для выбора последнего продукта, но не попадает в X.
    latest = raw.sort_values(['id', 'rn']).drop_duplicates('id', keep='last').set_index('id')
    for col in code_columns + flag_columns:
        values = latest[col].reindex(index)
        features[f'latest__{col}'] = values.map(
            lambda value: 'missing' if pd.isna(value) else f'code_{value:g}'
        )

    # Сравниваем статусы на равенство. Номер категории не задаёт тяжесть события.
    payments = raw[payment_columns]
    left = payments.iloc[:, :-1].to_numpy()
    right = payments.iloc[:, 1:].to_numpy()
    observed = pd.notna(left) & pd.notna(right)
    comparisons = observed.sum(axis=1)
    changes = ((left != right) & observed).sum(axis=1)
    change_share = np.divide(changes, comparisons,
                             out=np.full(len(raw), np.nan), where=comparisons > 0)
    changes_by_id = pd.Series(change_share, index=raw.index).groupby(raw.id)
    features['payment_change_share_mean'] = changes_by_id.mean()
    features['payment_change_share_max'] = changes_by_id.max()
    features['payment_pair_observed_share'] = pd.Series(
        observed.mean(axis=1), index=raw.index).groupby(raw.id).mean()

    numeric_columns = features.select_dtypes(include='number').columns
    # Ноль допустим для отсутствующих частот; NaN в среднем наблюдавшегося флага
    # (все значения пропущены) сохраняем для будущего train-only imputer.
    no_history = features.has_history.eq(0)
    features.loc[no_history, numeric_columns] = 0
    share_columns = [col for col in features if col.endswith('_share') and '__code_' in col]
    features[share_columns] = features[share_columns].fillna(0)
    features[numeric_columns] = features[numeric_columns].astype('float32')
    return features
def make_application_features(raw, application_ids, vocabulary):
    features = make_base_features(raw, application_ids, vocabulary)
    count_features = {}
    for flag in flag_columns:
        for value in [0, 1]:
            count = raw[flag].eq(value).groupby(raw.id).sum()
            count_features[f'extra__{flag}__count_{value}'] = count.reindex(features.index, fill_value=0)
    features = features.join(pd.DataFrame(count_features, index=features.index))
    ordered = raw.sort_values(['id', 'rn'])
    diversity_fields = [col for col in ['enc_loans_credit_type', 'enc_loans_credit_status',
        'pre_loans_outstanding', 'pre_util', 'pre_loans3060', 'pre_loans6090', 'pre_loans90'] if col in raw]
    for k in [3, 5]:
        recent = ordered.groupby('id', sort=False).tail(k)
        recent_groups = recent.groupby('id', sort=False)
        shares = recent_groups[flag_columns].mean().add_prefix(f'extra__last_{k}__').add_suffix('__share')
        diversity = recent_groups[diversity_fields].nunique().add_prefix(f'extra__last_{k}__').add_suffix('__nunique')
        features = features.join([shares, diversity])
    extras = [col for col in features if col.startswith('extra__')]
    features.loc[features.has_history.eq(0), extras] = 0
    features[extras] = features[extras].astype('float32')
    return features

