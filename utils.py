import numpy as np
import matplotlib.pyplot as plt
from lm_polygraph.utils.manager import UEManager
import sklearn
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import logging
from lm_polygraph.ue_metrics.pred_rej_area import PredictionRejectionArea
from lm_polygraph.ue_metrics.ue_metric import (
    get_random_scores,
    normalize_metric,
)
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

ue_metric = PredictionRejectionArea(max_rejection=0.5)

log = logging.getLogger("lm_polygraph")
log.setLevel(logging.ERROR)


def build_rejection_curve(ues, metrics):
    order = np.argsort(ues)
    sorted_metrics = metrics[order]
    sum_rej_metrics = np.cumsum(sorted_metrics)
    num_points_left = np.arange(1, len(sum_rej_metrics) + 1)

    rej_metrics = sum_rej_metrics / num_points_left
    rej_rates = 1 - num_points_left / len(sum_rej_metrics)

    return rej_metrics[::-1], rej_rates[::-1]



def score_ues(ues, metric):
    ues_nans = np.isnan(ues)
    metric_nans = np.isnan(metric)
    total_nans = ues_nans | metric_nans

    filtered_ues = ues[~total_nans]
    filtered_metric = metric[~total_nans]

    oracle_score = ue_metric(-filtered_metric, filtered_metric)
    random_score = get_random_scores(ue_metric, filtered_metric)

    raw_ue_metric_val = ue_metric(filtered_ues, filtered_metric)

    raw_score = normalize_metric(raw_ue_metric_val, oracle_score, random_score)

    return raw_score


def load_managers(dataset, model='llama'):
    manager = UEManager.load(f'./processed_mans/{model}_{dataset}_test_full_enriched.man')
    train_manager = UEManager.load(f'./processed_mans/{model}_{dataset}_train_full_enriched.man')
    return manager, train_manager


def extract_and_prepare_data(dataset, methods_dict, all_metrics, model='llama'):
    manager, train_manager = load_managers(dataset, model)

    full_ue_methods = list(methods_dict.keys())
    ue_methods = list(methods_dict.values())

    sequences = manager.stats['greedy_tokens']

    train_sequences = train_manager.stats['greedy_tokens']

    train_gen_lengths = np.array([len(seq) for seq in train_sequences])
    gen_lengths = np.array([len(seq) for seq in sequences])

    # Get train and test values for metrics and UE, remove union of nans
    test_nans = []
    train_nans = []

    train_metric_values = {}
    test_metric_values = {}
    for metric in all_metrics:
        values = np.array(manager.gen_metrics[('sequence', metric)])
        test_metric_values[metric] = np.array(values)
        test_nans.extend(np.argwhere(np.isnan(values)).flatten())

        train_values = np.array(train_manager.gen_metrics[('sequence', metric)])
        train_metric_values[metric] = np.array(train_values)
        train_nans.extend(np.argwhere(np.isnan(train_values)).flatten())

    train_ue_values = {}
    test_ue_values = {}
    for i, method in enumerate(full_ue_methods):
        train_values = np.array(train_manager.estimations[('sequence', method)])
        train_ue_values[ue_methods[i]] = train_values
        train_nans.extend(np.argwhere(np.isnan(train_values)).flatten())

        values = np.array(manager.estimations[('sequence', method)])
        test_ue_values[ue_methods[i]] = values
        test_nans.extend(np.argwhere(np.isnan(values)).flatten())

    train_nans = np.unique(train_nans).astype(int)
    test_nans = np.unique(test_nans).astype(int)

    # Remove nans
    for metric in all_metrics:
        test_metric_values[metric] = np.delete(test_metric_values[metric], test_nans)
        train_metric_values[metric] = np.delete(train_metric_values[metric], train_nans)

    for method in ue_methods:
        test_ue_values[method] = np.delete(test_ue_values[method], test_nans)
        train_ue_values[method] = np.delete(train_ue_values[method], train_nans)

    train_gen_lengths = np.delete(train_gen_lengths, train_nans)
    gen_lengths = np.delete(gen_lengths, test_nans)

    return train_ue_values, test_ue_values, train_metric_values, test_metric_values, train_gen_lengths, gen_lengths



def detrend_ue(datasets, model, all_metrics, ue_methods, methods_dict, return_scores =False):
    ue_scores = defaultdict(list)
    ue_coefs = defaultdict(list)
    ave_test_metric_values = {}

    if len(all_metrics) == 1 and len(datasets) > 1:
        all_metrics = all_metrics * len(datasets)
    elif len(all_metrics) != len(datasets):
        raise ValueError('Number of metrics and datasets must be the same')

    for metric, dataset in zip(all_metrics, datasets):
        train_ue_values, \
        test_ue_values, \
        train_metric_values, \
        test_metric_values, \
        train_gen_lengths, \
        gen_lengths = extract_and_prepare_data(dataset, methods_dict, [metric], model=model)

        ave_test_metric_values[dataset] = np.mean(test_metric_values[metric])

        upper_q = np.quantile(train_gen_lengths, 0.95)
        lower_q = np.quantile(train_gen_lengths, 0.05)
        below_q_ids = (train_gen_lengths < upper_q) & (train_gen_lengths > lower_q)
        print(f'{model} {dataset} Below q ids: {below_q_ids.sum()}')
        train_gen_lengths = train_gen_lengths[below_q_ids]

        for method in ue_methods:
            train_ue_values[method] = train_ue_values[method][below_q_ids]

        train_normalized_ue_values = {}
        test_normalized_ue_values = {}

        ue_residuals = {}

        for method in ue_methods:
            gen_length_scaler = MinMaxScaler()
            train_gen_lengths_normalized = gen_length_scaler.fit_transform(train_gen_lengths[:, np.newaxis]).squeeze()
            test_gen_lengths_normalized = gen_length_scaler.transform(gen_lengths[:, np.newaxis]).squeeze()

            scaler = MinMaxScaler()
            train_normalized_ue_values[method] = scaler.fit_transform(train_ue_values[method][:, np.newaxis]).squeeze()
            test_normalized_ue_values[method] = scaler.transform(test_ue_values[method][:, np.newaxis]).squeeze()

            linreg = sklearn.linear_model.LinearRegression()
            linreg.fit(train_gen_lengths_normalized[:, np.newaxis], train_normalized_ue_values[method])
            ue_coefs[method].append(linreg.coef_[0])

            ue_residuals[method] = test_normalized_ue_values[method] - linreg.predict(test_gen_lengths_normalized[:, np.newaxis])
            scaler = MinMaxScaler()
            norm_residuals = scaler.fit_transform(ue_residuals[method][:, np.newaxis]).squeeze()
            linreg = sklearn.linear_model.LinearRegression()
            linreg.fit(test_gen_lengths_normalized[:, np.newaxis], norm_residuals)
            ue_coefs[method].append(linreg.coef_[0])

            met_vals = test_metric_values[metric]
            raw_score = score_ues(test_ue_values[method], met_vals)
            raw_norm_score = score_ues(test_normalized_ue_values[method], met_vals)
            detrended_score = score_ues(ue_residuals[method], met_vals)

            ue_scores[f'{method}_raw'].append(raw_score)
            ue_scores[f'{method}_detr'].append(detrended_score)
    if return_scores:
        return test_normalized_ue_values, ue_residuals, test_gen_lengths_normalized
    return ue_scores, ue_coefs, ave_test_metric_values





def detrend_ue_w_quality(datasets, model, all_metrics, ue_methods, methods_dict, quality_fit_sample_size=None, random_state= 42, max_bins= 10 ):
    ue_scores = defaultdict(list)
    ue_coefs = defaultdict(list)
    ave_test_metric_values = {}
    rng = np.random.default_rng(random_state)

    if len(all_metrics) == 1 and len(datasets) > 1:
        all_metrics = all_metrics * len(datasets)
    elif len(all_metrics) != len(datasets):
        raise ValueError('Number of metrics and datasets must be the same')

    for metric, dataset in zip(all_metrics, datasets):
        train_ue_values, \
        test_ue_values, \
        train_metric_values, \
        test_metric_values, \
        train_gen_lengths, \
        gen_lengths = extract_and_prepare_data(dataset, methods_dict, [metric], model=model)

        ave_test_metric_values[dataset] = np.mean(test_metric_values[metric])

        upper_q = np.quantile(train_gen_lengths, 0.95)
        lower_q = np.quantile(train_gen_lengths, 0.05)
        below_q_ids = (train_gen_lengths < upper_q) & (train_gen_lengths > lower_q)
        print(f'{model} {dataset} Below q ids: {below_q_ids.sum()}')
        train_gen_lengths = train_gen_lengths[below_q_ids]

        for method in ue_methods:
            train_ue_values[method] = train_ue_values[method][below_q_ids]

        train_normalized_ue_values = {}
        test_normalized_ue_values = {}

        train_normalized_metric_values = {}
        test_normalized_metric_values = {}


        for method in ue_methods:
            gen_length_scaler = MinMaxScaler()
            train_gen_lengths_normalized = gen_length_scaler.fit_transform(train_gen_lengths[:, np.newaxis]).squeeze()
            test_gen_lengths_normalized = gen_length_scaler.transform(gen_lengths[:, np.newaxis]).squeeze()

            scaler = MinMaxScaler()
            train_normalized_ue_values[method] = scaler.fit_transform(train_ue_values[method][:, np.newaxis]).squeeze()
            test_normalized_ue_values[method] = scaler.transform(test_ue_values[method][:, np.newaxis]).squeeze()

            scaler = MinMaxScaler()
            train_normalized_metric_values[method] = scaler.fit_transform(train_metric_values[metric][:, np.newaxis]).squeeze()
            test_normalized_metric_values[method] = scaler.transform(test_metric_values[metric][:, np.newaxis]).squeeze()

           
            if quality_fit_sample_size is not None and quality_fit_sample_size < len(train_gen_lengths):
                # Filter to only non-outliers
                filtered_metrics = train_normalized_metric_values[method][below_q_ids]
                filtered_gen_lengths_normalized = train_gen_lengths_normalized

                # Adaptive binning using KMeans
                n_bins = 10
                est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
                bin_ids = est.fit_transform(filtered_gen_lengths_normalized.reshape(-1, 1)).astype(int).squeeze()

                sample_per_bin = quality_fit_sample_size // n_bins
                stratified_indices = []

                for bin_id in np.unique(bin_ids):
                    bin_indices = np.where(bin_ids == bin_id)[0]
                    n = min(sample_per_bin, len(bin_indices))
                    if n > 0:
                        stratified_indices.extend(rng.choice(bin_indices, size=n, replace=False))

                stratified_indices = np.array(stratified_indices)

                quality_reg = sklearn.linear_model.LinearRegression()
                quality_reg.fit(
                    filtered_gen_lengths_normalized[stratified_indices, np.newaxis],
                    filtered_metrics[stratified_indices]
                )

            else:
                quality_reg = sklearn.linear_model.LinearRegression()
                quality_reg.fit(
                    train_gen_lengths_normalized[:, np.newaxis],
                    train_normalized_metric_values[method][below_q_ids]
                )

            # Fit UE ~ length
            linreg = sklearn.linear_model.LinearRegression()
            linreg.fit(train_gen_lengths_normalized[:, np.newaxis], train_normalized_ue_values[method])
            ue_slope = linreg.coef_[0]
            ue_coefs[method].append(ue_slope)

            predicted_quality_trend = quality_reg.predict(test_gen_lengths_normalized[:, np.newaxis])
            # Predict UE trend on test
            predicted_ue_trend = linreg.predict(test_gen_lengths_normalized[:, np.newaxis])

            adjusted_ue = test_normalized_ue_values[method] - predicted_ue_trend - predicted_quality_trend

            residual_reg = sklearn.linear_model.LinearRegression()
            residual_reg.fit(test_gen_lengths_normalized[:, np.newaxis], adjusted_ue)
            ue_coefs[method].append(residual_reg.coef_[0])

            met_vals = test_metric_values[metric]
            raw_score = score_ues(test_ue_values[method], met_vals)
            raw_norm_score = score_ues(test_normalized_ue_values[method], met_vals)
            detrended_score = score_ues(adjusted_ue, met_vals)
            ue_scores[f'{method}_raw'].append(raw_score)
            ue_scores[f'{method}_detr'].append(detrended_score)

    return ue_scores, ue_coefs, ave_test_metric_values


def detrend_ue_degreed(
    datasets,
    model,
    all_metrics,
    ue_methods,
    methods_dict):
   
    ue_scores = defaultdict(list)   # keys: f'{method}_raw', f'{method}_deg1/2/3'
    ue_coefs  = defaultdict(list)   # keys: f'{method}_deg1/2/3' -> list of coef arrays
    ave_test_metric_values = {}

    # Align metrics to datasets
    if len(all_metrics) == 1 and len(datasets) > 1:
        all_metrics = all_metrics * len(datasets)
    elif len(all_metrics) != len(datasets):
        raise ValueError('Number of metrics and datasets must be the same')

    for metric, dataset in zip(all_metrics, datasets):
        (train_ue_values,
         test_ue_values,
         train_metric_values,
         test_metric_values,
         train_gen_lengths,
         gen_lengths) = extract_and_prepare_data(
            dataset, methods_dict, [metric],
            model=model
        )

        ave_test_metric_values[dataset] = float(np.mean(test_metric_values[metric]))

        # Trim extreme lengths (5th–95th percentile) on TRAIN only
        upper_q = np.quantile(train_gen_lengths, 0.95)
        lower_q = np.quantile(train_gen_lengths, 0.05)
        keep_ids = (train_gen_lengths < upper_q) & (train_gen_lengths > lower_q)

        train_gen_lengths = train_gen_lengths[keep_ids]
        for method in ue_methods:
            train_ue_values[method] = train_ue_values[method][keep_ids]

        len_scaler = MinMaxScaler()
        train_gl_norm = len_scaler.fit_transform(train_gen_lengths[:, None]).squeeze()
        test_gl_norm  = len_scaler.transform(gen_lengths[:, None]).squeeze()

        # Prepare polynomial feature generators for degrees 1..3
        poly_by_deg = {d: PolynomialFeatures(degree=d, include_bias=False) for d in (1, 2, 3)}
        train_feats_by_deg = {d: poly.fit_transform(train_gl_norm[:, None]) for d, poly in poly_by_deg.items()}
        test_feats_by_deg  = {d: poly_by_deg[d].transform(test_gl_norm[:, None]) for d in (1, 2, 3)}

        for method in ue_methods:
            ue_scaler = MinMaxScaler()
            train_ue_norm = ue_scaler.fit_transform(train_ue_values[method][:, None]).squeeze()
            test_ue_norm  = ue_scaler.transform(test_ue_values[method][:, None]).squeeze()

            met_vals = test_metric_values[metric]
            raw_score = score_ues(test_ue_values[method], met_vals)
            ue_scores[f'{method}_raw'].append(float(raw_score))

            for d in (1, 2, 3):
                linreg = sklearn.linear_model.LinearRegression()
                linreg.fit(train_feats_by_deg[d], train_ue_norm)
                ue_coefs[f'{method}_deg{d}'].append(linreg.coef_.copy())

                # Residuals on TEST (using normalized UE)
                pred_norm = linreg.predict(test_feats_by_deg[d])
                residuals = test_ue_norm - pred_norm

                # Score residuals vs metric
                detr_score = score_ues(residuals, met_vals)
                ue_scores[f'{method}_deg{d}'].append(float(detr_score))

    return ue_scores, ue_coefs, ave_test_metric_values

