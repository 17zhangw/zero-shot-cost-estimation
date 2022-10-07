#!/bin/bash

set -ex

#python3 run_benchmark.py --run_workload --source data/raw/tpcc_sf100_ff50_128MB_0 --target data/raw_plans/tpcc/tpcc_sf100_ff50_128MB_0.json --database postgres_csv
#python3 run_benchmark.py --run_workload --source data/raw/tpcc_sf1_ff10_1hr --target data/raw_plans/tpcc/tpcc_sf1_ff10.json --database postgres_csv
#python3 run_benchmark.py --run_workload --source data/raw/tpcc_sf1_ff50_1hr --target data/raw_plans/tpcc/tpcc_sf1_ff50.json --database postgres_csv
#python3 run_benchmark.py --run_workload --source data/raw/tpcc_sf1_ff100_1hr --target data/raw_plans/tpcc/tpcc_sf1_ff100.json --database postgres_csv

#python3 dbgensetup.py --parse_all_queries --raw_dir data/raw_plans --parsed_plan_dir data/parsed_plans --workloads tpcc_sf100_ff50_128MB_0.json --target_stats_path data/statistics/tpcc_sf100_ff50_128MB_0_stats.csv --min_query_ms 0 --cap_queries 1000000000000 --include_zero_card
#python3 dbgensetup.py --parse_all_queries --raw_dir data/raw_plans --parsed_plan_dir data/parsed_plans --workloads tpcc_sf1_ff10.json --target_stats_path data/statistics/tpcc_sf1_ff10_stats.csv --min_query_ms 0 --cap_queries 1000000000000 --include_zero_card
#python3 dbgensetup.py --parse_all_queries --raw_dir data/raw_plans --parsed_plan_dir data/parsed_plans --workloads tpcc_sf1_ff50.json --target_stats_path data/statistics/tpcc_sf1_ff50_stats.csv --min_query_ms 0 --cap_queries 1000000000000 --include_zero_card
#python3 dbgensetup.py --parse_all_queries --raw_dir data/raw_plans --parsed_plan_dir data/parsed_plans --workloads tpcc_sf1_ff100.json --target_stats_path data/statistics/tpcc_sf1_ff100_stats.csv --min_query_ms 0 --cap_queries 1000000000000 --include_zero_card
#
#python3 train.py --gather_feature_statistics --workload_runs tpcc_sf100_ff50_128MB_0.json --raw_dir data/parsed_plans --target data/parsed_plans/statistics_128MB_bp.json

#python3 train.py --train_model --workload_runs data/parsed_plans/tpcc/tpcc_sf100_ff50_128MB_0.json \
#                               --statistics_file data/parsed_plans/statistics_128MB_bp.json \
#                               --target evaluation/tpcc_bp \
#                               --hyperparameter_path setup/tuned_hyperparameters/tune_best_config.json \
#                               --max_epoch_tuples 100000 --loss_class_name QLoss --device cpu \
#                               --filename_model tpcc_sf100_128MB_bp \
#                               --num_workers 16 \
#                               --database postgres \
#                               --seed 12122022 \
#                               --device cpu
#
python3 train.py --train_model --workload_runs data/parsed_plans/tpcc/tpcc_sf100_ff50_128MB_0.json \
                               --statistics_file data/parsed_plans/statistics_128MB_bp.json \
                               --target evaluation/tpcc_bp_est \
                               --hyperparameter_path setup/tuned_hyperparameters/tune_est_best_config.json \
                               --max_epoch_tuples 100000 --loss_class_name QLoss --device cpu \
                               --filename_model tpcc_sf100_128MB_bp \
                               --num_workers 16 \
                               --database postgres \
                               --seed 12122022 \
                               --device cpu

#python3 run_benchmark.py --run_workload --source data/raw/tpcc_sf100_ff50_128MB_1 --target data/raw_plans/tpcc/tpcc_sf100_ff50_128MB_1.json --database postgres_csv
#python3 run_benchmark.py --run_workload --source data/raw/tpcc_sf1_ff_part10 --target data/raw_plans/tpcc/tpcc_sf1_ff_part10.json --database postgres_csv
#python3 run_benchmark.py --run_workload --source data/raw/tpcc_sf1_ff_part50 --target data/raw_plans/tpcc/tpcc_sf1_ff_part50.json --database postgres_csv
#python3 run_benchmark.py --run_workload --source data/raw/tpcc_sf1_ff_part100 --target data/raw_plans/tpcc/tpcc_sf1_ff_part100.json --database postgres_csv

#python3 dbgensetup.py --parse_all_queries --raw_dir data/raw_plans --parsed_plan_dir data/parsed_plans --workloads tpcc_sf100_ff50_128MB_1.json --target_stats_path data/statistics/tpcc_sf100_ff50_128MB_1_stats.csv --min_query_ms 0 --cap_queries 1000000000000 --include_zero_card
#python3 dbgensetup.py --parse_all_queries --raw_dir data/raw_plans --parsed_plan_dir data/parsed_plans --workloads tpcc_sf1_ff_part10.json --target_stats_path data/statistics/tpcc_sf1_ff_part10_stats.csv --min_query_ms 0 --cap_queries 1000000000000 --include_zero_card
#python3 dbgensetup.py --parse_all_queries --raw_dir data/raw_plans --parsed_plan_dir data/parsed_plans --workloads tpcc_sf1_ff_part50.json --target_stats_path data/statistics/tpcc_sf1_ff_part50_stats.csv --min_query_ms 0 --cap_queries 1000000000000 --include_zero_card
#python3 dbgensetup.py --parse_all_queries --raw_dir data/raw_plans --parsed_plan_dir data/parsed_plans --workloads tpcc_sf1_ff_part100.json --target_stats_path data/statistics/tpcc_sf1_ff_part100_stats.csv --min_query_ms 0 --cap_queries 1000000000000 --include_zero_card

#python3 train.py --gather_feature_statistics --workload_runs tpcc_sf100_ff50_128MB_1.json --raw_dir data/parsed_plans --target data/parsed_plans/test_statistics_128MB_bp.json

python3 train.py --train_model --target evaluation/tpcc_bp_est \
                               --workload_runs data/parsed_plans/tpcc/tpcc_sf1_ff_part50.json \
                               --statistics_file data/parsed_plans/test_statistics_128MB_bp.json \
                               --hyperparameter_path setup/tuned_hyperparameters/tune_est_best_config.json \
                               --test_workload_runs data/parsed_plans/tpcc/tpcc_sf100_ff50_128MB_1.json \
                               --max_epoch_tuples 100000 --loss_class_name QLoss --device cpu \
                               --filename_model tpcc_sf100_128MB_bp \
                               --num_workers 16 \
                               --database postgres \
                               --seed 12122022 \
                               --device cpu \
                               --skip_train
