from tqdm import tqdm
import math
import glob
import json
import os
import random
import re
import shutil
import time
from json.decoder import JSONDecodeError

from tqdm import tqdm
import pandas as pd
import numpy as np

from cross_db_benchmark.benchmark_tools.load_database import create_db_conn
from cross_db_benchmark.benchmark_tools.postgres.check_valid import check_valid
from cross_db_benchmark.benchmark_tools.utils import load_json

def transform_dicts(column_stats_names, column_stats_rows):
    return [{k: v for k, v in zip(column_stats_names, row)} for row in column_stats_rows]


def collect_db_statistics(workload_path):
    pg_stats_cols = ["tablename", "attname", "null_frac", "avg_width", "n_distinct", "correlation", "data_type"]
    pg_stats = pd.read_csv(f"{workload_path}/pg_stats.csv.0")
    pg_stats.rename(columns={"extract": "time"}, inplace=True)
    pg_stats = pg_stats[pg_stats.time == pg_stats.iloc[0].time]
    pg_stats = pg_stats[pg_stats_cols]
    pg_stats_rows = pg_stats.values.tolist()
    column_stats = transform_dicts(pg_stats_cols, pg_stats_rows)

    table_stats_cols = ["relname", "reltuples", "relpages", "fillfactor"]
    pg_class = pd.read_csv(f"{workload_path}/pg_class.csv.0")
    pg_class.rename(columns={"extract": "time"}, inplace=True)
    pg_class = pg_class[pg_class.time == pg_class.iloc[0].time]

    table_stats_rows = []
    for row in pg_class.itertuples():
        if row.reloptions is None or (isinstance(row.reloptions, float) and np.isnan(row.reloptions)):
            table_stats_rows.append((row.relname, row.reltuples, row.relpages, 100))
        else:
            ff = 100
            for reloption in row.reloptions:
                for key, value in re.findall(r'(\w+)=(\w*)', reloption):
                    if key == "fillfactor":
                        # Fix fillfactor options.
                        ff = int(value)
            table_stats_rows.append((row.relname, row.reltuples, row.relpages, ff))
    table_stats = transform_dicts(table_stats_cols, table_stats_rows)
    return dict(column_stats=column_stats, table_stats=table_stats)


def extract_pg_csv_workload(workload_path, database, target_path, run_kwargs):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # extract column statistics. we assume this "segment" is consistent.
    # FIXME: database stats can change over time. we just take this to be a given here.
    # The possible solution is to distinctly extract every "different" table as its own object.
    database_stats = collect_db_statistics(workload_path)

    col_names = [
        "log_time",
        "user_name",
        "database_name",
        "process_id",
        "connection_from",
        "session_id",
        "session_line_num",
        "command_tag",
        "session_start_time",
        "virtual_transaction_id",
        "transaction_id",
        "error_severity",
        "sql_state_code",
        "message",
        "detail",
        "hint",
        "internal_query",
        "internal_query_pos",
        "context",
        "query",
        "query_pos",
        "location",
        "application_name",
        "backend_type",
        "leader_pid",
        "query_id",
    ]

    query_list = []
    files = sorted(glob.glob(f"{workload_path}/log.0/*.csv"))
    actual_regex = re.compile('\(actual time=(?P<act_startup_cost>\d+.\d+)..(?P<act_time>\d+.\d+) rows=(?P<act_card>\d+)')
    for f in tqdm(files):
        it = pd.read_csv(f, names=col_names, chunksize=8192, usecols=["user_name", "command_tag", "message"])
        for chunk in tqdm(it):
            chunk = chunk[chunk.user_name == "admin"]
            cmd_tags = ["SELECT", "INSERT", "UPDATE", "DELETE", "BIND"]
            chunk = chunk[chunk.command_tag.isin(cmd_tags)]
            for t in chunk.itertuples():
                qs = t.message.split("\n")
                if len(qs) == 0:
                    continue

                if not qs[0].startswith("Query Text"):
                    continue

                q = [q for q in qs if q.startswith("elapsed_us")]
                if len(q) > 0:
                    elapsed_milli = float(q[0].split(": ")[1]) / 1000.0

                q = [q for q in qs if q.startswith("start_time")]
                if len(q) > 0:
                    start_time = q[0].split(": ")[1]

                q = [q for q in qs if q.startswith("query_id")]
                if len(q) > 0:
                    query_id = q[0].split(": ")[1]

                q = [q for q in qs if q.startswith("txn")]
                if len(q) > 0:
                    txn = q[0].split(": ")[1]

                QUERY_CONTENT_BLOCK = [
                    "pg_",
                    "version()",
                    "current_schema()",
                    "pgstattuple_approx",
                ]

                block = any([b in qs[0] for b in QUERY_CONTENT_BLOCK])
                if block:
                    continue

                # So adjust the "root"
                if "fold_runtime" in kwargs:
                    if qs[0].split(": ")[1].startswith("UPDATE"):
                        for i in range(len(qs)):
                            if qs[i].startswith("Update"):
                                # We assume this is the root...
                                match = actual_regex.search(qs[i])
                                if match is not None:
                                    start = match.start('act_time')
                                    end = match.end('act_time')
                                    qs[i] = qs[i][:start] + "{:.3f}".format(float(elapsed_milli)) + qs[i][end:]
                                break
                    elif qs[0].split(": ")[1].startswith("INSERT"):
                        for i in range(len(qs)):
                            if qs[i].startswith("Insert"):
                                # We assume this is the root...
                                match = actual_regex.search(qs[i])
                                if match is not None:
                                    start = match.start('act_time')
                                    end = match.end('act_time')
                                    qs[i] = qs[i][:start] + "{:.3f}".format(float(elapsed_milli)) + qs[i][end:]
                                break

                qs = [[q] for q in qs]
                curr_statistics = dict(analyze_plans=[qs.copy()], verbose_plan=qs)
                curr_statistics.update(sql=qs[0][len("Query Text"):])
                curr_statistics.update(query_id=query_id)
                curr_statistics.update(txn=txn)
                curr_statistics.update(start_time=start_time)
                query_list.append(curr_statistics)

    run_stats = dict(query_list=query_list, database_stats=database_stats, run_kwargs=run_kwargs)
    save_workload(run_stats, target_path)
    print(f"Extracted workload {workload_path}")


def save_workload(run_stats, target_path):
    target_temp_path = os.path.join(os.path.dirname(target_path), f'{os.path.basename(target_path)}_temp')
    with open(target_temp_path, 'w') as outfile:
        json.dump(run_stats, outfile)
    shutil.move(target_temp_path, target_path)
