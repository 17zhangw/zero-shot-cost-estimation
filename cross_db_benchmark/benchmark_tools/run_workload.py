from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.postgres.run_workload import run_pg_workload
from cross_db_benchmark.benchmark_tools.postgres_csv.run_workload import extract_pg_csv_workload


def run_workload(workload_path, database, db_name, database_conn_args, database_kwarg_dict, target_path, run_kwargs,
                 repetitions_per_query, timeout_sec, hints=None, with_indexes=False, cap_workload=None,
                 min_runtime=100):
    if database == DatabaseSystem.POSTGRES:
        run_pg_workload(workload_path, database, db_name, database_conn_args, database_kwarg_dict, target_path,
                        run_kwargs, repetitions_per_query, timeout_sec, hints=hints, with_indexes=with_indexes,
                        cap_workload=cap_workload, min_runtime=min_runtime)
    elif database == DatabaseSystem.POSTGRES_CSV:
        extract_pg_csv_workload(workload_path, database, target_path, run_kwargs)
    else:
        raise NotImplementedError
