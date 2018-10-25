# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import pandas as pd
import pyspark.sql.functions as F

from mozanalysis.stats import bootstrap


class ExperimentAnalysis(object):
    def __init__(self, dataset):
        self.sc = dataset._sc
        self._dataset = dataset
        self._metrics = None
        self._aggregate_by = "client_id"
        self._split_by = "experiment_branch"

    def metrics(self, *metrics):
        self._metrics = metrics
        return self

    def aggregate_by(self, column):
        self._aggregate_by = column
        return self

    def split_by(self, column):
        self._split_by = column
        return self

    def run(self):
        """
        Run the full analysis on the provided dataset.

        This will perform the per-client per-day aggregation, as well as the
        analysis aggregations, across all metrics.

        """
        df = self.aggregate_per_client_daily(self._dataset)
        return self.analyze(df)

    def aggregate_per_client_daily(self, dataset):
        cols = set([self._aggregate_by, self._split_by, "submission_date_s3"])
        aggs = set()

        for m in self._metrics:
            cols.update(m.daily_columns)
            aggs.update(m.daily_aggregations)

        df = (
            dataset.select(*cols)
            .groupBy(self._aggregate_by, self._split_by, "submission_date_s3")
            .agg(*aggs)
        )
        return df

    def analyze(self, dataset):
        data = []

        splits = [
            r[self._split_by]
            for r in dataset.select(self._split_by).distinct().collect()
        ]

        for m in self._metrics:
            # TODO: Use a lib or make this more robust.
            metric_name = m.name.replace(" ", "_").lower()

            agg_df = (
                dataset.select(*([self._aggregate_by, self._split_by] + m.columns))
                .groupBy(self._aggregate_by, self._split_by)
                .agg(*m.aggregations)
                .select(
                    *[
                        self._aggregate_by,
                        self._split_by,
                        m.final_expression.alias(metric_name),
                    ]
                )
            )
            for split in splits:
                for stat in m.stats:
                    bs = bootstrap(
                        self.sc,
                        agg_df.filter(F.col(self._split_by) == split)
                        .select(metric_name)
                        .collect(),
                        stat,
                    )
                    data.append(
                        {
                            "branch": split,
                            "metric_name": metric_name,
                            "stat_name": stat.func_name,
                            "stat_value": bs["calculated_value"],
                            "ci_low": bs["confidence_low"],
                            "ci_high": bs["confidence_high"],
                        }
                    )

        return pd.DataFrame(
            data,
            columns=[
                "branch",
                "metric_name",
                "stat_name",
                "stat_value",
                "ci_low",
                "ci_high",
            ],
        )
