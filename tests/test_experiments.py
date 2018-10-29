# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import mock
import numpy as np

# NOTE: The metrics are not imported here b/c they are evaluated and require a
# spark context, which isn't available outside the test functions.
from mozanalysis.experiments import ExperimentAnalysis


def test_chaining():
    dataset = mock.Mock()
    dataset._sc = "sc"
    exp = (
        ExperimentAnalysis(dataset)
        .metrics("metric1", "metric2")
        .date_aggregate_by("other_date_column")
        .split_by("split")
    )
    assert exp._date_aggregate_by == "other_date_column"
    assert exp._aggregate_by == "client_id"  # The default.
    assert exp._metrics == ("metric1", "metric2")
    assert exp._split_by == "split"


def _generate_data(spark, client_branches):
    """
    Given a dict of clients->branches, generate predictive data for testing.
    """
    submission_date = "2018010%d"  # Fill in date with digits 1-9.

    incrementer = 1
    data_rows = []
    for client in sorted(client_branches.keys()):
        for branch in sorted(client_branches.values()):
            for date in range(1, 10):
                data_rows.append(
                    [
                        client,
                        submission_date % date,
                        "mozanalysis-tests",  # experiment_id
                        client_branches[client],  # experiment_branch
                        2 * incrementer,  # URI counts
                        100 * incrementer,  # subsession length
                        150 * incrementer,  # active ticks
                    ]
                )
                incrementer += 1

    return spark.createDataFrame(
        data_rows,
        [
            "client_id",
            "submission_date_s3",
            "experiment_id",
            "experiment_branch",
            "scalar_parent_browser_engagement_total_uri_count",
            "subsession_length",
            "active_ticks",
        ],
    )


def test_engagement_metrics(spark):
    # Testing all engagement metrics in one pass to reduce amount Spark testing time.
    from mozanalysis.metrics import (
        EngagementAvgDailyHours,
        EngagementAvgDailyActiveHours,
        EngagementHourlyUris,
        EngagementIntensity,
    )

    # Only compute the means across all stats.
    metrics = [
        EngagementAvgDailyHours,
        EngagementAvgDailyActiveHours,
        EngagementHourlyUris,
        EngagementIntensity,
    ]
    # Only calculate the means to reduce bootstrap time during testing.
    for m in metrics:
        m.stats = [np.mean]

    df = _generate_data(spark, {"aaaa": "control", "bbbb": "variant"})
    pdf = ExperimentAnalysis(df).metrics(*metrics).run()

    # Get the control rows and "stat_value" columns.
    control_values = pdf[pdf.branch == "control"].to_dict()["stat_value"].values()
    variant_values = pdf[pdf.branch == "variant"].to_dict()["stat_value"].values()

    # engagement_daily_hours
    # Note: 4.75 is the sum of the subsession lengths / 3600 over 9 days.
    assert np.allclose(control_values[0], 4.75 / 9)
    # Note: 13.75 is the sum of the subsession lengths / 3600 over 9 days.
    assert np.allclose(variant_values[0], 13.75 / 9)

    # engagement_daily_active_hours
    # Note: 35.625 is the sum of the active ticks * 5 / 3600 over 9 days.
    assert np.allclose(control_values[1], 35.625 / 9)
    # Note: 103.125 is the sum of the active ticks * 5 / 3600 over 9 days.
    assert np.allclose(variant_values[1], 103.125 / 9)

    # engagement_uris_per_active_hour
    # Note: 342 is the sum of the URIs, 35.625 is the sum of the active hours.
    assert np.allclose(control_values[2], 342 / (1 / 3600.0 + 35.625))
    # Note: 990 is the sum of the URIs, 103.125 is the sum of the active hours.
    assert np.allclose(variant_values[2], 990 / (1 / 3600.0 + 103.125))

    # engagement_intensity
    # Note: 4.75 is the sum of the total hours, 35.625 is the sum of the active hours.
    assert np.allclose(control_values[3], 4.75 / (1 / 3600.0 + 35.625))
    # Note: 13.75 is the sum of the total hours, 103.125 is the sum of the active hours.
    assert np.allclose(variant_values[3], 13.75 / (1 / 3600.0 + 103.125))
