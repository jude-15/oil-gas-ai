WELL_INTELLIGENCE_QUERY = """
SELECT
    well_id,

    COUNT(*) AS total_readings,

    SUM(failure) AS failure_count,

    ROUND(
        (100.0 * SUM(failure) / COUNT(*))::numeric,
        2
    ) AS failure_rate_percent,

    ROUND(
        AVG(temperature_c)::numeric,
        2
    ) AS avg_temperature,

    ROUND(
        AVG(vibration_mm_s)::numeric,
        2
    ) AS avg_vibration,

    ROUND(
        AVG(production_rate_bbl_day)::numeric,
        2
    ) AS avg_production,

    CASE
        WHEN (100.0 * SUM(failure) / COUNT(*)) >= 2.5
            THEN 'HIGH_RISK'

        WHEN (100.0 * SUM(failure) / COUNT(*)) >= 1.5
            THEN 'MEDIUM_RISK'

        ELSE 'LOW_RISK'
    END AS risk_level

FROM industrial_sensor_data

GROUP BY well_id

ORDER BY failure_rate_percent DESC;
"""     