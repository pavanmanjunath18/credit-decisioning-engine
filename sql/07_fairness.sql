-- Fairness check on 2015 decisions, by income band, home ownership, and state.
-- Input table `decisions`: one row per 2015 loan with the approve flag for an operating point.
--
-- For each group:
--   approval_rate          share of the group approved
--   approval_ratio         approval_rate / highest group approval_rate in that dimension
--                          (the "four-fifths rule": below 0.80 is a conventional red flag)
--   default_rate_approved  actual default rate among the approved - are approved groups equally safe?
--   mean_pd_approved       what the model predicted for them - under/over-prediction by group
--
-- No protected attributes (race, sex, age...) exist in this data and none are inferred.
WITH banded AS (
    SELECT
        *,
        CASE
            WHEN annual_inc < 40000  THEN '1: <$40K'
            WHEN annual_inc < 60000  THEN '2: $40-60K'
            WHEN annual_inc < 80000  THEN '3: $60-80K'
            WHEN annual_inc < 100000 THEN '4: $80-100K'
            WHEN annual_inc < 150000 THEN '5: $100-150K'
            ELSE '6: $150K+'
        END AS income_band
    FROM decisions
),
long AS (
    SELECT operating_point, 'income_band'    AS dimension, income_band    AS grp, approved, is_default, pd FROM banded
    UNION ALL
    SELECT operating_point, 'home_ownership' AS dimension, home_ownership AS grp, approved, is_default, pd FROM banded
    UNION ALL
    SELECT operating_point, 'addr_state'     AS dimension, addr_state     AS grp, approved, is_default, pd FROM banded
),
by_group AS (
    SELECT
        operating_point,
        dimension,
        grp,
        count(*)                                                  AS n_applicants,
        avg(CASE WHEN approved THEN 1.0 ELSE 0.0 END)             AS approval_rate,
        avg(is_default)                                           AS default_rate_all,
        avg(CASE WHEN approved THEN is_default END)               AS default_rate_approved,
        avg(CASE WHEN approved THEN pd END)                       AS mean_pd_approved
    FROM long
    GROUP BY operating_point, dimension, grp
)
SELECT
    *,
    -- The reference ("best") group must be large enough to be meaningful: a group of 1 applicant approved
    -- at 100% (home_ownership = OTHER) would otherwise become the benchmark for everyone else.
    approval_rate / max(CASE WHEN n_applicants >= 1000 THEN approval_rate END)
                    OVER (PARTITION BY operating_point, dimension)                   AS approval_ratio,
    default_rate_approved - mean_pd_approved                                           AS underprediction_pp,
    n_applicants < 1000                                                                AS small_group
FROM by_group
ORDER BY operating_point, dimension, grp;
