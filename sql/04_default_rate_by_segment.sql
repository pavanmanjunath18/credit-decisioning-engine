-- Default rate and average net profit by segment, in one long table:
-- (dimension, bucket, sort_key, n_loans, default_rate, avg_net_profit)
WITH b AS (
    SELECT
        *,
        CASE
            WHEN fico_mid < 680 THEN '<680'
            WHEN fico_mid < 700 THEN '680-699'
            WHEN fico_mid < 720 THEN '700-719'
            WHEN fico_mid < 740 THEN '720-739'
            WHEN fico_mid < 760 THEN '740-759'
            ELSE '760+'
        END AS fico_band,
        CASE
            WHEN dti IS NULL THEN 'missing'
            WHEN dti < 5  THEN '00-05'
            WHEN dti < 10 THEN '05-10'
            WHEN dti < 15 THEN '10-15'
            WHEN dti < 20 THEN '15-20'
            WHEN dti < 25 THEN '20-25'
            WHEN dti < 30 THEN '25-30'
            ELSE '30+'
        END AS dti_band
    FROM loans
)
SELECT 'issue_year' AS dimension, CAST(issue_year AS VARCHAR) AS bucket, issue_year AS sort_key,
       count(*) AS n_loans, avg(is_default) AS default_rate, avg(net_profit) AS avg_net_profit
FROM b GROUP BY issue_year
UNION ALL
SELECT 'grade', grade, ascii(grade), count(*), avg(is_default), avg(net_profit)
FROM b GROUP BY grade
UNION ALL
SELECT 'fico_band', fico_band, min(fico_mid), count(*), avg(is_default), avg(net_profit)
FROM b GROUP BY fico_band
UNION ALL
SELECT 'dti_band', dti_band, coalesce(min(dti), 999), count(*), avg(is_default), avg(net_profit)
FROM b GROUP BY dti_band
UNION ALL
SELECT 'purpose', purpose, -count(*), count(*), avg(is_default), avg(net_profit)
FROM b GROUP BY purpose
ORDER BY dimension, sort_key, bucket;
