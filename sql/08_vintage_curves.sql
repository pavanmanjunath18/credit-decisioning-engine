-- Vintage curves: cumulative share of each issue year's 36-month loans that had charged off by
-- N months after issue. Denominator = every loan issued that year, whatever its status today.
--
-- Charge-off month is approximated as last payment month + 5 (LendingClub generally charges off at
-- 150 days past due; the same assumption as the cost-of-funds calculation in 03_base_loans.sql).
-- Curves stop at the last month observable before the data snapshot (payments recorded through Mar-2019),
-- so young vintages are short -
-- which is exactly why only 2007-2015 vintages are used for modeling.
WITH loans AS (
    SELECT
        year(strptime(issue_d, '%b-%Y'))                            AS issue_year,
        strptime(issue_d, '%b-%Y')                                  AS issue_date,
        loan_status LIKE '%Charged Off'                             AS charged_off,
        coalesce(date_diff('month', strptime(issue_d, '%b-%Y'), try_strptime(last_pymnt_d, '%b-%Y')), 0) + 5
                                                                    AS chargeoff_month
    FROM raw_loans
    WHERE term LIKE '%36%'
),
cohort AS (
    SELECT issue_year,
           count(*) AS n_loans,
           -- months observable for the youngest loan in the vintage (issued in December)
           date_diff('month', max(issue_date), DATE '2019-03-01') AS months_observable
    FROM loans
    GROUP BY issue_year
),
months AS (SELECT range AS month FROM range(1, 61))
SELECT
    c.issue_year,
    m.month                                          AS months_since_issue,
    c.n_loans,
    sum(CASE WHEN l.charged_off AND l.chargeoff_month <= m.month THEN 1 ELSE 0 END) * 1.0 / c.n_loans
                                                     AS cumulative_default_rate
FROM cohort c
CROSS JOIN months m
JOIN loans l ON l.issue_year = c.issue_year
WHERE c.issue_year >= 2009
  AND m.month <= c.months_observable
GROUP BY c.issue_year, m.month, c.n_loans
ORDER BY c.issue_year, m.month;
