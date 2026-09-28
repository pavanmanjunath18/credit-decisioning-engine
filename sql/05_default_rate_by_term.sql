-- Term comparison on matured loans only (the modeling table is 36-month only, so term is
-- compared here on the raw data). A loan is "matured" if issue date + term <= Dec-2018 (before the Mar-2019 data snapshot):
--   36-month loans issued through 2015, 60-month loans issued through 2013.
WITH t AS (
    SELECT
        CAST(trim(replace(term, 'months', '')) AS INTEGER) AS term_months,
        year(strptime(issue_d, '%b-%Y'))                    AS issue_year,
        loan_status
    FROM raw_loans
    WHERE loan_status IN ('Fully Paid', 'Charged Off')
)
SELECT
    term_months,
    min(issue_year) AS first_year,
    max(issue_year) AS last_year,
    count(*)        AS n_loans,
    avg(CASE WHEN loan_status = 'Charged Off' THEN 1.0 ELSE 0.0 END) AS default_rate
FROM t
WHERE (term_months = 36 AND issue_year <= 2015)
   OR (term_months = 60 AND issue_year <= 2013)
GROUP BY term_months
ORDER BY term_months;
