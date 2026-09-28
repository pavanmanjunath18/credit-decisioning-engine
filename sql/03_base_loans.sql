-- Modeling base table: one row per resolved, fully matured 36-month loan.
--
-- Scope (see sql/01_maturity_check.sql for the evidence):
--   * loan_status exactly 'Fully Paid' or 'Charged Off'
--   * 36-month term, issued 2007-2015 (every such loan reached maturity by Dec-2018, before the
--     data snapshot: payments are recorded through Mar-2019)
--
-- Columns are split into three groups:
--   1. keys / time          -> used for the time-based split only
--   2. application features -> the only columns a model may use
--   3. outcomes             -> used only to measure results
--
-- Profit (sql/02_payment_check.sql shows total_pymnt already includes recoveries):
--   net_profit = total_pymnt - collection_recovery_fee - funded_amnt
--
-- Capital tied up, for an optional cost-of-funds charge (cost = annual_rate * dollar_years_outstanding):
--   months paying        = issue month -> last payment month (0 if no payment was ever made)
--   months after default = 5 for charged-off loans (LendingClub generally charges off at 150 days past due)
--   average balance      = funded - principal repaid / 2 while paying (straight-line approximation),
--                          then funded - principal repaid until charge-off
WITH base AS (
    SELECT
        *,
        strptime(issue_d, '%b-%Y')                          AS issue_date,
        try_strptime(earliest_cr_line, '%b-%Y')             AS earliest_cr_date,
        CAST(trim(replace(term, 'months', '')) AS INTEGER)  AS term_months,
        coalesce(date_diff('month', strptime(issue_d, '%b-%Y'), try_strptime(last_pymnt_d, '%b-%Y')), 0)
                                                            AS months_paying,
        CASE WHEN loan_status = 'Charged Off' THEN 5 ELSE 0 END AS months_after_default
    FROM raw_loans
    WHERE loan_status IN ('Fully Paid', 'Charged Off')
)
SELECT
    -- 1. keys / time
    id,
    issue_date,
    year(issue_date)                                         AS issue_year,

    -- 2. application-time features
    loan_amnt,
    term_months,
    int_rate,
    installment,
    grade,
    sub_grade,
    -- A1 = 1 ... G5 = 35: ordinal version of LendingClub's sub-grade
    (ascii(grade) - ascii('A')) * 5 + CAST(right(sub_grade, 1) AS INTEGER) AS sub_grade_num,
    CASE
        WHEN emp_length IS NULL OR emp_length = 'n/a' THEN NULL
        WHEN emp_length = '< 1 year'  THEN 0
        WHEN emp_length = '10+ years' THEN 10
        ELSE CAST(regexp_extract(emp_length, '(\d+)', 1) AS INTEGER)
    END                                                      AS emp_length_yrs,
    CASE WHEN home_ownership IN ('MORTGAGE', 'RENT', 'OWN') THEN home_ownership
         ELSE 'OTHER' END                                    AS home_ownership,
    annual_inc,
    verification_status,
    purpose,
    addr_state,
    left(zip_code, 3)                                        AS zip3,
    dti,
    delinq_2yrs,
    date_diff('month', earliest_cr_date, issue_date)         AS credit_history_months,
    fico_range_low,
    fico_range_high,
    (fico_range_low + fico_range_high) / 2.0                 AS fico_mid,
    inq_last_6mths,
    open_acc,
    pub_rec,
    revol_bal,
    revol_util,
    total_acc,
    mort_acc,
    pub_rec_bankruptcies,
    -- ratios built only from application fields
    loan_amnt / nullif(annual_inc, 0)                        AS loan_to_income,
    installment * 12 / nullif(annual_inc, 0)                 AS payment_to_income,

    -- 3. outcomes (never model inputs)
    loan_status,
    CASE WHEN loan_status = 'Charged Off' THEN 1 ELSE 0 END  AS is_default,
    funded_amnt,
    total_pymnt,
    recoveries,
    collection_recovery_fee,
    total_pymnt - collection_recovery_fee - funded_amnt      AS net_profit,
    months_paying + months_after_default                     AS months_outstanding,
    (months_paying * (funded_amnt - total_rec_prncp / 2)
     + months_after_default * (funded_amnt - total_rec_prncp)) / 12.0 AS dollar_years_outstanding
FROM base
WHERE term_months = 36
  AND year(issue_date) <= 2015
ORDER BY issue_date, id;
