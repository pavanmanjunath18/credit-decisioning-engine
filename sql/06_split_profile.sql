-- Profile each issue year so the time split can be judged on data, not assumption:
-- volume, default rate, and the borrower/pricing mix that a model would have to generalize across.
SELECT
    issue_year,
    count(*)                                            AS n_loans,
    avg(is_default)                                     AS default_rate,
    avg(CASE WHEN grade IN ('A', 'B') THEN 1.0 ELSE 0.0 END)      AS share_grade_ab,
    avg(CASE WHEN grade IN ('E', 'F', 'G') THEN 1.0 ELSE 0.0 END) AS share_grade_efg,
    avg(int_rate)                                       AS avg_int_rate,
    avg(fico_mid)                                       AS avg_fico,
    min(fico_mid)                                       AS min_fico,
    avg(dti)                                            AS avg_dti,
    median(annual_inc)                                  AS median_income,
    avg(loan_amnt)                                      AS avg_loan_amnt,
    avg(CASE WHEN purpose = 'debt_consolidation' THEN 1.0 ELSE 0.0 END) AS share_debt_consol,
    avg(CASE WHEN mort_acc IS NULL THEN 1.0 ELSE 0.0 END)          AS share_mort_acc_missing,
    avg(CASE WHEN emp_length_yrs IS NULL THEN 1.0 ELSE 0.0 END)    AS share_emp_missing,
    avg(net_profit)                                     AS avg_net_profit
FROM loans
GROUP BY issue_year
ORDER BY issue_year;
