-- Survivorship check: share of loans still unfinished at the data snapshot (payments recorded through
-- Mar-2019), by issue year and term.
-- "Finished" = a terminal status (paid off or charged off). Anything else (Current, Late,
-- In Grace Period, Default) had not resolved by the end of the data.
-- Also shows the default rate among only the finished loans, which is biased for young vintages
-- because the loans that finish early are disproportionately early payoffs and early defaults.
WITH loans AS (
    SELECT
        year(strptime(issue_d, '%b-%Y'))            AS issue_year,
        CAST(trim(replace(term, 'months', '')) AS INTEGER) AS term_months,
        loan_status,
        loan_status IN (
            'Fully Paid', 'Charged Off',
            'Does not meet the credit policy. Status:Fully Paid',
            'Does not meet the credit policy. Status:Charged Off'
        ) AS is_finished,
        loan_status LIKE '%Charged Off' AS is_charged_off
    FROM raw_loans
)
SELECT
    issue_year,
    term_months,
    count(*)                                              AS n_loans,
    sum(CASE WHEN NOT is_finished THEN 1 ELSE 0 END)      AS n_unfinished,
    round(avg(CASE WHEN NOT is_finished THEN 1.0 ELSE 0.0 END), 4) AS share_unfinished,
    round(sum(CASE WHEN is_charged_off THEN 1.0 ELSE 0.0 END)
          / nullif(sum(CASE WHEN is_finished THEN 1 ELSE 0 END), 0), 4) AS default_rate_among_finished
FROM loans
GROUP BY issue_year, term_months
ORDER BY term_months, issue_year;
