-- Does total_pymnt already include recoveries?
-- If total_pymnt = principal + interest + late fees + recoveries, then recoveries are already
-- inside total_pymnt and must NOT be added again when computing cash received.
WITH resolved AS (
    SELECT
        loan_status,
        total_pymnt,
        total_rec_prncp + total_rec_int + total_rec_late_fee               AS pymnt_ex_recoveries,
        total_rec_prncp + total_rec_int + total_rec_late_fee + recoveries  AS pymnt_with_recoveries,
        recoveries
    FROM raw_loans
    WHERE loan_status IN ('Fully Paid', 'Charged Off')
)
SELECT
    loan_status,
    count(*)                                                                    AS n_loans,
    sum(CASE WHEN recoveries > 0 THEN 1 ELSE 0 END)                             AS n_with_recoveries,
    round(avg(CASE WHEN abs(total_pymnt - pymnt_with_recoveries) < 0.05 THEN 1.0 ELSE 0.0 END), 4)
                                                                                AS share_match_incl_recoveries,
    round(avg(CASE WHEN abs(total_pymnt - pymnt_ex_recoveries)  < 0.05 THEN 1.0 ELSE 0.0 END), 4)
                                                                                AS share_match_excl_recoveries,
    -- Same test restricted to loans that actually have a recovery (where the two differ)
    round(avg(CASE WHEN recoveries > 0 AND abs(total_pymnt - pymnt_with_recoveries) < 0.05 THEN 1.0
                   WHEN recoveries > 0 THEN 0.0 END), 4)                        AS share_match_incl_when_recovery_gt0,
    round(avg(CASE WHEN recoveries > 0 AND abs(total_pymnt - pymnt_ex_recoveries) < 0.05 THEN 1.0
                   WHEN recoveries > 0 THEN 0.0 END), 4)                        AS share_match_excl_when_recovery_gt0
FROM resolved
GROUP BY loan_status
ORDER BY loan_status;
