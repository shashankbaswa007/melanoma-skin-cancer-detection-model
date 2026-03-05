package com.finsight.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class MonthlySummaryResponse {
    private Integer month;
    private Integer year;
    private BigDecimal totalIncome;
    private BigDecimal totalExpense;
    private BigDecimal netSavings;
    private double savingsRate;
}
