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
public class BudgetStatusResponse {
    private Long budgetId;
    private Long categoryId;
    private String categoryName;
    private BigDecimal monthlyLimit;
    private BigDecimal amountSpent;
    private BigDecimal remaining;
    private double percentageUsed;
    private boolean exceeded;
    private Integer month;
    private Integer year;
}
