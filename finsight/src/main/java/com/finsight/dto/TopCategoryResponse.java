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
public class TopCategoryResponse {
    private String categoryName;
    private BigDecimal totalAmount;
    private double percentage;
}
